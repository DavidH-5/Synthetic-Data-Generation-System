"""
agents/orchestrator.py

Orchestration Agent — the single entry point for the full pipeline.

Responsibilities:
  - Drive the end-to-end generation loop
  - Delegate to Planner, Code Writer, Validator via tool calls
  - Interpret failure signals and make replan decisions
  - Update the todo list via native tools
  - Declare generation complete when all tasks are done

Uses CodeMode so it can orchestrate multiple sub-agent calls in a single
round-trip using Python loops and conditionals inside the Monty sandbox.

Native tools (read_todo, write_status, etc.) remain outside the sandbox
so the orchestrator always has state visibility regardless of sandbox state.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine, TypeVar

from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from agents.code_writer import run_code_writer
from agents.planner import run_planner
from agents.validator import run_validator
from core.models import GenerationManifest, ReplanDecision, TaskStatus
from db.store import DataStore
from tools.native_tools import make_native_tools


T = TypeVar("T")
_executor = ThreadPoolExecutor(max_workers=4)


def _run_async(coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine from a sync tool when an event loop is already active."""
    future = _executor.submit(asyncio.run, coro)
    return future.result()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """
<role>
You are the orchestration agent for a synthetic data generation pipeline.
You own the full lifecycle: planning → code generation → validation → done.
You interpret failure signals from sub-agents and make routing decisions.
You never generate data or write code directly.
</role>

<tools>

  <native_tools description="Always available. Not sandboxed. Call these directly.">
    <tool name="read_todo_list()">
      Returns all tasks with their current status, attempt counts, and failure details.
      Call this at the start of each loop iteration to assess what to do next.
    </tool>
    <tool name="write_task_status(table_name, status, error?)">
      Update a task's status. Use for intermediate state updates only.
      Sub-agents update status themselves on success/failure — don't double-set.
    </tool>
    <tool name="record_replan_decision(decision_json)">
      Persist a ReplanDecision and apply its action to the todo list.
      Always call this when routing a failed task — creates the audit trail.
    </tool>
    <tool name="get_fk_context(table_name)">
      Returns {column_name: [pk_values]} for a table's FK columns.
      Available after parent tables are DONE.
    </tool>
    <tool name="get_manifest_summary()">
      Returns a compact view of the full manifest. Use for orientation.
    </tool>
    <tool name="export_summary()">
      Returns row counts for all DONE tables. Call at completion.
    </tool>
  </native_tools>

  <sandboxed_tools description="Run inside Monty sandbox. Can be batched with asyncio.gather.">
    <tool name="run_planner_tool(revised_hints?)">
      Runs the Planner Agent against the full manifest.
      Call once at the start, or again with revised_hints when a full replan is needed.
      Returns: "Planning complete: N tables planned" or "Planner error: ..."
    </tool>
    <tool name="run_code_writer_tool(table_name, revised_hints?)">
      Runs the Code Writer Agent for one table.
      Returns: "ok:N rows" or "failed:error_classification:self_fixable=True/False"
    </tool>
    <tool name="run_validator_tool(table_name)">
      Runs the Validation Agent for one table.
      Returns: "passed" or "failed:failure_classification:suggestions"
    </tool>
  </sandboxed_tools>

</tools>

<loop_strategy>
  <step n="1" name="orient">
    Call read_todo_list() and get_manifest_summary() to understand current state.
    If no tasks exist yet, run_planner_tool() first.
  </step>
  <step n="2" name="generate">
    Find all tasks with status="plan_ready" whose dependencies are all "done".
    Batch independent tables: use asyncio.gather([run_code_writer_tool(t) for t in ready])
    NEVER run a child table before all its dependencies are "done".
  </step>
  <step n="3" name="validate">
    For each table that just became "data_ready", run run_validator_tool(table_name).
    These can also be batched if they have no interdependencies.
  </step>
  <step n="4" name="handle_failures">
    For any task with status="failed", apply the failure routing rules below.
  </step>
  <step n="5" name="loop">
    Repeat steps 2–4 until all tasks are "done" or "unresolvable".
  </step>
  <step n="6" name="complete">
    Call export_summary() and report results.
  </step>
</loop_strategy>

<failure_routing>

  Read BOTH signals before deciding:
  - code_block.self_fixable and code_block.error_classification (from Code Writer)
  - validation_result.failure_classification and validation_result.suggestions (from Validator)

  <decision signal="code_bug" self_fixable="true" attempt_count_lt_3">
    action: retry_code
    Call run_code_writer_tool(table_name, revised_hints=validation_result.suggestions)
    The hints must include the specific error — not just "try again".
  </decision>

  <decision signal="code_bug" self_fixable="false">
    action: replan_table
    The code is a faithful implementation of a bad plan.
    Call record_replan_decision with action="replan_table" and clear reasoning.
    Then run_planner_tool() will be triggered for that table.
  </decision>

  <decision signal="plan_error">
    action: replan_table
    Call record_replan_decision with action="replan_table".
    Include the validation suggestions as revised_hints for the Planner.
  </decision>

  <decision signal="spec_conflict">
    action: mark_unresolvable
    Call record_replan_decision with action="mark_unresolvable".
    Document the exact conflict in the reasoning field.
    Do not retry. Surface to user at completion.
  </decision>

  <circuit_breakers>
    If attempt_count >= 3: escalate to replan_table regardless of self_fixable.
    If replan_count >= 2: escalate to mark_unresolvable regardless of failure type.
    These are hard limits — do not override them.
  </circuit_breakers>

</failure_routing>

<replan_decision_schema>
  When calling record_replan_decision, provide a JSON string with:
  {
    "table_name": "...",
    "action": "retry_code" | "replan_table" | "replan_full" | "mark_unresolvable",
    "reasoning": "specific explanation referencing the error and signals observed",
    "revised_hints": "concrete guidance for the next agent call (Planner or Code Writer)"
  }
</replan_decision_schema>
"""


# ---------------------------------------------------------------------------
# Sandboxed sub-agent dispatcher tools
# These wrap the async sub-agent entry points as synchronous tool calls
# that CodeMode can dispatch inside Monty.
# ---------------------------------------------------------------------------

def make_dispatcher_tools(store: DataStore, manifest: GenerationManifest):
    """
    Returns sandboxed tool functions that the orchestration agent calls
    from within its CodeMode sandbox to invoke sub-agents.
    """

    def run_planner_tool(revised_hints: str | None = None) -> str:
        """
        Run the Planner Agent against the full manifest.
        revised_hints: optional guidance when replanning.
        Returns: "ok" or error string.
        """
        try:
            plan = _run_async(run_planner(manifest, store, revised_hints))
            return f"Planning complete: {len(plan.tables)} tables planned"
        except Exception as e:
            return f"Planner error: {e}"

    def run_code_writer_tool(table_name: str, revised_hints: str | None = None) -> str:
        """
        Run the Code Writer Agent for one table.
        table_name: the table to generate.
        revised_hints: optional guidance for this specific table.
        Returns: "ok:{rows}" or "failed:{error_classification}"
        """
        try:
            success = _run_async(run_code_writer(table_name, store, revised_hints))
            task = store.get_task(table_name)
            if success:
                return f"ok:{task.table_plan.volume if task and task.table_plan else '?'} rows"
            else:
                ec = task.code_block.error_classification if task and task.code_block else "unknown"
                sf = task.code_block.self_fixable if task and task.code_block else True
                return f"failed:{ec}:self_fixable={sf}"
        except Exception as e:
            return f"error:{e}"

    def run_validator_tool(table_name: str) -> str:
        """
        Run the Validator Agent for one table.
        Returns: "passed" or "failed:{failure_classification}:{suggestions}"
        """
        try:
            result = _run_async(run_validator(table_name, store, manifest))
            if result is None:
                return "error:no validation result returned"
            if result.passed:
                return "passed"
            return f"failed:{result.failure_classification}:{result.suggestions or 'no suggestions'}"
        except Exception as e:
            return f"error:{e}"

    return [run_planner_tool, run_code_writer_tool, run_validator_tool]


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def make_orchestration_agent(store: DataStore, manifest: GenerationManifest) -> Agent:
    native_tools = make_native_tools(store, manifest)
    dispatcher_tools = make_dispatcher_tools(store, manifest)

    # Native tools stay outside CodeMode — always visible to the agent
    # Dispatcher tools go into the sandbox — can be batched
    agent = Agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        capabilities=[
            CodeMode(
                tools=lambda ctx, td: td.name in {
                    "run_planner_tool",
                    "run_code_writer_tool",
                    "run_validator_tool",
                },
                max_retries=3,
            )
        ],
    )

    # Register native tools (not sandboxed)
    for tool_fn in native_tools:
        agent.tool_plain(tool_fn)

    # Register dispatcher tools (sandboxed via CodeMode selector above)
    for tool_fn in dispatcher_tools:
        agent.tool_plain(tool_fn)

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_pipeline(manifest: GenerationManifest, store: DataStore) -> dict:
    """
    Run the full synthetic data generation pipeline.
    Returns a summary dict of {table_name: {status, rows}}.
    """
    agent = make_orchestration_agent(store, manifest)

    table_names = [t.name for t in manifest.tables]

    prompt = f"""
<task>
Run the full synthetic data generation pipeline.
Follow the loop_strategy and failure_routing rules from your instructions exactly.
</task>

<pipeline_scope>
  <tables>{table_names}</tables>
  <seed>{manifest.seed}</seed>
</pipeline_scope>

<initial_state_check>
Call read_todo_list() first.
- If the list is empty: call run_planner_tool() before anything else.
- If tasks already exist (resuming a previous run): skip planning and continue
  from the current task statuses.
</initial_state_check>

<completion_criteria>
All tasks must reach status "done" or "unresolvable" before reporting results.
Call export_summary() as the final step, then report:
- Which tables succeeded and their row counts.
- Which tables are unresolvable and why (cite the exact error and decision reasoning).
</completion_criteria>
"""

    result = await agent.run(prompt)
    return result.output if isinstance(result.output, dict) else {"summary": str(result.output)}
