"""
agents/code_writer.py

Code Writer Agent — generates a Python function for one table at a time.

Responsibilities:
  - Read a TablePlan from the todo list
  - Write a self-contained generate() function
  - Classify its own errors (self_fixable vs strategy_mismatch)
  - Execute the generated code in CodeMode (Monty sandbox)
  - Write resulting DataFrame to SQLite via the store

The generated function always has this exact signature:
    def generate(n: int, seed: int, fk_context: dict[str, list]) -> pd.DataFrame

CodeMode is used here so the agent can:
  - Write the generate() function
  - Call execute_code() to run it inside Monty
  - Call write_dataframe() to persist the result
  - All in a single model round-trip
"""

from __future__ import annotations

import textwrap

import pandas as pd
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from core.models import CodeBlock, TablePlan, TaskStatus
from db.store import DataStore


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CODE_WRITER_SYSTEM_PROMPT = """
<role>
You are a Python data generation engineer. You write clean, correct, reproducible
Python functions that generate synthetic tabular data.
Allowed libraries: faker, pandas, numpy, random, uuid, datetime.
Nothing else.
</role>

<function_contract>
Every function you write MUST match this exact signature:

  def generate(n: int, seed: int, fk_context: dict[str, list]) -> pd.DataFrame:
      ...

<parameters>
  <param name="n">Number of rows to generate.</param>
  <param name="seed">Integer seed. Apply to ALL random sources for reproducibility.</param>
  <param name="fk_context">
    Dict of {column_name: [pk_values]} for FK columns.
    Example: fk_context["customer_id"] contains all valid customer IDs to sample from.
    If a column uses fk_sample strategy, its values MUST come from this dict.
  </param>
</parameters>

<return>
  A pandas DataFrame with EXACTLY the columns listed in the TablePlan.
  Column names must match the plan exactly (case-sensitive).
  No extra columns. No missing columns.
</return>
</function_contract>

<allowed_imports>
  import pandas as pd
  import numpy as np
  from faker import Faker
  import random
  import uuid
  from datetime import date, datetime, timedelta

  ANY other import is a sandbox violation and will fail immediately.
</allowed_imports>

<implementation_rules>

  <rule name="seeding">
    Apply seed to EVERY random source at the top of the function:
      np.random.seed(seed)
      random.seed(seed)
      fake = Faker(); fake.seed_instance(seed)
    Never skip seeding — it breaks reproducibility.
  </rule>

  <rule name="strategy_faker">
    Use the faker_provider from the plan.
    Generate as a list comprehension: [getattr(fake, provider)() for _ in range(n)]
    For fake.unique providers: [fake.unique.email() for _ in range(n)]
  </rule>

  <rule name="strategy_numpy">
    Use the numpy_call expression from the plan verbatim.
    Always clip to min_value/max_value if specified in the plan.
    Round floats to 2 decimal places unless plan specifies otherwise.
  </rule>

  <rule name="strategy_fk_sample">
    Always use random.choices(fk_context["column_name"], k=n)
    NEVER hardcode PK values. NEVER generate new UUIDs for FK columns.
    If fk_context["column_name"] is empty, raise ValueError with a clear message.
  </rule>

  <rule name="strategy_pandas_choice">
    For columns with distribution.type="choice", use:
      random.choices(values, weights=weights, k=n)
    where values and weights come from the plan's distribution spec.
  </rule>

  <rule name="nullable_columns">
    For columns with nullable > 0.0:
      mask = np.random.random(n) < nullable_fraction
      series = pd.array(values, dtype="object")
      series[mask] = None
    Apply AFTER generating the column values, not before.
  </rule>

  <rule name="unique_columns">
    For unique=True columns using faker:
      Use fake.unique.<provider>() inside the list comprehension.
    For unique=True columns using numpy:
      Generate extra values (n * 1.5) and take the first n unique ones.
    Always verify len(set(values)) == n before returning the DataFrame.
  </rule>

</implementation_rules>

<workflow>
  <step n="1">Call get_table_plan(table_name) to read the full plan as JSON.</step>
  <step n="2">Write the generate() function following all rules above.</step>
  <step n="3">
    Call execute_code(table_name, code, n, seed, fk_context_json).
    fk_context_json must be a valid JSON string of the fk_context dict.
  </step>
  <step n="4">
    If execute_code returns success=False:
    - Read the error message carefully.
    - Fix the specific issue — do not rewrite the whole function unnecessarily.
    - Retry up to 3 times total.
  </step>
  <step n="5">If execute_code returns success=True: call write_result(table_name).</step>
  <step n="6">
    If all retries exhausted and still failing:
    Call report_code_error() with an accurate error_classification:
      - "syntax_error": Python syntax is invalid
      - "runtime_error": valid syntax but fails at runtime (e.g. KeyError, TypeError)
      - "strategy_mismatch": the plan's strategy is incompatible with the column spec
      - "constraint_conflict": the spec constraints are mutually incompatible
    Set self_fixable=False ONLY if the problem is in the plan, not the code.
  </step>
</workflow>

<error_classification_guide>
  <when_self_fixable_true>
    - SyntaxError you introduced
    - Wrong variable name or typo
    - Off-by-one in array size
    - Missing .tolist() or .values conversion
    - Incorrect dtype cast you can fix
  </when_self_fixable_true>
  <when_self_fixable_false>
    - fk_context["col"] is empty (parent table not generated yet — plan order wrong)
    - Plan specifies a faker provider that does not exist
    - Distribution parameters produce values outside the column's min/max range
      AND no clipping can resolve it
    - unique=True but volume exceeds the provider's cardinality
  </when_self_fixable_false>
</error_classification_guide>
"""


# ---------------------------------------------------------------------------
# Sandboxed tools — run inside Monty via CodeMode
# ---------------------------------------------------------------------------

def make_sandboxed_tools(store: DataStore):
    """Returns sandboxed tool functions bound to the store."""

    def execute_code(table_name: str, code: str, n: int, seed: int, fk_context_json: str) -> dict:
        """
        Execute the generated Python code in the sandbox.
        code: the complete generate() function definition as a string.
        fk_context_json: JSON string of {column_name: [values]}.
        Returns: {"success": bool, "error": str | None, "row_count": int | None}
        """
        import json as _json
        import traceback

        fk_context = _json.loads(fk_context_json)

        # Build execution namespace with allowed imports only
        namespace: dict = {}
        try:
            exec(code, namespace)  # noqa: S102 — Monty sandbox enforces restrictions
        except SyntaxError as e:
            return {"success": False, "error": f"SyntaxError: {e}", "row_count": None}

        if "generate" not in namespace:
            return {"success": False, "error": "Function 'generate' not found in code", "row_count": None}

        try:
            df = namespace["generate"](n=n, seed=seed, fk_context=fk_context)
            if not hasattr(df, "to_sql"):
                return {"success": False, "error": "generate() did not return a DataFrame", "row_count": None}

            # Temporarily store df on store object keyed by table_name for write_result
            store._pending_df = getattr(store, "_pending_df", {})
            store._pending_df[table_name] = df

            return {"success": True, "error": None, "row_count": len(df)}
        except Exception:
            return {"success": False, "error": traceback.format_exc(), "row_count": None}

    def write_result(table_name: str) -> dict:
        """
        Persist the successfully generated DataFrame to SQLite.
        Must be called after a successful execute_code() call.
        Returns: {"success": bool, "rows_written": int}
        """
        pending = getattr(store, "_pending_df", {})
        df = pending.get(table_name)
        if df is None:
            return {"success": False, "rows_written": 0}

        # Get column dtype info from the task's table plan
        task = store.get_task(table_name)
        col_dtypes: dict[str, str] = {}
        if task and task.table_plan:
            manifest_table = None
            # Match back to original spec for dtype info
            # (TablePlan has column_strategies; we need the original dtype)
            # We store what we can from the plan notes
            col_dtypes = {}  # DataStore will handle basic type inference

        store.write_table(table_name, df, col_dtypes if col_dtypes else None)

        # Update task status
        task = store.get_task(table_name)
        if task:
            task.status = TaskStatus.DATA_READY
            task.attempt_count += 1
            store.upsert_task(task)
            store.log_event(table_name, "data_written", f"{len(df)} rows")

        # Clean up pending
        pending.pop(table_name, None)
        return {"success": True, "rows_written": len(df)}

    def get_table_plan(table_name: str) -> str:
        """
        Retrieve the TablePlan for a table as a JSON string.
        The code writer uses this to understand what to generate.
        """
        task = store.get_task(table_name)
        if not task or not task.table_plan:
            return f"ERROR: no plan found for {table_name}"
        return task.table_plan.model_dump_json(indent=2)

    def report_code_error(table_name: str, code: str, error: str, self_fixable: bool, error_classification: str) -> str:
        """
        Report that code generation failed after exhausting retries.
        Updates the todo list so the orchestration agent can decide what to do.
        """
        task = store.get_task(table_name)
        if not task:
            return f"ERROR: task {table_name} not found"

        task.status = TaskStatus.FAILED
        task.last_error = f"[{error_classification}] {error[:500]}"
        if task.code_block:
            task.code_block.error_classification = error_classification  # type: ignore[assignment]
            task.code_block.self_fixable = self_fixable
            task.code_block.error_detail = error[:500]
        else:
            from core.models import CodeBlock
            task.code_block = CodeBlock(
                table_name=table_name,
                code=code,
                imports=[],
                error_classification=error_classification,  # type: ignore[arg-type]
                self_fixable=self_fixable,
                error_detail=error[:500],
            )
        store.upsert_task(task)
        store.log_event(table_name, "code_error", f"self_fixable={self_fixable} [{error_classification}]")
        return f"Error recorded for {table_name}"

    return [execute_code, write_result, get_table_plan, report_code_error]


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

def make_code_writer_agent(store: DataStore) -> Agent:
    sandboxed_tools = make_sandboxed_tools(store)

    agent = Agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt=CODE_WRITER_SYSTEM_PROMPT,
        capabilities=[CodeMode(tools="all", max_retries=3)],
    )

    for tool_fn in sandboxed_tools:
        agent.tool_plain(tool_fn)

    return agent


# ---------------------------------------------------------------------------
# Entry point called by the Orchestration Agent
# ---------------------------------------------------------------------------

async def run_code_writer(
    table_name: str,
    store: DataStore,
    revised_hints: str | None = None,
) -> bool:
    """
    Run the Code Writer Agent for one table.
    Returns True if data was successfully generated and written.
    """
    task = store.get_task(table_name)
    if not task or not task.table_plan:
        raise ValueError(f"No plan found for table '{table_name}'")

    task.status = TaskStatus.GENERATING
    store.upsert_task(task)

    agent = make_code_writer_agent(store)

    fk_context = _build_fk_context(table_name, task.table_plan, store)

    prompt = textwrap.dedent(f"""
        <task>
        Generate synthetic data for table: <table_name>{table_name}</table_name>
        Follow the workflow in your instructions exactly.
        </task>

        <fk_context_preview>
        The following parent PK values are available for FK sampling.
        The full lists are accessible via execute_code's fk_context_json parameter.
        {fk_context}
        </fk_context_preview>

        <generation_parameters>
          <n>{task.table_plan.volume}</n>
          <seed>42</seed>
        </generation_parameters>

        {f"<revision_hints>{revised_hints}</revision_hints>" if revised_hints else ""}
    """)

    result = await agent.run(prompt)

    # Check final status
    updated_task = store.get_task(table_name)
    return updated_task is not None and updated_task.status == TaskStatus.DATA_READY


def _build_fk_context(table_name: str, plan: TablePlan, store: DataStore) -> dict:
    """Pre-fetch FK context so it's available in the prompt."""
    fk_context: dict[str, list] = {}
    for col in plan.column_strategies:
        if col.method == "fk_sample" and col.fk_source_table:
            values = store.get_pk_values(
                col.fk_source_table,
                col.fk_source_column or "id",
            )
            fk_context[col.column_name] = values[:10]  # truncate for prompt; full passed via tool
    return fk_context
