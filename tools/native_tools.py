"""
tools/native_tools.py

Native (non-sandboxed) tools available to the Orchestration Agent.
These handle state management and I/O — they never execute generated code.

In CodeMode terminology: these stay at the top-level agent scope.
Sandboxed tools (run_planner, run_code_writer, run_validator) live in their
respective agent modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.models import (
    GenerationManifest,
    ReplanDecision,
    TaskStatus,
    TodoTask,
)
from db.store import DataStore


def make_native_tools(store: DataStore, manifest: GenerationManifest):
    """
    Factory that binds a DataStore and manifest to the tool functions.
    Returns a list of callables to register on the orchestration agent.

    Using a factory (rather than globals) means the store reference is
    explicit and testable.
    """

    def read_todo_list() -> list[dict]:
        """
        Read the current state of all tasks from the todo list.
        Returns a list of task dicts ordered by generation_order.
        Use this to understand what still needs to be done.
        """
        tasks = store.get_all_tasks()
        return [
            {
                "table_name": t.table_name,
                "status": t.status.value,
                "generation_order": t.generation_order,
                "dependencies": t.dependencies,
                "attempt_count": t.attempt_count,
                "replan_count": t.replan_count,
                "last_error": t.last_error,
                "has_plan": t.table_plan is not None,
                "has_code": t.code_block is not None,
                "validation_passed": (
                    t.validation_result.passed if t.validation_result else None
                ),
                "validation_failure_type": (
                    t.validation_result.failure_classification
                    if t.validation_result
                    else None
                ),
                "validation_suggestions": (
                    t.validation_result.suggestions if t.validation_result else None
                ),
            }
            for t in tasks
        ]

    def write_task_status(table_name: str, status: str, error: str | None = None) -> str:
        """
        Update the status of a task. Optionally record an error message.
        Valid statuses: pending, planning, plan_ready, generating, code_ready,
                        data_ready, validating, done, failed, unresolvable
        """
        task = store.get_task(table_name)
        if not task:
            return f"ERROR: task '{table_name}' not found in todo list"
        task.status = TaskStatus(status)
        if error:
            task.last_error = error
        store.upsert_task(task)
        store.log_event(table_name, "status_change", f"→ {status}" + (f": {error}" if error else ""))
        return f"Updated {table_name} → {status}"

    def record_replan_decision(decision_json: str) -> str:
        """
        Record a ReplanDecision and update the affected task's status accordingly.
        decision_json: JSON string of a ReplanDecision model.
        """
        decision = ReplanDecision.model_validate_json(decision_json)
        task = store.get_task(decision.table_name)
        if not task:
            return f"ERROR: task '{decision.table_name}' not found"

        task.replan_count += 1
        task.last_error = f"[replan:{decision.action}] {decision.reasoning}"
        task.replan_action = decision.action
        task.replan_reasoning = decision.reasoning
        task.replan_hints = decision.revised_hints

        if decision.action == "mark_unresolvable":
            task.status = TaskStatus.UNRESOLVABLE
        elif decision.action in ("retry_code", "replan_table"):
            task.status = TaskStatus.PLAN_READY  # re-enter at code generation
            if decision.action == "replan_table":
                task.table_plan = None            # force planner to re-run
                task.status = TaskStatus.PENDING
        elif decision.action == "replan_full":
            # Reset all tasks to pending
            for t in store.get_all_tasks():
                t.status = TaskStatus.PENDING
                t.table_plan = None
                t.code_block = None
                t.validation_result = None
                store.upsert_task(t)
            store.log_event(decision.table_name, "full_replan", decision.reasoning)
            return "Full replan triggered — all tasks reset to pending"

        store.upsert_task(task)
        store.log_event(
            decision.table_name,
            f"replan:{decision.action}",
            decision.reasoning,
        )
        return f"Replan recorded for {decision.table_name}: {decision.action}"

    def get_fk_context(table_name: str) -> dict:
        """
        Get PK values from parent tables that this table depends on.
        Returns a dict of {column_name: [values]} ready to pass as fk_context
        to the generate() function.
        """
        task = store.get_task(table_name)
        if not task or not task.table_plan:
            return {}

        fk_context: dict[str, list] = {}
        for col_strategy in task.table_plan.column_strategies:
            if col_strategy.method == "fk_sample" and col_strategy.fk_source_table:
                pk_values = store.get_pk_values(
                    col_strategy.fk_source_table,
                    col_strategy.fk_source_column or "id",
                )
                fk_context[col_strategy.column_name] = pk_values

        return fk_context

    def get_manifest_summary() -> dict:
        """
        Return a summary of the original manifest for context.
        Useful for the orchestration agent to understand the full picture.
        """
        return {
            "tables": [
                {
                    "name": t.name,
                    "volume": t.volume,
                    "columns": [c.name for c in t.columns],
                    "has_fk": any(c.foreign_key for c in t.columns),
                }
                for t in manifest.tables
            ],
            "seed": manifest.seed,
        }

    def export_summary() -> dict:
        """
        Return a summary of what has been generated so far.
        Shows row counts and table names for all completed tables.
        """
        summary = {}
        for task in store.get_tasks_by_status(TaskStatus.DONE):
            if store.table_exists(task.table_name):
                with __import__("sqlite3").connect(store.db_path) as conn:
                    row = conn.execute(
                        f"SELECT COUNT(*) as n FROM data_{task.table_name}"
                    ).fetchone()
                summary[task.table_name] = {"rows": row[0], "status": "done"}
        return summary

    return [
        read_todo_list,
        write_task_status,
        record_replan_decision,
        get_fk_context,
        get_manifest_summary,
        export_summary,
    ]
