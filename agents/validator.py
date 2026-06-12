"""
agents/validator.py

Validation Agent — validates generated tables against the original spec.

Responsibilities:
  - Row count check
  - Column dtype checks
  - Null rate checks
  - Distribution checks (KS test for numeric columns)
  - Referential integrity via SQL JOIN (not pandas merge)
  - Uniqueness checks
  - Classify failure type to guide Orchestration Agent's routing decision

Uses CodeMode so multiple validation checks can be batched in one round-trip.
Reads directly from SQLite — no CSV, no dtype loss.
"""

from __future__ import annotations

import textwrap

from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from core.models import (
    GenerationManifest,
    TableSpec,
    TaskStatus,
    ValidationResult,
)
from db.store import DataStore


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

VALIDATOR_SYSTEM_PROMPT = """
<role>
You are a data quality validator. You validate generated synthetic data against
the original schema specification and produce a structured, actionable result.
Your output directly drives the Orchestration Agent's routing decision —
accurate classification is more important than being lenient.
</role>

<workflow>
  <step n="1">
    Call run_validation_checks(table_name).
    This executes all checks (dtype, nulls, range, distribution, FK integrity,
    uniqueness) and returns a result dict.
  </step>
  <step n="2">
    Review the result dict carefully:
    - Is the failure_classification correct given the evidence?
    - Is the suggestion specific enough for the Orchestration Agent to act on?
    If not, adjust before persisting.
  </step>
  <step n="3">
    Call write_validation_result(table_name, result_json)
    where result_json is the (possibly adjusted) result dict as a JSON string.
  </step>
  <step n="4">
    Return a brief plain-text summary: one line per check, pass/fail with detail.
  </step>
</workflow>

<failure_classification_rules>

  <classification value="ok">
    All checks passed. No action needed.
  </classification>

  <classification value="code_bug">
    Use when the data is wrong in a way the Code Writer can fix by rewriting:
    - Column dtype does not match spec (e.g. float where int expected)
    - Null rate is more than 20 percentage points off the spec
    - Unique violation (duplicates found in a unique=True column)
    - Range violation (values outside min_value/max_value)
    - Row count wrong
    The plan is fine; only the implementation is wrong.
  </classification>

  <classification value="plan_error">
    Use when the code is likely correct but the plan itself is wrong:
    - FK integrity broken (orphaned rows) — parent may not have been generated first,
      or the FK source table/column is wrong in the plan
    - Distribution fundamentally incompatible with column dtype
      (e.g. normal distribution on a boolean column)
    - numpy_call produces values structurally incompatible with the column
    The Code Writer cannot fix this — the Planner must revise the strategy.
  </classification>

  <classification value="spec_conflict">
    Use ONLY when the original spec is self-contradictory and cannot be resolved
    by either replanning or rewriting:
    - unique=True on a column where the volume exceeds the provider's cardinality
      (e.g. 10M rows of unique Australian postcodes — impossible)
    - min_value > max_value
    - Distribution parameters that make the constraint literally unachievable
    Surface to the user. Do not retry.
  </classification>

</failure_classification_rules>

<suggestion_format>
  The suggestions field must be specific and actionable. Examples:

  Good: "retry Code Writer: account_id column has dtype object instead of int64,
         use .astype('Int64') after generation"

  Good: "replan: FK customer_id has 312 orphaned rows — check that customers
         table is generated before accounts (generation_order must be lower)"

  Good: "spec conflict: unique=True on postcode with volume=5000000 exceeds
         Australian postcode space (~2800 codes) — remove unique constraint"

  Bad: "validation failed" (not actionable)
  Bad: "please fix the code" (not specific)
</suggestion_format>
"""


# ---------------------------------------------------------------------------
# Sandboxed tools
# ---------------------------------------------------------------------------

def make_validator_tools(store: DataStore, manifest: GenerationManifest):

    # Build a lookup of table specs for quick access
    spec_by_name: dict[str, TableSpec] = {t.name: t for t in manifest.tables}

    def run_validation_checks(table_name: str) -> dict:
        """
        Run all validation checks for a generated table.
        Returns a detailed result dict that can be serialised to ValidationResult.
        """
        import json as _json
        import sqlite3

        import numpy as np
        import pandas as pd
        from scipy import stats  # type: ignore[import]

        spec = spec_by_name.get(table_name)
        if not spec:
            return {"error": f"No spec found for table '{table_name}'"}

        if not store.table_exists(table_name):
            return {"error": f"Table 'data_{table_name}' not found in SQLite"}

        df = store.read_table(table_name)
        results = []

        # --- Row count ---
        row_count_ok = len(df) == spec.volume

        # --- Column checks ---
        for col_spec in spec.columns:
            col_name = col_spec.name
            result = {"column_name": col_name, "passed": True, "failure_type": "ok", "detail": None}

            if col_name not in df.columns:
                result.update({"passed": False, "failure_type": "dtype_mismatch", "detail": f"Column '{col_name}' missing from DataFrame"})
                results.append(result)
                continue

            series = df[col_name]

            # Dtype check
            expected_dtype = col_spec.dtype
            actual_null_rate = series.isna().mean()
            non_null = series.dropna()

            dtype_ok = True
            if expected_dtype == "int" and not pd.api.types.is_integer_dtype(non_null):
                dtype_ok = False
            elif expected_dtype == "float" and not pd.api.types.is_float_dtype(non_null):
                dtype_ok = False
            elif expected_dtype == "bool":
                # SQLite stores bool as INTEGER — accept bool OR int64 containing only 0/1
                is_bool = pd.api.types.is_bool_dtype(non_null)
                is_int_bool = (
                    pd.api.types.is_integer_dtype(non_null)
                    and non_null.isin([0, 1]).all()
                )
                if not (is_bool or is_int_bool):
                    dtype_ok = False

            if not dtype_ok:
                result.update({
                    "passed": False,
                    "failure_type": "dtype_mismatch",
                    "detail": f"Expected {expected_dtype}, got {non_null.dtype}",
                })
                results.append(result)
                continue

            # Null rate check (allow 20% tolerance)
            expected_null_rate = col_spec.nullable
            if abs(actual_null_rate - expected_null_rate) > 0.20:
                result.update({
                    "passed": False,
                    "failure_type": "null_rate_exceeded",
                    "detail": f"Expected ~{expected_null_rate:.0%} nulls, got {actual_null_rate:.0%}",
                })
                results.append(result)
                continue

            # Uniqueness check
            if col_spec.unique and series.dropna().duplicated().any():
                result.update({
                    "passed": False,
                    "failure_type": "unique_violation",
                    "detail": f"Column has {series.dropna().duplicated().sum()} duplicate values",
                })
                results.append(result)
                continue

            # Range check
            if col_spec.min_value is not None and pd.api.types.is_numeric_dtype(non_null):
                if (non_null < col_spec.min_value).any():
                    result.update({
                        "passed": False,
                        "failure_type": "range_violation",
                        "detail": f"Values below min_value={col_spec.min_value}",
                    })
                    results.append(result)
                    continue

            if col_spec.max_value is not None and pd.api.types.is_numeric_dtype(non_null):
                if (non_null > col_spec.max_value).any():
                    result.update({
                        "passed": False,
                        "failure_type": "range_violation",
                        "detail": f"Values above max_value={col_spec.max_value}",
                    })
                    results.append(result)
                    continue

            # Distribution check (KS test for numeric columns with distribution spec)
            if col_spec.distribution and pd.api.types.is_numeric_dtype(non_null) and len(non_null) > 30:
                dist = col_spec.distribution
                if dist.type == "normal" and dist.mean is not None and dist.std is not None:
                    _, p_value = stats.kstest(
                        non_null.astype(float),
                        "norm",
                        args=(dist.mean, dist.std),
                    )
                    if p_value < 0.01:  # strict: p < 1%
                        result.update({
                            "passed": False,
                            "failure_type": "distribution_mismatch",
                            "detail": f"KS test failed for normal({dist.mean}, {dist.std}): p={p_value:.4f}",
                        })
                        results.append(result)
                        continue

            results.append(result)

        # --- Referential integrity via SQL ---
        fk_integrity_ok = True
        fk_details = []
        for col_spec in spec.columns:
            if col_spec.foreign_key:
                fk = col_spec.foreign_key
                orphans = store.validate_fk(
                    table_name, col_spec.name,
                    fk.references_table, fk.references_column,
                )
                if orphans > 0:
                    fk_integrity_ok = False
                    fk_details.append(f"{col_spec.name}→{fk.references_table}.{fk.references_column}: {orphans} orphans")

        # --- Failure classification ---
        all_col_passed = all(r["passed"] for r in results)
        passed = row_count_ok and fk_integrity_ok and all_col_passed

        if passed:
            failure_classification = "ok"
            suggestions = None
        elif not fk_integrity_ok:
            failure_classification = "plan_error"
            suggestions = f"FK integrity failed: {'; '.join(fk_details)}. Check generation order."
        else:
            failed_cols = [r for r in results if not r["passed"]]
            failure_types = {r["failure_type"] for r in failed_cols}
            if "distribution_mismatch" in failure_types or "strategy_mismatch" in failure_types:
                failure_classification = "plan_error"
                suggestions = f"Distribution mismatch in: {[r['column_name'] for r in failed_cols if r['failure_type'] == 'distribution_mismatch']}"
            else:
                failure_classification = "code_bug"
                suggestions = f"Fix columns: {[(r['column_name'], r['failure_type'], r['detail']) for r in failed_cols]}"

        return {
            "table_name": table_name,
            "passed": passed,
            "row_count_ok": row_count_ok,
            "referential_integrity_ok": fk_integrity_ok,
            "column_results": results,
            "failure_classification": failure_classification,
            "suggestions": suggestions,
        }

    def write_validation_result(table_name: str, result_json: str) -> str:
        """
        Persist a ValidationResult to the todo list.
        result_json: JSON string of the dict returned by run_validation_checks().
        """
        import json as _json
        result_dict = _json.loads(result_json)
        validation = ValidationResult(**result_dict)

        task = store.get_task(table_name)
        if not task:
            return f"ERROR: task {table_name} not found"

        task.validation_result = validation
        task.status = TaskStatus.DONE if validation.passed else TaskStatus.FAILED
        store.upsert_task(task)
        store.log_event(
            table_name,
            "validation_complete",
            f"passed={validation.passed} [{validation.failure_classification}]",
        )
        return f"Validation written for {table_name}: passed={validation.passed}"

    return [run_validation_checks, write_validation_result]


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

def make_validator_agent(store: DataStore, manifest: GenerationManifest) -> Agent:
    validator_tools = make_validator_tools(store, manifest)

    agent = Agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt=VALIDATOR_SYSTEM_PROMPT,
        capabilities=[CodeMode(tools="all", max_retries=2)],
    )

    for tool_fn in validator_tools:
        agent.tool_plain(tool_fn)

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_validator(
    table_name: str,
    store: DataStore,
    manifest: GenerationManifest,
) -> ValidationResult | None:
    """
    Run validation for one table.
    Returns the ValidationResult, or None if the task/data is missing.
    """
    task = store.get_task(table_name)
    if not task or task.status != TaskStatus.DATA_READY:
        return None

    task.status = TaskStatus.VALIDATING
    store.upsert_task(task)

    agent = make_validator_agent(store, manifest)

    prompt = textwrap.dedent(f"""
        <task>
        Validate the generated data for table: <table_name>{table_name}</table_name>
        Follow the workflow in your instructions exactly.
        Classify failures precisely — your output drives the orchestrator's next action.
        </task>
    """)

    await agent.run(prompt)

    # Return the persisted result
    updated_task = store.get_task(table_name)
    return updated_task.validation_result if updated_task else None
