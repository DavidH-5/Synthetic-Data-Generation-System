"""Unit tests for db/store.py TodoTask persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.models import (
    CodeBlock,
    ColumnStrategy,
    ColumnValidationResult,
    TablePlan,
    TaskStatus,
    TodoTask,
    ValidationResult,
)
from db.store import DataStore


@pytest.fixture
def store(tmp_path: Path) -> DataStore:
    db = DataStore(tmp_path / "test.db")
    db.initialise()
    return db


def _sample_table_plan() -> TablePlan:
    return TablePlan(
        table_name="customers",
        volume=100,
        generation_order=0,
        column_strategies=[
            ColumnStrategy(
                column_name="customer_id",
                method="faker",
                faker_provider="uuid4",
            ),
        ],
        dependencies=[],
    )


def _sample_code_block() -> CodeBlock:
    return CodeBlock(
        table_name="customers",
        code="def generate(n, seed, fk_context):\n    pass",
        imports=["import pandas as pd"],
        error_classification="none",
        self_fixable=True,
    )


def _sample_validation_result() -> ValidationResult:
    return ValidationResult(
        table_name="customers",
        passed=True,
        row_count_ok=True,
        referential_integrity_ok=True,
        column_results=[
            ColumnValidationResult(column_name="customer_id", passed=True),
        ],
        failure_classification="ok",
    )


def test_upsert_and_read_round_trip(store: DataStore) -> None:
    task = TodoTask(
        table_name="customers",
        status=TaskStatus.PLAN_READY,
        generation_order=0,
        dependencies=[],
        table_plan=_sample_table_plan(),
        code_block=_sample_code_block(),
        validation_result=_sample_validation_result(),
        replan_action="retry_code",
        replan_reasoning="fix dtype",
        replan_hints="use Int64",
        attempt_count=1,
        replan_count=0,
        last_error=None,
    )

    store.upsert_task(task)
    loaded = store.get_task("customers")

    assert loaded is not None
    assert loaded.table_name == "customers"
    assert loaded.status == TaskStatus.PLAN_READY
    assert loaded.table_plan is not None
    assert loaded.table_plan.volume == 100
    assert loaded.code_block is not None
    assert loaded.code_block.code.startswith("def generate")
    assert loaded.code_block.error_classification == "none"
    assert loaded.validation_result is not None
    assert loaded.validation_result.passed is True
    assert loaded.replan_action == "retry_code"
    assert loaded.replan_reasoning == "fix dtype"
    assert loaded.replan_hints == "use Int64"
    assert loaded.attempt_count == 1


def test_log_event(store: DataStore) -> None:
    store.log_event("customers", "plan_written", "order=0")

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute(
            "SELECT table_name, event, detail FROM generation_log"
        ).fetchone()

    assert row is not None
    assert row[0] == "customers"
    assert row[1] == "plan_written"
    assert row[2] == "order=0"


def test_code_block_plain_string_fallback(store: DataStore) -> None:
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            """
            INSERT INTO todo_list (
                table_name, status, generation_order, dependencies,
                generated_code, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy_table",
                "plan_ready",
                0,
                "[]",
                "def generate(n, seed, fk_context):\n    return None",
                "2026-01-01T00:00:00",
            ),
        )

    loaded = store.get_task("legacy_table")

    assert loaded is not None
    assert loaded.code_block is not None
    assert loaded.code_block.table_name == "legacy_table"
    assert "def generate" in loaded.code_block.code
    assert loaded.code_block.imports == []
