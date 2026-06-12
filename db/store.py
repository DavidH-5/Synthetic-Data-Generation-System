"""
db/store.py

SQLite-backed store for:
  1. todo_list        — shared blackboard for all agents
  2. generated tables — typed intermediate and final storage

All reads/writes go through this class. Agents never touch SQLite directly.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import pandas as pd

from core.models import (
    CodeBlock,
    TablePlan,
    TaskStatus,
    TodoTask,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Dtype mapping: our dtype string → SQLite affinity / pandas dtype
# ---------------------------------------------------------------------------

DTYPE_TO_SQLITE: dict[str, str] = {
    "int":      "INTEGER",
    "float":    "REAL",
    "str":      "TEXT",
    "bool":     "INTEGER",   # SQLite has no BOOL; store as 0/1
    "date":     "TEXT",      # ISO-8601
    "datetime": "TEXT",      # ISO-8601
    "uuid":     "TEXT",
}

DTYPE_TO_PANDAS: dict[str, str] = {
    "int":      "Int64",     # nullable integer
    "float":    "float64",
    "str":      "object",
    "bool":     "boolean",   # nullable bool
    "date":     "object",
    "datetime": "object",
    "uuid":     "object",
}


class DataStore:
    def __init__(self, db_path: str | Path = "output.db"):
        self.db_path = str(db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create tables if they don't exist. Safe to call multiple times."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS todo_list (
                    table_name               TEXT PRIMARY KEY,
                    status                   TEXT NOT NULL DEFAULT 'pending',
                    generation_order         INTEGER NOT NULL DEFAULT 0,
                    dependencies             TEXT NOT NULL DEFAULT '[]',

                    -- Agent artefacts (JSON blobs)
                    table_plan               TEXT,
                    generated_code           TEXT,
                    validation_result        TEXT,

                    -- Code Writer error signals
                    code_error_classification TEXT NOT NULL DEFAULT 'none',
                    code_self_fixable        INTEGER NOT NULL DEFAULT 1,

                    -- Orchestration routing decision
                    replan_action            TEXT NOT NULL DEFAULT 'none',
                    replan_reasoning         TEXT,
                    replan_hints             TEXT,

                    -- Circuit breaker counters
                    attempt_count            INTEGER NOT NULL DEFAULT 0,
                    replan_count             INTEGER NOT NULL DEFAULT 0,

                    -- Audit
                    last_error               TEXT,
                    updated_at               TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS generation_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name   TEXT NOT NULL,
                    event        TEXT NOT NULL,
                    detail       TEXT,
                    created_at   TEXT NOT NULL
                );

                -- Stores original dtype strings per column so read_table
                -- can correctly cast bool/int columns after SQLite round-trip.
                CREATE TABLE IF NOT EXISTS _table_meta (
                    table_name  TEXT NOT NULL,
                    column_name TEXT NOT NULL,
                    dtype       TEXT NOT NULL,
                    PRIMARY KEY (table_name, column_name)
                );
            """)

    # ------------------------------------------------------------------
    # TodoList operations
    # ------------------------------------------------------------------

    def upsert_task(self, task: TodoTask) -> None:
        """Insert or update a task row. Always refreshes updated_at."""
        task.updated_at = datetime.utcnow()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO todo_list (
                    table_name, status, generation_order, dependencies,
                    table_plan, generated_code, validation_result,
                    code_error_classification, code_self_fixable,
                    replan_action, replan_reasoning, replan_hints,
                    attempt_count, replan_count, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(table_name) DO UPDATE SET
                    status                    = excluded.status,
                    generation_order          = excluded.generation_order,
                    dependencies              = excluded.dependencies,
                    table_plan                = excluded.table_plan,
                    generated_code            = excluded.generated_code,
                    validation_result         = excluded.validation_result,
                    code_error_classification = excluded.code_error_classification,
                    code_self_fixable         = excluded.code_self_fixable,
                    replan_action             = excluded.replan_action,
                    replan_reasoning          = excluded.replan_reasoning,
                    replan_hints              = excluded.replan_hints,
                    attempt_count             = excluded.attempt_count,
                    replan_count              = excluded.replan_count,
                    last_error                = excluded.last_error,
                    updated_at                = excluded.updated_at
            """, self._task_to_row(task))

    def get_task(self, table_name: str) -> TodoTask | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM todo_list WHERE table_name = ?", (table_name,)
            ).fetchone()
        return self._row_to_task(row) if row else None

    def get_all_tasks(self) -> list[TodoTask]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM todo_list ORDER BY generation_order ASC"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def get_tasks_by_status(self, status: TaskStatus) -> list[TodoTask]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM todo_list WHERE status = ? ORDER BY generation_order ASC",
                (status.value,),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def all_done(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as n FROM todo_list
                WHERE status NOT IN ('done', 'unresolvable')
            """).fetchone()
        return row["n"] == 0

    def log_event(self, table_name: str, event: str, detail: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO generation_log (table_name, event, detail, created_at) "
                "VALUES (?, ?, ?, ?)",
                (table_name, event, detail, datetime.utcnow().isoformat()),
            )

    def _task_to_row(self, task: TodoTask) -> tuple:
        """Map a TodoTask to SQLite column values."""
        if task.code_block:
            generated_code = task.code_block.model_dump_json()
            code_error_classification = task.code_block.error_classification
            code_self_fixable = int(task.code_block.self_fixable)
        else:
            generated_code = None
            code_error_classification = "none"
            code_self_fixable = 1

        return (
            task.table_name,
            task.status.value,
            task.generation_order,
            json.dumps(task.dependencies),
            task.table_plan.model_dump_json() if task.table_plan else None,
            generated_code,
            task.validation_result.model_dump_json() if task.validation_result else None,
            code_error_classification,
            code_self_fixable,
            task.replan_action,
            task.replan_reasoning,
            task.replan_hints,
            task.attempt_count,
            task.replan_count,
            task.last_error,
            task.updated_at.isoformat(),
        )

    def _parse_code_block(self, table_name: str, generated_code: str | None) -> CodeBlock | None:
        if not generated_code:
            return None
        try:
            return CodeBlock.model_validate_json(generated_code)
        except Exception:
            return CodeBlock(
                table_name=table_name,
                code=generated_code,
                imports=[],
            )

    def _row_to_task(self, row: sqlite3.Row) -> TodoTask:
        return TodoTask(
            table_name=row["table_name"],
            status=TaskStatus(row["status"]),
            generation_order=row["generation_order"],
            dependencies=json.loads(row["dependencies"]),
            table_plan=(
                TablePlan.model_validate_json(row["table_plan"])
                if row["table_plan"] else None
            ),
            code_block=self._parse_code_block(row["table_name"], row["generated_code"]),
            validation_result=(
                ValidationResult.model_validate_json(row["validation_result"])
                if row["validation_result"] else None
            ),
            replan_action=row["replan_action"],
            replan_reasoning=row["replan_reasoning"],
            replan_hints=row["replan_hints"],
            attempt_count=row["attempt_count"],
            replan_count=row["replan_count"],
            last_error=row["last_error"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ------------------------------------------------------------------
    # Generated data tables
    # ------------------------------------------------------------------

    def write_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        column_dtypes: dict[str, str] | None = None,
    ) -> None:
        """
        Write a DataFrame to SQLite as data_{table_name}.
        Drops and recreates on retry — safe for replanning.
        column_dtypes maps column name → our dtype string ("int", "str", "bool", etc.)

        Persists dtype metadata to _table_meta so read_table can correctly
        cast bool and other columns after the SQLite round-trip.
        """
        sqlite_dtype_map = {}
        if column_dtypes:
            sqlite_dtype_map = {
                col: DTYPE_TO_SQLITE.get(dtype, "TEXT")
                for col, dtype in column_dtypes.items()
            }
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(
                name=f"data_{table_name}",
                con=conn,
                if_exists="replace",
                index=False,
                dtype=sqlite_dtype_map or None,
            )

            # Persist dtype metadata for correct round-trip casting
            if column_dtypes:
                conn.execute(
                    "DELETE FROM _table_meta WHERE table_name = ?", (table_name,)
                )
                conn.executemany(
                    "INSERT INTO _table_meta (table_name, column_name, dtype) VALUES (?, ?, ?)",
                    [(table_name, col, dtype) for col, dtype in column_dtypes.items()],
                )

    def read_table(
        self,
        table_name: str,
        column_dtypes: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Read a generated table back as a DataFrame with correct dtypes.

        SQLite has no native BOOL type — it stores booleans as INTEGER (0/1),
        so pandas always reads them back as int64. This method fixes that by:
        1. Reading stored dtype metadata from the _table_meta table (written by write_table).
        2. Casting columns accordingly, with caller-supplied column_dtypes taking priority.
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM data_{table_name}", conn)

            # Read stored dtype metadata if available
            stored_dtypes: dict[str, str] = {}
            try:
                meta_rows = conn.execute(
                    "SELECT column_name, dtype FROM _table_meta WHERE table_name = ?",
                    (table_name,),
                ).fetchall()
                stored_dtypes = {row[0]: row[1] for row in meta_rows}
            except sqlite3.OperationalError:
                pass  # _table_meta doesn't exist yet (older db)

        # Merge: caller-supplied overrides stored metadata
        effective_dtypes = {**stored_dtypes, **(column_dtypes or {})}

        for col, dtype in effective_dtypes.items():
            if col not in df.columns:
                continue
            pandas_dtype = DTYPE_TO_PANDAS.get(dtype, "object")
            try:
                df[col] = df[col].astype(pandas_dtype)
            except (ValueError, TypeError):
                pass  # leave as-is; validator will catch the mismatch

        return df

    def table_exists(self, table_name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (f"data_{table_name}",),
            ).fetchone()
        return row is not None

    def get_pk_values(self, table_name: str, pk_column: str) -> list:
        """Fetch PK values from a generated table for FK injection."""
        if not self.table_exists(table_name):
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT {pk_column} FROM data_{table_name}"
            ).fetchall()
        return [r[0] for r in rows]

    def validate_fk(
        self,
        child_table: str,
        fk_col: str,
        parent_table: str,
        pk_col: str,
    ) -> int:
        """
        Returns count of orphaned FK rows via SQL JOIN.
        0 = referential integrity holds.
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(f"""
                SELECT COUNT(*) as orphans
                FROM data_{child_table} c
                LEFT JOIN data_{parent_table} p ON c.{fk_col} = p.{pk_col}
                WHERE p.{pk_col} IS NULL
            """).fetchone()
        return row[0]
