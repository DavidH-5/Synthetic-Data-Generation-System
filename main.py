"""
main.py

Entry point for the synthetic data generation pipeline.

Usage:
    python main.py --input examples/banking_schema.json --db output.db
    python main.py --input examples/banking_schema.json --db output.db --reset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from core.models import GenerationManifest, TaskStatus
from db.store import DataStore
from agents.orchestrator import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument("--input", required=True, help="Path to JSON manifest file")
    parser.add_argument("--db", default="output.db", help="Path to SQLite output database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the todo list (start fresh even if a previous run exists)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # --- Load and validate manifest ---
    manifest_path = Path(args.input)
    if not manifest_path.exists():
        print(f"ERROR: manifest file not found: {manifest_path}")
        sys.exit(1)

    with manifest_path.open() as f:
        raw = json.load(f)

    try:
        manifest = GenerationManifest.model_validate(raw)
    except Exception as e:
        print(f"ERROR: invalid manifest: {e}")
        sys.exit(1)

    print(f"Loaded manifest: {len(manifest.tables)} tables")
    for t in manifest.tables:
        print(f"  {t.name}: {t.volume} rows, {len(t.columns)} columns")

    # --- Initialise store ---
    store = DataStore(args.db)
    store.initialise()

    if args.reset:
        # Wipe todo_list so we start fresh
        import sqlite3
        with sqlite3.connect(args.db) as conn:
            conn.execute("DELETE FROM todo_list")
            conn.execute("DELETE FROM generation_log")
        print("Reset: todo_list cleared")

    # --- Run pipeline ---
    print(f"\nStarting pipeline → {args.db}")
    print("-" * 50)

    await run_pipeline(manifest, store)

    tasks = store.get_all_tasks()
    if not tasks:
        print("\nERROR: pipeline produced no tasks — check API key and orchestrator logs.")
        sys.exit(1)

    incomplete = [
        t for t in tasks
        if t.status not in (TaskStatus.DONE, TaskStatus.UNRESOLVABLE)
    ]
    if incomplete:
        print("\nERROR: pipeline did not finish all tables:")
        for t in incomplete:
            print(f"  {t.table_name}: {t.status.value}")
            if t.last_error:
                print(f"    → {t.last_error}")
        sys.exit(1)

    # --- Print results ---
    print("\n" + "=" * 50)
    print("GENERATION COMPLETE")
    print("=" * 50)

    for task in sorted(tasks, key=lambda t: t.generation_order):
        status_icon = "✓" if task.status.value == "done" else "✗"
        rows = ""
        if store.table_exists(task.table_name):
            import sqlite3
            with sqlite3.connect(args.db) as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM data_{task.table_name}"
                ).fetchone()
                rows = f" ({row[0]} rows)"
        print(f"  {status_icon} {task.table_name}{rows} [{task.status.value}]")
        if task.last_error and task.status.value in ("failed", "unresolvable"):
            print(f"      → {task.last_error}")

    print(f"\nOutput database: {args.db}")
    print("Tables prefixed with 'data_' contain the generated data.")
    print("Table 'todo_list' contains the full audit trail.")


if __name__ == "__main__":
    asyncio.run(main())
