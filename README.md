# Synthetic Data Generation System

A multi-agent LLM pipeline that generates production-quality synthetic data from a JSON schema definition — no manual coding required.

---

## What Does This Project Do?

You define your data model in a JSON file: table names, column types, statistical distributions, foreign key relationships, and row volumes. The pipeline reads that spec and produces a fully populated SQLite database with referentially intact, statistically faithful synthetic data.

```
{
  "tables": [
    {
      "name": "customers",
      "volume": 1000,
      "columns": [
        { "name": "customer_id", "dtype": "uuid", "primary_key": true },
        { "name": "email",       "dtype": "str",  "unique": true, "semantic_hint": "email" },
        { "name": "balance",     "dtype": "float", "distribution": { "type": "normal", "mean": 5000, "std": 1500 } }
      ]
    }
  ]
}
```

Run it with:

```bash
uv run python main.py --input schema.json --db output.db
```

The pipeline automatically:

- Analyses your schema and plans a generation strategy per column
- Topologically sorts tables so parent rows exist before child FK references are sampled
- Writes Python generation code using `faker`, `pandas`, and `numpy`
- Executes that code in an isolated sandbox
- Validates the output against your spec (dtypes, null rates, distributions, referential integrity)
- Self-heals on failure — retrying code errors or replanning strategy mismatches before surfacing unresolvable conflicts to you

The final output is a SQLite database. Every generated table is prefixed `data_` and every decision, retry, and error is logged to the `todo_list` and `generation_log` tables for full auditability.

---

## Why This Over a Traditional Python Package?

Traditional synthetic data packages like `SDV`, `Faker`, or `mimesis` require you to write Python code that calls their APIs. You learn the API, wire up the relationships, handle edge cases, and debug failures yourself. For a three-table schema with FK relationships that takes hours. For a twenty-table schema it takes days.

This system inverts that. You write JSON, not Python. The LLM writes the Python.

| | Traditional Package | This System |
|---|---|---|
| **Setup** | Write Python code per table | Write a JSON spec |
| **FK relationships** | Wire manually, sample manually | Declared in spec, handled automatically |
| **Statistical distributions** | Call distribution API per column | Declare `"distribution": {"type": "normal", ...}` |
| **Semantic realism** | Manually select faker providers | LLM infers provider from column name and semantic hint |
| **Schema changes** | Rewrite affected code | Edit the JSON |
| **Error handling** | Debug your own code | Pipeline self-heals; surfaces genuine spec conflicts |
| **New table types** | Learn new API surface | Describe it in JSON |

The LLM also makes decisions a package cannot: it reads `"semantic_hint": "bsb_number"` and knows to call `faker.bsb_number()`, or reads a column named `merchant_category` with a choice distribution and maps the weights correctly. That kind of contextual mapping has no equivalent in a static package API.

The cost trade-off is real: LLM calls cost tokens. But the pipeline minimises this deliberately — the LLM only writes code, it never generates rows. All row generation runs locally via `faker`/`pandas`/`numpy` at zero marginal cost per row. A 50,000-row transaction table costs the same in LLM tokens as a 50-row one.

---

## What Makes This Solution Robust and Sophisticated?

### Multi-agent architecture with clear separation of concerns

Four specialised agents each do one thing. The **Planner** analyses the full schema and produces a generation strategy. The **Code Writer** implements that strategy for one table at a time. The **Validator** checks the output against the spec. The **Orchestration Agent** interprets failure signals and decides what to do next. None of these agents know about each other's internals — they communicate only through a shared SQLite blackboard.

### Blackboard pattern for shared state

Every agent reads from and writes to a central `todo_list` table in SQLite. Task status, generated code, validation results, replan decisions, and error messages all live in one place. This means the full pipeline state is inspectable at any point, restartable after a crash, and auditable after completion — you can query the database to understand exactly why any table succeeded or failed.

### Intelligent failure routing, not naive retries

When something fails, the Orchestration Agent reads two independent signals: the Code Writer's `error_classification` and `self_fixable` flag, and the Validator's `failure_classification`. It then routes accordingly:

- **Code bug, self-fixable** → retry the Code Writer with the specific error as a hint
- **Code bug, not self-fixable** → the plan is wrong; send back to the Planner
- **Plan error** → FK integrity or strategy mismatch; replan the table
- **Spec conflict** → the spec itself is self-contradictory; surface to the user

This is the opposite of retry loops. The system understands *why* something failed before deciding what to do about it.

### Hard circuit breakers

No table can cycle indefinitely. Each table has a maximum of 3 code generation attempts and 2 replan cycles. Once either limit is hit the table is marked `unresolvable` and the pipeline continues with the remaining tables. Downstream tables that depend on an unresolvable table are also marked by dependency cascade — no orphaned FK references.

### Sandboxed code execution via Monty

LLM-generated code runs inside the Monty sandbox provided by `pydantic-ai-harness`. The generated `generate()` function cannot import arbitrary packages, access the filesystem, or affect the host process. Only `faker`, `pandas`, `numpy`, `random`, `uuid`, and `datetime` are permitted. A static AST import check runs before execution as an additional gate.

### SQLite as typed intermediate storage

Data is written to SQLite, not CSV. This preserves column types across the write/read round-trip and enables FK validation as a single SQL `LEFT JOIN` query rather than an in-memory pandas merge. A `_table_meta` table stores the original dtype declarations so `bool` columns survive the SQLite `INTEGER` round-trip and come back as `boolean` dtype in pandas.

### Structured inter-agent contracts

Nothing passes between agents as raw strings or untyped dicts. Every handoff is a Pydantic model: `GenerationPlan`, `TablePlan`, `ValidationResult`, `TodoTask`. Schema violations are caught at the boundary — a malformed planner output fails immediately with a validation error rather than silently producing wrong data three steps later.

### XML-structured prompts

All agent system prompts use XML tags to separate role definition, output contracts, rules, strategy options, and examples. This is not cosmetic — structured prompts produce more consistent structured outputs, make rule boundaries unambiguous across long context windows, and reduce strategy misclassification (the most common planner failure mode).

### Resumable runs

If a pipeline run is interrupted, re-running without `--reset` picks up from the current `todo_list` state. Tables already marked `done` are skipped. Tables mid-generation retry from their last known state.

---

## Architecture

```
input.json (GenerationManifest)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│               Orchestration Agent                     │
│   CodeMode: sandboxes dispatcher tools only           │
│   Native tools: read/write todo list, export          │
└──────────────────────────────────────────────────────┘
          │ (via Monty sandbox)
          ▼
┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐
│  Planner    │   │  Code Writer     │   │  Validator      │
│  Agent      │   │  Agent           │   │  Agent          │
│             │   │  CodeMode: all   │   │  CodeMode: all  │
│  outputs    │   │                  │   │                 │
│  GenerationP│   │  writes          │   │  SQL FK checks  │
│  lan        │   │  generate() fn   │   │  KS dist tests  │
│             │   │  to SQLite       │   │  dtype checks   │
└─────────────┘   └──────────────────┘   └─────────────────┘
          │               │                      │
          └───────────────┴──────────────────────┘
                          │
              ┌───────────────────────┐
              │      SQLite DB        │
              │  todo_list            │ ← blackboard
              │  generation_log       │ ← audit trail
              │  _table_meta          │ ← dtype metadata
              │  data_<table_name>    │ ← generated data
              └───────────────────────┘
```

---

## Project Structure

```
synth_data_gen/
├── main.py                        # CLI entry point
├── core/
│   └── models.py                  # All Pydantic models (inter-agent contracts)
├── db/
│   └── store.py                   # SQLite read/write layer
├── agents/
│   ├── orchestrator.py            # Orchestration Agent
│   ├── planner.py                 # Planner Agent
│   ├── code_writer.py             # Code Writer Agent
│   └── validator.py               # Validation Agent
├── tools/
│   └── native_tools.py            # Non-sandboxed tools for orchestrator
└── examples/
    └── banking_schema.json        # Sample schema: customers → accounts → transactions
```

---

## Installation

```bash
uv add "pydantic-ai-harness[code-mode]"
uv add "pydantic-ai-slim[anthropic]"
uv add faker pandas numpy scipy
```

---

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...

# Run the pipeline
uv run python main.py --input examples/banking_schema.json --db output.db

# Start fresh (clears todo_list and generation_log)
uv run python main.py --input examples/banking_schema.json --db output.db --reset
```

Query the results directly:

```bash
sqlite3 output.db "SELECT COUNT(*) FROM data_transactions;"
sqlite3 output.db "SELECT * FROM todo_list;"
sqlite3 output.db "SELECT * FROM generation_log ORDER BY created_at DESC LIMIT 20;"
```

---

## Input Schema Reference

| Field | Type | Description |
|---|---|---|
| `name` | string | Table name |
| `volume` | int | Number of rows to generate |
| `columns[].name` | string | Column name |
| `columns[].dtype` | string | `int`, `float`, `str`, `bool`, `date`, `datetime`, `uuid` |
| `columns[].nullable` | float | Fraction of NULL values (0.0–1.0) |
| `columns[].distribution` | object | `normal`, `uniform`, `poisson`, `exponential`, or `choice` |
| `columns[].primary_key` | bool | Marks this column as the PK |
| `columns[].unique` | bool | Enforces uniqueness |
| `columns[].foreign_key` | object | `references_table`, `references_column`, `cardinality` |
| `columns[].semantic_hint` | string | Guides faker provider selection (e.g. `"email"`, `"bsb_number"`) |
| `columns[].min_value` | float | Minimum value for numeric columns |
| `columns[].max_value` | float | Maximum value for numeric columns |

See `examples/banking_schema.json` for a complete three-table example with FK relationships, weighted distributions, and nullable columns.
