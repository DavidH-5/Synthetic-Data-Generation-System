"""
core/models.py

All Pydantic models used across agents.
These are the inter-agent contracts — nothing passes between agents as raw dicts or strings.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input manifest — parsed from the user's JSON spec
# ---------------------------------------------------------------------------

class DistributionSpec(BaseModel):
    """Optional statistical distribution for numeric columns."""
    type: Literal["normal", "uniform", "poisson", "exponential", "choice"]
    # normal
    mean: float | None = None
    std: float | None = None
    # uniform
    low: float | None = None
    high: float | None = None
    # poisson
    lam: float | None = None
    # exponential
    scale: float | None = None
    # choice (enum values)
    values: list[Any] | None = None
    weights: list[float] | None = None


class ColumnSpec(BaseModel):
    """Definition of a single column."""
    name: str
    dtype: Literal["int", "float", "str", "bool", "date", "datetime", "uuid"]
    nullable: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of NULLs")
    distribution: DistributionSpec | None = None
    # Constraints
    min_value: float | None = None
    max_value: float | None = None
    regex: str | None = None
    unique: bool = False
    primary_key: bool = False
    foreign_key: ForeignKeySpec | None = None
    # Hints for the planner
    semantic_hint: str | None = Field(
        default=None,
        description="e.g. 'email', 'full_name', 'transaction_amount' — guides faker selection"
    )


class ForeignKeySpec(BaseModel):
    """Describes a FK relationship to another table."""
    references_table: str
    references_column: str
    cardinality: Literal["1:1", "1:N", "M:N"] = "1:N"


class TableSpec(BaseModel):
    """Definition of a single table."""
    name: str
    volume: int = Field(ge=1, description="Number of rows to generate")
    columns: list[ColumnSpec]


class GenerationManifest(BaseModel):
    """Root input model. Parsed from the user's JSON file."""
    tables: list[TableSpec]
    seed: int = 42
    description: str | None = None


# ---------------------------------------------------------------------------
# Planner output — the GenerationPlan
# ---------------------------------------------------------------------------

class ColumnStrategy(BaseModel):
    """How the planner intends to generate a single column."""
    column_name: str
    method: Literal["faker", "numpy", "pandas", "fk_sample", "constant"]
    faker_provider: str | None = Field(
        default=None,
        description="e.g. 'email', 'name', 'uuid4' — exact faker method name"
    )
    numpy_call: str | None = Field(
        default=None,
        description="e.g. 'np.random.normal(50000, 10000, n)'"
    )
    fk_source_table: str | None = None
    fk_source_column: str | None = None
    notes: str | None = None


class TablePlan(BaseModel):
    """The planner's complete strategy for one table."""
    table_name: str
    volume: int
    generation_order: int = Field(description="Lower = generate first. Parents before children.")
    column_strategies: list[ColumnStrategy]
    dependencies: list[str] = Field(
        default_factory=list,
        description="Table names that must be generated before this one"
    )
    notes: str | None = None


class GenerationPlan(BaseModel):
    """Full plan produced by the Planner Agent. One TablePlan per table."""
    tables: list[TablePlan]
    seed: int


# ---------------------------------------------------------------------------
# Code Writer output
# ---------------------------------------------------------------------------

class CodeBlock(BaseModel):
    """
    The Code Writer Agent's output for one table.
    Always produces a function with this exact signature:
        def generate(n: int, seed: int, fk_context: dict[str, list]) -> pd.DataFrame
    """
    table_name: str
    code: str = Field(description="Complete Python function as a string, including imports")
    imports: list[str] = Field(description="Top-level imports declared in the code")
    error_classification: Literal[
        "none",
        "syntax_error",
        "runtime_error",
        "strategy_mismatch",
        "constraint_conflict",
    ] = "none"
    self_fixable: bool = Field(
        default=True,
        description="False means the plan itself is wrong, not just the code"
    )
    error_detail: str | None = None


# ---------------------------------------------------------------------------
# Validation output
# ---------------------------------------------------------------------------

class ColumnValidationResult(BaseModel):
    column_name: str
    passed: bool
    failure_type: Literal[
        "dtype_mismatch",
        "null_rate_exceeded",
        "distribution_mismatch",
        "unique_violation",
        "range_violation",
        "regex_violation",
        "ok",
    ] = "ok"
    detail: str | None = None


class ValidationResult(BaseModel):
    """
    Output from the Validation Agent for one table.
    Failure classifications guide the Orchestration Agent's routing decision.
    """
    table_name: str
    passed: bool
    row_count_ok: bool
    referential_integrity_ok: bool
    column_results: list[ColumnValidationResult]
    failure_classification: Literal[
        "ok",
        "code_bug",           # fix by retrying Code Writer
        "plan_error",         # fix by asking Planner to revise
        "spec_conflict",      # unresolvable — surface to user
    ] = "ok"
    suggestions: str | None = Field(
        default=None,
        description="Free-text guidance for the Orchestration Agent's replanning decision"
    )


# ---------------------------------------------------------------------------
# TodoList — the shared blackboard in SQLite
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    PLAN_READY = "plan_ready"
    GENERATING = "generating"
    CODE_READY = "code_ready"
    DATA_READY = "data_ready"
    VALIDATING = "validating"
    DONE = "done"
    FAILED = "failed"
    UNRESOLVABLE = "unresolvable"


class TodoTask(BaseModel):
    """One row in the todo_list table."""
    table_name: str
    status: TaskStatus = TaskStatus.PENDING
    generation_order: int = 0
    dependencies: list[str] = Field(default_factory=list)

    # Artefacts written by each agent
    table_plan: TablePlan | None = None
    code_block: CodeBlock | None = None
    validation_result: ValidationResult | None = None

    # Orchestration audit (mirrors todo_list replan_* columns)
    replan_action: str = "none"
    replan_reasoning: str | None = None
    replan_hints: str | None = None

    # Tracking
    attempt_count: int = 0
    replan_count: int = 0
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ReplanDecision(BaseModel):
    """
    Output of the Orchestration Agent when it decides what to do after a failure.
    Explicit and structured so the decision is auditable.
    """
    table_name: str
    action: Literal[
        "retry_code",        # send back to Code Writer with error context
        "replan_table",      # send back to Planner for this table only
        "replan_full",       # send back to Planner for the full manifest
        "mark_unresolvable", # give up and surface to user
    ]
    reasoning: str
    revised_hints: str | None = Field(
        default=None,
        description="Guidance injected into the next Planner or Code Writer call"
    )
