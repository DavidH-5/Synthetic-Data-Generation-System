"""
agents/planner.py

Planner Agent — analyses the full GenerationManifest and produces a GenerationPlan.

Responsibilities:
  - Understand cross-table FK relationships
  - Determine generation order (topological sort)
  - Choose per-column generation strategy (faker / numpy / pandas / fk_sample)
  - Write TablePlan entries to the todo list

Does NOT write code. Does NOT execute anything.
Output is a strict GenerationPlan Pydantic model.
"""

from __future__ import annotations

import json

from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

from core.models import (
    GenerationManifest,
    GenerationPlan,
    TablePlan,
    TaskStatus,
    TodoTask,
)
from db.store import DataStore


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """
<role>
You are a data generation planner. Your sole job is to analyse a data schema
manifest and produce a precise, table-by-table generation plan.
You do NOT write code. You do NOT execute anything.
</role>

<output_contract>
Return ONLY a valid GenerationPlan JSON object.
No explanation text. No markdown fences. No preamble.
The JSON must be parseable by GenerationPlan.model_validate_json().
</output_contract>

<rules>

  <rule id="1" name="generation_order">
    Perform a topological sort of tables based on FK relationships.
    - Tables with no FK dependencies receive generation_order = 0.
    - A child table must have a strictly higher generation_order than all its parents.
    - Two tables with no dependency between them may share the same generation_order
      (they can be generated in parallel).
  </rule>

  <rule id="2" name="column_strategy_selection">
    For each column choose exactly one method from the options below.
    Think through each column carefully before assigning — wrong strategy
    is the most common cause of downstream code failures.

    <strategy name="faker">
      Use when: semantic_hint is set, OR column name implies a real-world entity
      (name, email, phone, address, company, date of birth, etc.)
      Set faker_provider to the EXACT faker method string, e.g.:
        "email", "name", "uuid4", "date_of_birth", "date_this_decade",
        "bsb_number" (Australian banking), "aba" (US routing number),
        "credit_card_number", "company", "job", "city", "postcode"
    </strategy>

    <strategy name="numpy">
      Use when: column has a numeric dtype AND a distribution spec
      (normal, uniform, poisson, exponential).
      Set numpy_call to the FULL expression including array size, e.g.:
        "np.random.normal(50000, 10000, n).round(2)"
        "np.random.exponential(scale=15000, size=n).clip(0, 500000).round(2)"
        "np.random.poisson(lam=3, size=n)"
    </strategy>

    <strategy name="fk_sample">
      Use when: column has a foreign_key spec.
      Set fk_source_table and fk_source_column exactly as declared in the manifest.
      The code writer will sample from the parent table's already-generated PK values.
      NEVER use faker or numpy for FK columns.
    </strategy>

    <strategy name="pandas">
      Use when: column is a simple sequential integer PK, a constant, or a
      derived column (e.g. pd.RangeIndex, pd.Categorical from an enum list).
    </strategy>

    <strategy name="constant">
      Use when: column takes a single fixed value for all rows.
    </strategy>
  </rule>

  <rule id="3" name="primary_keys">
    - UUID PKs (dtype="uuid"): use faker, faker_provider="uuid4", unique=True.
    - Integer PKs (dtype="int"): use pandas (range index), unique=True.
    - PKs are NEVER nullable.
  </rule>

  <rule id="4" name="dependencies">
    The dependencies list contains TABLE NAMES only (not column names).
    It must include every table whose PK is referenced by an FK in this table.
  </rule>

  <rule id="5" name="notes">
    Populate the notes field if:
    - A column has an unusual or non-obvious strategy choice.
    - A constraint interaction could cause issues for the code writer.
    - Volume combined with unique=True is close to the faker provider's cardinality limit.
  </rule>

</rules>

<faker_provider_reference>
  Personal:  name, first_name, last_name, prefix, suffix
  Contact:   email, phone_number, safe_email
  Location:  address, city, state, postcode, country, street_address
  Company:   company, job, bs, catch_phrase
  Identity:  uuid4, ssn, passport_number
  Finance:   iban, bsb_number, aba, credit_card_number, currency_code
  DateTime:  date_of_birth, date_this_decade, date_this_year,
             past_date, future_date, date_time_this_year
  Text:      sentence, paragraph, word, lexify, numerify, bothify
  Numeric:   random_int, random_element, pyfloat, pyint
</faker_provider_reference>
"""


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

planner_agent = Agent(
    model="anthropic:claude-sonnet-4-6",
    output_type=GenerationPlan,
    system_prompt=PLANNER_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Entry point called by the Orchestration Agent
# ---------------------------------------------------------------------------

async def run_planner(
    manifest: GenerationManifest,
    store: DataStore,
    revised_hints: str | None = None,
) -> GenerationPlan:
    """
    Run the Planner Agent against the full manifest.
    Writes resulting TablePlans to the todo list.

    revised_hints: optional guidance from the Orchestration Agent when replanning.
    """
    prompt = f"""
<task>
Analyse the manifest below and produce a complete GenerationPlan.
Apply all rules from your instructions. Think through FK relationships
and generation order before writing the output.
</task>

<manifest>
{manifest.model_dump_json(indent=2)}
</manifest>

{f"<revision_hints>{revised_hints}</revision_hints>" if revised_hints else ""}
"""

    result = await planner_agent.run(prompt)
    plan: GenerationPlan = result.output

    # Write each TablePlan to the todo list
    for table_plan in sorted(plan.tables, key=lambda t: t.generation_order):
        existing = store.get_task(table_plan.table_name)

        if existing:
            # Update existing task with new plan (replanning scenario)
            existing.table_plan = table_plan
            existing.generation_order = table_plan.generation_order
            existing.dependencies = table_plan.dependencies
            existing.status = TaskStatus.PLAN_READY
            store.upsert_task(existing)
        else:
            # First time — create new task
            task = TodoTask(
                table_name=table_plan.table_name,
                status=TaskStatus.PLAN_READY,
                generation_order=table_plan.generation_order,
                dependencies=table_plan.dependencies,
                table_plan=table_plan,
            )
            store.upsert_task(task)

        store.log_event(
            table_plan.table_name,
            "plan_written",
            f"order={table_plan.generation_order}, deps={table_plan.dependencies}",
        )

    return plan
