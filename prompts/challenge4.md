# Challenge 4 — Configurable scoring dimensions

Moves the four scoring dimensions out of code and into `dimensions.yaml`. The
Pydantic scoring schema and the scoring prompt are now both generated from that
one file. New: `dimensions.yaml`, `src/dimensions.py`, `tests/test_dimensions.py`,
`tests/test_dimension_wiring.py`. Modified: `src/models.py`, `src/prompts.py`,
and `pyproject.toml` (adds `pyyaml`).

## The problem

Four dimensions were written out by hand in two places that had to agree but had
nothing enforcing it:

```python
# src/models.py                  # src/prompts.py
skills_match: DimensionScore     - `skills_match`: overlap between ...
experience_match: DimensionScore - `experience_match`: does the candidate's ...
role_relevance: DimensionScore   - `role_relevance`: how closely do the ...
overall_fit: DimensionScore      - `overall_fit`: your holistic judgment ...
```

Plus a third copy in `ScoredCandidate`. Renaming a dimension in the schema
without renaming it in the prompt gives Claude a tool whose fields don't match
the instructions — a failure that shows up as degraded scores, not as an error.

## The shape of the fix

`dimensions.yaml` holds `key`, `label`, `prompt_description` per dimension.
`src/dimensions.py` loads and validates it at import into frozen `Dimension`
records. Then:

- `models.py` builds one dynamic base with `create_model()` and has both
  `ScoreReport` and `ScoredCandidate` inherit it, so the dimension fields exist
  in exactly one place.
- `prompts.py` renders the bullet block and interpolates it into
  `SCORING_SYSTEM_PROMPT`.

`src/dimensions.py` imports nothing from the rest of `src` — not even Pydantic.
That is load-bearing: `models.py` imports `DIMENSIONS`, so any import back the
other way would be a cycle.

## Why it was built in three scoped stages

The natural way to do this is one pass over all the files at once. That was
avoided deliberately, because this refactor has an unusual risk profile: it
touches the prompt, and prompt wording moves eval scores. A change that silently
reflows a sentence costs a full eval run to detect and is easy to misattribute.

So the work was cut so that each stage could be verified before the next
started:

1. **Stage 0 — read-only investigation.** Map the blast radius before writing
   anything.
2. **Stage 1 — new files only.** Config, loader, tests. Nothing imports it yet,
   so nothing can break.
3. **Stage 2 — wire it in.** Two files, guarded by golden snapshots written
   *before* the refactor.

Each stage ended with "stop for review." The scope fence was restated in every
prompt as an explicit file list, including the negative list.

## Authoring prompts

Stage 0 is reproduced verbatim. Stages 1 and 2 are condensed from a multi-turn
session; substance preserved.

**0 — Investigation and mapping (read-only).**

> I'm working on Challenge 4 from `CONTRIBUTING.MD`: making the scoring
> dimensions configurable from a single source of truth. Before we write any
> code, I want you to map how the four scoring dimensions (skills, experience,
> role relevance, overall fit) are currently wired into the codebase. **Do not
> change any files yet** — this is investigation only.
>
> Please find and show me:
>
> 1. Where the Pydantic schema for a candidate's scores is defined (file + the
>    class).
> 2. Where the scoring prompt is built, and the exact text that mentions the four
>    dimensions.
> 3. Anywhere the dimensions are referenced elsewhere — score aggregation, the
>    markdown/JSON report generation, and the eval cases in `tests/evals/`.
> 4. How "overall fit" is produced specifically: does the LLM score it directly,
>    or is it computed/aggregated from the other three?
>
> For each, quote the relevant lines and give me the file path. Then summarize in
> a few sentences: how many distinct places would I have to edit today to add or
> rename one dimension? End with your recommendation for where a single source of
> truth (a config file) should plug in, but don't implement it yet — I'll review
> your map first.

Question 4 is the one that paid off. `overall_fit` is scored directly by the LLM
as a fourth peer dimension — the prompt calls it "your holistic judgment — not a
simple average of the above" — rather than computed from the other three. Had it
been derived, it could not have lived in the same flat config list as the
others, and the whole design would have needed a distinction between scored and
computed dimensions. Asking before building is what kept that out of the design.

The demand to *quote lines with file paths* rather than summarize is what
produced the count of hardcoded sites that later set the stage boundaries.

**1 — Config, loader, tests (new files only).** Stage one is new files only —
do not modify any existing file. Create three files:

- `dimensions.yaml` — the four current dimensions in their existing order. Each
  entry has `key` (the field name exactly as used today), `label` (human
  readable), and `prompt_description` (the exact sentence currently used for
  that dimension in `SCORING_SYSTEM_PROMPT`, copied verbatim — we must not
  change prompt wording, since it affects eval scores).
- `src/dimensions.py` — loads and validates `dimensions.yaml` at import and
  exposes a `DIMENSIONS` list of small immutable records. Keep this module pure
  data: it must **not** import from `models.py` or depend on Pydantic model
  classes, so stage two can import `DIMENSIONS` into `models.py` without a
  circular import. Validation must reject, with clear error messages: an empty
  dimension list, duplicate keys, and any entry missing or with an empty `key`
  or `prompt_description`.
- `tests/test_dimensions.py` — pytest, no LLM calls. Assert the default file
  loads to exactly the four expected keys in order, each record has a non-empty
  `prompt_description`, and each invalid case above raises.

Read the current `SCORING_SYSTEM_PROMPT` to copy the descriptions exactly, then
run the new tests and show the results and the final `dimensions.yaml`.

**2 — Wire it in, behind snapshots.** Touch only `src/models.py`,
`src/prompts.py`, plus one new test file. If anything else seems to need
changing to keep the app working, **stop and say so** — do not expand scope.

First, lock in the safety net *before* refactoring anything. Create
`tests/test_dimension_wiring.py` capturing two golden snapshots of current
behavior: the exact current value of `SCORING_SYSTEM_PROMPT` (read it from
`src/prompts.py` and embed it verbatim as the expected value), and the exact
current `ScoreReport.model_json_schema()`.

Then refactor. In `models.py`, build a single dynamic base and have both classes
inherit it:

```python
DimensionScores = create_model(
    "DimensionScores",
    **{d.key: (DimensionScore, ...) for d in DIMENSIONS},
)

class ScoreReport(DimensionScores):
    reasoning: str
    gaps: list[Gap]
```

Preserve every non-dimension field exactly as it is today.
`CritiqueReport.revised_scores` stays typed as `ScoreReport`. In `prompts.py`,
render the bullet block from `DIMENSIONS` — format each line exactly as today
(`` - `{key}`: {prompt_description} ``, same order) and keep the entire rest of
the prompt static.

Then assert the snapshots hold: the rendered `SCORING_SYSTEM_PROMPT` must equal
the golden string byte-for-byte, `ScoreReport.model_json_schema()` must equal
the golden schema, and both models must expose exactly the expected field names.
Run the full suite, not just the new file. Do not run the live LLM evals.

## The lesson worth recording: snapshot before you refactor

The ordering matters more than the snapshots do. `tests/test_dimension_wiring.py`
was written and **run green against the unrefactored code** before `models.py`
or `prompts.py` were touched. Only then did the refactor start.

Written the other way round — refactor first, snapshot after — the test would
have captured whatever the new code produced and passed trivially, including a
reflowed prompt or a reordered schema. It would look like a safety net and catch
nothing.

Two properties keep it honest:

- The goldens are **hand-written longhand**, not generated from `DIMENSIONS`. A
  snapshot derived from the code it is checking proves nothing. This is why the
  test file repeats the four dimension names and the full prompt text that the
  rest of the change works to eliminate — the duplication is the point.
- The schema golden is the exact dict `scorer.py` passes as the `record_score`
  `input_schema`, so it pins the actual API payload: property order, `required`
  order, `$defs`, and the `reasoning` description.

The snapshots caught the em dash in `overall_fit`'s description surviving the
YAML folded-scalar round-trip — a detail that no eval run would have isolated.

## What stayed hardcoded, and why that's the right scope

`CONTRIBUTING.MD` asks for one thing: *"The prompt and the Pydantic schema
should both update from one source of truth."* That is what shipped.

Seven other places still name the dimensions explicitly — `scorer.py` and
`critique.py` (explicit kwargs), `reporter.py` (`_DIMENSIONS`, the markdown
breakdown, the drift table), `bias_auditor.py` (`_DIMENSIONS`), `evals.py`,
`app.py`, and the `expected_*` fields in `tests/evals/cases.py`. **Adding a
fifth dimension to the YAML today updates the schema and the prompt, then fails
at runtime** when `scorer.py` constructs a `ScoredCandidate` without the new
required field.

That boundary was found in stage 0 and left in place on purpose: making those
seven dynamic is a much larger change, and several of them are genuinely better
explicit (`cases.py` expected ranges are per-dimension assertions; `app.py`
column labels are layout). Documenting the boundary is worth more than a
half-finished generalization — a reader who believes "edit the YAML and you're
done" will be surprised at runtime, so `CLAUDE.md` and the README both say so
plainly.

## Verification

```
uv run pytest
63 passed
```

Up from 34: 16 in `test_dimensions.py`, 13 in `test_dimension_wiring.py`.

Both goldens hold — the prompt is byte-for-byte identical and the `record_score`
schema is unchanged, so no eval rerun was required for this change. Live evals
were deliberately not run: with a provably identical prompt and schema, a score
difference could only be model nondeterminism, which would be noise, not signal.

One incidental behavior change: `ScoredCandidate`'s field order shifted, because
Pydantic places base-class fields first.

```
before: profile, skills_match, ..., overall_fit, reasoning, gaps, source_file
after:  skills_match, ..., overall_fit, profile, reasoning, gaps, source_file
```

Cosmetic — it only affects key order inside each candidate object in
`results.json`. Every construction site passes keywords, and `ScoreReport`'s
order (the one on the wire) is untouched.

## Possible follow-up

Derive `reporter._DIMENSIONS` and `bias_auditor._DIMENSIONS` from
`DIMENSION_KEYS`, and have `scorer.py` / `critique.py` copy dimension fields
generically rather than by name. That would close most of the gap above and make
a fifth dimension a genuinely YAML-only change everywhere except the eval cases.
`label` is already carried through `dimensions.yaml` for exactly this — the
markdown report and the Streamlit column headers currently hardcode strings it
could supply.
