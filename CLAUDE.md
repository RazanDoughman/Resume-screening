# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching project that scores resume PDFs against a job description using the Claude API. Each resume goes through a pipeline: PDF parse -> structured extraction (LLM) -> scoring (LLM) -> optional self-critique (LLM) -> optional bias audit (LLM × N swaps). Output is a ranked JSON file, markdown report, token/cost usage JSON, optional CSV export, and optional bias audit JSON.

## Commands

```bash
# Install dependencies
uv sync

# Generate sample resume PDFs (not checked in)
uv run python scripts/generate_sample_resumes.py

# Run the pipeline
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/

# Override output directory (default ./output) or model (default DEFAULT_MODEL)
python main.py --jd ... --resumes ... --output out/ --model claude-sonnet-4-6

# Run with self-critique (adds one extra LLM call per resume)
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/ --self-critique

# Run with bias audit (re-scores each candidate with demographic signals swapped)
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/ --bias-audit

# Run with CSV export (adds a flat results.csv — deterministic, no extra LLM calls)
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/ --csv

# Run the Streamlit web UI
uv run streamlit run app.py

# Unit tests (no LLM calls, fast)
uv run pytest

# Run a single test
uv run pytest tests/test_pdf_parser.py

# Evals (makes real LLM calls, checks score ranges + bias stability)
uv run python -m src.evals

# Run only bias audit eval cases
uv run python -m src.evals --bias
```

## Environment

Requires `ANTHROPIC_API_KEY` in a `.env` file (loaded via `python-dotenv`). See `.env.example`.

## Architecture

**Core pattern repeated in every LLM module:** define a tool from a Pydantic model's JSON schema -> force Claude to call it via `tool_choice` -> validate the response back into a Pydantic object. This appears in `extractor.py`, `scorer.py`, `critique.py`, and the optional summarization call in `bias_auditor.py`.

**Two entry points:**
- `main.py` — CLI that loops over resume PDFs sequentially
- `app.py` — Streamlit web UI with file upload

Both entry points define their own `process_one()` that wires the same parse → extract → score (→ critique) (→ bias audit) pipeline. They are **not** sharing a helper — if you change the pipeline shape, update both files. Both `main.py` and `app.py` return `tuple[Result, BiasAuditReport | None]` from `process_one()` and fully support bias audit.

**Pipeline stages (each in its own module under `src/`):**
1. `pdf_parser.py` — pdfplumber text extraction, no LLM
2. `extractor.py` — resume text -> `CandidateProfile` (LLM call #1, tool: `record_candidate`)
3. `scorer.py` — profile + JD -> `ScoredCandidate` (LLM call #2, tool: `record_score`). Uses prompt caching on the JD text block (`cache_control: ephemeral`)
4. `critique.py` — optional review pass (LLM call #3, tool: `record_critique`)
5. `bias_auditor.py` — optional bias audit (LLM call × N swap variants). Re-scores the candidate with demographic signals swapped (name, graduation year, location) and computes score drift per dimension.
6. `reporter.py` — writes `results.json`, `report.md`, `usage.json`, (if `--csv`) `results.csv`, and (if audit ran) `bias_audit.json`. No LLM. `report.md` opens with an ASCII histogram of the `overall_fit` distribution — always on, no flag.

**Cross-cutting, not a stage:** `usage.py` meters every LLM call in the pipeline by wrapping the client, not by participating in it. See the `MeteredClient` convention below.

**Key files:**
- `dimensions.yaml` — the scoring dimensions (`key`, `label`, `prompt_description`), in the order they appear in the prompt and the report. Single source of truth for the scoring schema and the scoring prompt.
- `src/dimensions.py` — loads and validates `dimensions.yaml` at import time, exposing `DIMENSIONS` (frozen `Dimension` records) and `DIMENSION_KEYS`. Deliberately pure data: it imports nothing from the rest of `src`, so `models.py` can import it without a cycle. Malformed config raises `DimensionConfigError` on import rather than failing later mid-run.
- `src/models.py` — all Pydantic models (`CandidateProfile`, `ScoreReport`, `ScoredCandidate`, `CritiqueReport`, `ProcessingError`, `BiasVariant`, `BiasAuditReport`). These are the contracts between modules.
- `src/prompts.py` — all LLM prompts in one place. Edit prompts here, not in the module files. The one exception is the per-dimension bullet block in `SCORING_SYSTEM_PROMPT`, which is rendered from `dimensions.yaml`.
- `src/bias_auditor.py` — swap sets (`NAME_SWAPS`, `GRAD_YEAR_SWAPS`, `LOCATION_SWAPS`, `ALL_SWAPS`), mutation logic, drift math, and the `run_bias_audit()` entry point.
- `src/usage.py` — token accounting and cost estimation. Owns `APIUsage` (one normalized record per API response), `UsageTracker`, `MeteredClient`, `MODEL_PRICING`, `estimate_cost()`, `build_report()`, and `format_summary()`. Imports nothing from the rest of `src`, and writes no files.
- `src/evals.py` — eval harness runner (scoring cases + bias stability cases)
- `tests/evals/cases.py` — golden eval cases (`ALL_CASES`) and bias stability cases (`ALL_BIAS_CASES`)

## Design conventions

- Default model is `claude-sonnet-4-6`, set in `src/extractor.py` as `DEFAULT_MODEL`. `scorer.py`, `critique.py`, and `bias_auditor.py` redefine the same constant locally — keep them in sync.
- Errors in individual resumes produce a `ProcessingError` (`stage` is one of `parse`/`extract`/`score`) instead of crashing the batch. A failure inside the optional critique or bias audit pass does **not** produce a `ProcessingError`; it logs and skips, keeping whatever valid work was already done.
- CSV export is opt-in via `--csv` and fully deterministic — no extra LLM calls. `write_csv()` flattens each `ScoredCandidate` into one row: nested `DimensionScore` objects become `{dimension}_score` + `{dimension}_reasoning` columns, and list fields (skills, education, gaps) join into one cell with `_LIST_SEP`. `ProcessingError` entries are omitted by design — a CSV holds one schema per file, so errors stay in `results.json`. Column order is derived from `reporter.py`'s own `_DIMENSIONS` tuple — note this is a separate hardcoded list, not `dimensions.yaml`.
- `scorer.py` also exposes `score_candidate_ensemble()` which runs N scoring calls and returns median scores per dimension. Reasoning text is taken from the first run only — don't try to merge text across runs.
- Evals test score ranges (e.g., 80-100 for strong match) because LLM output varies between calls. Add new scoring eval cases by appending to `ALL_CASES` in `tests/evals/cases.py`. Add new bias stability cases by appending to `ALL_BIAS_CASES` using the `BiasAuditEvalCase` dataclass.
- Bias audit eval cases use only `NAME_SWAPS` (4 variants) to keep cost manageable. Production runs use `ALL_SWAPS` (name + grad year = 5 variants). `LOCATION_SWAPS` is defined in `bias_auditor.py` but excluded from `ALL_SWAPS` by default — pass it explicitly via `run_bias_audit(swaps=...)` to opt in.
- Scoring dimensions are configured in `dimensions.yaml`, not in code. Both consumers derive from it: `models.py` builds a dynamic `DimensionScores` base with `create_model()` that `ScoreReport` and `ScoredCandidate` inherit, and `prompts.py` renders the `` - `key`: description `` bullet block inside `SCORING_SYSTEM_PROMPT` from the same list. **Don't hand-edit dimension fields in the models or dimension bullets in the prompt** — both are generated, and an edit there will be silently overwritten by the config. Change the YAML.
- `prompt_description` is prompt text, so wording changes move eval scores. Treat an edit there as a prompt change: rerun `uv run python -m src.evals` and surface it in the PR like any other prompt edit. `key` is worse to change casually — it is the field name in `ScoreReport`, in `results.json`, and the `{key}_score` / `{key}_reasoning` CSV column prefix.
- `tests/test_dimension_wiring.py` pins the exact `SCORING_SYSTEM_PROMPT` string and the exact `ScoreReport.model_json_schema()` as hand-written golden snapshots. They are deliberately *not* derived from `DIMENSIONS` — a snapshot generated from the code it checks proves nothing. An intentional dimension change should update the goldens in the same commit; an unintentional one fails the suite.
- **The single source of truth currently covers the schema and the prompt only.** Call sites that reference dimensions by name are still hardcoded: `scorer.py` and `critique.py` (explicit kwargs), `reporter.py` (`_DIMENSIONS` plus the markdown breakdown and drift table), `bias_auditor.py` (`_DIMENSIONS`), `evals.py`, `app.py`, and the `expected_*` fields in `tests/evals/cases.py`. Adding a fifth dimension to the YAML updates the schema and prompt but will fail at runtime until those are updated too — see `prompts/challenge4.md`.
- **Usage is metered at the API-client boundary, not in the pipeline.** `main.py` and `app.py` build a `UsageTracker` and pass `MeteredClient(Anthropic(), tracker)` where the raw client used to go. `MeteredClient` exposes only `.messages`, calls through to the real client, records `response.usage`, and returns the *same* response object. The five pipeline modules are unchanged and know nothing about it. Do **not** add usage to return types as tuples, thread a `tracker=` parameter through the pipeline, or introduce a module-level counter — all three were considered and rejected. Recording happens *before* the caller pulls out the tool_use block and validates it, so a billed call whose response fails `model_validate` is still counted; the report describes API work observed, not successful candidates.
- **Usage never belongs in a domain model.** Nothing in `src/models.py` — `CandidateProfile`, `ScoreReport`, `ScoredCandidate`, `ProcessingError` — carries token counts or cost. Usage is operational telemetry; putting it on a candidate would leak it into `results.json`, the CSV columns, and the tool schemas Claude is handed. `usage.py` does not import from `models.py` and must not start.
- **`usage.input_tokens` is the uncached input only.** The real prompt size is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`, exposed as `total_prompt_tokens`. Both cache fields are `Optional[int]` in the SDK and arrive as `None` on uncached calls; `APIUsage.from_response()` is the single place that normalizes them to `0`. Report cached *token counts* — the API gives no cache-hit event count, so don't invent one.
- Pricing lives in `MODEL_PRICING` in `src/usage.py` as a plain dict of `Decimal` rates, with a comment recording the official source URL and the date verified. It is deliberately not a YAML config: unlike `dimensions.yaml` it has one consumer and changing it alters no model behavior. Update the date whenever you touch the numbers. Cost is computed per record from `response.model` (the *resolved* model, not the requested one) and is always called an **estimate**; an unknown model yields `None`, never `0.0`.
- `MeteredClient` is passed where the pipeline annotates `client: Anthropic`. That mismatch is intentional and harmless — annotations aren't enforced here, and `.messages` is the only client attribute any pipeline module touches. Widening those annotations would mean editing five files to satisfy a hint nothing checks.
- Per `CONTRIBUTING.MD`, prompts are treated as the teaching artifact: any prompt change should be visible in the PR description (or saved under `prompts/`), and new conventions should be reflected back into this file.