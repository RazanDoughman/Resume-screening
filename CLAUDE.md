# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching project that scores resume PDFs against a job description using the Claude API. Each resume goes through a pipeline: PDF parse -> structured extraction (LLM) -> scoring (LLM) -> optional self-critique (LLM) -> optional bias audit (LLM × N swaps). Output is a ranked JSON file, markdown report, and optional bias audit JSON.

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
6. `reporter.py` — writes `results.json`, `report.md`, and (if audit ran) `bias_audit.json`. No LLM.

**Key files:**
- `src/models.py` — all Pydantic models (`CandidateProfile`, `ScoreReport`, `ScoredCandidate`, `CritiqueReport`, `ProcessingError`, `BiasVariant`, `BiasAuditReport`). These are the contracts between modules.
- `src/prompts.py` — all LLM prompts in one place. Edit prompts here, not in the module files.
- `src/bias_auditor.py` — swap sets (`NAME_SWAPS`, `GRAD_YEAR_SWAPS`, `LOCATION_SWAPS`, `ALL_SWAPS`), mutation logic, drift math, and the `run_bias_audit()` entry point.
- `src/evals.py` — eval harness runner (scoring cases + bias stability cases)
- `tests/evals/cases.py` — golden eval cases (`ALL_CASES`) and bias stability cases (`ALL_BIAS_CASES`)

## Design conventions

- Default model is `claude-sonnet-4-6`, set in `src/extractor.py` as `DEFAULT_MODEL`. `scorer.py`, `critique.py`, and `bias_auditor.py` redefine the same constant locally — keep them in sync.
- Errors in individual resumes produce a `ProcessingError` (`stage` is one of `parse`/`extract`/`score`) instead of crashing the batch. A failure inside the optional critique or bias audit pass does **not** produce a `ProcessingError`; it logs and skips, keeping whatever valid work was already done.
- `scorer.py` also exposes `score_candidate_ensemble()` which runs N scoring calls and returns median scores per dimension. Reasoning text is taken from the first run only — don't try to merge text across runs.
- Evals test score ranges (e.g., 80-100 for strong match) because LLM output varies between calls. Add new scoring eval cases by appending to `ALL_CASES` in `tests/evals/cases.py`. Add new bias stability cases by appending to `ALL_BIAS_CASES` using the `BiasAuditEvalCase` dataclass.
- Bias audit eval cases use only `NAME_SWAPS` (4 variants) to keep cost manageable. Production runs use `ALL_SWAPS` (name + grad year = 5 variants). `LOCATION_SWAPS` is defined in `bias_auditor.py` but excluded from `ALL_SWAPS` by default — pass it explicitly via `run_bias_audit(swaps=...)` to opt in.
- Per `CONTRIBUTING.MD`, prompts are treated as the teaching artifact: any prompt change should be visible in the PR description (or saved under `prompts/`), and new conventions should be reflected back into this file.