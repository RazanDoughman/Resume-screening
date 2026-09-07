# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching project that scores resume PDFs against a job description using the Claude API. Each resume goes through a pipeline: PDF parse -> structured extraction (LLM) -> scoring (LLM) -> optional self-critique (LLM) -> optional bias audit (LLM × N swaps). After the whole batch is scored, an optional deep-dive stage ranks the field and writes a hiring-manager briefing for the top N candidates (LLM × N selected). A separate offline tool (`scripts/gen_eval_cases.py`) generates synthetic resumes at four match levels for a JD, saving them as unreviewed proposals that become eval cases only after a human approves them. Output is a ranked JSON file, markdown report, token/cost usage JSON, optional CSV export, and optional bias audit JSON.

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

# Run with deep-dive briefings for the top N candidates (adds N LLM calls total)
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/ --deep-dive-top 3

# Run the Streamlit web UI
uv run streamlit run app.py

# Override the concurrency limit (default 5; 1 = sequential)
python main.py --jd ... --resumes ... --concurrency 5

# Unit tests (no LLM calls, fast)
uv run pytest

# Benchmark one pipeline configuration (PAID — makes real API calls)
uv run python -m scripts.benchmark --label sequential --concurrency 1
uv run python -m scripts.benchmark --dry-run    # prints the plan, spends nothing

# Run a single test
uv run pytest tests/test_pdf_parser.py

# Evals (makes real LLM calls, checks score ranges + bias stability)
uv run python -m src.evals

# Run only bias audit eval cases
uv run python -m src.evals --bias

# Run only deep-dive eval cases
uv run python -m src.evals --deep-dive

# Generate synthetic eval-case proposals for a JD (PAID — 4 LLM calls)
uv run python -m scripts.gen_eval_cases --jd sample_data/sample_jd.txt
uv run python -m scripts.gen_eval_cases --jd ... --dry-run   # prints the plan, spends nothing
```

## Environment

Requires `ANTHROPIC_API_KEY` in a `.env` file (loaded via `python-dotenv`). See `.env.example`.

## Architecture

**Core pattern repeated in every LLM module:** define a tool from a Pydantic model's JSON schema -> force Claude to call it via `tool_choice` -> validate the response back into a Pydantic object. This appears in `extractor.py`, `scorer.py`, `critique.py`, and the optional summarization call in `bias_auditor.py`.

**Two entry points:**
- `main.py` — CLI that processes resume PDFs concurrently (`--concurrency`, default 5)
- `app.py` — Streamlit web UI with file upload

Both entry points define their own `process_one()` that wires the same parse → extract → score (→ critique) (→ bias audit) pipeline. They are **not** sharing a helper — if you change the pipeline shape, update both files. Both `main.py` and `app.py` return `tuple[Result, BiasAuditReport | None]` from `process_one()` and fully support bias audit.

**Pipeline stages (each in its own module under `src/`):**
1. `pdf_parser.py` — pdfplumber text extraction, no LLM
2. `extractor.py` — resume text -> `CandidateProfile` (LLM call #1, tool: `record_candidate`)
3. `scorer.py` — profile + JD -> `ScoredCandidate` (LLM call #2, tool: `record_score`). Uses prompt caching on the JD text block (`cache_control: ephemeral`)
4. `critique.py` — optional review pass (LLM call #3, tool: `record_critique`)
5. `bias_auditor.py` — optional bias audit (LLM call × N swap variants). Re-scores the candidate with demographic signals swapped (name, graduation year, location) and computes score drift per dimension.
6. `deep_dive.py` — optional hiring-manager briefing (LLM call × N selected, tool: `record_deep_dive`). **Not a per-resume stage.** It runs once, after the whole batch, on the top N candidates only, so its cost is O(N selected) rather than O(resumes). Reads a `ScoredCandidate` — profile, every dimension's score and reasoning, and the gaps — never the raw resume text, which has gone out of scope by then in both entry points.
7. `reporter.py` — writes `results.json`, `report.md`, `usage.json`, (if `--csv`) `results.csv`, and (if audit ran) `bias_audit.json`. No LLM. `report.md` opens with an ASCII histogram of the `overall_fit` distribution — always on, no flag. Also owns `rank_candidates()`, the project's single ranking definition.

**Cross-cutting, not a stage:** `usage.py` meters every LLM call in the pipeline by wrapping the client, not by participating in it. See the `MeteredClient` convention below.

**Key files:**
- `dimensions.yaml` — the scoring dimensions (`key`, `label`, `prompt_description`), in the order they appear in the prompt and the report. Single source of truth for the scoring schema and the scoring prompt.
- `src/dimensions.py` — loads and validates `dimensions.yaml` at import time, exposing `DIMENSIONS` (frozen `Dimension` records) and `DIMENSION_KEYS`. Deliberately pure data: it imports nothing from the rest of `src`, so `models.py` can import it without a cycle. Malformed config raises `DimensionConfigError` on import rather than failing later mid-run.
- `src/models.py` — all Pydantic models (`CandidateProfile`, `ScoreReport`, `ScoredCandidate`, `CritiqueReport`, `ProcessingError`, `BiasVariant`, `BiasAuditReport`, `DeepDiveReport`). These are the contracts between modules.
- `src/prompts.py` — all LLM prompts in one place. Edit prompts here, not in the module files. The one exception is the per-dimension bullet block in `SCORING_SYSTEM_PROMPT`, which is rendered from `dimensions.yaml`.
- `src/bias_auditor.py` — swap sets (`NAME_SWAPS`, `GRAD_YEAR_SWAPS`, `LOCATION_SWAPS`, `ALL_SWAPS`), mutation logic, drift math, and the `run_bias_audit()` entry point.
- `src/deep_dive.py` — `generate_deep_dive()` and the `_screening_results()` serializer. One LLM call, one candidate, nothing else: no ranking, no file I/O, no rendering, no client construction. `main.py` owns selection and hands it whichever candidate to brief.
- `src/usage.py` — token accounting and cost estimation. Owns `APIUsage` (one normalized record per API response), `UsageTracker`, `MeteredClient`, `MODEL_PRICING`, `estimate_cost()`, `build_report()`, and `format_summary()`. Imports nothing from the rest of `src`, and writes no files.
- `src/evals.py` — eval harness runner (scoring cases + bias stability cases)
- `src/eval_gen.py` — Challenge 7 generation: `generate_resume_proposal()` (one LLM call, one match level) and `generate_proposal_set()` (four levels, sequential). Writes no files, constructs no client.
- `src/eval_fixtures.py` — writes/reads the raw proposal artifacts under `tests/evals/generated/`. No LLM, no Anthropic import — `reporter.py`'s counterpart for generation.
- `src/generated_cases.py` — the human-review gate. `promote_proposal()`, `write_generated_cases()`, `load_eval_case_kwargs()`. No LLM.
- `scripts/gen_eval_cases.py` — Challenge 7 CLI. The only place that builds an `Anthropic()` for generation.
- `tests/evals/cases.py` — `HANDWRITTEN_CASES` (hand-written), `GENERATED_CASES` (reviewed generated), `ALL_CASES` = both; plus `ALL_BIAS_CASES` and `ALL_DEEP_DIVE_CASES`
- `tests/evals/generated_cases.json` — reviewed generated cases (the promotion output)
- `tests/evals/generated/` — raw unreviewed proposals (generator provenance)

## Design conventions

- Default model is `claude-sonnet-4-6`, set in `src/extractor.py` as `DEFAULT_MODEL`. `scorer.py`, `critique.py`, and `bias_auditor.py` redefine the same constant locally — keep them in sync.
- Errors in individual resumes produce a `ProcessingError` (`stage` is one of `parse`/`extract`/`score`) instead of crashing the batch. A failure inside the optional critique, bias audit, or deep-dive pass does **not** produce a `ProcessingError`; it logs and skips, keeping whatever valid work was already done. Do not widen the `stage` Literal to cover them — a failed briefing is not a failed resume, and widening it would change the `results.json` contract.
- CSV export is opt-in via `--csv` and fully deterministic — no extra LLM calls. `write_csv()` flattens each `ScoredCandidate` into one row: nested `DimensionScore` objects become `{dimension}_score` + `{dimension}_reasoning` columns, and list fields (skills, education, gaps) join into one cell with `_LIST_SEP`. `ProcessingError` entries are omitted by design — a CSV holds one schema per file, so errors stay in `results.json`. Column order is derived from `reporter.py`'s own `_DIMENSIONS` tuple — note this is a separate hardcoded list, not `dimensions.yaml`.
- **Deep-dive is a post-scoring batch stage, not a pipeline stage.** It lives in `run()`, after the resume loop and before `build_report()`, never inside `process_one()` — "top N" is undefined until the whole field has been scored. That placement is load-bearing twice over: below `build_report()` the calls would be billed but missing from `usage.json`, and inside `process_one()` there would be no ranking to select from. `tests/test_main_deep_dive.py` pins both orderings.
- **`rank_candidates()` in `reporter.py` is the one definition of rank.** `results.json`, `results.csv`, `report.md` (all via `_split`), the deep-dive selection in `main.py`, and `app.py` all call it, so the briefed candidates are always exactly the top rows of the report. Sort key is `overall_fit` descending, then `source_file` ascending. The explicit filename tiebreak replaced an implicit one — a stable sort over `main.py`'s already-filename-sorted input — because rank now decides where paid calls go, and that shouldn't rest on two facts in two files. Do not re-implement the sort in `main.py` or `deep_dive.py`.
- **Top-N is strict.** `--deep-dive-top 3` makes exactly 3 calls even when candidates 3 and 4 tie; the tied candidate is cut and named in a terminal notice. Including ties would make a flag that says 3 spend 10 calls on a field that clusters in one score bin — which `reporter.py`'s histogram comments note is a real observed outcome. `0` and omission both mean off; negatives and non-integers are rejected by `_non_negative_int` at parse time, before the API-key check.
- **Deep-dive results never touch a domain model.** `DeepDiveReport` is carried in a `dict[str, DeepDiveReport]` keyed by `source_file` and joined in the reporter — the same sidecar pattern as `BiasAuditReport`. Do **not** add a `deep_dive` field to `ScoredCandidate`: it would put `"deep_dive": null` on every candidate in `results.json` for a briefing most never receive, and `tests/test_dimension_wiring.py` asserts that model's field set exactly, so it fails immediately if you try.
- **Deep-dive renders in `report.md` only.** Per candidate, after gaps and before the bias drift table. `write_markdown` takes `deep_dives=None` by default so every existing call site is unaffected; `write_json` and `write_csv` don't take the parameter at all, which is what makes the JSON and CSV schemas structurally incapable of changing. Pros/cons/questions are bullets, not a table, so a `|` in model prose needs no escaping. Candidates outside the top N render nothing — no placeholder.
- **Deep-dive needed zero changes to `usage.py`.** It takes the client it is handed, which is the `MeteredClient` both entry points already build, so its calls land in `usage.json` by construction. This is the Challenge 8 design working as intended — a new stage is metered because it accepts a client, not because anyone wired metering into it. Never construct an `Anthropic()` inside a pipeline module.
- **Deep-dive gets its own cache entry, and that's correct.** A cache prefix covers the tools and system prompt as well as the message blocks, and both differ from `scorer.py`'s, so it cannot and need not reuse the scorer's entry. The JD is block 0 with `cache_control: ephemeral`; candidate profile and screening results follow it, so the prefix stays identical across candidates in a run. No `ttl` is set, keeping the 5-minute assumption `MODEL_PRICING` documents. Expect aggregate `cache_creation_input_tokens` to roughly double when the flag is on — two cached prefixes instead of one, not a regression.
- `scorer.py` also exposes `score_candidate_ensemble()` which runs N scoring calls and returns median scores per dimension. Reasoning text is taken from the first run only — don't try to merge text across runs.
- Evals test score ranges (e.g., 80-100 for strong match) because LLM output varies between calls. Add new scoring eval cases by appending to `ALL_CASES` in `tests/evals/cases.py`. Add new bias stability cases by appending to `ALL_BIAS_CASES` using the `BiasAuditEvalCase` dataclass.
- Bias audit eval cases use only `NAME_SWAPS` (4 variants) to keep cost manageable. Production runs use `ALL_SWAPS` (name + grad year = 5 variants). `LOCATION_SWAPS` is defined in `bias_auditor.py` but excluded from `ALL_SWAPS` by default — pass it explicitly via `run_bias_audit(swaps=...)` to opt in.
- Scoring dimensions are configured in `dimensions.yaml`, not in code. Both consumers derive from it: `models.py` builds a dynamic `DimensionScores` base with `create_model()` that `ScoreReport` and `ScoredCandidate` inherit, and `prompts.py` renders the `` - `key`: description `` bullet block inside `SCORING_SYSTEM_PROMPT` from the same list. **Don't hand-edit dimension fields in the models or dimension bullets in the prompt** — both are generated, and an edit there will be silently overwritten by the config. Change the YAML.
- `prompt_description` is prompt text, so wording changes move eval scores. Treat an edit there as a prompt change: rerun `uv run python -m src.evals` and surface it in the PR like any other prompt edit. `key` is worse to change casually — it is the field name in `ScoreReport`, in `results.json`, and the `{key}_score` / `{key}_reasoning` CSV column prefix.
- `tests/test_dimension_wiring.py` pins the exact `SCORING_SYSTEM_PROMPT` string and the exact `ScoreReport.model_json_schema()` as hand-written golden snapshots. They are deliberately *not* derived from `DIMENSIONS` — a snapshot generated from the code it checks proves nothing. An intentional dimension change should update the goldens in the same commit; an unintentional one fails the suite.
- **The single source of truth currently covers the schema and the prompt only.** Call sites that reference dimensions by name are still hardcoded: `scorer.py` and `critique.py` (explicit kwargs), `reporter.py` (`_DIMENSIONS` plus the markdown breakdown and drift table), `bias_auditor.py` (`_DIMENSIONS`), `evals.py`, `app.py`, and the `expected_*` fields in `tests/evals/cases.py`. **`deep_dive.py` is the exception and the model to copy**: it iterates `DIMENSIONS` to serialize scores and names no dimension anywhere, in code or in its prompt, so a fifth dimension reaches the briefing with no edit. `tests/test_deep_dive.py` asserts that no dimension key appears in either file. Adding a fifth dimension to the YAML updates the schema and prompt but will fail at runtime until those are updated too — see `prompts/challenge4.md`.
- **Usage is metered at the API-client boundary, not in the pipeline.** `main.py` and `app.py` build a `UsageTracker` and pass `MeteredClient(Anthropic(), tracker)` where the raw client used to go. `MeteredClient` exposes only `.messages`, calls through to the real client, records `response.usage`, and returns the *same* response object. The five pipeline modules are unchanged and know nothing about it. Do **not** add usage to return types as tuples, thread a `tracker=` parameter through the pipeline, or introduce a module-level counter — all three were considered and rejected. Recording happens *before* the caller pulls out the tool_use block and validates it, so a billed call whose response fails `model_validate` is still counted; the report describes API work observed, not successful candidates.
- **Usage never belongs in a domain model.** Nothing in `src/models.py` — `CandidateProfile`, `ScoreReport`, `ScoredCandidate`, `ProcessingError` — carries token counts or cost. Usage is operational telemetry; putting it on a candidate would leak it into `results.json`, the CSV columns, and the tool schemas Claude is handed. `usage.py` does not import from `models.py` and must not start.
- **`usage.input_tokens` is the uncached input only.** The real prompt size is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`, exposed as `total_prompt_tokens`. Both cache fields are `Optional[int]` in the SDK and arrive as `None` on uncached calls; `APIUsage.from_response()` is the single place that normalizes them to `0`. Report cached *token counts* — the API gives no cache-hit event count, so don't invent one.
- Pricing lives in `MODEL_PRICING` in `src/usage.py` as a plain dict of `Decimal` rates, with a comment recording the official source URL and the date verified. It is deliberately not a YAML config: unlike `dimensions.yaml` it has one consumer and changing it alters no model behavior. Update the date whenever you touch the numbers. Cost is computed per record from `response.model` (the *resolved* model, not the requested one) and is always called an **estimate**; an unknown model yields `None`, never `0.0`.
- `MeteredClient` is passed where the pipeline annotates `client: Anthropic`. That mismatch is intentional and harmless — annotations aren't enforced here, and `.messages` is the only client attribute any pipeline module touches. Widening those annotations would mean editing five files to satisfy a hint nothing checks.
- **The unit of concurrency is one candidate.** A worker owns `parse → extract → score (→ critique) (→ bias audit)` — that is `process_one()`, unchanged and unsplit. Do not parallelize stages *within* a candidate and do not nest pools. Ranking, top-N selection, the deep-dive stage, `build_report()` and every file write stay sequential, after the pool drains, because each needs the whole field.
- **Prime, then fan out.** The first resume is processed alone; only then do the rest go through `ThreadPoolExecutor(max_workers=concurrency)`. `scorer.py`'s JD prefix is written by the first scoring call of a run, and starting cold at concurrency 5 measured **4 cache writes instead of 1** — the first wave races. Running one candidate first fixes it with ordering alone, costing ~10s and eliminating 3 redundant writes. Do not replace this with all-at-once concurrency without new benchmark evidence, and do not "improve" it into a warm-up request, a `max_tokens: 0` call, or a score latch — all were considered and rejected (see `prompts/challenge6.md`). If the primer fails, the batch fans out anyway: a failed prime degrades cost, never correctness.
- **Deep-dive stays sequential on purpose.** It has its own separate cached prefix, and a pool around it would reintroduce upstream's race. It was also the experimental control that isolated concurrency as the cause of the scorer's cache behavior.
- **Threads, not asyncio.** Every pipeline function is synchronous, `pdfplumber` blocks, `MeteredClient` wraps a sync client, and the tests are sync `MagicMock`. Async would touch six modules, both entry points and three test files to reach identical throughput at this request volume. Don't convert isolated paths casually.
- **Execution is concurrent; collection is deterministic.** Futures are submitted together and read back in *submission order* — never `as_completed()`. Scored candidates would survive either way (`rank_candidates()` re-sorts), but `ProcessingError` entries are emitted in list order by the reporter, so completion-order collection would shuffle `results.json`'s `errors` array and the "Could not process" table between runs of identical input; same for `audits` insertion order, which `bias_audit.json` iterates. No original-index field is needed — submission order is the index. Workers return their outcome and mutate nothing shared: `results`, `audits` and all printing belong to the collecting thread.
- **One shared, lock-protected `UsageTracker`.** One `Anthropic`, one `MeteredClient`, one tracker for every worker. `record()`'s append and the `records` snapshot are guarded by a `threading.Lock`; `APIUsage.from_response()` runs outside it. Do not add per-worker trackers, a merge API, running totals, worker IDs, or sequence numbers. Under concurrency the `calls` array in `usage.json` is in completion order and is no longer a chronological trace — totals, cost and models are all order-independent, so leave it; run sequentially if you need the trace.
- **`main.py` and `app.py` both implement this, separately.** Neither shares a helper, per the rule above, so a change to the concurrency shape must land in both. `tests/test_app_parity.py` inspects `app.py`'s source (an established pattern — see `tests/test_deep_dive.py`) and fails if the two drift on priming, pool count, `as_completed`, or the concurrency default.
- **Challenge 6 performance numbers are benchmark observations, not guarantees.** They come from one fixed configuration (10 sample resumes, `sample_data/sample_jd.txt`, `claude-sonnet-4-6`, `deep_dive_top=3`) on one machine, recorded in `benchmarks/*.json` and written up in `prompts/challenge6.md`. Don't generalize them, and don't re-derive them from a different input set without saying so. `benchmarks/parallel-naive.json` records a code state that no longer exists — priming is unconditional, so that arm is not reproducible from the current tree without reverting it. A `--no-prime` flag was deliberately not added.
- `scripts/benchmark.py` owns wall-clock timing and is not part of the pipeline. Timing must never move into `src/usage.py`, `UsageReport`, or `usage.json`: that module records one normalized record per API *response*, and elapsed time is a property of a *run*.
- **Generated eval metadata is a proposal, never ground truth.** `GeneratedResumeProposal` carries the level, score range and dimension hints the *generator* was aiming for. `promote_proposal()` requires `expected_ranges` from the caller and never reads the proposal's own range — that range is stored beside it as `proposed_score_range`, for comparison only. Using it would let the generator write the resume, set the passing grade, and be graded against it. The qualitative `dimensions_expected_high`/`_low` lists are never converted into numeric ranges; there is no honest arithmetic for that.
- **`reviewed` gates the eval suite, in code.** `load_eval_case_kwargs()` is the only path from `tests/evals/generated_cases.json` into `ALL_CASES` and it filters on `reviewed` with no override; `reviewed` is keyword-only on `promote_proposal()` and defaults to `False`. Unreviewed cases stay loadable (a reviewer must read them) but cannot reach the runner. **Do not add `reviewed` to `EvalCase`** — review is an ingestion concern, and the runner receives a plain `EvalCase` that cannot be told apart from a hand-written one.
- **Raw proposals and reviewed cases are separate files on purpose.** `tests/evals/generated/` is generator provenance and is never edited; `tests/evals/generated_cases.json` is the reviewed layer. Do not write review state back into `proposals.json`.
- **Generation is one level per call, sequential.** All four levels in one response would make them compete for output budget and let the model tune them against each other. Sequential ordering is what lets the four calls share one cached prefix (tool schema + system prompt + JD, breakpoint after the JD) — measured 1 write / 3 reads. Do not add a thread pool. The level order comes from `MatchLevel` via `get_args`, not a second list.
- **Two generated eval cases fail on purpose.** `generated_partial_payments` and `generated_adversarial_payments` are scorer findings, not stale expectations — a live run reports 6/8. Do not widen their ranges to go green; they should pass as a consequence of fixing the scorer. See `prompts/challenge7.md`.
- Per `CONTRIBUTING.MD`, prompts are treated as the teaching artifact: any prompt change should be visible in the PR description (or saved under `prompts/`), and new conventions should be reflected back into this file.