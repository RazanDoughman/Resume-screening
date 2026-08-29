# Resume-to-JD Matching Agent

A teaching project that scores resume PDFs against a job description using
the Claude API.

Each resume goes through three steps, each a separate Python module:

1. **Parse** the PDF into plain text (`pdf_parser.py`, no LLM)
2. **Extract** structured info — name, skills, past roles, years of experience
   (`extractor.py`, one LLM call)
3. **Score** the candidate against the JD across four dimensions
   (`scorer.py`, one LLM call) — the dimensions are configured in
   [dimensions.yaml](dimensions.yaml)

The output is a ranked JSON file plus a readable markdown report.

## How the code runs (end to end)

### Step 1: Entry point (`main.py`)

```
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/
```

Python hits `if __name__ == "__main__"` and calls `main()`.

### Step 2: Parse arguments and validate

`main()` reads the command-line args (`--jd`, `--resumes`, `--output`, `--model`), then checks:

- Is `ANTHROPIC_API_KEY` set?
- Does the JD file exist?
- Does the resumes directory exist?

If all good, it calls `run()`.

### Step 3: Setup

`run()` does two things:

1. **Reads the job description** from the text file into a string using `open()`
2. **Lists the resumes directory** using `os.listdir()` and collects all `.pdf` files

Then it creates an Anthropic API client and starts looping through each PDF.

### Step 4: Process each resume

For each PDF, `process_one()` runs a 3-stage pipeline. If any stage fails,
it records the error and moves on to the next resume.

**Stage A — Parse the PDF** (`src/pdf_parser.py`)

- Uses `pdfplumber` to extract raw text from the PDF
- No LLM involved here, just text extraction
- Input: `alice_chen.pdf` -> Output: a big string of resume text

**Stage B — Extract candidate profile** (`src/extractor.py`) — LLM call #1

- Sends the resume text to Claude with a system prompt saying "you are a resume parser"
- Uses **tool use**: defines a tool called `record_candidate` whose schema matches
  `CandidateProfile` (name, skills, years_experience, past_roles, education)
- Forces Claude to call that tool via `tool_choice` — so Claude **must** return
  structured JSON, not free text
- Validates the JSON with Pydantic to get a typed `CandidateProfile` object
- Input: raw text -> Output: structured profile like `{name: "Alice Chen", skills: ["Python", "ML"], ...}`

**Stage C — Score against the JD** (`src/scorer.py`) — LLM call #2

- Sends both the job description and the candidate profile to Claude
- System prompt says "you are a hiring evaluator"
- Again uses tool use: a tool called `record_score` matching `ScoreReport` schema
- Claude scores 4 dimensions (0-100 each): skills_match, experience_match, role_relevance, overall_fit
- Also returns reasoning and gaps
- Input: JD + profile -> Output: `ScoredCandidate` with scores like `{overall_fit: 85, skills_match: 90, ...}`

**Stage D — Self-critique (optional)** (`src/critique.py`) — LLM call #3

- Only runs if you pass `--self-critique`
- Sends the JD + profile + the scores from Stage C to Claude as a "senior hiring manager"
- Claude reviews the scores and decides: are they reasonable, or off by more than 10 points?
- If `did_revise: false` -> keeps original scores
- If `did_revise: true` -> returns corrected scores

**Stage E — Bias audit (optional)** (`src/bias_auditor.py`) — LLM call × N swap variants

- Only runs if you pass `--bias-audit`
- Takes the extracted `CandidateProfile` and creates copies with one demographic signal swapped at a time: candidate name (4 variants covering gender × ethnicity) and graduation year (1 variant probing age bias). Location swaps (`LOCATION_SWAPS`) are defined but excluded from the default set — opt in by passing them explicitly to `run_bias_audit(swaps=...)`
- Re-scores each mutated copy against the JD using the same scoring LLM
- Computes the absolute score delta per dimension for each variant vs the baseline
- If any delta exceeds the threshold (default 10 pts), the candidate is flagged
- Output: a drift table per candidate in `report.md` and a full `bias_audit.json`

### Step 5: Write output (`src/reporter.py`)

After all resumes are processed, `run()`:

1. Creates the output directory
2. Writes `results.json` — all candidates ranked by overall_fit score, plus any errors
3. Writes `report.md` — an ASCII histogram of the `overall_fit` distribution, then a readable markdown table with rankings, per-candidate score breakdowns, and gaps
4. Writes `results.csv` — only when `--csv` is passed. One flat row per
   successfully scored candidate, for spreadsheets. No LLM calls.

### The big picture

```
PDF file
   |  pdfplumber
Raw text
   |  Claude + tool use (LLM #1)
Structured profile
   |  Claude + tool use (LLM #2)
Scores + reasoning
   |  Claude + tool use (LLM #3, optional)
Reviewed scores
   |  Claude × N swaps (LLM #4…N, optional)
Bias audit report
   |
results.json + report.md + results.csv + bias_audit.json
```

Every stage above runs once per resume. One stage doesn't: after the whole
batch is scored, `--deep-dive-top N` ranks the field and makes one more call
for each of the top N candidates only.

```
all resumes scored
   |  rank by overall_fit, take top N
   |  Claude + tool use (one call per selected candidate, optional)
Hiring-manager briefings -> report.md
```

The key pattern that repeats in every LLM call: **define a tool from a Pydantic
schema -> force Claude to call it -> validate the response back into a Pydantic
object**. This guarantees structured, typed output every time.

## What's being taught

- Reading structured data out of LLM responses using Claude's **tool use**
  feature + **Pydantic** for validation.
- A **prompt** module where all LLM instructions live in one place, so you
  can iterate on wording without touching the surrounding code.
- **Prompt caching** — the JD is identical across every resume in a run, so
  Claude serves it from cache at ~10% of the normal cost.
- An optional **self-critique** pass behind a flag that reviews scores.
- An optional **bias audit** that probes for demographic sensitivity by
  re-scoring the same candidate with swapped name, graduation year, and
  location signals, then measuring score drift across dimensions.
- An **eval harness** for testing LLM pipelines, because you can't assert
  exact-score equality on a probabilistic output. Includes both scoring
  range evals and bias stability evals.
- A **cost reporter** that reads the `usage` field most engineers never look
  at, showing where the tokens actually went — and why "input tokens" is not
  the same number as "prompt tokens" once caching is on.

No asyncio, no frameworks, no agent libraries. Everything is plain
synchronous Python you can read top-to-bottom in one sitting.

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with your API key:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Generate the sample resumes (they're not checked in):

```bash
uv run python scripts/generate_sample_resumes.py
```

This creates five fictional candidates in `sample_data/resumes/`. Edit
[scripts/generate_sample_resumes.py](scripts/generate_sample_resumes.py)
and re-run to experiment.

## Run

Basic run — score all resumes against the job description:

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --output output/
```

Resumes are processed concurrently by default. Set the limit with
`--concurrency`:

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --concurrency 5
```

- Default `5`, minimum `1`. `0`, negatives and non-integers are rejected.
- `--concurrency 1` processes candidates one at a time — the original
  sequential behaviour.
- It controls how many resumes run at once **after** the first one. It changes
  wall-clock only: the number of LLM calls a run makes is identical at every
  setting.

### How concurrency works: prime, then fan out

The first resume is processed on its own, and only then do the rest run in
parallel. That is deliberate. `scorer.py` sends the job description as a cached
prompt prefix, so the first scoring call of a run writes that cache entry and
every later call reads it. Starting all workers at once from a cold cache had
four of the first five scoring calls race and each write its own copy — paying
the cache-write premium four times instead of once. Running one candidate first
fixes it with ordering alone, with no change to any prompt.

Ranking, the top-N deep-dive, the usage report and all file writes stay
sequential, after every resume has been scored.

### Benchmark

Measured on the 10 sample resumes with `--deep-dive-top 3` (see
[prompts/challenge6.md](prompts/challenge6.md) for the full write-up and
[benchmarks/](benchmarks/) for the raw results):

| | Median wall-clock | Speedup | Cold scorer cache |
| --- | ---: | ---: | --- |
| Sequential (`--concurrency 1`) | 268.49 s | 1.00× | 1 write / 9 reads |
| Naive parallel (no priming) | 116.30 s | 2.309× | 4 writes / 6 reads |
| **Primed parallel (current)** | **127.12 s** | **2.112×** | **1 write / 9 reads** |

Priming costs about 10 s against naive concurrency and eliminates three
redundant cache writes, worth roughly $0.017 per cold run at this size.

These are observations from one fixed benchmark on one machine and network —
not performance guarantees. Your timings will differ.

With bias audit — re-scores each candidate with demographic signals swapped
and reports score drift per dimension:

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --bias-audit
```

With both self-critique and bias audit:

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --self-critique \
    --bias-audit
```

With CSV export — same scoring run, plus a flat file you can open in a
spreadsheet:

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --csv
```

With deep-dive briefings — after everyone is scored and ranked, write a
hiring-manager briefing for the top 3:

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --deep-dive-top 3
```

Flags:

- `--model MODEL_ID` — override the Claude model (default: `claude-sonnet-4-6`)
- `--output DIR` — override the output directory (default: `./output`)
- `--self-critique` — add one more LLM call per resume that reviews and
  may revise the scores. +1 call per resume.
- `--bias-audit` — re-score each candidate with demographic signals swapped
  (name, graduation year) and report score drift. Adds 5 extra LLM calls per
  resume (4 name variants covering gender × ethnicity + 1 grad year variant
  probing age bias). Pass `swaps=LOCATION_SWAPS` to `run_bias_audit()` to
  also include location variants.
- `--csv` — also write `results.csv`, one flat row per successfully scored
  candidate. Pure logic, no extra LLM calls, so it costs nothing to turn on.
- `--deep-dive-top N` — after scoring, write a hiring-manager briefing for the
  top N ranked candidates. Adds **N LLM calls in total, not N per resume** —
  see [Deep-dive briefings](#deep-dive-briefings) below. Default 0 (off).

Outputs:

- `output/results.json` — machine-readable ranked list plus any errors
- `output/report.md` — human-readable markdown report. Opens with an ASCII
  histogram of `overall_fit` across the scored candidates (failed resumes are
  excluded and noted beneath the chart); includes a per-candidate drift table
  when `--bias-audit` is used, and a hiring-manager briefing for each selected
  candidate when `--deep-dive-top` is used
- `output/results.csv` — flat one-row-per-candidate export for spreadsheets
  (only written when `--csv` is used). Scores and reasoning get one column per
  dimension; skills, education, and gaps are joined into single cells. Failed
  resumes are omitted — they stay in `results.json`
- `output/bias_audit.json` — full audit data per candidate (only written when
  `--bias-audit` is used)
- `output/usage.json` — token counts and estimated cost for the run. Always
  written; see [Cost reporting](#cost-reporting) below

## Scoring dimensions

The four dimensions Claude scores are configuration, not code. They live in
[dimensions.yaml](dimensions.yaml) at the project root:

```yaml
- key: skills_match
  label: Skills match
  prompt_description: >-
    overlap between the candidate's skills and the JD's required/nice-to-have
    skills.
```

- `key` — the field name used everywhere downstream: in the JSON schema Claude
  is handed, in `results.json`, and as the `{key}_score` / `{key}_reasoning`
  column prefix in the CSV. Renaming it renames all of those.
- `label` — the human-readable name, for reports.
- `prompt_description` — the sentence describing the dimension to Claude. It is
  dropped verbatim into the scoring prompt as `` - `key`: description ``.

Both the Pydantic scoring schema (`ScoreReport` / `ScoredCandidate` in
`src/models.py`) and the scoring prompt (`SCORING_SYSTEM_PROMPT` in
`src/prompts.py`) are built from this file at import time, so the two can't
drift apart. `src/dimensions.py` loads and validates it — an empty list, a
duplicate `key`, or a missing `key` / `prompt_description` fails loudly on
startup instead of producing a half-built schema.

`prompt_description` is prompt text, so editing it changes scores. Rerun the
evals after any change:

```bash
uv run python -m src.evals
```

**Adding a fifth dimension needs more than the YAML.** The schema and the prompt
follow automatically, but `scorer.py`, `critique.py`, `reporter.py`,
`bias_auditor.py`, `evals.py`, `app.py`, and the `expected_*` fields in
`tests/evals/cases.py` still name the four dimensions explicitly and must be
updated alongside it. Editing the wording or the label of an existing dimension
is the safe, YAML-only change.

Parsing uses `pyyaml`, which `uv sync` installs along with everything else.

## Cost reporting

Every run prints a token and cost breakdown at the end and writes the same
numbers to `output/usage.json`. There is no flag — it costs nothing, because it
only reads the `usage` field the API already returns on every response.

```text
API Usage
--------------------------------
Model:  claude-sonnet-4-6

API calls:                   12
Uncached input tokens:   12,480
Cache read tokens:        9,900
Cache creation tokens:    1,240
Total prompt tokens:     23,620
Output tokens:            8,150

Estimated cost:         $0.1639
```

The four token categories are not interchangeable, and the difference is the
main thing this report exists to show:

- **Uncached input tokens** — prompt tokens processed at the full input price.
  This is the API's `input_tokens` field, and on its own it is **not** the size
  of the prompt: when prompt caching is working, most of the prompt shows up in
  the two cache lines instead.
- **Cache read tokens** — prompt tokens served from the cache, at roughly a
  tenth of the input price. The job description is cached, so re-scoring the
  same JD against many resumes should land here.
- **Cache creation tokens** — prompt tokens written into the cache, at a small
  premium over the input price. Expect these on the first scoring call of a run.
- **Total prompt tokens** — the sum of the three above. This is the real prompt
  size.
- **Output tokens** — everything Claude generated.

The API reports cached *token counts*, not a count of cache-hit events, so this
report shows token counts and does not invent a "cache hits" number.

The cost is an **estimate**, not an invoice. It is computed from a local pricing
table in [src/usage.py](src/usage.py) that records Anthropic's published list
prices along with the date they were verified. Your actual bill can differ
because prices change, because negotiated or batch rates differ from list, and
because the figure is rounded. If a run uses a model that is not in the table,
the token counts are still reported in full and the cost reads
`unavailable - no pricing entry for '<model>'` rather than `$0.00` — a zero
would be indistinguishable from a free run.

Every API response that comes back is counted, including calls for resumes that
later failed to parse or validate. The report describes API work that actually
happened, not just the candidates that made it into `results.json`. A request
that never returns a response — a connection error, say — has no usage to read
and cannot be counted.

The Streamlit UI shows the same summary at the bottom of a run. It writes no
files, so there is no `usage.json` there.

## Deep-dive briefings

Scoring answers *who is at the top*. It doesn't answer *what do I do with
them*. `--deep-dive-top N` adds that second question, for the shortlist only.

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --deep-dive-top 3
```

After every resume has been scored and the field ranked, the top N candidates
each get one more LLM call that returns a briefing for a hiring manager:

- a paragraph on what the candidate would bring and where the uncertainty is
- **strengths** — concrete, tied to their actual background
- **risks** — shortfalls and things to verify, seeded from the screener's gaps
- **suggested interview questions** — aimed at those specific risks

**It is opt-in.** Omit the flag, or pass `0`, and nothing changes: no extra
calls, no extra output, byte-identical reports.

**It costs O(N selected), not O(resumes).** Every other optional stage
multiplies by the resume count — `--bias-audit` on 10 resumes adds 60 calls.
Deep-dive adds exactly N, whether you screened 10 resumes or 500. That's the
point of the feature: screen wide and cheap, then spend real tokens only on
the people you might actually interview.

```
10 resumes, no flags:            20 calls
10 resumes, --deep-dive-top 3:   23 calls
500 resumes, --deep-dive-top 3: 1003 calls
```

**Where it shows up.** `report.md` only, in each selected candidate's section,
after their gaps and before their bias-audit table. Candidates outside the top
N are rendered exactly as before — no placeholder.

`results.json` and `results.csv` are **unchanged**, by design. A briefing is an
optional analysis of a handful of candidates, not a property of a candidate, so
it never touches the machine-readable schemas that every other candidate shares.

**Ties are cut strictly.** Ranking is `overall_fit` descending, then
`source_file` ascending, so it is deterministic. `--deep-dive-top 3` always
means exactly three calls, even when candidates 3 and 4 share a score — the run
prints a note naming who was cut and what N would have included them.

## Test

Fast local tests for the deterministic parts of the pipeline (parser and
reporter) — no LLM calls:

```bash
uv run pytest
```

## Evals

LLM calls can't be unit-tested with exact equality — scores move a few
points call to call. So we test *properties* of the output: a strong-match
resume should score 80+ overall, a weak-match resume should score under 45,
and so on.

```bash
uv run python -m src.evals
```

Runs the golden cases in [tests/evals/cases.py](tests/evals/cases.py)
through the real `extract -> score` pipeline and checks each dimension
against an expected range. Also runs bias stability cases that assert a
clearly qualified candidate's scores don't shift when demographic signals
are swapped. Exits nonzero on any failure.

```bash
uv run python -m src.evals --bias
```

Run only the bias stability cases (faster when iterating on audit prompts).

```bash
uv run python -m src.evals --deep-dive
```

Run only the deep-dive cases. These check the *shape* of a briefing rather
than its prose: a summary of real length, at least two strengths, at least one
risk, at least three interview questions, and at least one distinctive term
from that candidate's own resume — which is what separates a real analysis
from a generic template. Prose quality would need a judge model; these
assertions survive normal LLM variation and still fail on a prompt regression.

Add a scoring case by appending to `ALL_CASES`. Add a bias stability case
by appending to `ALL_BIAS_CASES` using the `BiasAuditEvalCase` dataclass. Add
a deep-dive case by appending to `ALL_DEEP_DIVE_CASES` using the
`DeepDiveEvalCase` dataclass.
When you iterate on a prompt, run the evals to see whether the change moved
scores in the right direction.

## Project layout

```
resume_matcher/
├── main.py              # CLI entry point — concurrent resume processing
├── pyproject.toml
├── dimensions.yaml      # scoring dimensions — schema and prompt derive from this
├── src/
│   ├── dimensions.py    # loads + validates dimensions.yaml (no LLM)
│   ├── pdf_parser.py    # PDF -> plain text (no LLM)
│   ├── extractor.py     # text -> CandidateProfile (LLM call)
│   ├── scorer.py        # (profile, JD) -> ScoredCandidate (LLM call)
│   ├── critique.py      # optional 2nd-pass review (LLM call)
│   ├── deep_dive.py     # optional hiring-manager briefing for the top N (LLM call)
│   ├── bias_auditor.py  # optional bias audit — re-scores with swapped demographic signals
│   ├── reporter.py      # results -> JSON + markdown + optional CSV (+ bias drift table)
│   ├── usage.py         # token accounting + pricing + cost estimate (no LLM)
│   ├── prompts.py       # every LLM prompt in one place
│   ├── models.py        # Pydantic contracts
│   └── evals.py         # eval harness runner (scoring + bias stability)
├── tests/
│   ├── test_pdf_parser.py
│   ├── test_reporter.py
│   ├── test_usage.py            # token math, pricing, MeteredClient
│   ├── test_deep_dive.py        # deep-dive tool wiring, caching, top-N selection
│   ├── test_main_deep_dive.py   # deep-dive orchestration in main.run()
│   ├── test_dimensions.py       # dimensions.yaml loading + validation
│   ├── test_dimension_wiring.py # golden prompt + schema snapshots
│   └── evals/
│       └── cases.py     # golden eval cases (ALL_CASES + ALL_BIAS_CASES)
├── sample_data/
│   ├── sample_jd.txt
│   └── resumes/         # generated by scripts/generate_sample_resumes.py
├── scripts/
│   ├── generate_sample_resumes.py
│   └── benchmark.py     # Challenge 6 time/cost benchmark harness (no production role)
├── benchmarks/          # committed benchmark results (sequential / naive / primed)
└── output/
```

## Design notes

- **One LLM call per stage, not per field.** Structured output via tool use
  means one call returns every field at once. Students see the full
  `messages.create()` call with `tools`, `tool_choice`, and `model_validate`
  in one place (`extractor.py`).
- **Prompts are data, code is logic.** All prompts live in
  [src/prompts.py](src/prompts.py). You can change wording without touching
  any function.
- **Scoring dimensions are configuration, not code.** The prompt bullets and
  the Pydantic score fields are both generated from
  [dimensions.yaml](dimensions.yaml), so they cannot describe one set of
  dimensions while the schema enforces another. Golden snapshot tests pin the
  exact prompt string and the exact tool schema, so the refactor that made them
  dynamic is provably a no-op on the wire.
- **Bad PDFs don't crash the run.** Each resume is wrapped in try/except. A
  failure at any stage becomes a `ProcessingError` in the output instead of
  stopping the batch.
- **Opt-in quality flag.** `--self-critique` is off by default so a basic
  run costs two LLM calls per resume. Turn it on when you want to add a
  reflection pass.
- **Deep-dive spends tokens where the decision is.** Every other optional
  stage costs O(resumes); `--deep-dive-top N` costs O(N selected). Screening
  stays cheap enough to run wide, and the expensive analysis lands only on the
  shortlist. It is also purely additive: the briefings live in `report.md`
  alongside a sidecar dict in memory, so `ScoredCandidate`, `results.json` and
  the CSV columns are untouched whether the flag is on or off.
- **Bias audit is additive, not destructive.** `--bias-audit` appends a
  drift table to each candidate's section in `report.md` and writes a
  separate `bias_audit.json`. It never changes scores — it only reports
  whether the model would have scored differently with different demographic
  signals. The audit fails safely: a broken variant is logged and skipped,
  the scored candidate is always returned regardless.
