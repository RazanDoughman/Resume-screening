# Challenge 8 — Cost reporter

Prints a token and cost breakdown after every run and writes it to
`output/usage.json`. New: `src/usage.py`, `tests/test_usage.py`. Modified:
`main.py`, `app.py`, `src/reporter.py`. **The five pipeline modules
(`extractor.py`, `scorer.py`, `critique.py`, `bias_auditor.py`, `models.py`)
have a zero-line diff** — that constraint drove the whole design.

No prompt changes, no dimension changes, no scoring-behavior changes, and no new
dependencies. Evals were not rerun because nothing that could move a score was
touched.

## The problem

Every Anthropic response carries a `usage` field. In this repo, all four call
sites threw it away on the very next line:

```python
response = client.messages.create(...)          # usage lives here
tool_use = next(b for b in response.content if b.type == "tool_use")
return CandidateProfile.model_validate(tool_use.input)   # response is now garbage
```

So a run that made 60+ API calls (10 resumes x 6 calls with `--bias-audit`)
could tell you nothing about what it cost.

## Why a client wrapper and not tuple returns

The obvious approach is `return profile, usage` — explicit, easy to read. Two
things ruled it out.

**It loses usage on failure.** `model_validate` on the line after the API call
can raise, and `test_extractor.py` already asserts that it does. When it raises,
the function never returns, so the tuple never comes back — and you were still
billed for that call. Challenge 8 is supposed to report API work that happened,
not API work that succeeded.

**It spreads.** `_score_report` feeds `score_candidate`, `score_candidate_ensemble`,
and `run_bias_audit`, which calls it 6 times inside a loop and then does drift
math over the results. Threading tuples through that means editing five modules,
both `process_one` implementations, and `test_extractor.py`.

The alternative — passing a `tracker=` parameter down — fixes the failure case
but still touches every signature, and correctness then depends on each new call
site remembering to record before validating.

The observation that made a third option work: **every pipeline module already
takes the client as its first argument, and every one of them reaches it exactly
one way.** A grep for `client.<attr>` across the repo returns 8 hits, all
`client.messages`. Nothing touches `.beta`, `.with_options`, or anything else.

So wrap the client:

```python
client = MeteredClient(Anthropic(), tracker)   # main.py, one line
```

`MeteredClient.messages.create()` calls through, records `response.usage`, and
returns the *same* response object. The pipeline is unchanged. Usage is recorded
before anything downstream can raise, so the failure case is handled by
construction rather than by convention. And when Challenge 5 adds a deep-dive
stage, it gets metered for free — it takes a client, so it's already covered.

`MeteredClient` deliberately exposes **only** `.messages`. A catch-all
`__getattr__` would future-proof `client.beta.messages.create(...)` while letting
it silently escape metering; raising `AttributeError` is the better failure.

## Two numbers that are easy to get wrong

**`input_tokens` is not the prompt size.** It is the *uncached* remainder. The
JD is cached in `scorer.py` and `critique.py`, so when caching works most of the
prompt is reported as cache reads instead. Labelling `input_tokens` as "total
input tokens" would under-report by exactly the amount caching saved you — the
one number the report exists to reveal. Hence `total_prompt_tokens`, and hence
the four separate lines in the output.

**The cache fields are `None`, not `0`.** In the installed SDK,
`cache_creation_input_tokens` and `cache_read_input_tokens` are
`Optional[int] = None`. Every extraction call returns `None` for both, so
`a + b` is a guaranteed `TypeError`. `APIUsage.from_response()` is the single
normalization boundary; nothing downstream ever sees `None`.

## "Cache hits" is not a real field

The challenge asks for "cache hits". The API doesn't report one, and at the call
level the concept doesn't hold: a single request can read 1,100 cached tokens
*and* process 400 fresh ones. It isn't a hit or a miss. So the report shows the
fields that actually exist — cache read tokens and cache creation tokens — plus
their sum. Inventing `Cache hits: 4` would be a number with no definition, and
it would be read as "4 requests were free".

## Estimated, not billed

Cost is computed in `Decimal` from a table of Anthropic's published list prices,
with the source URL and verification date in a comment above it. It is called an
**estimate** everywhere because the local table can go stale, list price is not
everyone's price, and the figure is rounded. `Decimal` converts to a plain JSON
number in exactly one place, `UsageReport.to_dict()`.

An unknown model reports full token counts and
`unavailable - no pricing entry for '<model>'`. Never `$0.00` — a zero is
indistinguishable from a free run, and would be believed.

Mixed-model runs fall out for free: each record is priced with its own model's
rates and the results summed, because the record stores `response.model` (the
*resolved* model) rather than whatever was requested.

## Structure

```text
Anthropic()            usage.py                        reporter.py
     |                    |                                 |
MeteredClient --> UsageTracker (per-call APIUsage records)   |
     |                    |                                 |
 pipeline           build_report --> UsageReport ------------+--> usage.json
 (untouched)                             |
                                   format_summary --> terminal / st.code
```

`usage.py` owns the data, the pricing, the math, and the formatting, and writes
no files. `reporter.py` owns file writing and knows nothing about pricing — its
`write_usage` is three lines over `UsageReport.to_dict()`, which keeps the JSON
schema next to the data it describes.

## Built for Challenge 6, without building Challenge 6

Parallel processing is coming. Rather than adding a lock nothing can exercise
today, the design just avoids the hazards:

- no module-level mutable state — the tracker is created inside `run()`
- `APIUsage` is frozen, so combining records from several sources is safe
- append-only records with totals derived on demand — no shared counter to
  read-modify-write
- `build_report()` takes any sequence of records, so per-worker trackers merge
  by concatenation with no new API
- nothing infers meaning from record order

A lock would be untestable in a sequential suite, and currently unfalsifiable —
`list.append` is atomic under the GIL, so a shared tracker would happen to work
either way. The structural choices are what actually prevent the debt.

## Tests

21 tests in `tests/test_usage.py`, all mocked, no API key, no network. The ones
that matter most:

- `None` cache fields normalize to `0` — the real shape of every extraction call
- `total_prompt_tokens` sums all three input categories, and differs from
  `input_tokens`
- unknown model gives `None`, and is asserted `!= 0`
- `MeteredClient` returns the response by identity (`is`), not a copy
- **usage is still recorded when `model_validate` raises** — this one drives the
  real `extract_candidate` with a mocked client, so it also proves the wrapper
  is a drop-in for the untouched pipeline modules
- the summary is pure ASCII, because a box-drawing rule would raise
  `UnicodeEncodeError` on a non-UTF-8 console and take down the last line of
  every run
