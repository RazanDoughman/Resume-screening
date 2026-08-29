# Challenge 6 — Parallel processing

Processes resumes concurrently behind a configurable limit, with the first
candidate deliberately run alone so the scorer's cached job description is warm
before the rest fan out. New: `scripts/benchmark.py`, `scripts/__init__.py`,
`tests/test_benchmark.py`, `tests/test_main_concurrency.py`,
`tests/test_app_parity.py`, `benchmarks/*.json`. Modified: `main.py`, `app.py`,
`src/usage.py` (one lock).

**`scorer.py`, `deep_dive.py`, `extractor.py`, `critique.py`, `bias_auditor.py`,
`prompts.py`, `models.py`, `reporter.py`, `dimensions.py` and `dimensions.yaml`
have a zero-line diff.** No prompt changed, no `cache_control` marker moved, no
tool schema changed. That constraint is the whole point of the result below: the
cache behaviour changed three times across this challenge, and *execution
ordering* was the only variable. Evals were not rerun — nothing that could move
a score was touched.

## Goal

> Process N resumes concurrently with a configurable concurrency limit. Prompt
> caching on the JD becomes an architectural decision, not an afterthought.
> Submit benchmarks: time and cost, before vs after.

Four deliverables: concurrency, a limit you can set, a deliberate answer to what
concurrency does to the JD cache, and measurements on both sides.

## Before: sequential, and accidentally optimal

The old loop ran one resume to completion, then the next:

```
candidate 1: parse → extract → score
candidate 2: parse → extract → score
...
```

`scorer.py` sends the job description as a separate content block tagged
`cache_control: ephemeral`, so the first scoring call of a run writes that
prefix and every later one reads it. Sequentially that ordering is free — there
is one thread, so call #2 cannot start before call #1 has returned. The measured
baseline shows it exactly: **1 cache write of 1,667 tokens, then 9 reads.**

Nobody designed that ordering. It was a side effect of being slow. Concurrency
takes it away, and that is the interesting part of this challenge.

## The concurrency boundary

One resume is the unit of work:

```
parse → extract → score → (optional critique) → (optional bias audit)
```

`process_one()` already *was* that unit — it takes a client and a path, mutates
nothing outside itself, and returns `(Result, BiasAuditReport | None)`. No
pre-concurrency refactor was needed, which is Challenge 5 and 8's tidiness
paying a dividend.

Everything that needs the whole field stays sequential, after the pool drains:

- **Ranking** — `rank_candidates()` needs every candidate.
- **Top-N selection and deep-dive** — "top N" is undefined mid-batch.
- **`build_report()`** — must sit below every LLM call or the calls go unbilled
  in `usage.json`.
- **All file writes.**

Deep-dive stays sequential for a second reason: it has its own separate cached
prefix (1,863 tokens), and a pool around it would reintroduce the very race
being fixed upstream. It also turned out to be a useful experimental control —
see below.

## Threads, not async

`ThreadPoolExecutor`. Every function in the pipeline is synchronous, `pdfplumber`
blocks with no async API, `MeteredClient` wraps a sync client, and the test suite
is built on sync `MagicMock`. Converting would mean touching six pipeline
modules, both entry points, `usage.py`, `evals.py` and three test files — to
reach identical throughput, because 5–20 concurrent HTTPS requests is nowhere
near where an event loop starts to win. Threads changed one file.

## Deterministic collection

Futures are submitted together and read back **in submission order** — never
`as_completed()`:

```python
futures = [executor.submit(process_one, ...) for pdf in pdfs[1:]]
for i, (pdf, future) in enumerate(zip(pdfs[1:], futures), start=2):
    accept(i, pdf, future.result())
```

Scored candidates would survive either way, since `rank_candidates()` re-sorts
them. `ProcessingError` entries would not: the reporter emits them in list
order, so completion-order collection would shuffle the "Could not process"
table and the `errors` array in `results.json` between runs of identical input.
The same applies to `audits` insertion order, which `bias_audit.json` iterates.

Nothing downstream streams, so `as_completed()` would trade reproducible output
for no gain. **No original-index bookkeeping was needed** — submission order
*is* the index.

Workers return their outcome and touch nothing shared. `results`, `audits` and
every printed line are owned by the collecting thread, which is why neither
collection needs a lock.

## Usage tracking

One `Anthropic` client, one `MeteredClient`, one `UsageTracker` — shared by
every worker, with a `threading.Lock` around the append in `record()` and the
tuple build in `records`.

The lock is arguably unnecessary: the tracker is append-only, `APIUsage` is
frozen, totals are derived on demand rather than kept as counters, and
`build_report()` is called after every worker has joined. `list.append` is
atomic under CPython, so nothing is lost without it. It was added anyway,
because the critical section is a single append held for nanoseconds between
API calls that take seconds, and because the alternative is a safety property
that only exists in a comment. `APIUsage.from_response()` runs *outside* the
lock so parsing one response never blocks another thread.

Per-worker trackers with a merge step were considered and rejected: they need a
second client, a merge API, and buy nothing.

One consequence is documented rather than fixed: under concurrency the `calls`
array in `usage.json` is in completion order, so it is no longer a chronological
trace. Totals, cost and the model list are all order-independent, so nothing
downstream cares. Run sequentially if you want the trace.

## The cache experiment

This is the centre of the challenge, and it was run as three measured stages
rather than one design decision.

### Stage 0 — sequential baseline

```
scorer:     1 write  /  9 reads
deep-dive:  1 write  /  2 reads
```

Stage 0 also discovered something that shaped every later measurement: **runs
inside the 5-minute TTL are not independent.** Back-to-back runs 2 and 3 showed
zero cache writes because run 1's entries were still alive and every read
refreshes the clock. Cached tokens were conserved at exactly 22,259 across all
three runs — the same prefix, moved from the write column to the read column.
Every cold-cache comparison after that was taken from a first run with a
verified idle gap.

### Stage 1 — naive parallel, deliberately shipped unfixed

Concurrency was added with **no** priming, on purpose. Priming before measuring
would have destroyed the experiment. Cold, at concurrency 5:

```
scorer:     4 writes  /  6 reads
deep-dive:  1 write   /  2 reads
```

The per-call trace shows the race directly — five extractions launch together,
then the first wave of scoring calls arrives at a cold cache:

```
 6 score WRITE      9 score WRITE
 7 score WRITE     10 score WRITE
 8 score read            ← the one worker that arrived after a write had landed
```

Extraction latency staggered exactly one worker out of five. Partial protection,
not protection.

**This was measured, not assumed.** The hypothesis was recorded before the paid
run, with an explicit falsification criterion (exactly 1 write would have meant
priming was unnecessary here). Three independent confirmations: the direct
count; the exact arithmetic `8,531 = 4×1,667 + 1×1,863` with the 22,259 total
still conserved; and a cost reconciliation matching the write-vs-read price
differential to within $0.0000006.

Deep-dive stayed at 1 write / 2 reads — the control. It was the one stage that
stayed sequential, and the one prefix that behaved.

### Stage 2 — primed parallel

```
first real candidate, alone:  parse → extract → score   ← writes the prefix
then the rest:                ThreadPoolExecutor(max_workers=concurrency)
```

```
scorer:     1 write  /  9 reads      ← restored
deep-dive:  1 write  /  2 reads
```

The primer is a **real candidate**, not a synthetic warm-up. No
`max_tokens: 0` request, no `threading.Event`, no score latch, no semaphore, no
split of `process_one` into phases. Its extraction is serialized as a side
effect — that is the entire cost — and the work itself had to happen anyway.

A `max_tokens: 0` pre-warm was investigated and rejected on a repo-specific
ground: it cannot carry `tool_choice: {"type": "tool"}`, and changing
`tool_choice` invalidates the *messages*-level cache, which is exactly where
`scorer.py` puts the JD. It would have warmed the tool schema and system prompt
and left the largest part of the prefix cold.

Reproduced on two independent cold runs, hours apart: 3,530 creation and 18,729
read tokens, digit-identical.

**If the primer fails**, the batch fans out anyway. A failed prime means the
cache may not have been written, so the run degrades to the naive profile — more
writes, identical results. That is a cost, never a wrong answer. Promoting the
next candidate to primer would mean a retry loop that serializes the whole batch
when every resume is bad, and "did the cache get written?" is not answerable in
`run()` without reading token counts, which would make orchestration cache-aware
and break the boundary that keeps metering out of the pipeline.

## Benchmark

| Metric | Sequential | Naive parallel | Primed parallel |
| --- | ---: | ---: | ---: |
| Median wall-clock | 268.49 s | 116.30 s | 127.12 s |
| Speedup vs sequential | 1.00× | 2.309× | 2.112× |
| Cold scorer writes | 1 | 4 | 1 |
| Cold scorer reads | 9 | 6 | 9 |
| Cold cache creation tokens | 3,530 | 8,531 | 3,530 |
| Cold cache read tokens | 18,729 | 13,728 | 18,729 |
| Cold estimated cost | $0.295030 | $0.315248 | $0.288679 |

### Read the cost column carefully

**Primed is not inherently cheaper than sequential.** Their cache economics are
*identical* — same writes, same reads, same token counts. The raw cost
difference is output-token nondeterminism, and it lands on either side depending
on the run. Decomposing two independent cold primed runs against the sequential
baseline:

| | vs sequential cold | of which cache |
| --- | ---: | ---: |
| Primed cold run 1 | −$0.006351 | **$0.000000** |
| Primed cold run 4 | +$0.002364 | **$0.000000** |

Opposite signs, cache component exactly zero both times. That is the robust
claim; the raw difference is noise.

Against naive the saving *is* architectural. Of the −$0.026569 observed
difference, output nondeterminism accounts for −$0.009060 and input for
−$0.000255. The real figure is:

```
3 redundant scorer cache writes eliminated
5,001 tokens moved from the $3.75/Mtok write rate to the $0.30/Mtok read rate
≈ $0.017254 per cold run  (5.47% of the naive cold cost)
```

API call count was exactly 23 in every valid run of all three arms. Concurrency
changes *when* calls happen, never how many.

### The tradeoff

```
Priming penalty vs naive:     +10.82 s   (+9.30%)
Primed gain vs sequential:    141.37 s saved, 52.65% faster, 2.112×
```

Priming gives back roughly one serialized candidate — just under 10% of the
naive wall-clock — and buys back three of four redundant cache writes. Two
things settle it in priming's favour for this project: production runs are cold
by definition (one batch per job posting, minutes or hours apart), so the naive
penalty is the normal case rather than the exception; and the write penalty
scales with concurrency while the priming cost stays fixed at one candidate, so
the trade improves as the field or the limit grows.

**Prime-then-fan-out is the final architecture.**

## Methodology

Fixed across all three arms: the 10 sample resumes (regenerated from the
checked-in dict in `scripts/generate_sample_resumes.py`), `sample_data/sample_jd.txt`,
`claude-sonnet-4-6`, `deep_dive_top=3`, self-critique and bias-audit off, CSV
off, concurrency 5 for both parallel arms.

Three timed runs per arm, **median** wall-clock — LLM latency is right-skewed by
slow calls and by invisible SDK-internal retries, so a mean overweights an
outlier the median ignores. Token and cost figures are reported per run rather
than averaged, because a difference there is a signal about caching, not noise.
Cold-cache comparisons use a first run with a verified idle gap; cold and warm
metrics are never averaged together.

`scripts/benchmark.py` owns the clock — a `time.perf_counter()` bracket around
the real `main.run(...)`, so what is measured is the application, not a
lookalike. **Everything else came from Challenge 8's `usage.json` with no new
instrumentation.** Because the prefix sizes are fixed constants for this
configuration, the per-call array yields an exact cache-write count:
`writes = count of calls with cache_creation_input_tokens == 1,667`. The cost
reporter turned out to be the instrument that made this challenge's central
claim falsifiable. Those constants live only in the benchmark script, as labels
for grouping — prefix groups are always *derived* from each run's own data, and
nothing in `src/` knows either number.

### One invalid run, kept

The third primed run hit an exhausted Anthropic credit balance during the
deep-dive stage. All three briefings returned HTTP 400 — not retryable, and
raised before a response existed, so no usage was recorded: 20 calls instead of
23, with the elapsed time missing the whole deep-dive stage. It is excluded from
every statistic and **preserved in `benchmarks/parallel-primed.json`** with a
`valid: false` flag and a reason. A replacement run was performed at the
identical configuration and the primed median uses three complete runs.

Worth noting what it accidentally demonstrated: the pipeline degraded exactly as
designed under a real outage — briefings logged and skipped, no
`ProcessingError`, exit 0, all 10 candidates intact.

### Reproducibility caveat

`benchmarks/parallel-naive.json` records a code state that **no longer exists in
the tree**. Priming is unconditional, not a flag, so re-running the harness with
`--label parallel-naive` today would produce a primed run. Reproducing that arm
means reverting the priming change. A `--no-prime` flag was deliberately *not*
added: permanent product complexity to re-enable a worse architecture, for the
sake of benchmark archaeology, is a bad trade. The artifact stands as historical
evidence, and this caveat is the honest label on it.

## The lesson

Concurrency improved latency. Prompt caching controlled cost. Each one changed
the assumptions the other was built on — the sequential pipeline was getting
correct cache behaviour for free precisely *because* it was slow, and the first
thing concurrency did was silently take that away.

The fix was not more parallelism, and not a cleverer cache. It was ordering: run
one candidate first. The measured cost of that decision is about ten seconds;
the measured benefit is three-quarters of the redundant cache writes.

The sequence matters as much as the answer. Stage 1 shipped the naive
architecture *knowing* it was probably wrong, because a fix applied before the
measurement would have left nothing to measure — and the falsification criterion
was written down first, so the data could have said priming was unnecessary. It
said the opposite, with an exact token count.
