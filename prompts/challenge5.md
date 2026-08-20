# Challenge 5 — Deep-dive stage

After scoring, ranks the field and writes a hiring-manager briefing — summary,
pros, cons, interview questions — for the top N candidates only. New:
`src/deep_dive.py`, `tests/test_deep_dive.py`, `tests/test_main_deep_dive.py`.
Modified: `main.py`, `app.py`, `src/models.py`, `src/prompts.py`,
`src/reporter.py`, `src/evals.py`, `tests/evals/cases.py`, `tests/test_reporter.py`.

`src/usage.py` has a **zero-line diff**, and so do `scorer.py`, `extractor.py`,
`critique.py`, `bias_auditor.py`, `pdf_parser.py`, `dimensions.py` and
`dimensions.yaml`. The scoring prompt is untouched, so no score can move; the
evals were written but not rerun for that reason.

## The problem

Scoring answers *who is at the top*. It does not answer *what do I do with
them*. A recruiter reading `report.md` gets a ranked table and one sentence per
dimension, then has to build the interview themselves from the gaps list.

The obvious fix — have the scorer write more prose — is the wrong one. It would
cost tokens on every resume, including the 90% nobody will ever call back.

## The shape of the feature is its economics

Every other optional stage in this repo multiplies by resume count.
`--bias-audit` on 10 resumes adds 60 calls. Deep-dive adds exactly N, whether
the field is 10 resumes or 500:

```
total = R × (2 + [1 if --self-critique] + [6 if --bias-audit]) + min(N, scored)
```

That asymmetry is the whole point, and it's why the flag is `--deep-dive-top N`
rather than a bare `--deep-dive`: the count *is* the feature. One flag, one
concept, and no way to express "deep-dive enabled but for how many?"

## Ranking had to move before the feature could exist

Ranking lived in `reporter._split()`, private, called three times — once each by
`write_json`, `write_csv` and `write_markdown` — and every one of those runs
*after* the point where a deep-dive stage needs to know who the top N are.
`main.py` never held a ranked list at all.

So `rank_candidates()` came out of `_split()` as a public function, and `_split`
delegates to it. Pure refactor, no behavior change, and all 84 existing tests
passed unmodified — which is the proof it was one.

The extraction also let the tiebreak become explicit:

```python
candidates.sort(key=lambda c: (-c.overall_fit.score, c.source_file))
```

Ties were already deterministic in the CLI, but only accidentally: a stable
sort meeting `main.py`'s already-filename-sorted input. That's two facts in two
files holding each other up, and it was fine only while rank had no
consequence. Now rank decides which candidates a paid call is spent on, so the
order became a property of the function. It also makes `app.py`, whose input is
browser upload order, deterministic for the first time.

## Strict N, and saying so

With scores 94, 91, 88, 88, 82 and `--deep-dive-top 3`, the fourth candidate is
cut even though they tie the third.

Including ties was the alternative, and it's the fairer-sounding one. It's also
unbounded: `reporter.py`'s own histogram comments note that everyone landing in
one score bin is a real observed outcome, and in that run a flag that says 3
spends 10 calls. In the repo that shipped a cost reporter one challenge ago,
that's the wrong trade.

The honest resolution isn't to pick fairness or predictability — it's to keep
predictability and stop hiding the seam:

```
Deep-dive: briefing top 3 of 10...
  note: d.pdf also scored 88 but was not included (raise --deep-dive-top to 4).
```

## What the briefing reads, and what it can't

`generate_deep_dive()` takes a `ScoredCandidate` — profile, every dimension's
score and reasoning, the overall reasoning, and the gaps. Not the raw resume
text.

That wasn't purely a preference. In `main.py` the parsed text is a local inside
`process_one()` and falls out of scope; in `app.py` the uploaded PDF is
`os.unlink`'d immediately after processing. Threading raw text through would
mean restructuring both entry points, and only one of them could be done
cleanly.

The consolation is that the alternative was overrated. Raw text can't tell you
what the screener already concluded, and a briefing's risks and interview
questions are built precisely from the gaps and the low-scoring dimensions. A
`resume_text: str | None = None` parameter is left on the signature as a
one-line upgrade path if an eval ever shows the questions going generic.

## Dimensions arrive dynamically, and a test enforces it

Challenge 4 made `dimensions.yaml` the source of truth for the schema and the
prompt, but seven call sites still name dimensions literally. This is the
eighth call site, and it doesn't:

```python
"dimension_scores": {
    d.key: {"label": d.label,
            "score": getattr(candidate, d.key).score,
            "reasoning": getattr(candidate, d.key).reasoning}
    for d in DIMENSIONS
}
```

The system prompt says "each scoring dimension" and never names one either — so
a fifth dimension in the YAML reaches the briefing with no code and no prompt
edit. `test_dimension_names_are_not_hardcoded_in_the_module_or_prompt` asserts
no dimension key appears in either file, so the property can't quietly rot.

Rendering the names into the prompt from `DIMENSIONS` was the tempting
alternative and would have been worse: `SCORING_SYSTEM_PROMPT` is already
generated, `test_dimension_wiring.py` pins it byte-for-byte because generated
prompt text is dangerous, and a second generated prompt doubles that surface
for no gain.

## Metered by accepting an argument

`prompts/challenge8.md` predicted this in writing: *"when Challenge 5 adds a
deep-dive stage, it gets metered for free — it takes a client, so it's already
covered."*

It was right. `generate_deep_dive(client, ...)` receives the `MeteredClient`
that `run()` already built, calls `client.messages.create(...)`, and the calls
appear in `usage.json`. `src/usage.py` diff: zero lines.

The one thing that had to be got right is ordering — the stage sits after the
resume loop but *above* `build_report(tracker.records)`. Below it, the calls
would still be billed and still be missing from the report. That's invisible to
unit tests, so `test_deep_dive_calls_happen_before_the_usage_report_is_built`
spies on `build_report` and asserts the briefing count at the moment it runs.

## Caching: its own prefix, deliberately

The intuition that deep-dive would reuse the scorer's cached JD is wrong. A
cache prefix covers the tools and system prompt too, and both differ, so this
is a separate entry from the first byte. It doesn't disturb the scorer's — that
cache has already been read nine times by the time this stage starts.

Block order is the scorer's, for the same reason:

```
[0] <job_description>  + cache_control: ephemeral    ← identical every candidate
[1] <candidate_profile>                              ← after the breakpoint
[2] <screening_results>                              ← after the breakpoint
```

Anything before the breakpoint is part of the prefix, so a candidate name
placed there would make the prefix unique per candidate and silently turn
caching off. `test_cached_prefix_is_identical_across_candidates` briefs two
different people and asserts block 0 is byte-identical while block 1 is not.

Estimated prefix is ~1500 tokens against the 1024-token minimum — it should
cache, with less margin than the scorer's. Deliberately not padded to buy
headroom; the smoke test reads `usage.json` and will say.

No `ttl` is set, which keeps the 5-minute assumption `MODEL_PRICING` documents
true. Expect aggregate `cache_creation_input_tokens` to roughly double when the
flag is on — two cached prefixes now, not a regression.

## A sidecar, because the schema is a contract

`DeepDiveReport` is **not** a field on `ScoredCandidate`. It travels in a
`dict[str, DeepDiveReport]` keyed by `source_file` and is joined in the
reporter, exactly as `BiasAuditReport` already does.

Nesting it would put `"deep_dive": null` into every candidate object in
`results.json` — for a briefing most candidates never get. It would also fail
`test_scored_candidate_exposes_exactly_the_expected_fields`, which asserts that
model's field set by exact equality. Challenge 4 installed that tripwire; this
challenge is the first to walk into it, and it held.

`write_markdown` grew a `deep_dives=None` keyword. `write_json` and `write_csv`
didn't grow anything, which is what makes the JSON and CSV schemas structurally
incapable of changing rather than merely unchanged.

## Failure is not a processing error

A briefing that fails logs a warning and skips. The candidate keeps their
score, the run keeps its exit code, every output still gets written. The
`ProcessingError.stage` Literal stays `parse`/`extract`/`score` — widening it
would change the `results.json` contract to describe something that isn't a
resume-processing failure at all.

## Tests

157 passing, up from 84. The 84 are unmodified: `test_dimension_wiring.py` and
`test_usage.py` were not touched, and their staying green is the regression
signal for "did the model change" and "did metering change".

Notable ones:

- `test_markdown_is_byte_identical_when_no_deep_dives_are_passed` — the sidecar
  is inert when absent, compared three ways (default, explicit `None`, `{}`).
- `test_cached_prefix_is_identical_across_candidates` — the caching contract,
  checked for free.
- `test_deep_dive_calls_happen_before_the_usage_report_is_built` — the ordering
  no unit test can see.
- `test_selection_matches_the_first_n_of_the_reports_ranking` — selection and
  the written report can never disagree about who is on top.
- `test_module_constructs_no_anthropic_client` — greps the source, because the
  failure mode is silent unmetered calls.

The eval is one `DeepDiveEvalCase` reusing `STRONG_MATCH`'s resume. Assertions
are structural — a summary of real length, ≥2 pros, ≥1 con, ≥3 questions, and
≥1 distinctive term from that resume. `min_cons=1` is the load-bearing one:
handed a 95-scoring candidate, a model will happily return an empty cons list,
which makes the briefing worthless exactly where it's needed. Prose quality
would need a judge model, which is a bigger commitment than this challenge
warrants and has no precedent here.

Not yet run — it costs real calls, and the implementation is up for review
first.
