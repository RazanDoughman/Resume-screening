# Challenge 7 — Eval set generator

Ask Claude to write synthetic resumes at four match levels, save them as
reviewable artifacts, and let a human — never the generator — decide which ones
become trusted eval cases.

The prompt itself is documented separately in
[`prompts/synth_resume.md`](synth_resume.md). This page records what happened
when the generated cases were first run against the real scorer, and what we
decided to do about it.

## What this adds

| Piece | Role |
| --- | --- |
| `GeneratedResumeProposal` (`src/models.py`) | what the generator returns: the resume plus its *stated intent* — level, expected range, expected-high/low dimensions, rationale |
| `generate_resume_proposal()` (`src/eval_gen.py`) | one LLM call, one match level, one validated proposal |
| `generate_proposal_set()` (`src/eval_gen.py`) | asks for each level in turn — four calls, sequential |
| `src/eval_fixtures.py` | writes and reads the raw proposal artifacts. No LLM |
| `GeneratedEvalCase` + `src/generated_cases.py` | the review gate: a proposal plus human-approved ranges plus `reviewed` |
| `scripts/gen_eval_cases.py` | the CLI. Reads a JD, builds a `MeteredClient`, calls the two functions above, reports where the output went |

Nothing in `src/` gained a dependency; the CLI is the only new entry point.

### Four levels, one call each

`strong` satisfies most important requirements with convincing evidence.
`partial` has meaningful overlap but real gaps. `weak` has limited relevant
overlap while remaining a coherent, employable career. `adversarial` is
engineered to *look* qualified — JD vocabulary without the substance, inflated
and unfalsifiable claims, adjacent-but-not-qualifying experience, titles the
described work does not support.

The order comes from `MatchLevel` itself (`get_args`), not a second hardcoded
list. One level per call is deliberate: asking for all four in one response
makes them compete for the same output budget and lets the model tune them
against each other, which is exactly the self-consistency this challenge is
trying to avoid.

The calls are sequential and share one cached prefix — tool schema, system
prompt, job description, in that order, with the cache breakpoint at the end of
the JD and the level named *after* it. The first live run measured **1 cache
write and 3 reads** (3,216 tokens written once, read three times), which is the
whole reason not to put a thread pool around it.

### The CLI

```bash
uv run python -m scripts.gen_eval_cases --jd sample_data/sample_jd.txt --dry-run
uv run python -m scripts.gen_eval_cases --jd sample_data/sample_jd.txt
```

`--jd` (required), `--out-dir` (default `tests/evals/generated`), `--model`
(default `claude-sonnet-4-6`), `--dry-run`. There is no `--levels` or `--n`:
`generate_proposal_set()` owns the level list, so a flag would either duplicate
it or spend the calls and then discard the results.

`--dry-run` prints the plan and exits *before* the API-key check and before a
client exists, so it works on a machine with no credentials — which is what
makes it useful for checking wiring.

A run writes `proposals.json` (exact records, the round-trip source),
`NN-<level>.md` (one per proposal, for a human to read), and `usage.json`. The
Markdown is a view, never parsed back — the same JSON-for-machines /
Markdown-for-people split `reporter.py` already uses.

## The rule the whole design exists to enforce

A generator that writes a resume *and* states how good it is has an obvious
failure mode: it can make the two agree by construction, and the resulting eval
set measures nothing except the generator's self-consistency. So the generated
metadata — intended level, expected score range, expected-high/low dimensions —
is treated as a **proposal**, and a human supplies the ranges the suite actually
enforces.

That boundary is enforced by code, not convention. `promote_proposal()` requires
`expected_ranges` from the caller and never reads the proposal's own range;
`reviewed` is keyword-only and defaults to `False`; and
`load_eval_case_kwargs()` — the only path from generated data into `ALL_CASES` —
filters on `reviewed` with no override. The generator's proposed range is kept
beside the human's as `proposed_score_range`, for comparison, and is never used
as an expectation.

## First live run

Four resumes generated from `sample_data/sample_jd.txt`, reviewed by hand,
promoted with human-approved ranges, then scored by the real pipeline.

Result: **4/4 hand-written cases passed, 0/4 generated cases passed.** The
failures were not equivalent, and the reason they were not equivalent is the
whole return on this challenge.

| Case | skills | exp | role | overall | Adjudication |
| --- | ---: | ---: | ---: | ---: | --- |
| `generated_strong_payments` | 98 | 97 | 99 | 98 | expectation too strict |
| `generated_partial_payments` | 85 | 88 | 90 | 87 | **scorer finding — kept failing** |
| `generated_weak_payments` | 8 | 12 | 5 | 7 | expectation too generous |
| `generated_adversarial_payments` | 98 | 96 | 99 | 97 | **scorer defect — kept failing** |

Two of these were our error and were corrected. Two were the suite telling us
something about the scorer, and were left red on purpose.

### strong — our ceiling was wrong

Every dimension passed except `overall_fit`, which came in at 98 against a
ceiling of 95. The scorer's reasoning was specific and correct, and it caught
the one deliberate imperfection in the resume ("Go is listed as 'working
knowledge' rather than production-level"). The hand-written `strong_match_payments`
case allows 80–100 and also scored 98 in the same run, so a 95 ceiling was out
of line with how this scorer treats a near-perfect candidate.

**Changed `overall_fit` to 82–100.** Nothing else.

### weak — our floors were wrong

The scorer went *below* our floors on three dimensions. The disagreement was
concentrated in `experience_match`: we read it as tenure alignment (six years,
clears the "5+ years" bar), the scorer read it as *relevant* experience — "While
she has 6 years of experience (meeting the tenure requirement), all of it is in
frontend/UI engineering". `dimensions.yaml` phrases the dimension as "does the
candidate's years of experience and seniority align with the JD?", which
honestly supports both readings. The scorer's position is defensible and ours
was generous.

**Lowered to skills 5–25, experience 10–35, role 5–25, overall 5–25.**

**Review note, recorded here and in the case's `description`:** this case is
retained, but the resume states several of its own weaknesses outright ("no
schema design or production DBA experience", "configuring hosted widgets, not
building custom integrations"). Real candidates do not volunteer disqualifiers
like that, and the scorer quoted them straight back in its gaps. The case is
therefore **easier than an ideal weak candidate** and discriminates less than it
could. Worth regenerating with a prompt tweak at some point; not worth
regenerating now.

### partial — kept failing

`role_relevance` scored 90 against an expected 45–68, and `overall_fit` 87
against 50–70. The scorer justified it as "she owned subscription billing flows
(auth, refund, reconciliation)" — but *that phrasing is the JD's, not the
resume's*. The resume says proration logic, dunning workflows, invoice
generation, and Stripe card charging. The scorer restated e-commerce
subscription billing in payments-platform terms and then scored the restatement.

It did identify four real gaps (Kafka only at personal-project level, limited
Kubernetes/IaC, no non-Stripe rails, no PCI/SOC 2) and then declined to let any
of them move the score. Our range may have been somewhat strict; the loose
mapping of adjacent experience into JD vocabulary is the more interesting half,
and it is the same mechanism that produced the adversarial failure below.

**No range changed.** The failure stays visible.

### adversarial — kept failing, and this is the finding

The adversarial resume scored **98 / 96 / 99 / 97** with **`gaps: []`**.

For comparison, the genuinely strong candidate scored 98 / 97 / 99 / 98 with one
gap. **The scorer found fewer problems with the fabricated résumé than with the
real one.**

What it missed:

- **It read a transcription of the JD as substantiated skill.** The resume's
  "Distributed Systems" line is a near-verbatim copy of the JD's bullet; the
  scorer credited it back as "all distributed-systems fundamentals (idempotency,
  retries, at-least-once delivery, eventual consistency, circuit breakers)".
- **It accepted unfalsifiable claims.** "Architected and owned the core payments
  platform, delivering industry-leading reliability" — no metric, no team size,
  no architectural decision — became "his most recent position was literally
  building a payments platform".
- **It did not distinguish analytics from payments infrastructure.** The only
  verifiable depth is reporting pipelines and webhook *ingestion* at Cloudspark.
  The scorer never mentioned that role in any of its four reasonings.
- **It ignored every hedge.** The resume says Go "(familiar)", "PCI DSS
  awareness", "SOC 2 exposure". The scorer listed these as plain matches.
- **It applied more scrutiny to the honest resume.** It flagged the strong
  candidate's "working knowledge" Go as a gap, and said nothing about the
  adversarial candidate's *weaker* "familiar" Go.

**No range changed.** Widening this to 90–100 would delete the only test in the
suite that catches this, and the case would then certify the bug as correct
behaviour.

## What this means for the suite

`uv run python -m src.evals` is expected to report **6/8 scoring cases passing**
until the scorer changes. That is the intended state, not an outstanding bug in
the eval data:

- `generated_partial_payments` fails while the scorer over-credits adjacent
  experience described in JD language.
- `generated_adversarial_payments` fails while the scorer cannot tell recited
  vocabulary from demonstrated evidence.

Both should go green as a *consequence* of fixing the scorer — never by editing
the range. A prompt change that makes the adversarial case pass without also
keeping the strong case high is the signal this suite exists to give.

Fixing the scorer is deliberately out of scope for Challenge 7. Challenge 7's
job was to produce evidence, and it did that on its first paid run.

## Known, out of scope

**`src/evals.py` builds a raw `Anthropic()`** ([src/evals.py:203](../src/evals.py)),
not the `MeteredClient` that `main.py` and `app.py` use. Live eval runs are
therefore the one place in the project that spends money without recording a
token — the 16-call run above had to be metered from outside the runner to get
the numbers quoted here. Wiring Challenge 8's seam into the eval runner is a
one-line change but it is Challenge 8's, not Challenge 7's, and nothing in this
challenge required it.

**The weak case is easier than ideal**, for the reason recorded above. Both are
noted rather than fixed, so the next person finds them written down instead of
rediscovering them.

## A note on cost

Generating the four proposals cost **$0.097** (4 calls, one cached prefix
written and read three times). Running the 8 scoring cases cost **$0.231** (16
calls). The adversarial finding cost about thirty cents to discover.
