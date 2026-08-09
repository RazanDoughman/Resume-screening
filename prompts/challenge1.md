# Challenge 1 — Add a 5th eval case

Adds `CAREER_CHANGER_MATCH` to `ALL_CASES` in `tests/evals/cases.py`: a senior
SRE/platform engineer scored against the existing `_PAYMENTS_JD`. Only that file
changed — no architecture, prompt, or scoring-logic changes.

## What it tests

The existing three cases vary skills and experience, but always move role
relevance along with them:

| | High experience | Low experience |
|---|---|---|
| **High skills** | `STRONG_MATCH` (also high relevance) | `MIXED_MATCH` |
| **Low skills** | `WEAK_MATCH` | — |

Nothing tested **high skills + high experience + low role relevance**.
`MIXED_MATCH` is the nearest case but varies the wrong axis: Priya Ramesh has the
right domain and the right role type, and is held back only by two years of
experience.

`CAREER_CHANGER_MATCH` is the first case where a candidate is simultaneously very
well qualified and a poor role match. It tests whether the scorer can separate
transferable technical strength from actual relevance to this specific role.

## Why high skills and experience but lower relevance

This follows from the JD's structure, not from a hiring opinion. All six
**Required** bullets are generic senior backend — Python, PostgreSQL, REST/gRPC,
distributed-systems fundamentals, on-call maturity — and a platform engineer
genuinely satisfies them. Payments appears once, under **Nice to have**.

The scoring rubric in `src/prompts.py` therefore routes the mismatch to a
specific dimension:

- `skills_match` — overlap with required/nice-to-have skills. Stays high; the
  transferable skills are real.
- `experience_match` — years and seniority only. Stays high; 8 years clears a
  5+ bar.
- `role_relevance` — domain, responsibilities, and scope. All three are where
  the gap lives.

Marcus Feld satisfies every Required bullet plus Kafka, Kubernetes, and
Terraform. What he lacks is payments domain experience and product-backend
ownership — his services are consumed by internal operators and platform teams,
not customers. Neither absence is stated in the resume; both must be inferred
from the work history.

A scorer that penalises `skills_match` here would be wrong, not strict. The case
asserts the penalty lands in the right place.

## Expected ranges

| Dimension | Range | Role in the case |
|---|---|---|
| `skills_match` | `(75, 100)` | guardrail — must not punish transferable skills |
| `experience_match` | `(75, 100)` | guardrail — must not punish seniority |
| **`role_relevance`** | **`(55, 79)`** | **the assertion** |
| `overall_fit` | `(55, 85)` | guardrail — must not equate with the payments native |

Every bound is anchored to another case's band or to the JD, not fitted to an
observed score:

- `role_relevance` floor **55** > `WEAK_MATCH`'s 40 ceiling — fleet operations at
  Staff scope are clearly more relevant than frontend work.
- `role_relevance` ceiling **79** < `STRONG_MATCH`'s 80 floor — platform
  enablement, however senior, must not read as relevant as payments-native
  experience.
- `overall_fit` floor **55** > `WEAK_MATCH`'s 45 ceiling.
- `overall_fit` ceiling **85** — anchored to `STRONG_MATCH`'s *observed* ~98
  rather than its nominal 80 floor, requiring 13+ points of separation while
  allowing that a candidate meeting every requirement is a genuinely strong hire.

`skills_match` and `experience_match` are deliberately wide. They carry no claim,
so a boundary hit there would confound the result rather than inform it.

The `role_relevance` band was revised once during calibration. An earlier
`(40, 65)` failed consistently at 72 — that band encoded a stricter hiring
philosophy than either the JD or the rubric supports. It was re-derived from the
ordering properties above rather than widened to fit the observation. It would
fail if a future prompt change made the model treat platform work as fully
role-relevant, or as disqualifying: sensitivity in both directions is what makes
it a property rather than a recording of one run.

## What calibration taught us

**The model weighs the domain gap modestly, and defensibly.** It detected the
missing payments experience in every dimension's rationale and reported it as the
sole gap, reasoning explicitly that it is "listed as nice-to-have rather than
required." The initial ranges assumed a 30-40 point relevance penalty; the model
applies roughly 17. On the JD as written, that is a fair reading.

**The model does not weigh who consumes an API.** The first draft resume
described an "internal provisioning control plane" and a "REST admin API for
internal operators." The word *internal* survived extraction into the scored
profile and was never referenced in any dimension's reasoning — "gRPC service
ownership" was credited at face value. Rewriting the resume to name consumers
three times and to bind the work to node lifecycle operations moved
`role_relevance` from a stable 78-80 to a stable 72. The signal registers when
made structural, but internal-platform and product-backend ownership are not
distinguished by default. That distinction is material to this role.

**Realistic detail can carry unintended weight.** The `SERIALIZABLE isolation`
detail — included so the case would not test a second technical weakness —
became the model's strongest bridge to payments relevance. It coined
"transactional PostgreSQL" and used it as headline evidence.

## Authoring prompts

Condensed from a multi-turn session; substance preserved.

**1 — Scope and plan.** Start Challenge 1, but do not implement or modify any
files yet. Inspect the repository and give a clear implementation plan. The
scenario: a technically strong candidate with several years of experience and
strong overlap with the JD in Python, PostgreSQL, distributed systems,
infrastructure/platform engineering, and on-call — but most experience is
SRE/platform, with little payments-domain experience and limited ownership of
customer-facing product backend APIs. The purpose is to test whether the model
can distinguish transferable technical skills and strong overall engineering
experience from actual relevance to this specific role. Confirm whether
`_PAYMENTS_JD` should be reused, whether the runner loops `ALL_CASES`
automatically, and what documentation requirements apply. Do not choose expected
score ranges yet.

**2 — Range derivation, before writing the resume.** Settle the expected ranges
first. Compare the candidate against each required and nice-to-have bullet one by
one. Show how the existing three cases are ranged so we stay consistent with the
project's calibration style. Propose ranges and explain why each is defensible
**from the JD rather than from expected model output**. Point out any range that
would overlap too much with the existing cases. Do not run the model.

**3 — Resume design.** Draft the exact `resume_text`. It must contain no payments
or fintech domain experience, avoid explicitly saying the candidate lacks
payments experience, avoid phrases like "career changer" or "looking to
transition", and make the domain mismatch something the model must infer from the
actual work history. Follow the formatting and approximate length of the existing
eval resumes.

**4 — Calibration.** Run the eval three times using the exact same code, resume,
ranges, and model. Report the four scores per run and whether the case passed. Do
not modify the ranges, resume, prompts, or scoring logic based on the results.

**5 — Diagnosis.** Retrieve the model's reasoning using a one-off scratchpad
script only; do not modify any project files. Analyse especially why the model
considers `role_relevance` around 78-80 despite the SRE/platform background and
zero payments experience.

**6 — Refinement.** The case mixes two questions: how much missing
payments-domain experience should matter when payments is only a nice-to-have,
and whether the model distinguishes internal platform/SRE API ownership from
product-backend ownership. The second is the sharper blind spot. Propose how to
refine the case to test it more cleanly — keep the candidate technically strong
and senior, keep the same JD, and avoid simply making the candidate weaker.
Identify which parts of the resume create too much semantic similarity to the
payments role.

**7 — Whether a failing case is acceptable.** Inspect `CONTRIBUTING.MD`, README
guidance, and the existing eval conventions, and answer one question: is an eval
contribution expected to leave the full suite passing, or is it acceptable for a
newly contributed case to fail consistently as a documented model finding?
Support the answer with the exact relevant repository guidance, then recommend
whether to keep the failing range or recalibrate.

**8 — Final range derivation.** Determine a defensible `overall_fit` range using
the JD, the scoring rubric, the existing case ranges, and the same
ordering-property approach — rather than simply fitting the observed score.

## Verification

```
uv run pytest
34 passed
```

```
uv run python -m src.evals
4/4 scoring cases passed
1/1 bias cases passed
```

`CAREER_CHANGER_MATCH` final scores:

| Dimension | Score | Expected | |
|---|---|---|---|
| `skills_match` | 92 | 75-100 | ✓ |
| `experience_match` | 90 | 75-100 | ✓ |
| `role_relevance` | 72 | 55-79 | ✓ |
| `overall_fit` | 81 | 55-85 | ✓ |

Stability on the final resume: `role_relevance` 72 in every complete run,
`overall_fit` 80-81. Existing cases and the bias case are unaffected.

## Possible follow-up

A paired candidate with the same domain gap but genuine product-backend role
shape. The delta between the two cases would isolate role shape from domain more
cleanly than a single case can.
