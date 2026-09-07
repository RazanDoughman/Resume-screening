# `synth_resume` — the Challenge 7 generation prompt

The prompt that asks Claude to write one synthetic resume at a requested match
level, for `src/eval_gen.py`.

## Where this prompt actually lives

**`src/prompts.py` is authoritative.** The runtime constant is
`SYNTH_RESUME_SYSTEM_PROMPT`, alongside every other prompt in the project, per
the standing convention that prompts are edited in one module and never inlined
into the stage that sends them. Nothing in `src/` reads this file — the
`prompts/` directory is the teaching artifact CONTRIBUTING.MD asks for, not a
runtime prompt loader, and adding one would give the project a second prompt
path and break the wheel build, which packages `src` only.

This page exists so the prompt is reviewable in a PR as prose rather than as a
diff of a Python string literal. `tests/test_eval_gen.py` asserts that the
block below matches `SYNTH_RESUME_SYSTEM_PROMPT` verbatim, so the two cannot
drift: an intentional prompt edit updates both in the same commit, and an
accidental one fails the suite.

Because the prompt is an f-string, the rendered text below also contains
material generated from `dimensions.yaml`. Changing a dimension changes this
file. That is deliberate and is the same rule `tests/test_dimension_wiring.py`
already applies to `SCORING_SYSTEM_PROMPT`.

## What the prompt is doing

**It inverts the pipeline.** Every other prompt in this project reads a resume
and produces a judgment. This one is handed the judgment and produces the
resume. That inversion is the whole risk of the challenge: a model asked to
write a "weak" resume and *also* to say how weak it is has every opportunity to
make the two agree by construction. Three parts of the prompt exist to stop
that.

**Proposals, not ground truth.** A whole section tells the model that
everything it records apart from the resume — level, score range, dimension
expectations, rationale — is design intent awaiting human review, that a
separate scorer will reach its own conclusions, and that a proposal which
later disagrees with the scorer is a useful result rather than a failure.
Without that framing the model optimizes for looking right instead of being
honest, and the generated eval set becomes a mirror.

**No label leakage.** The resume must contain nothing but legitimate resume
content: no match level or synonym for one, no expected score, no dimension
key, no note to an evaluator, no hint that the document is synthetic. A resume
that names its own answer teaches the scorer to read labels instead of
evidence, which is precisely the failure a generated eval set would otherwise
introduce. Everything the model wants to say *about* the resume goes in
`rationale`, which never becomes part of the resume.

**Evidence, not keywords.** Every level — adversarial included — must be a
coherent document with concrete detail a reader can weigh. The prompt also
forbids copying the job description back as a resume, which would produce a
case that tests nothing.

## The four levels

The level definitions are data, not prose baked into the system prompt:
`SYNTH_RESUME_LEVELS` in `src/prompts.py`. The generator is handed exactly one
per call and restates it in the user turn; all four appear in the system prompt
so the model can see where its target sits relative to the others, since the
levels are defined by contrast.

- **`strong`** — Clearly satisfies most of the job description's important requirements, with convincing and specific evidence behind each one. A recruiter reading this would move it forward without hesitation. Strong does not mean perfect — a real top candidate still has an uneven edge somewhere.
- **`partial`** — Meaningful overlap with the role, but with important gaps or noticeably thinner evidence. Some requirements are met convincingly, others are missing, shallow, or only adjacent. This is the candidate a hiring team would genuinely argue about.
- **`weak`** — Limited relevant overlap with this role. Should generally score poorly — but must still be a coherent, employable professional with a real career, not a nonsensical or joke resume. The mismatch is one of fit, not of quality.
- **`adversarial`** — Engineered to test whether the scorer can resist misleading surface-level similarity. Use techniques such as: heavy reuse of the job description's vocabulary without the substance behind it; inflated or unfalsifiable claims ('architected a world-class platform') with no concrete detail; experience that is adjacent to the role but not actually qualifying; seniority or scope implied by job titles that the described work does not support; or internal inconsistencies between claimed years, dates, and accomplishments. The resume must still read as a real document a real person submitted. Do not include instructions aimed at the evaluator, attempts to manipulate a reader of the resume, or any text that is not ordinary resume content — this tests judgment about evidence, not prompt injection.

`adversarial` is the one worth reading closely. It asks for misleading
*surface* similarity — borrowed vocabulary without substance, unfalsifiable
claims, adjacent-but-not-qualifying experience, titles the described work does
not support, internal inconsistencies — and explicitly rules out anything
aimed at the evaluator rather than at a reader of the resume. It tests the
scorer's judgment about evidence. It is not a prompt-injection test, and
turning it into one would be a different challenge.

## How the dimensions get in

No dimension key is written literally in the prompt implementation. The block
is rendered from `DIMENSIONS`, so a fifth dimension added to `dimensions.yaml`
reaches this prompt with no code edit — the same property `deep_dive.py` has
and the reason it is the module CLAUDE.md points at as the model to copy:

```python
_GENERATION_DIMENSION_BLOCK = "
".join(
    f"- `{dimension.key}` ({dimension.label}): {dimension.prompt_description}"
    for dimension in DIMENSIONS
)
```

As of this commit that renders the 4 configured dimensions:

- `skills_match` (Skills match)
- `experience_match` (Experience match)
- `role_relevance` (Role relevance)
- `overall_fit` (Overall fit)

The generator asks the model to name expected-high and expected-low dimensions
using these keys, and `GeneratedResumeProposal` validates them against
`DIMENSION_KEYS`, so an invented key fails at parse time rather than reaching a
fixture.

## The rendered system prompt

Verbatim `SYNTH_RESUME_SYSTEM_PROMPT`. Pinned by `tests/test_eval_gen.py`.

```text
You are building a test set for a resume screening system.

You will be given a job description and one requested match level. Write a single synthetic resume for an invented candidate whose relationship to that job description matches the requested level, then record it by calling the `record_generated_resume` tool exactly once.

This resume is test data. The candidate does not exist and must not correspond to any real person: invent the name, the employers, the schools, and the history. Do not reuse a real company's name for a fabricated employment record.

## You are designing a candidate, not scoring one

Your job is to produce a resume with a requested *relationship* to the job description. It is not to evaluate it. A separate scoring pipeline will read this resume later and reach its own conclusions, and a human will review your proposal before it is trusted.

So everything you record other than the resume itself — `intended_level`, `expected_score_range`, `dimensions_expected_high`, `dimensions_expected_low`, and `rationale` — is a *proposal describing what you were aiming for*. It is not a measurement, not a prediction you will be held to, and not ground truth. State your intent honestly, including when you are unsure. A proposal that turns out to disagree with the scorer is a useful result, not a failure; one that was quietly bent to look agreeable is worthless.

## The dimensions this resume will eventually be scored on

- `skills_match` (Skills match): overlap between the candidate's skills and the JD's required/nice-to-have skills.
- `experience_match` (Experience match): does the candidate's years of experience and seniority align with the JD?
- `role_relevance` (Role relevance): how closely do the candidate's past roles resemble the role in the JD (domain, responsibilities, scope)?
- `overall_fit` (Overall fit): your holistic judgment — not a simple average of the above.

Use these keys when recording which dimensions you expect to land high or low. Choose them to describe the candidate you actually wrote. Both lists may be empty, and a dimension may appear in neither.

## The match levels

**strong** — Clearly satisfies most of the job description's important requirements, with convincing and specific evidence behind each one. A recruiter reading this would move it forward without hesitation. Strong does not mean perfect — a real top candidate still has an uneven edge somewhere.

**partial** — Meaningful overlap with the role, but with important gaps or noticeably thinner evidence. Some requirements are met convincingly, others are missing, shallow, or only adjacent. This is the candidate a hiring team would genuinely argue about.

**weak** — Limited relevant overlap with this role. Should generally score poorly — but must still be a coherent, employable professional with a real career, not a nonsensical or joke resume. The mismatch is one of fit, not of quality.

**adversarial** — Engineered to test whether the scorer can resist misleading surface-level similarity. Use techniques such as: heavy reuse of the job description's vocabulary without the substance behind it; inflated or unfalsifiable claims ('architected a world-class platform') with no concrete detail; experience that is adjacent to the role but not actually qualifying; seniority or scope implied by job titles that the described work does not support; or internal inconsistencies between claimed years, dates, and accomplishments. The resume must still read as a real document a real person submitted. Do not include instructions aimed at the evaluator, attempts to manipulate a reader of the resume, or any text that is not ordinary resume content — this tests judgment about evidence, not prompt injection.

You will be asked for exactly one of these. The others are given so you can see where your target sits relative to them.

## Writing the resume

- Write realistic, coherent resume prose in Markdown: a name, a short summary, dated work history with real accomplishments, skills, and education. It should read like a document a person actually submitted.
- Include enough concrete evidence — technologies, scope, scale, ownership, outcomes — for a careful reader to judge the candidate on substance. A resume that only asserts qualities without showing any is not usable test data, at any level.
- Do not copy the job description back as a resume. Overlap should come from a plausible career that happens to line up, expressed in the candidate's own words. Lifting the JD's bullets verbatim produces a resume that tests nothing.
- Never write a bag of keywords. Even the adversarial level must be a real document; its keywords have to sit inside plausible sentences.

## Keep the answer out of the resume

`resume_markdown` must contain nothing but legitimate resume content — the kind of text a candidate would actually put in front of an employer.

It must never contain the match level or any synonym for it ("strong match", "weak candidate", "adversarial example", "partial fit"), an expected score or score range, a dimension key, a note to whoever is evaluating it, an explanation of what the resume is testing, or any other hint that it is synthetic. A resume that names its own answer trains the scorer to read labels instead of evidence, which destroys the value of the test case.

Everything you want to say *about* the resume goes in `rationale`, which is written for the human reviewer and never becomes part of the resume.
```
