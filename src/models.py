"""
Pydantic models used across the pipeline.

These classes are the contracts between modules. Every LLM call in this
project returns structured data shaped like one of these models — that's
how we guarantee the agent's output is valid JSON with the right fields
instead of free-form text we'd have to parse ourselves.

When Claude is asked to "call the record_candidate tool", we hand it a
JSON schema generated from CandidateProfile. Claude's response must match
that schema. Pydantic then validates it on our side.
"""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from src.dimensions import DIMENSION_KEYS, DIMENSIONS


class Role(BaseModel):
    title: str
    company: str
    duration_months: int = Field(
        ge=0,
        description="Total time in the role, in months. Zero if unknown.",
    )
    description: str


class CandidateProfile(BaseModel):
    name: str
    years_experience: float = Field(
        ge=0,
        description="Total professional experience in years. May be fractional.",
    )
    skills: list[str]
    past_roles: list[Role]
    education: list[str]
    raw_summary: str = Field(
        description="One-paragraph summary of the candidate in their own words.",
    )


class DimensionScore(BaseModel):
    score: int = Field(ge=0, le=100)
    reasoning: str


class Gap(BaseModel):
    category: Literal["skill", "experience", "other"]
    detail: str


# The per-dimension fields, built once from dimensions.yaml and inherited by
# both scoring models. This base is the single source for the scoring *schema* —
# ScoreReport (the JSON schema Claude is handed as the record_score tool) and
# ScoredCandidate. Together with the bullet block prompts.py renders from the
# same DIMENSIONS list, that covers the schema and the prompt.
#
# It does not reach further: reporter.py keeps its own _DIMENSIONS tuple for the
# CSV columns and markdown breakdown, and bias_auditor.py, evals.py and app.py
# likewise name the dimensions independently. Adding a dimension to the YAML
# updates the schema and the prompt, but those call sites need updating too.
#
# Field order follows DIMENSIONS, and base-class fields come first in Pydantic,
# so the dimensions lead in both models below. No docstring on purpose: it would
# become the schema `description` of any subclass that lacks its own.
DimensionScores = create_model(
    "DimensionScores",
    **{d.key: (DimensionScore, ...) for d in DIMENSIONS},
)


class ScoreReport(DimensionScores):
    """What the scorer LLM returns. The orchestrator wraps this into a ScoredCandidate."""

    reasoning: str = Field(
        description="Two-sentence overall reasoning for the scores.",
    )
    gaps: list[Gap]


class ScoredCandidate(DimensionScores):
    profile: CandidateProfile
    reasoning: str
    gaps: list[Gap]
    source_file: str


class CritiqueReport(BaseModel):
    """A second-pass review of a ScoreReport — used by the self-critique flow."""

    did_revise: bool = Field(
        description="True if the critic changed at least one score.",
    )
    critique: str = Field(
        description="One or two sentences explaining the critic's decision.",
    )
    revised_scores: ScoreReport


class ProcessingError(BaseModel):
    """Emitted when a resume fails at any stage. Keeps bad PDFs from crashing the run."""

    source_file: str
    stage: Literal["parse", "extract", "score"]
    message: str


class BiasVariant(BaseModel):
    """One mutated version of a resume, re-scored to probe for demographic drift."""

    label: str = Field(
        description="Short identifier for this variant, e.g. 'name_female_ethnic'.",
    )
    swapped_field: Literal["name", "grad_year", "location"]
    swapped_value: str = Field(
        description="The actual value substituted into the candidate profile.",
    )
    scores: ScoreReport


class BiasAuditReport(BaseModel):
    """
    Collected drift analysis across demographic variants for one candidate.

    baseline holds the original scores. variants holds re-scored copies with
    individual signals swapped. flagged is True when any dimension drifts more
    than drift_threshold points from baseline.
    """

    baseline: ScoreReport
    variants: list[BiasVariant]
    drift_threshold: int = Field(
        default=10,
        description="Max allowed point difference before a variant is flagged.",
    )
    max_score_drift: float = Field(
        description="Largest absolute score delta observed across all variants and dimensions.",
    )
    flagged: bool = Field(
        description="True if any variant exceeded drift_threshold on any dimension.",
    )
    flag_reason: str | None = Field(
        default=None,
        description="Plain-English explanation of what drifted and by how much.",
    )


class DeepDiveReport(BaseModel):
    """A hiring-manager briefing for one shortlisted candidate.

    Produced only for the top N candidates (`--deep-dive-top`), after scoring
    has finished and the field is ranked.

    Deliberately *not* a field on ScoredCandidate. A deep-dive is an optional
    analysis of a subset of candidates, not a property of one: nesting it here
    would put a `"deep_dive": null` key into every candidate object in
    results.json, and widen the CSV's schema, for a briefing most candidates
    never receive. It travels beside the results instead, in a dict keyed by
    source_file — the same sidecar pattern BiasAuditReport already uses.
    """

    summary: str = Field(
        description=(
            "One paragraph written for a hiring manager deciding whether to "
            "interview this candidate. Reference this candidate's actual "
            "experience — no generic praise."
        ),
    )
    pros: list[str] = Field(
        description=(
            "Concrete strengths for this specific role, each tied to something "
            "in the candidate's background. 2-4 items."
        ),
    )
    cons: list[str] = Field(
        description=(
            "Concrete risks, shortfalls, or things to verify before hiring. "
            "2-4 items. Never empty — if the candidate is exceptional, name "
            "what still needs checking."
        ),
    )
    interview_questions: list[str] = Field(
        description=(
            "3-5 questions targeting this candidate's specific risks, gaps, "
            "and uncertainties. A question that could be asked of any "
            "candidate for this role does not belong here."
        ),
    )


# ---------------------------------------------------------------------------
# Challenge 7: synthetic eval-case generation
#
# Everything in this section describes a *proposal* produced by an LLM, never a
# verified fact. The generator writes a resume and states what it was aiming
# for; whether it actually hit that target is exactly the question the scoring
# pipeline exists to answer, and nothing here may be treated as ground truth.
# A proposal becomes a trusted eval case only after a human has reviewed it.
# ---------------------------------------------------------------------------

# The four difficulty levels a synthetic resume can be generated at.
MatchLevel = Literal["strong", "partial", "weak", "adversarial"]

# One end of an inclusive score range, in the same 0-100 domain as
# DimensionScore.score above.
_ScoreBound = Annotated[int, Field(ge=0, le=100)]

# A scoring dimension's `key`, built from the live dimensions.yaml rather than
# retyped. A Literal (not a plain str plus a validator) so the allowed keys land
# in `model_json_schema()` as an `enum` — this schema is handed to Claude as a
# tool's input_schema, so the model reads the real dimension keys instead of
# guessing at them, and a fifth dimension added to the YAML reaches the tool
# with no edit here. Subscripting Literal with a runtime tuple is deliberate;
# it is the same "derive the schema from config" move `DimensionScores` makes
# with create_model() above.
DimensionKey = Literal[tuple(DIMENSION_KEYS)]


class GeneratedResumeProposal(BaseModel):
    """One synthetic resume proposed by the eval-set generator.

    Produced by an LLM from a job description, to be saved as an eval fixture
    and promoted to a trusted eval case only after human review.

    Every expectation on this model is *generation intent*, not measurement.
    `intended_level`, `expected_score_range`, and the two dimension lists say
    what the generator was aiming for while writing `resume_markdown` — they
    are a claim to be checked against the real scorer, not a label to be
    trusted. Treating them as ground truth would let the generator grade its
    own homework, which is the one failure mode a generated eval set must not
    have.

    Structural validity is enforced here (score bounds in range, low <= high,
    dimension keys that actually exist). Semantic correctness — does this
    resume genuinely read as `weak`? — deliberately is not, because that is a
    judgment for the scorer and the human reviewer, not for a schema.
    """

    resume_markdown: str = Field(
        description=(
            "The full synthetic resume as Markdown: name, summary, work "
            "history with dates, skills, and education. Write a realistic "
            "document a person could plausibly have submitted, not a sketch "
            "or a bulleted outline of one. Invent the candidate entirely — "
            "never reuse a real person's name, employer, or history."
        ),
    )
    intended_level: MatchLevel = Field(
        description=(
            "The match level this resume was written to represent. 'strong' "
            "meets essentially every requirement; 'partial' meets some and "
            "clearly misses others; 'weak' is a poor fit for the role; "
            "'adversarial' is engineered to look better than it is — heavy "
            "keyword overlap with the job description, but without the "
            "substance behind it. This is the target you were writing to, not "
            "a prediction of the score it will receive."
        ),
    )
    expected_score_range: tuple[_ScoreBound, _ScoreBound] = Field(
        description=(
            "Proposed inclusive [low, high] range for this candidate's "
            "overall_fit score, 0-100. Both bounds must be within 0-100 and "
            "low must not exceed high. Make the range wide enough to absorb "
            "normal scoring variation — a range narrower than about 15 points "
            "will fail intermittently for reasons that have nothing to do "
            "with the resume. This is a proposal awaiting human review, not a "
            "measured result."
        ),
    )
    dimensions_expected_high: list[DimensionKey] = Field(
        description=(
            "Scoring dimension keys this resume was written to score well on. "
            "May be empty — a weak or adversarial candidate need not be strong "
            "anywhere. List each key at most once."
        ),
    )
    dimensions_expected_low: list[DimensionKey] = Field(
        description=(
            "Scoring dimension keys this resume was written to score poorly "
            "on. May be empty. List each key at most once. For an adversarial "
            "candidate this is the important field: name the dimensions where "
            "the surface keyword match should fail to hold up."
        ),
    )
    rationale: str = Field(
        description=(
            "Two or three sentences for the human reviewer, explaining what "
            "this resume is testing and why it belongs at the intended level. "
            "Point at the specific details that make it so — the missing "
            "requirement, the shallow-but-keyword-rich role — so a reviewer "
            "can check your reasoning against the resume instead of taking "
            "the level on trust."
        ),
    )

    @field_validator("dimensions_expected_high", "dimensions_expected_low")
    @classmethod
    def _no_duplicate_dimensions(cls, value: list[str]) -> list[str]:
        """Reject a dimension key listed twice in the same field.

        The Literal type already rejects keys that aren't configured; this
        catches the other way a list can be malformed. A repeated key carries
        no extra meaning and would double-count if anything downstream ever
        tallies these, so it is a mistake worth failing on rather than
        silently deduplicating.
        """

        duplicates = sorted({key for key in value if value.count(key) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate dimension keys: {', '.join(duplicates)}; "
                f"list each key at most once."
            )
        return value

    @model_validator(mode="after")
    def _ordered_score_range(self) -> "GeneratedResumeProposal":
        """Reject an inverted range, e.g. (80, 60).

        Each bound is individually constrained to 0-100 by `_ScoreBound`, but
        that says nothing about their order, and an inverted range is a range
        no score can satisfy — it would silently turn into an eval case that
        can never pass.
        """

        low, high = self.expected_score_range
        if low > high:
            raise ValueError(
                f"expected_score_range lower bound {low} exceeds upper bound "
                f"{high}; the range must be ordered [low, high]."
            )
        return self


class GeneratedEvalCase(BaseModel):
    """A generated resume that a human has been asked to sign off on as an eval case.

    The record that sits between a `GeneratedResumeProposal` and the eval
    suite. A proposal says what the generator was *aiming* for; this says what
    a person is willing to *assert*, and only once `reviewed` is True does it
    reach `ALL_CASES`.

    The split matters because the two carry different authority:

      - `expected_ranges` is the assertion the eval suite will enforce. It
        comes from the reviewer, never from the proposal. The generator's
        proposed range is kept below under `proposed_overall_fit_range` as
        provenance and is never read when building an `EvalCase` — letting it
        flow through would be the generator grading its own homework.
      - `reviewed` is that person's decision. It defaults to False, and
        nothing sets it to True as a side effect of anything.

    `expected_ranges` must cover every configured dimension: an `EvalCase`
    needs a range per dimension and there is no defensible way to invent a
    missing one, so an incomplete set is an error rather than a gap quietly
    filled in.
    """

    name: str = Field(
        description="Stable, unique eval case identifier, e.g. 'gen_payments_partial'.",
    )
    description: str = Field(
        description="One line explaining what this case tests, for the runner's output.",
    )
    jd: str = Field(description="The job description this resume was generated against.")
    resume_text: str = Field(
        description="The generated resume, exactly as it will be handed to the extractor.",
    )
    expected_ranges: dict[DimensionKey, tuple[_ScoreBound, _ScoreBound]] = Field(
        description=(
            "Inclusive [low, high] score range per dimension key — the "
            "reviewer's assertion, not the generator's proposal. Must contain "
            "an entry for every configured dimension."
        ),
    )
    reviewed: bool = Field(
        default=False,
        description=(
            "True only when a human has read the resume and accepted the "
            "ranges above. False keeps this case out of the eval suite."
        ),
    )

    # --- Provenance. Recorded so a reviewer can see what the generator claimed
    # --- and compare it against what they accepted. Never used to build an
    # --- EvalCase, and never treated as an assertion about anything.
    source_level: MatchLevel = Field(
        description="The match level the generator was asked to write.",
    )
    proposed_score_range: tuple[_ScoreBound, _ScoreBound] = Field(
        description=(
            "Whatever `GeneratedResumeProposal.expected_score_range` held — "
            "see that field for what it covers. Provenance only: kept so a "
            "reviewer can see the generator's guess beside their own judgment. "
            "Never used as an expectation, and named after its source field "
            "rather than after a dimension so it cannot go stale if the "
            "dimension list changes."
        ),
    )

    @model_validator(mode="after")
    def _ranges_are_complete_and_ordered(self) -> "GeneratedEvalCase":
        """Every configured dimension present, every range low <= high.

        Completeness is checked against the live `DIMENSION_KEYS`, so adding a
        dimension to dimensions.yaml invalidates existing generated cases
        immediately and loudly, rather than letting them build an `EvalCase`
        that is missing a field.
        """

        missing = [key for key in DIMENSION_KEYS if key not in self.expected_ranges]
        if missing:
            raise ValueError(
                f"expected_ranges is missing a range for: {', '.join(missing)}. "
                f"A reviewer must supply one per dimension; they cannot be inferred."
            )

        for key, (low, high) in self.expected_ranges.items():
            if low > high:
                raise ValueError(
                    f"expected_ranges[{key!r}] lower bound {low} exceeds upper "
                    f"bound {high}; the range must be ordered [low, high]."
                )

        return self
