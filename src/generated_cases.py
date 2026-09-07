"""
Challenge 7: the human-review boundary between a generated proposal and the
eval suite.

`eval_gen.py` writes proposals. `eval_fixtures.py` stores them. Neither may
make one count for anything — and this module is the gate that enforces it.
Its whole job is one rule:

    A generated case runs in the eval suite if, and only if, a human set
    `reviewed` to True on it.

That rule is enforced by code, not by a convention someone has to remember.
`load_eval_case_kwargs()` is the only path from this file's data into
`ALL_CASES`, and it filters on `reviewed` before returning anything. An
unreviewed case is still loadable — a reviewer has to be able to read it — but
it cannot reach the runner.

Two things this module deliberately does not do:

1. **It never derives an expectation from a proposal.** A
   `GeneratedResumeProposal` carries the range the *generator* proposed;
   promoting one requires the reviewer to pass `expected_ranges` explicitly.
   The proposed range is copied into the case as provenance and is never read
   again. If it were used, the generator would be writing the resume, setting
   the passing grade, and then being graded against it.

2. **It never converts the qualitative dimension hints into numbers.**
   `dimensions_expected_high` / `_low` say which dimensions the generator was
   aiming to push up or down. They are not ranges and there is no honest
   arithmetic that turns them into ranges, so nothing here tries.

Data lives in a JSON file next to the hand-written cases, not in generated
Python source. Review is then a data edit a person can read in a diff, and a
malformed edit fails on load instead of at import of a Python module nobody
can safely regenerate. This mirrors `dimensions.py`: validate on load, raise a
dedicated error immediately, never half-build something that fails later.

No LLM calls, no Anthropic import, no scoring. Whether the reviewer's ranges
are *correct* is a question for the eval run, not for this module.
"""

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import ValidationError

from src.dimensions import DIMENSION_KEYS
from src.models import GeneratedEvalCase, GeneratedResumeProposal, MatchLevel

# Beside the hand-written cases, so a reviewer finds both in one directory.
# Resolved from this file like dimensions.py's DIMENSIONS_PATH, so it does not
# depend on the working directory a test or script happens to run from.
GENERATED_CASES_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "evals" / "generated_cases.json"
)

_CASES_KEY = "cases"


class GeneratedCaseError(ValueError):
    """Raised when generated case data is missing required structure or invalid.

    Deliberately loud, and deliberately not caught anywhere. This file is
    hand-edited during review, and a case that quietly vanished from the suite
    because of a typo is the worst outcome available: the suite would still
    pass, with one fewer thing being checked and nobody aware of it.
    """


def promote_proposal(
    proposal: GeneratedResumeProposal,
    jd: str,
    name: str,
    expected_ranges: dict[str, tuple[int, int]],
    *,
    reviewed: bool = False,
    description: str | None = None,
) -> GeneratedEvalCase:
    """Turn a proposal into a reviewable eval case. Does not approve anything.

    `expected_ranges` is required and comes from the reviewer — one inclusive
    `(low, high)` per configured dimension. It is the one piece of information
    that cannot be inferred: the generator's own proposed range is an opinion
    about work it just did, so using it would make the case self-certifying.
    An incomplete mapping raises rather than being filled in.

    `reviewed` is keyword-only and defaults to False, so approving a case takes
    the literal words `reviewed=True` at a call site a person wrote. Conversion
    on its own approves nothing.

    `name` is supplied by the caller rather than generated, so the same inputs
    always produce the same case and nothing depends on a counter, a clock or
    the order files happened to be written in.

    The proposal contributes exactly three things: the resume text, and — as
    provenance only — the level it was written at and the range it proposed.
    """

    try:
        return GeneratedEvalCase(
            name=name,
            description=description or _default_description(proposal),
            jd=jd,
            resume_text=proposal.resume_markdown,
            expected_ranges=expected_ranges,
            reviewed=reviewed,
            source_level=proposal.intended_level,
            proposed_score_range=proposal.expected_score_range,
        )
    except ValidationError as exc:
        raise GeneratedCaseError(
            f"Cannot promote proposal into eval case {name!r}: {exc}"
        ) from exc


def _default_description(proposal: GeneratedResumeProposal) -> str:
    """A description for the runner's output, when the reviewer gives none.

    Names the level as *generated intent* rather than as fact, because this
    string is printed beside a pass/fail and would otherwise read as a claim
    about the case that nobody verified.
    """

    return f"Generated at intended level '{proposal.intended_level}'. {proposal.rationale}"


def eval_case_kwargs(case: GeneratedEvalCase) -> dict[str, Any]:
    """The keyword arguments for constructing an `EvalCase` from a case.

    Returns kwargs rather than an `EvalCase` so this module never imports
    `tests.evals.cases`. That keeps the dependency one-directional — the case
    file imports this, not the other way round — and leaves `EvalCase` itself
    untouched by anything to do with generation or review.

    The `expected_*` field names are built from `DIMENSION_KEYS`, not written
    out. `EvalCase` happens to name its range fields `expected_{key}` for every
    configured dimension, so this mapping is exact and stays exact if the
    dimension list changes and `EvalCase` is updated to match.

    Note what does *not* appear in the output: `reviewed`, `source_level` and
    `proposed_score_range` all stop here. The runner receives a plain
    `EvalCase` and cannot tell a generated case from a hand-written one, which
    is the point — review state is an ingestion concern, not a scoring one.
    """

    return {
        "name": case.name,
        "description": case.description,
        "jd": case.jd,
        "resume_text": case.resume_text,
        **{f"expected_{key}": case.expected_ranges[key] for key in DIMENSION_KEYS},
    }


def reviewed_cases(cases: Iterable[GeneratedEvalCase]) -> list[GeneratedEvalCase]:
    """Just the approved ones, in the order given."""

    return [case for case in cases if case.reviewed]


def write_generated_cases(
    cases: Sequence[GeneratedEvalCase],
    path: str | Path = GENERATED_CASES_PATH,
) -> Path:
    """Write cases to `path` as JSON. Reviewed and unreviewed alike.

    Both are written because the file is the reviewer's working document: an
    unreviewed case has to be visible in it to be reviewed at all. The filter
    happens on the way *out*, in `load_eval_case_kwargs()`, not on the way in.

    Deterministic — order comes from the sequence, field order from the model,
    no timestamps, UTF-8 with a trailing newline — so re-writing an unchanged
    set produces an empty `git diff`.
    """

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {_CASES_KEY: [case.model_dump(mode="json") for case in cases]}
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def load_generated_cases(
    path: str | Path = GENERATED_CASES_PATH,
) -> list[GeneratedEvalCase]:
    """Load every generated case, reviewed or not, for review tooling.

    A missing file is not an error — it just means nothing has been generated
    yet, which is the state of a fresh checkout. Anything else wrong with the
    file is an error: bad JSON, the wrong top-level shape, or a case that no
    longer validates all raise `GeneratedCaseError` naming the file and, where
    it can, the offending case.
    """

    file = Path(path)
    if not file.exists():
        return []

    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GeneratedCaseError(f"{file} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or _CASES_KEY not in payload:
        raise GeneratedCaseError(
            f"{file} must be a JSON object with a {_CASES_KEY!r} key, "
            f"got {type(payload).__name__}."
        )

    entries = payload[_CASES_KEY]
    if not isinstance(entries, list):
        raise GeneratedCaseError(
            f"{file}: {_CASES_KEY!r} must be a list, got {type(entries).__name__}."
        )

    cases: list[GeneratedEvalCase] = []
    for index, entry in enumerate(entries):
        try:
            cases.append(GeneratedEvalCase.model_validate(entry))
        except ValidationError as exc:
            name = entry.get("name", "?") if isinstance(entry, dict) else "?"
            raise GeneratedCaseError(
                f"{file}: case #{index + 1} ({name!r}) is invalid: {exc}"
            ) from exc

    return cases


def load_eval_case_kwargs(
    reserved_names: Sequence[str] = (),
    path: str | Path = GENERATED_CASES_PATH,
) -> list[dict[str, Any]]:
    """The review gate. `EvalCase` kwargs for approved generated cases only.

    This is the single path from generated data into the eval suite, and the
    filter is not optional or overridable: an unreviewed case is dropped here
    and there is no argument that keeps it. Skipping this function means not
    getting the cases at all.

    `reserved_names` is the names already in use — the hand-written cases —
    checked so a generated case cannot shadow one. Duplicates among the
    generated cases themselves are caught too. Names end up in the runner's
    output and in `source_file`, so two cases sharing one would make a failure
    report ambiguous about which case actually failed.
    """

    cases = load_generated_cases(path)
    approved = reviewed_cases(cases)

    seen = {name: "hand-written" for name in reserved_names}
    for case in approved:
        if case.name in seen:
            raise GeneratedCaseError(
                f"{Path(path)}: duplicate eval case name {case.name!r} "
                f"(already used by a {seen[case.name]} case). Case names must "
                f"be unique — they identify the case in the runner's output."
            )
        seen[case.name] = "generated"

    return [eval_case_kwargs(case) for case in approved]


def review_summary(cases: Sequence[GeneratedEvalCase]) -> str:
    """One line per case for a human deciding what still needs attention."""

    if not cases:
        return "No generated cases."

    lines = [f"{len(cases)} generated case(s):"]
    for case in cases:
        mark = "reviewed" if case.reviewed else "UNREVIEWED - excluded"
        lines.append(f"  [{mark}] {case.name} (generated as {case.source_level})")
    return "\n".join(lines)


# Re-exported so a caller that only needs the level vocabulary does not have to
# reach into models.py for it.
__all__ = [
    "GENERATED_CASES_PATH",
    "GeneratedCaseError",
    "MatchLevel",
    "eval_case_kwargs",
    "load_eval_case_kwargs",
    "load_generated_cases",
    "promote_proposal",
    "review_summary",
    "reviewed_cases",
    "write_generated_cases",
]
