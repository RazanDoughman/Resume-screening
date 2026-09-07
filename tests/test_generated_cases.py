"""Tests for Challenge 7 Stage 4 — the human-review / promotion boundary.

No Anthropic client, no network, no scoring: this stage is about who is
allowed to assert what, and none of that needs a model. Filesystem work runs
under `tmp_path`; the repo's real generated_cases.json is never written.

Four groups:
  - promotion (what the generator contributes vs what a reviewer must supply)
  - the review gate (reviewed=True is the only way in)
  - persistence and loading (round-trip, determinism, loud failure)
  - eval wiring (EvalCase compatibility, hand-written cases intact)
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.dimensions import DIMENSION_KEYS
from src.generated_cases import (
    GENERATED_CASES_PATH,
    GeneratedCaseError,
    eval_case_kwargs,
    load_eval_case_kwargs,
    load_generated_cases,
    promote_proposal,
    review_summary,
    reviewed_cases,
    write_generated_cases,
)
from src.models import GeneratedEvalCase, GeneratedResumeProposal
from tests.evals.cases import (
    ALL_CASES,
    GENERATED_CASES,
    HANDWRITTEN_CASES,
    EvalCase,
)

_ROOT = Path(__file__).resolve().parent.parent

_JD = "Senior Backend Engineer — Payments. Python, PostgreSQL, 5+ years."

# What a reviewer supplies: one inclusive range per configured dimension.
# Built from DIMENSION_KEYS so the fixture follows dimensions.yaml.
_HUMAN_RANGES = {key: (40, 70) for key in DIMENSION_KEYS}


def _proposal(level: str = "partial", **overrides) -> GeneratedResumeProposal:
    payload = {
        "resume_markdown": "# Dana Vale\n\nBackend engineer, 4 years.\n",
        "intended_level": level,
        "expected_score_range": (55, 75),
        "dimensions_expected_high": ["skills_match"],
        "dimensions_expected_low": ["experience_match"],
        "rationale": "Right stack, short tenure.",
    }
    payload.update(overrides)
    return GeneratedResumeProposal(**payload)


def _case(name: str = "gen_partial_01", **overrides) -> GeneratedEvalCase:
    kwargs = {
        "proposal": _proposal(),
        "jd": _JD,
        "name": name,
        "expected_ranges": _HUMAN_RANGES,
    }
    kwargs.update(overrides)
    return promote_proposal(**kwargs)


# ---------------------------------------------------------------------------
# Promotion: what comes from the generator, what must come from a human
# ---------------------------------------------------------------------------


def test_promotion_preserves_the_resume_and_jd_exactly() -> None:
    proposal = _proposal(resume_markdown="# Ada Byrne\n\n- Built a ledger.\n")

    case = _case(proposal=proposal)

    assert case.resume_text == proposal.resume_markdown
    assert case.jd == _JD


def test_promotion_uses_the_human_ranges_not_the_proposals() -> None:
    """The load-bearing test of the whole challenge.

    The generator proposed 55-75. The reviewer said 10-30. The eval suite must
    enforce the reviewer's numbers.
    """

    human = {key: (10, 30) for key in DIMENSION_KEYS}
    proposal = _proposal(expected_score_range=(55, 75))

    case = _case(proposal=proposal, expected_ranges=human)

    assert case.expected_ranges == human
    for key in DIMENSION_KEYS:
        assert eval_case_kwargs(case)[f"expected_{key}"] == (10, 30)
    # The generator's number survives only as provenance.
    assert case.proposed_score_range == (55, 75)


def test_proposed_range_never_reaches_the_eval_case() -> None:
    proposal = _proposal(expected_score_range=(90, 100))
    case = _case(proposal=proposal, expected_ranges={k: (0, 20) for k in DIMENSION_KEYS})

    kwargs = eval_case_kwargs(case)

    assert (90, 100) not in kwargs.values()
    assert "proposed_score_range" not in kwargs
    assert "source_level" not in kwargs
    assert "reviewed" not in kwargs


def test_qualitative_dimension_hints_are_not_turned_into_ranges() -> None:
    """`dimensions_expected_high` is an intention, not a number.

    Nothing may manufacture a range from it: a case whose hints say
    skills_match is high must still get skills_match's range from the human,
    and identical hints must not produce different numbers.
    """

    proposal = _proposal(
        dimensions_expected_high=["skills_match", "overall_fit"],
        dimensions_expected_low=["experience_match"],
    )
    flat = {key: (45, 55) for key in DIMENSION_KEYS}

    case = _case(proposal=proposal, expected_ranges=flat)

    # Every dimension kept the reviewer's identical range — no hint nudged one.
    assert set(case.expected_ranges.values()) == {(45, 55)}


def test_promotion_does_not_mark_a_case_reviewed() -> None:
    """Conversion is not approval."""

    assert _case().reviewed is False


def test_review_requires_the_explicit_keyword() -> None:
    assert _case(reviewed=True).reviewed is True

    with pytest.raises(TypeError):
        promote_proposal(_proposal(), _JD, "gen_x", _HUMAN_RANGES, True)  # type: ignore[misc]


def test_case_name_is_caller_supplied_and_deterministic() -> None:
    """Same inputs, same case — no counter, clock or random id anywhere."""

    first = _case(name="gen_payments_partial")
    second = _case(name="gen_payments_partial")

    assert first.name == "gen_payments_partial"
    assert first == second


def test_source_level_is_recorded_as_provenance() -> None:
    assert _case(proposal=_proposal("adversarial")).source_level == "adversarial"


def test_description_defaults_to_generated_intent_not_a_claim() -> None:
    case = _case(proposal=_proposal("weak"))

    assert "intended level" in case.description
    assert "weak" in case.description
    assert "Right stack, short tenure." in case.description


def test_description_can_be_overridden_by_the_reviewer() -> None:
    assert _case(description="Reviewer's own words.").description == "Reviewer's own words."


# ---------------------------------------------------------------------------
# Promotion: incomplete or invalid review input fails loudly
# ---------------------------------------------------------------------------


def test_missing_a_dimension_range_fails_rather_than_being_guessed() -> None:
    partial = {key: (40, 70) for key in DIMENSION_KEYS[:-1]}

    with pytest.raises(GeneratedCaseError, match="missing a range"):
        _case(expected_ranges=partial)


def test_empty_ranges_fail() -> None:
    with pytest.raises(GeneratedCaseError, match="missing a range"):
        _case(expected_ranges={})


@pytest.mark.parametrize("bad", [(80, 60), (100, 0)])
def test_inverted_reviewer_range_is_rejected(bad: tuple[int, int]) -> None:
    ranges = {**_HUMAN_RANGES, DIMENSION_KEYS[0]: bad}

    with pytest.raises(GeneratedCaseError, match="exceeds upper bound"):
        _case(expected_ranges=ranges)


@pytest.mark.parametrize("bad", [(-1, 50), (0, 101)])
def test_out_of_domain_reviewer_range_is_rejected(bad: tuple[int, int]) -> None:
    ranges = {**_HUMAN_RANGES, DIMENSION_KEYS[0]: bad}

    with pytest.raises(GeneratedCaseError):
        _case(expected_ranges=ranges)


def test_unknown_dimension_key_in_ranges_is_rejected() -> None:
    """Still config-driven: the key set comes from dimensions.yaml."""

    with pytest.raises(GeneratedCaseError):
        _case(expected_ranges={**_HUMAN_RANGES, "culture_fit": (10, 20)})


# ---------------------------------------------------------------------------
# The review gate
# ---------------------------------------------------------------------------


def test_unreviewed_case_is_excluded_from_the_runnable_set(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_unreviewed", reviewed=False)], path)

    assert load_eval_case_kwargs(path=path) == []


def test_reviewed_case_becomes_eligible(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_reviewed", reviewed=True)], path)

    kwargs = load_eval_case_kwargs(path=path)

    assert [k["name"] for k in kwargs] == ["gen_reviewed"]


def test_the_gate_filters_a_mixed_file(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases(
        [
            _case("gen_a", reviewed=True),
            _case("gen_b", reviewed=False),
            _case("gen_c", reviewed=True),
        ],
        path,
    )

    assert [k["name"] for k in load_eval_case_kwargs(path=path)] == ["gen_a", "gen_c"]


def test_review_state_is_data_not_file_existence(tmp_path: Path) -> None:
    """Writing the file must not approve anything — presence is not consent."""

    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_present", reviewed=False)], path)

    assert path.exists()
    assert load_generated_cases(path)  # readable for review
    assert load_eval_case_kwargs(path=path) == []  # but not runnable


def test_unreviewed_cases_remain_loadable_for_review(tmp_path: Path) -> None:
    """A reviewer has to be able to see what they are being asked to approve."""

    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_pending", reviewed=False)], path)

    loaded = load_generated_cases(path)

    assert [c.name for c in loaded] == ["gen_pending"]
    assert loaded[0].reviewed is False
    assert reviewed_cases(loaded) == []


def test_gate_has_no_override_argument() -> None:
    """There must be no way to ask for unreviewed cases from the gate."""

    import inspect

    params = set(inspect.signature(load_eval_case_kwargs).parameters)

    assert params == {"reserved_names", "path"}


def test_review_summary_marks_excluded_cases() -> None:
    summary = review_summary([_case("gen_a", reviewed=True), _case("gen_b")])

    assert "reviewed] gen_a" in summary
    assert "UNREVIEWED - excluded] gen_b" in summary
    assert review_summary([]) == "No generated cases."


# ---------------------------------------------------------------------------
# Persistence and loading
# ---------------------------------------------------------------------------


def test_round_trip_preserves_everything_including_review_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generated_cases.json"
    original = [
        _case("gen_one", reviewed=True, expected_ranges={k: (10, 90) for k in DIMENSION_KEYS}),
        _case("gen_two", proposal=_proposal("adversarial"), reviewed=False),
    ]

    write_generated_cases(original, path)
    restored = load_generated_cases(path)

    assert restored == original
    assert [c.reviewed for c in restored] == [True, False]
    assert restored[1].source_level == "adversarial"


def test_write_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    cases = [_case("gen_a", reviewed=True), _case("gen_b")]

    write_generated_cases(cases, a)
    write_generated_cases(cases, b)

    assert a.read_bytes() == b.read_bytes()


def test_utf8_survives_the_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    case = _case(
        "gen_utf8",
        proposal=_proposal(resume_markdown="# Zoë Müller\n\n日本語 — Kraków\n"),
    )

    write_generated_cases([case], path)

    assert load_generated_cases(path)[0] == case
    assert "Zoë Müller" in path.read_text(encoding="utf-8")
    assert "日本語" in path.read_text(encoding="utf-8")


def test_missing_file_is_empty_not_an_error(tmp_path: Path) -> None:
    """A fresh checkout has no generated cases; that is normal, not broken."""

    assert load_generated_cases(tmp_path / "nope.json") == []
    assert load_eval_case_kwargs(path=tmp_path / "nope.json") == []


def test_malformed_json_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(GeneratedCaseError, match="not valid JSON"):
        load_generated_cases(path)


@pytest.mark.parametrize("payload", ['["a"]', '{"wrong": []}', '"text"', "42"])
def test_wrong_top_level_shape_fails_loudly(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "generated_cases.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(GeneratedCaseError, match="must be a JSON object"):
        load_generated_cases(path)


def test_cases_key_must_be_a_list(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    path.write_text('{"cases": {"a": 1}}', encoding="utf-8")

    with pytest.raises(GeneratedCaseError, match="must be a list"):
        load_generated_cases(path)


def test_an_invalid_case_fails_loudly_and_names_itself(tmp_path: Path) -> None:
    """A bad hand-edit must not make a case silently vanish from the suite."""

    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_ok", reviewed=True)], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["expected_ranges"][DIMENSION_KEYS[0]] = [90, 10]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GeneratedCaseError, match="gen_ok"):
        load_generated_cases(path)


def test_a_case_missing_a_dimension_fails_on_load(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_ok", reviewed=True)], path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["cases"][0]["expected_ranges"][DIMENSION_KEYS[0]]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GeneratedCaseError, match="missing a range"):
        load_generated_cases(path)


def test_write_creates_a_missing_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "generated_cases.json"

    write_generated_cases([_case()], path)

    assert path.exists()


# ---------------------------------------------------------------------------
# Eval wiring
# ---------------------------------------------------------------------------


def test_reviewed_case_constructs_a_valid_eval_case(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_wired", reviewed=True)], path)

    cases = [EvalCase(**kwargs) for kwargs in load_eval_case_kwargs(path=path)]

    assert len(cases) == 1
    assert isinstance(cases[0], EvalCase)
    assert cases[0].name == "gen_wired"
    assert cases[0].expected_overall_fit == _HUMAN_RANGES["overall_fit"]


def test_eval_case_kwargs_match_the_eval_case_signature() -> None:
    """Config-driven mapping: expected_{key} for every configured dimension."""

    from dataclasses import fields

    kwargs = eval_case_kwargs(_case())

    assert set(kwargs) == {f.name for f in fields(EvalCase)}
    for key in DIMENSION_KEYS:
        assert f"expected_{key}" in kwargs


def test_no_dimension_key_is_hardcoded_in_the_review_module() -> None:
    source = (_ROOT / "src" / "generated_cases.py").read_text(encoding="utf-8")

    for key in DIMENSION_KEYS:
        assert key not in source


def test_handwritten_cases_are_unchanged() -> None:
    """Stage 4 must not disturb the cases a person wrote."""

    assert [c.name for c in HANDWRITTEN_CASES] == [
        "strong_match_payments",
        "weak_match_frontend_to_payments",
        "mixed_match_strong_skills_short_experience",
        "career_changer_sre_to_payments",
    ]
    assert all(isinstance(c, EvalCase) for c in HANDWRITTEN_CASES)
    assert ALL_CASES[: len(HANDWRITTEN_CASES)] == HANDWRITTEN_CASES


def test_all_cases_is_handwritten_plus_reviewed_generated() -> None:
    assert ALL_CASES == [*HANDWRITTEN_CASES, *GENERATED_CASES]


def test_every_active_generated_case_in_the_repo_was_reviewed() -> None:
    """Nothing reaches the live suite without `reviewed: true` on disk.

    This test used to assert `GENERATED_CASES == []`, which was true only
    because no proposal had been reviewed yet. That was an assertion about the
    repo's contents on a particular day, not about the review gate, and it
    stopped being true the moment a human approved the first four cases.

    What is actually load-bearing is the correspondence: every generated case
    the suite will run traces back to an entry in the data file that is marked
    reviewed, and nothing marked unreviewed appears among them. That holds with
    zero reviewed cases, with four, and with any mix — including a file that
    contains unreviewed entries alongside reviewed ones.
    """

    if not GENERATED_CASES_PATH.exists():
        assert GENERATED_CASES == []
        return

    on_disk = load_generated_cases(GENERATED_CASES_PATH)
    approved = {case.name for case in on_disk if case.reviewed}
    rejected = {case.name for case in on_disk if not case.reviewed}
    active = {case.name for case in GENERATED_CASES}

    assert active == approved, "active generated cases must be exactly the reviewed ones"
    assert not (active & rejected), "an unreviewed case reached the eval suite"


def test_duplicate_generated_name_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases(
        [_case("gen_dup", reviewed=True), _case("gen_dup", reviewed=True)], path
    )

    with pytest.raises(GeneratedCaseError, match="duplicate eval case name"):
        load_eval_case_kwargs(path=path)


def test_generated_case_cannot_shadow_a_handwritten_name(tmp_path: Path) -> None:
    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("strong_match_payments", reviewed=True)], path)

    with pytest.raises(GeneratedCaseError, match="hand-written"):
        load_eval_case_kwargs(
            reserved_names=[c.name for c in HANDWRITTEN_CASES], path=path
        )


def test_an_unreviewed_duplicate_does_not_trip_the_name_check(tmp_path: Path) -> None:
    """Only cases that would actually run need unique names."""

    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("strong_match_payments", reviewed=False)], path)

    assert (
        load_eval_case_kwargs(
            reserved_names=[c.name for c in HANDWRITTEN_CASES], path=path
        )
        == []
    )


def test_the_runner_cannot_tell_a_generated_case_apart(tmp_path: Path) -> None:
    """Review state stops at ingestion; the scorer sees a plain EvalCase.

    Where a case came from must not change how it is scored.
    """

    from dataclasses import fields

    path = tmp_path / "generated_cases.json"
    write_generated_cases([_case("gen_opaque", reviewed=True)], path)
    case = EvalCase(**load_eval_case_kwargs(path=path)[0])

    names = {f.name for f in fields(case)}
    assert not names & {"reviewed", "source_level", "proposed_score_range"}


def test_review_module_makes_no_api_calls_and_imports_no_eval_cases() -> None:
    """It is an ingestion boundary, not a pipeline stage — and not circular."""

    import ast

    tree = ast.parse((_ROOT / "src" / "generated_cases.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(a.name for a in node.names)

    assert not any("anthropic" in n.lower() for n in imported)
    # cases.py imports this module; importing back would be a cycle.
    assert not any(n.startswith("tests") for n in imported)
    assert "EvalCase" not in imported


def test_generated_eval_case_model_rejects_a_bad_direct_construction() -> None:
    """The model is the contract even when promote_proposal is bypassed."""

    with pytest.raises(ValidationError):
        GeneratedEvalCase(
            name="x",
            description="d",
            jd=_JD,
            resume_text="r",
            expected_ranges={},
            source_level="strong",
            proposed_score_range=(0, 10),
        )
