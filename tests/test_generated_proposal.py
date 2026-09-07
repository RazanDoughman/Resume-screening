"""Tests for GeneratedResumeProposal — the Challenge 7 generation contract.

Pure schema tests: nothing here constructs an `Anthropic()`, imports a
generator, or reaches the network. The model is the only thing under test, and
it is deterministic, so this file runs in milliseconds with no API key.

Two properties are load-bearing and get their own tests rather than riding
along on a happy path:

  - the dimension enum is *derived* from dimensions.yaml, not retyped here, so
    a fifth dimension reaches the tool schema with no edit to models.py;
  - the schema this model produces is the one Claude will be handed as a tool's
    input_schema in Stage 2, so the fields, the enums and the required list are
    asserted directly.
"""

import json

import pytest
from pydantic import ValidationError

from src.dimensions import DIMENSION_KEYS
from src.models import GeneratedResumeProposal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_RESUME = """# Dana Whitfield

Senior Backend Engineer with 7 years building transaction systems.

## Experience
**Staff Engineer — Meridian Pay (2020–Present)**
Owned the settlement ledger in Go. Designed idempotent capture and refund
APIs. Primary on-call.

## Skills
Go, Python, PostgreSQL, Kafka, gRPC.

## Education
BS Computer Science, 2017.
"""


def _proposal(**overrides) -> GeneratedResumeProposal:
    """A valid proposal, with any field overridden for the case at hand."""

    payload = {
        "resume_markdown": _RESUME,
        "intended_level": "strong",
        "expected_score_range": (80, 100),
        "dimensions_expected_high": ["skills_match", "overall_fit"],
        "dimensions_expected_low": [],
        "rationale": (
            "Meets every required bullet with payments-native experience, so "
            "it should anchor the top of the field."
        ),
    }
    payload.update(overrides)
    return GeneratedResumeProposal(**payload)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_valid_proposal_parses() -> None:
    proposal = _proposal()

    assert proposal.intended_level == "strong"
    assert proposal.expected_score_range == (80, 100)
    assert proposal.dimensions_expected_high == ["skills_match", "overall_fit"]
    assert proposal.dimensions_expected_low == []
    assert "Meridian Pay" in proposal.resume_markdown


def test_parses_from_a_plain_dict_the_way_a_tool_result_arrives() -> None:
    """model_validate on a raw dict is how Stage 2 will build this object."""

    proposal = GeneratedResumeProposal.model_validate(
        {
            "resume_markdown": _RESUME,
            "intended_level": "weak",
            # A JSON array, not a Python tuple — this is what the API returns.
            "expected_score_range": [10, 35],
            "dimensions_expected_high": [],
            "dimensions_expected_low": ["role_relevance"],
            "rationale": "Wrong domain entirely.",
        }
    )

    assert proposal.expected_score_range == (10, 35)


def test_empty_dimension_lists_are_allowed() -> None:
    """A weak candidate need not be strong on anything."""

    proposal = _proposal(
        intended_level="weak",
        expected_score_range=(0, 30),
        dimensions_expected_high=[],
        dimensions_expected_low=[],
    )

    assert proposal.dimensions_expected_high == []
    assert proposal.dimensions_expected_low == []


# ---------------------------------------------------------------------------
# intended_level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level", ["strong", "partial", "weak", "adversarial"]
)
def test_all_four_levels_are_accepted(level: str) -> None:
    assert _proposal(intended_level=level).intended_level == level


@pytest.mark.parametrize(
    "level",
    [
        "mixed",  # a level from some other vocabulary
        "Strong",  # right word, wrong case
        "",
        "strong ",  # trailing space
        None,
        3,
    ],
)
def test_invalid_level_is_rejected(level: object) -> None:
    with pytest.raises(ValidationError):
        _proposal(intended_level=level)


def test_the_four_levels_are_exactly_the_challenge_7_levels() -> None:
    """No fifth level sneaks in, and none of the four goes missing."""

    enum = GeneratedResumeProposal.model_json_schema()["properties"][
        "intended_level"
    ]["enum"]

    assert enum == ["strong", "partial", "weak", "adversarial"]


# ---------------------------------------------------------------------------
# expected_score_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bounds", [(0, 0), (0, 100), (100, 100), (55, 55)])
def test_valid_score_ranges_are_accepted(bounds: tuple[int, int]) -> None:
    """Boundary values and a zero-width range are all legitimate."""

    assert _proposal(expected_score_range=bounds).expected_score_range == bounds


@pytest.mark.parametrize(
    "bounds",
    [
        (-1, 50),  # below the domain
        (50, 101),  # above the domain
        (-5, -1),
        (200, 300),
    ],
)
def test_out_of_domain_score_bounds_are_rejected(bounds: tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        _proposal(expected_score_range=bounds)


@pytest.mark.parametrize("bounds", [(80, 60), (100, 0), (51, 50)])
def test_inverted_score_range_is_rejected(bounds: tuple[int, int]) -> None:
    """A range no score can satisfy would become an eval case that can never pass."""

    with pytest.raises(ValidationError, match="exceeds upper bound"):
        _proposal(expected_score_range=bounds)


@pytest.mark.parametrize(
    "bounds", [(50,), (10, 20, 30), [], "50-80", 70, None]
)
def test_malformed_score_range_shapes_are_rejected(bounds: object) -> None:
    with pytest.raises(ValidationError):
        _proposal(expected_score_range=bounds)


def test_non_integer_score_bounds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _proposal(expected_score_range=("low", "high"))


# ---------------------------------------------------------------------------
# Dimension keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", DIMENSION_KEYS)
def test_every_configured_dimension_key_is_accepted(key: str) -> None:
    """Whatever dimensions.yaml currently configures must be usable here."""

    proposal = _proposal(
        dimensions_expected_high=[key], dimensions_expected_low=[]
    )
    assert proposal.dimensions_expected_high == [key]


@pytest.mark.parametrize(
    "key",
    [
        "culture_fit",  # a plausible dimension that isn't configured
        "skills",  # a near-miss on a real key
        "Skills_Match",  # right key, wrong case
        "",
    ],
)
@pytest.mark.parametrize(
    "field", ["dimensions_expected_high", "dimensions_expected_low"]
)
def test_unknown_dimension_key_is_rejected(field: str, key: str) -> None:
    with pytest.raises(ValidationError):
        _proposal(**{field: [key]})


@pytest.mark.parametrize(
    "field", ["dimensions_expected_high", "dimensions_expected_low"]
)
def test_duplicate_dimension_key_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError, match="duplicate dimension keys"):
        _proposal(**{field: ["skills_match", "skills_match"]})


def test_dimension_enum_is_derived_from_the_yaml_not_retyped() -> None:
    """The whole point of the Literal: one source of truth for the key list.

    If someone hardcodes a dimension list into models.py, this fails the moment
    dimensions.yaml and that list disagree — which is the failure that would
    otherwise surface as a confusing validation error at generation time.
    """

    schema = GeneratedResumeProposal.model_json_schema()

    for field in ("dimensions_expected_high", "dimensions_expected_low"):
        assert schema["properties"][field]["items"]["enum"] == DIMENSION_KEYS


# ---------------------------------------------------------------------------
# The tool schema
# ---------------------------------------------------------------------------


def test_schema_exposes_every_field_as_required() -> None:
    """This dict becomes a tool's input_schema, so its shape is the contract."""

    schema = GeneratedResumeProposal.model_json_schema()
    expected = [
        "resume_markdown",
        "intended_level",
        "expected_score_range",
        "dimensions_expected_high",
        "dimensions_expected_low",
        "rationale",
    ]

    assert list(schema["properties"]) == expected
    assert sorted(schema["required"]) == sorted(expected)
    assert schema["type"] == "object"


def test_every_field_carries_a_description_for_claude() -> None:
    """A field with no description is a field Claude has to guess at."""

    properties = GeneratedResumeProposal.model_json_schema()["properties"]

    for name, spec in properties.items():
        assert spec.get("description", "").strip(), f"{name} has no description"


def test_schema_is_json_serializable() -> None:
    """It has to survive being embedded in an API request body."""

    schema = GeneratedResumeProposal.model_json_schema()

    assert json.loads(json.dumps(schema)) == schema


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


def test_round_trips_through_json_without_losing_information() -> None:
    """Stage 2 writes proposals to disk and reads them back; nothing may drift."""

    original = _proposal(
        intended_level="adversarial",
        expected_score_range=(15, 40),
        dimensions_expected_high=["skills_match"],
        dimensions_expected_low=["role_relevance", "experience_match"],
    )

    restored = GeneratedResumeProposal.model_validate_json(
        original.model_dump_json()
    )

    assert restored == original
    assert restored.model_dump() == original.model_dump()


def test_json_dump_renders_the_range_as_a_two_element_array() -> None:
    """The on-disk shape a reviewer reads, and the shape the API returns."""

    dumped = json.loads(_proposal(expected_score_range=(60, 79)).model_dump_json())

    assert dumped["expected_score_range"] == [60, 79]


def test_round_trips_through_a_plain_dict() -> None:
    original = _proposal(intended_level="partial", expected_score_range=(40, 70))

    assert GeneratedResumeProposal.model_validate(original.model_dump()) == original
