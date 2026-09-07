"""Tests for the Challenge 7 generation stage — mocks the Anthropic client.

Nothing here constructs an `Anthropic()`, so no test can reach the network, and
none of these tests costs anything to run. Same MagicMock shape as
tests/test_deep_dive.py and tests/test_extractor.py.

Four groups:
  - the request (prompt inputs, tool wiring, forced tool_choice, block order)
  - the response (validation, the tuple/JSON-array path)
  - failure modes (no tool_use, wrong tool, bad payload, unknown level)
  - configuration-driven behaviour (dimensions and levels come from config)
"""

import json
from pathlib import Path
from typing import get_args
from unittest.mock import MagicMock

import pytest

from src.dimensions import DIMENSION_KEYS, DIMENSIONS
from src.eval_gen import (
    DEFAULT_MODEL,
    GENERATION_TOOL_NAME,
    EvalGenerationError,
    generate_resume_proposal,
)
from src.models import GeneratedResumeProposal, MatchLevel
from src.prompts import SYNTH_RESUME_LEVELS, SYNTH_RESUME_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_JD = """Senior Backend Engineer — Payments Platform

Required:
- 5+ years of professional backend engineering experience.
- Strong Python or Go in production.
- Solid PostgreSQL: schema design, indexes, transaction isolation.
"""

_VALID_TOOL_INPUT = {
    "resume_markdown": (
        "# Dana Whitfield\n\nSenior Backend Engineer, 7 years.\n\n"
        "## Experience\n**Staff Engineer — Meridian Pay (2020-Present)**\n"
        "Owned the settlement ledger in Go against a PostgreSQL store.\n\n"
        "## Skills\nGo, Python, PostgreSQL, Kafka.\n"
    ),
    "intended_level": "strong",
    "expected_score_range": [80, 100],
    "dimensions_expected_high": ["skills_match", "overall_fit"],
    "dimensions_expected_low": [],
    "rationale": "Meets every required bullet with payments-native evidence.",
}


def _tool_use_response(
    tool_input: dict | None = None,
    name: str = GENERATION_TOOL_NAME,
    blocks: list | None = None,
) -> MagicMock:
    """A response shaped like the SDK's: `.content` is a list of typed blocks."""

    if blocks is None:
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.input = _VALID_TOOL_INPUT if tool_input is None else tool_input
        blocks = [block]

    response = MagicMock()
    response.content = blocks
    return response


def _text_block(text: str = "I can't help with that.") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _client(**kwargs) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(**kwargs)
    return client


def _kwargs(client: MagicMock) -> dict:
    """The keyword arguments the generator passed to messages.create()."""

    return client.messages.create.call_args.kwargs


def _blocks(client: MagicMock) -> list[dict]:
    return _kwargs(client)["messages"][0]["content"]


def _request_text(client: MagicMock) -> str:
    """Everything the model was sent: system prompt plus every user block."""

    return _kwargs(client)["system"] + "\n".join(b["text"] for b in _blocks(client))


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_returns_a_validated_proposal() -> None:
    client = _client()

    proposal = generate_resume_proposal(client, _JD, "strong")

    assert isinstance(proposal, GeneratedResumeProposal)
    assert proposal.intended_level == "strong"
    assert proposal.expected_score_range == (80, 100)
    assert proposal.dimensions_expected_high == ["skills_match", "overall_fit"]
    assert "Meridian Pay" in proposal.resume_markdown


def test_makes_exactly_one_llm_call() -> None:
    client = _client()

    generate_resume_proposal(client, _JD, "strong")

    assert client.messages.create.call_count == 1


def test_uses_the_default_model_and_honours_an_override() -> None:
    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    assert _kwargs(client)["model"] == DEFAULT_MODEL

    client = _client()
    generate_resume_proposal(client, _JD, "strong", model="claude-haiku-4-5")
    assert _kwargs(client)["model"] == "claude-haiku-4-5"


def test_constructs_no_client_of_its_own() -> None:
    """It must meter through whatever client it is handed, like deep_dive.py."""

    source = Path("src/eval_gen.py").read_text(encoding="utf-8")

    assert "Anthropic()" not in source


def test_writes_no_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persistence is a later stage; the core generator is pure request/response.

    Run from an empty cwd with `open` booby-trapped, so any write attempt —
    relative path or absolute — fails the test instead of leaving a file
    somewhere for a later run to trip over.
    """

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: pytest.fail(f"generator opened a file: {a!r}"),
    )

    generate_resume_proposal(_client(), _JD, "strong")

    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# What reaches the model
# ---------------------------------------------------------------------------


def test_job_description_reaches_the_request() -> None:
    client = _client()

    generate_resume_proposal(client, _JD, "strong")

    assert _JD in _request_text(client)


def test_job_description_is_the_first_block_and_is_cached() -> None:
    """Same cached-prefix shape as scorer.py and deep_dive.py."""

    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    blocks = _blocks(client)

    assert "<job_description>" in blocks[0]["text"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    # The level varies per call, so it must sit after the breakpoint or it
    # would give each level its own prefix.
    assert "cache_control" not in blocks[1]


@pytest.mark.parametrize("level", sorted(SYNTH_RESUME_LEVELS))
def test_requested_level_and_its_definition_reach_the_request(level: str) -> None:
    client = _client(tool_input={**_VALID_TOOL_INPUT, "intended_level": level})

    generate_resume_proposal(client, _JD, level)

    sent = _request_text(client)
    assert f"level: {level}" in sent
    assert SYNTH_RESUME_LEVELS[level] in sent


def test_the_requested_level_is_named_after_the_cache_breakpoint() -> None:
    client = _client()

    generate_resume_proposal(client, _JD, "adversarial")

    assert "adversarial" in _blocks(client)[1]["text"]


def test_current_dimension_information_reaches_the_request() -> None:
    """Key, label and description for every configured dimension."""

    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    sent = _request_text(client)

    for dimension in DIMENSIONS:
        assert dimension.key in sent
        assert dimension.label in sent
        assert dimension.prompt_description in sent


def test_the_proposals_are_not_ground_truth_framing_reaches_the_model() -> None:
    """The load-bearing instruction of the whole challenge."""

    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    sent = _request_text(client).lower()

    assert "not ground truth" in sent
    assert "human" in sent and "review" in sent


def test_the_no_label_leakage_instruction_reaches_the_model() -> None:
    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    sent = _request_text(client).lower()

    assert "strong match" in sent  # named as a forbidden phrase
    assert "rationale" in sent


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_tool_schema_is_the_pydantic_schema_not_a_hand_written_copy() -> None:
    client = _client()

    generate_resume_proposal(client, _JD, "strong")
    tools = _kwargs(client)["tools"]

    assert len(tools) == 1
    assert tools[0]["name"] == GENERATION_TOOL_NAME
    assert tools[0]["input_schema"] == GeneratedResumeProposal.model_json_schema()


def test_tool_choice_forces_the_generation_tool() -> None:
    client = _client()

    generate_resume_proposal(client, _JD, "strong")

    assert _kwargs(client)["tool_choice"] == {
        "type": "tool",
        "name": GENERATION_TOOL_NAME,
    }


def test_tool_schema_is_json_serializable() -> None:
    """It has to survive being placed in an API request body."""

    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    schema = _kwargs(client)["tools"][0]["input_schema"]

    assert json.loads(json.dumps(schema)) == schema


def test_tool_schema_carries_the_score_range_bounds() -> None:
    """Pins the prefixItems shape Stage 1 flagged, so a silent change is caught.

    Pydantic renders `tuple[int, int]` as a JSON Schema 2020-12 `prefixItems`
    array. The Anthropic SDK documents `input_schema` as 2020-12 JSON Schema
    and passes the dict through untouched, so this is the shape that goes on
    the wire. Asserted here so that if anyone later rewrites the Stage 1 field,
    it is a deliberate change with a failing test attached rather than a quiet
    one.
    """

    client = _client()
    generate_resume_proposal(client, _JD, "strong")
    field = _kwargs(client)["tools"][0]["input_schema"]["properties"][
        "expected_score_range"
    ]

    assert field["type"] == "array"
    assert field["minItems"] == field["maxItems"] == 2
    assert [b["minimum"] for b in field["prefixItems"]] == [0, 0]
    assert [b["maximum"] for b in field["prefixItems"]] == [100, 100]


# ---------------------------------------------------------------------------
# Reading the response
# ---------------------------------------------------------------------------


def test_json_array_score_range_survives_the_api_to_pydantic_path() -> None:
    """The API returns a JSON array; the model exposes a tuple.

    Nothing converts this by hand — the point is that the tool payload as it
    actually arrives validates straight through.
    """

    client = _client(
        tool_input={**_VALID_TOOL_INPUT, "expected_score_range": [42, 58]}
    )

    proposal = generate_resume_proposal(client, _JD, "partial")

    assert proposal.expected_score_range == (42, 58)
    assert isinstance(proposal.expected_score_range, tuple)


def test_reads_the_tool_use_block_past_leading_text_blocks() -> None:
    """A thinking or preamble block before the tool call must not break it."""

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = GENERATION_TOOL_NAME
    tool_block.input = _VALID_TOOL_INPUT

    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(
        blocks=[_text_block("Here is the resume."), tool_block]
    )

    assert generate_resume_proposal(client, _JD, "strong").intended_level == "strong"


@pytest.mark.parametrize("level", sorted(SYNTH_RESUME_LEVELS))
def test_every_level_round_trips_end_to_end(level: str) -> None:
    client = _client(tool_input={**_VALID_TOOL_INPUT, "intended_level": level})

    assert generate_resume_proposal(client, _JD, level).intended_level == level


# ---------------------------------------------------------------------------
# Failure modes — all loud, none silent
# ---------------------------------------------------------------------------


def test_missing_tool_use_block_fails_loudly() -> None:
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(blocks=[_text_block()])

    with pytest.raises(EvalGenerationError, match="no tool_use block"):
        generate_resume_proposal(client, _JD, "strong")


def test_empty_content_fails_loudly() -> None:
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(blocks=[])

    with pytest.raises(EvalGenerationError, match="no tool_use block"):
        generate_resume_proposal(client, _JD, "strong")


def test_missing_tool_use_error_does_not_raise_bare_stop_iteration() -> None:
    """The older stage modules raise a message-less StopIteration here."""

    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(blocks=[_text_block()])

    with pytest.raises(EvalGenerationError) as excinfo:
        generate_resume_proposal(client, _JD, "weak")

    assert not isinstance(excinfo.value, StopIteration)
    assert "weak" in str(excinfo.value)  # names the level that failed


def test_wrong_tool_name_fails_loudly() -> None:
    client = _client(name="record_score")

    with pytest.raises(EvalGenerationError, match="record_score"):
        generate_resume_proposal(client, _JD, "strong")


@pytest.mark.parametrize(
    "bad_input",
    [
        {**_VALID_TOOL_INPUT, "intended_level": "mixed"},  # unknown level
        {**_VALID_TOOL_INPUT, "expected_score_range": [90, 20]},  # inverted
        {**_VALID_TOOL_INPUT, "expected_score_range": [0, 500]},  # out of domain
        {**_VALID_TOOL_INPUT, "dimensions_expected_low": ["culture_fit"]},  # unknown key
        {k: v for k, v in _VALID_TOOL_INPUT.items() if k != "rationale"},  # missing
        {},
    ],
)
def test_invalid_payload_fails_validation_loudly(bad_input: dict) -> None:
    client = _client(tool_input=bad_input)

    with pytest.raises(EvalGenerationError, match="did not validate"):
        generate_resume_proposal(client, _JD, "strong")


def test_validation_failure_chains_the_underlying_pydantic_error() -> None:
    """Wrapping must not hide what Pydantic actually objected to."""

    from pydantic import ValidationError

    client = _client(tool_input={**_VALID_TOOL_INPUT, "intended_level": "mixed"})

    with pytest.raises(EvalGenerationError) as excinfo:
        generate_resume_proposal(client, _JD, "strong")

    assert isinstance(excinfo.value.__cause__, ValidationError)


def test_unknown_requested_level_fails_before_spending_a_call() -> None:
    """A typo must not cost money."""

    client = _client()

    with pytest.raises(EvalGenerationError, match="Unknown match level"):
        generate_resume_proposal(client, _JD, "excellent")

    client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Configuration-driven, not hardcoded
# ---------------------------------------------------------------------------


def test_no_dimension_key_is_hardcoded_in_the_generator_or_the_prompt() -> None:
    """The property deep_dive.py established and CLAUDE.md points at.

    Neither file may name a dimension. The generator gets them by rendering
    DIMENSIONS, so a fifth dimension reaches the prompt with no edit to either.
    """

    generator = Path("src/eval_gen.py").read_text(encoding="utf-8")

    # Only the Challenge 7 section of prompts.py: the scoring prompt above it
    # carries a pre-existing comment showing an example rendered bullet, which
    # names a real key and is not this stage's to change.
    _, _, generation_prompts = Path("src/prompts.py").read_text(
        encoding="utf-8"
    ).partition("# Challenge 7: synthetic eval-case generation")

    assert generation_prompts, "the Challenge 7 section banner moved or was renamed"

    for source, label in ((generator, "src/eval_gen.py"), (generation_prompts, "src/prompts.py")):
        for key in DIMENSION_KEYS:
            assert key not in source, f"{label} hardcodes the dimension key {key!r}"


def test_level_definitions_cover_exactly_the_match_level_literal() -> None:
    """SYNTH_RESUME_LEVELS is keyed by string; MatchLevel is the type.

    They are declared in two modules on purpose — prompts.py imports only
    src.dimensions — so this is the tripwire that keeps them in step.
    """

    assert set(SYNTH_RESUME_LEVELS) == set(get_args(MatchLevel))


def test_prompt_markdown_matches_the_runtime_prompt() -> None:
    """prompts/synth_resume.md documents the prompt; src/prompts.py runs it.

    Same rule tests/test_dimension_wiring.py applies to SCORING_SYSTEM_PROMPT:
    an intentional edit updates both in the same commit, an accidental one
    fails here. Note this also means a dimensions.yaml change updates the
    markdown, because the prompt is rendered from DIMENSIONS.
    """

    documented = Path("prompts/synth_resume.md").read_text(encoding="utf-8")

    assert SYNTH_RESUME_SYSTEM_PROMPT in documented
