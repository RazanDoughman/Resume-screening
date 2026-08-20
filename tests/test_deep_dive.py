"""Tests for the deep-dive stage — mocks the Anthropic client, no real API calls.

Three groups:
  - the module itself (tool wiring, block order, dynamic dimensions)
  - selection (rank_candidates + strict top-N)
  - orchestration (main.run with everything mocked)

Nothing here constructs an `Anthropic()`, so no test can reach the network.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.deep_dive import (
    DEEP_DIVE_TOOL_NAME,
    DEFAULT_MODEL,
    generate_deep_dive,
)
from src.dimensions import DIMENSION_KEYS
from src.models import (
    CandidateProfile,
    DeepDiveReport,
    DimensionScore,
    Gap,
    ProcessingError,
    Role,
    ScoredCandidate,
)
from src.prompts import DEEP_DIVE_SYSTEM_PROMPT
from src.reporter import rank_candidates

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_JD = "Senior Backend Engineer. Python, PostgreSQL, distributed systems."

_VALID_TOOL_INPUT = {
    "summary": "Strong ledger background from Brex, with real on-call ownership.",
    "pros": ["Owned a card transaction ledger", "Five years of payments work"],
    "cons": ["No Go in production", "Team-lead scope is unproven"],
    "interview_questions": [
        "Walk through the idempotent write-ahead replay you built.",
        "How did you handle a settlement mismatch on call?",
        "What would you change about that ledger schema today?",
    ],
}


def _candidate(
    name: str = "Alex Rivera",
    overall: int = 91,
    source: str = "alex.pdf",
) -> ScoredCandidate:
    """A ScoredCandidate with every configured dimension populated.

    Dimension fields are built from DIMENSION_KEYS rather than named, so this
    helper keeps working if dimensions.yaml changes.
    """

    dimensions = {
        key: DimensionScore(score=overall, reasoning=f"{key} reasoning for {name}")
        for key in DIMENSION_KEYS
    }
    dimensions["overall_fit"] = DimensionScore(
        score=overall, reasoning=f"overall_fit reasoning for {name}"
    )
    return ScoredCandidate(
        profile=CandidateProfile(
            name=name,
            years_experience=9.0,
            skills=["Python", "PostgreSQL"],
            past_roles=[
                Role(
                    title="Senior Backend Engineer",
                    company="Brex",
                    duration_months=60,
                    description="Card transaction ledger.",
                )
            ],
            education=["MS Computer Science"],
            raw_summary="I build payment ledgers.",
        ),
        **dimensions,
        reasoning="Two-sentence overall reasoning.",
        gaps=[Gap(category="skill", detail="No Kafka experience")],
        source_file=source,
    )


def _tool_use_response(tool_input: dict | None = None) -> MagicMock:
    """A fake response carrying one tool_use block."""

    block = MagicMock()
    block.type = "tool_use"
    block.input = _VALID_TOOL_INPUT if tool_input is None else tool_input

    response = MagicMock()
    response.content = [block]
    return response


def _client(tool_input: dict | None = None) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(tool_input)
    return client


def _kwargs(client: MagicMock) -> dict:
    return client.messages.create.call_args.kwargs


def _blocks(client: MagicMock) -> list[dict]:
    return _kwargs(client)["messages"][0]["content"]


# ---------------------------------------------------------------------------
# Tool wiring
# ---------------------------------------------------------------------------


def test_tool_is_named_and_schemad_from_the_model() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    tools = _kwargs(client)["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == DEEP_DIVE_TOOL_NAME == "record_deep_dive"
    assert tools[0]["input_schema"] == DeepDiveReport.model_json_schema()


def test_tool_choice_forces_the_deep_dive_tool() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    assert _kwargs(client)["tool_choice"] == {
        "type": "tool",
        "name": DEEP_DIVE_TOOL_NAME,
    }


def test_valid_tool_result_becomes_a_deep_dive_report() -> None:
    report = generate_deep_dive(_client(), _candidate(), _JD)

    assert isinstance(report, DeepDiveReport)
    assert report.summary == _VALID_TOOL_INPUT["summary"]
    assert report.pros == _VALID_TOOL_INPUT["pros"]
    assert report.cons == _VALID_TOOL_INPUT["cons"]
    assert report.interview_questions == _VALID_TOOL_INPUT["interview_questions"]


def test_validation_failure_propagates() -> None:
    """A malformed tool result must raise, so run() can log and skip."""

    client = _client({"summary": "only a summary, missing the three lists"})

    with pytest.raises(ValidationError):
        generate_deep_dive(client, _candidate(), _JD)


def test_system_prompt_comes_from_prompts_module() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    assert _kwargs(client)["system"] == DEEP_DIVE_SYSTEM_PROMPT


def test_model_defaults_and_is_forwarded_when_overridden() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)
    assert _kwargs(client)["model"] == DEFAULT_MODEL

    other = _client()
    generate_deep_dive(other, _candidate(), _JD, model="claude-haiku-4-5")
    assert _kwargs(other)["model"] == "claude-haiku-4-5"


def test_exactly_one_api_call_per_candidate() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    assert client.messages.create.call_count == 1


def test_module_constructs_no_anthropic_client() -> None:
    """The stage must use the client it is handed, or it escapes metering."""

    from pathlib import Path

    source = Path("src/deep_dive.py").read_text(encoding="utf-8")
    assert "Anthropic()" not in source

    # And the annotation import is the only reference to the SDK class.
    with patch("src.deep_dive.Anthropic") as constructor:
        generate_deep_dive(_client(), _candidate(), _JD)
        constructor.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt caching contract
# ---------------------------------------------------------------------------


def test_job_description_is_the_first_content_block() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    blocks = _blocks(client)
    assert blocks[0]["type"] == "text"
    assert "<job_description>" in blocks[0]["text"]
    assert _JD in blocks[0]["text"]


def test_job_description_block_carries_ephemeral_cache_control() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    assert _blocks(client)[0]["cache_control"] == {"type": "ephemeral"}


def test_no_custom_ttl_is_set_on_the_cache_block() -> None:
    """usage.py prices cache writes at the 5-minute rate; a TTL would break that."""

    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    assert "ttl" not in _blocks(client)[0]["cache_control"]


def test_candidate_specific_blocks_come_after_the_cached_jd() -> None:
    """Anything before the breakpoint is part of the prefix and must be shared."""

    client = _client()
    generate_deep_dive(client, _candidate(name="Alex Rivera"), _JD)

    blocks = _blocks(client)
    assert len(blocks) == 3
    assert "<candidate_profile>" in blocks[1]["text"]
    assert "<screening_results>" in blocks[2]["text"]

    # The cached block names no candidate, and only it is cacheable.
    assert "Alex Rivera" not in blocks[0]["text"]
    assert "cache_control" not in blocks[1]
    assert "cache_control" not in blocks[2]


def test_cached_prefix_is_identical_across_candidates() -> None:
    """Two candidates, same JD — block 0 must be byte-identical or caching is moot."""

    first, second = _client(), _client()
    generate_deep_dive(first, _candidate("Alex Rivera", 91, "a.pdf"), _JD)
    generate_deep_dive(second, _candidate("Priya Ramesh", 84, "b.pdf"), _JD)

    assert _blocks(first)[0] == _blocks(second)[0]
    assert _blocks(first)[1] != _blocks(second)[1]


# ---------------------------------------------------------------------------
# Challenge 4: dimensions arrive dynamically
# ---------------------------------------------------------------------------


def test_every_configured_dimension_is_serialized() -> None:
    client = _client()
    candidate = _candidate()
    generate_deep_dive(client, candidate, _JD)

    payload = _blocks(client)[2]["text"]
    scores = json.loads(
        payload.split("<screening_results>\n")[1].split("\n</screening_results>")[0]
    )

    assert list(scores["dimension_scores"]) == DIMENSION_KEYS
    for key in DIMENSION_KEYS:
        entry = scores["dimension_scores"][key]
        assert entry["score"] == getattr(candidate, key).score
        assert entry["reasoning"] == getattr(candidate, key).reasoning
        assert entry["label"]


def test_dimension_names_are_not_hardcoded_in_the_module_or_prompt() -> None:
    """A fifth dimension must reach the briefing with no code or prompt edit."""

    from pathlib import Path

    source = Path("src/deep_dive.py").read_text(encoding="utf-8")
    for key in DIMENSION_KEYS:
        assert key not in source, f"{key} is hardcoded in src/deep_dive.py"
        assert key not in DEEP_DIVE_SYSTEM_PROMPT, f"{key} is hardcoded in the prompt"


def test_screening_results_carry_reasoning_and_gaps() -> None:
    client = _client()
    candidate = _candidate()
    generate_deep_dive(client, candidate, _JD)

    payload = _blocks(client)[2]["text"]
    scores = json.loads(
        payload.split("<screening_results>\n")[1].split("\n</screening_results>")[0]
    )

    assert scores["overall_reasoning"] == candidate.reasoning
    assert scores["gaps"] == [{"category": "skill", "detail": "No Kafka experience"}]


def test_candidate_profile_block_carries_the_full_profile() -> None:
    client = _client()
    generate_deep_dive(client, _candidate(), _JD)

    profile_text = _blocks(client)[1]["text"]
    assert "Brex" in profile_text
    assert "PostgreSQL" in profile_text
    assert "I build payment ledgers." in profile_text


# ---------------------------------------------------------------------------
# Selection: rank_candidates + strict top-N
# ---------------------------------------------------------------------------


def _select(results: list, n: int) -> list[str]:
    """The exact selection main.py performs, reduced to source_file names."""

    return [c.source_file for c in rank_candidates(results)[:n]]


def test_top_zero_selects_nobody() -> None:
    results = [_candidate("A", 90, "a.pdf"), _candidate("B", 80, "b.pdf")]
    assert _select(results, 0) == []


def test_top_one_selects_the_highest_scorer() -> None:
    results = [
        _candidate("B", 80, "b.pdf"),
        _candidate("A", 94, "a.pdf"),
        _candidate("C", 88, "c.pdf"),
    ]
    assert _select(results, 1) == ["a.pdf"]


def test_normal_n_selects_in_rank_order() -> None:
    results = [
        _candidate("B", 91, "b.pdf"),
        _candidate("E", 82, "e.pdf"),
        _candidate("A", 94, "a.pdf"),
        _candidate("C", 88, "c.pdf"),
    ]
    assert _select(results, 3) == ["a.pdf", "b.pdf", "c.pdf"]


def test_n_larger_than_candidate_count_selects_everyone() -> None:
    results = [_candidate("A", 90, "a.pdf"), _candidate("B", 80, "b.pdf")]
    assert _select(results, 99) == ["a.pdf", "b.pdf"]


def test_empty_results_select_nobody() -> None:
    assert _select([], 3) == []


def test_all_processing_errors_select_nobody() -> None:
    results = [
        ProcessingError(source_file="x.pdf", stage="parse", message="corrupt"),
        ProcessingError(source_file="y.pdf", stage="score", message="timeout"),
    ]
    assert _select(results, 3) == []


def test_errors_are_excluded_but_candidates_still_rank() -> None:
    results = [
        ProcessingError(source_file="x.pdf", stage="parse", message="corrupt"),
        _candidate("A", 90, "a.pdf"),
    ]
    assert _select(results, 3) == ["a.pdf"]


def test_ties_resolve_by_source_file_ascending() -> None:
    results = [
        _candidate("D", 88, "zeta.pdf"),
        _candidate("C", 88, "alpha.pdf"),
        _candidate("B", 88, "mid.pdf"),
    ]
    assert _select(results, 3) == ["alpha.pdf", "mid.pdf", "zeta.pdf"]


def test_tie_ordering_does_not_depend_on_input_order() -> None:
    """Reversing the input must not change the ranking."""

    forward = [
        _candidate("A", 88, "alpha.pdf"),
        _candidate("Z", 88, "zeta.pdf"),
    ]
    assert _select(forward, 2) == _select(list(reversed(forward)), 2)


def test_strict_n_cuts_a_candidate_tied_at_the_cutoff() -> None:
    """The spec example: 94, 91, 88, 88, 82 with N=3 briefs exactly three."""

    results = [
        _candidate("A", 94, "a.pdf"),
        _candidate("B", 91, "b.pdf"),
        _candidate("C", 88, "c.pdf"),
        _candidate("D", 88, "d.pdf"),
        _candidate("E", 82, "e.pdf"),
    ]
    selected = _select(results, 3)

    assert selected == ["a.pdf", "b.pdf", "c.pdf"]
    assert "d.pdf" not in selected


def test_selection_matches_the_first_n_of_the_reports_ranking(tmp_path) -> None:
    """Selection and the written report must never disagree about who is on top."""

    from src.reporter import write_json

    results = [
        _candidate("B", 91, "b.pdf"),
        _candidate("D", 88, "d.pdf"),
        _candidate("A", 94, "a.pdf"),
        _candidate("C", 88, "c.pdf"),
        ProcessingError(source_file="x.pdf", stage="parse", message="corrupt"),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)
    reported = [c["source_file"] for c in json.loads(out.read_text())["candidates"]]

    assert _select(results, 3) == reported[:3]


def test_single_candidate_is_selected_normally() -> None:
    assert _select([_candidate("A", 90, "a.pdf")], 3) == ["a.pdf"]
