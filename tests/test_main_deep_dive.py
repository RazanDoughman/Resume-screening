"""Orchestration tests for the deep-dive stage in main.run().

Everything that would touch the network or a PDF is patched, so these run in
milliseconds with no API key. What they actually pin down is the wiring the
unit tests can't see: that the stage runs after the resume loop, on the top N
only, before the usage report is built, and that a failure inside it doesn't
cost anyone their score.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import main as main_module
from src.models import (
    CandidateProfile,
    DeepDiveReport,
    DimensionScore,
    Gap,
    ProcessingError,
    Role,
    ScoredCandidate,
)
from src.dimensions import DIMENSION_KEYS

_REPORT = DeepDiveReport(
    summary="A briefing paragraph.",
    pros=["Owned the ledger"],
    cons=["No Go in production"],
    interview_questions=["Walk me through the replay logic.", "Why leave?", "Scope?"],
)


def _candidate(name: str, overall: int, source: str) -> ScoredCandidate:
    dimensions = {
        key: DimensionScore(score=overall, reasoning=f"{key} reasoning")
        for key in DIMENSION_KEYS
    }
    dimensions["overall_fit"] = DimensionScore(score=overall, reasoning="overall")
    return ScoredCandidate(
        profile=CandidateProfile(
            name=name,
            years_experience=5.0,
            skills=["Python"],
            past_roles=[
                Role(
                    title="Engineer",
                    company="Acme",
                    duration_months=24,
                    description="APIs.",
                )
            ],
            education=["BS CS"],
            raw_summary="Summary.",
        ),
        **dimensions,
        reasoning="Reasoning.",
        gaps=[Gap(category="skill", detail="No Kafka")],
        source_file=source,
    )


def _usage_response(content: list | None = None) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    response = MagicMock()
    response.model = "claude-sonnet-4-6"
    response.usage = usage
    response.content = [] if content is None else content
    return response


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[str, str, str]:
    """A JD file, a resumes dir holding three PDFs, and an output dir."""

    jd = tmp_path / "jd.txt"
    jd.write_text("Senior Backend Engineer. Python, PostgreSQL.", encoding="utf-8")

    resumes = tmp_path / "resumes"
    resumes.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (resumes / name).write_bytes(b"%PDF-1.4 stub")

    return str(jd), str(resumes), str(tmp_path / "out")


def _patched_run(
    workspace: tuple[str, str, str],
    deep_dive_top: int,
    *,
    scores: dict[str, int] | None = None,
    deep_dive: MagicMock | None = None,
    inner_client: MagicMock | None = None,
) -> tuple[int, MagicMock]:
    """Run main.run() with the pipeline mocked. Returns (exit code, deep-dive mock)."""

    jd_path, resumes_dir, output_dir = workspace
    scores = scores or {"a.pdf": 94, "b.pdf": 88, "c.pdf": 88}
    deep_dive = deep_dive or MagicMock(return_value=_REPORT)

    client = inner_client or MagicMock()
    if inner_client is None:
        client.messages.create.return_value = _usage_response()

    def _score(_client, _profile, _jd, source_file, model=None):
        return _candidate(source_file.replace(".pdf", ""), scores[source_file], source_file)

    with (
        patch.object(main_module, "Anthropic", return_value=client),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate") as extract,
        patch.object(main_module, "score_candidate", side_effect=_score),
        patch.object(main_module, "generate_deep_dive", deep_dive),
    ):
        extract.return_value = _candidate("X", 0, "x.pdf").profile
        code = main_module.run(
            jd_path,
            resumes_dir,
            output_dir,
            "claude-sonnet-4-6",
            False,
            False,
            False,
            deep_dive_top,
        )

    return code, deep_dive


# ---------------------------------------------------------------------------
# The stage runs, and only on the top N
# ---------------------------------------------------------------------------


def test_disabled_by_default_makes_zero_deep_dive_calls(workspace) -> None:
    code, deep_dive = _patched_run(workspace, 0)

    assert code == 0
    deep_dive.assert_not_called()


def test_only_the_top_n_candidates_are_briefed(workspace) -> None:
    code, deep_dive = _patched_run(workspace, 2)

    assert code == 0
    assert deep_dive.call_count == 2
    briefed = [call.args[1].source_file for call in deep_dive.call_args_list]
    assert briefed == ["a.pdf", "b.pdf"]


def test_strict_n_cuts_the_tied_candidate(workspace) -> None:
    """b.pdf and c.pdf both score 88; N=2 briefs b.pdf only, by filename."""

    _, deep_dive = _patched_run(workspace, 2)

    briefed = [call.args[1].source_file for call in deep_dive.call_args_list]
    assert "c.pdf" not in briefed


def test_n_larger_than_the_field_briefs_everyone(workspace) -> None:
    _, deep_dive = _patched_run(workspace, 99)

    assert deep_dive.call_count == 3


def test_deep_dive_receives_the_scored_candidate_and_jd(workspace) -> None:
    _, deep_dive = _patched_run(workspace, 1)

    call = deep_dive.call_args_list[0]
    assert isinstance(call.args[1], ScoredCandidate)
    assert call.args[1].source_file == "a.pdf"
    assert "Senior Backend Engineer" in call.args[2]
    assert call.kwargs["model"] == "claude-sonnet-4-6"


def test_deep_dive_uses_the_same_client_the_pipeline_uses(workspace) -> None:
    """The client handed to the stage must be the metered one main.run built."""

    _, deep_dive = _patched_run(workspace, 1)

    from src.usage import MeteredClient

    assert isinstance(deep_dive.call_args_list[0].args[0], MeteredClient)


# ---------------------------------------------------------------------------
# Ordering: after scoring, before build_report
# ---------------------------------------------------------------------------


def test_deep_dive_runs_after_every_resume_is_scored(workspace) -> None:
    """A post-scoring batch stage: nothing may be briefed mid-loop."""

    order: list[str] = []

    def _score(_client, _profile, _jd, source_file, model=None):
        order.append(f"score:{source_file}")
        return _candidate(source_file, 90, source_file)

    def _brief(_client, candidate, _jd, model=None):
        order.append(f"brief:{candidate.source_file}")
        return _REPORT

    jd_path, resumes_dir, output_dir = workspace
    client = MagicMock()
    client.messages.create.return_value = _usage_response()

    with (
        patch.object(main_module, "Anthropic", return_value=client),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(main_module, "score_candidate", side_effect=_score),
        patch.object(main_module, "generate_deep_dive", side_effect=_brief),
    ):
        main_module.run(
            jd_path, resumes_dir, output_dir, "m", False, False, False, 2
        )

    scored = [i for i, step in enumerate(order) if step.startswith("score:")]
    briefed = [i for i, step in enumerate(order) if step.startswith("brief:")]
    assert max(scored) < min(briefed)


def test_deep_dive_calls_happen_before_the_usage_report_is_built(workspace) -> None:
    """Calls recorded after build_report would be missing from usage.json."""

    seen_at_build_time: list[int] = []
    deep_dive = MagicMock(return_value=_REPORT)

    real_build_report = main_module.build_report

    def _spy(records):
        seen_at_build_time.append(deep_dive.call_count)
        return real_build_report(records)

    with patch.object(main_module, "build_report", side_effect=_spy):
        _patched_run(workspace, 2, deep_dive=deep_dive)

    assert seen_at_build_time == [2]


def test_deep_dive_calls_appear_in_usage_json(workspace) -> None:
    """End to end through the real deep_dive module and the real MeteredClient."""

    block = MagicMock()
    block.type = "tool_use"
    block.input = _REPORT.model_dump()

    client = MagicMock()
    client.messages.create.side_effect = lambda **kwargs: _usage_response(
        content=[block]
    )

    jd_path, resumes_dir, output_dir = workspace

    def _score(_client, _profile, _jd, source_file, model=None):
        return _candidate(source_file, 90, source_file)

    # generate_deep_dive is NOT patched here — the real one runs.
    with (
        patch.object(main_module, "Anthropic", return_value=client),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(main_module, "score_candidate", side_effect=_score),
    ):
        code = main_module.run(
            jd_path, resumes_dir, output_dir, "claude-sonnet-4-6", False, False, False, 2
        )

    assert code == 0

    # extract_candidate and score_candidate are patched out and never touch the
    # client, so every recorded call here is a deep-dive call. That isolation is
    # the point: 2 briefings requested, 2 calls metered, with nothing in this
    # test teaching usage.py how to count them.
    usage = json.loads(Path(output_dir, "usage.json").read_text(encoding="utf-8"))
    assert usage["call_count"] == 2

    report = Path(output_dir, "report.md").read_text(encoding="utf-8")
    assert report.count("**Hiring Manager Deep-Dive**") == 2


def test_deep_dive_contributes_no_calls_when_disabled(workspace) -> None:
    """The mirror of the test above: same mocks, flag off, nothing metered."""

    block = MagicMock()
    block.type = "tool_use"
    block.input = _REPORT.model_dump()

    client = MagicMock()
    client.messages.create.side_effect = lambda **kwargs: _usage_response(
        content=[block]
    )

    jd_path, resumes_dir, output_dir = workspace

    def _score(_client, _profile, _jd, source_file, model=None):
        return _candidate(source_file, 90, source_file)

    with (
        patch.object(main_module, "Anthropic", return_value=client),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(main_module, "score_candidate", side_effect=_score),
    ):
        main_module.run(
            jd_path, resumes_dir, output_dir, "claude-sonnet-4-6", False, False, False, 0
        )

    usage = json.loads(Path(output_dir, "usage.json").read_text(encoding="utf-8"))
    assert usage["call_count"] == 0


# ---------------------------------------------------------------------------
# Failure behaviour
# ---------------------------------------------------------------------------


def test_a_failed_briefing_keeps_the_candidate_and_the_run(workspace) -> None:
    deep_dive = MagicMock(side_effect=RuntimeError("api exploded"))

    code, _ = _patched_run(workspace, 2, deep_dive=deep_dive)
    _, _, output_dir = workspace

    assert code == 0

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert len(data["candidates"]) == 3
    assert data["errors"] == []

    report = Path(output_dir, "report.md").read_text(encoding="utf-8")
    assert "**Hiring Manager Deep-Dive**" not in report
    assert "## Ranked candidates" in report


def test_one_failed_briefing_does_not_stop_the_others(workspace) -> None:
    deep_dive = MagicMock(side_effect=[RuntimeError("boom"), _REPORT])

    code, deep_dive = _patched_run(workspace, 2, deep_dive=deep_dive)
    _, _, output_dir = workspace

    assert code == 0
    assert deep_dive.call_count == 2

    report = Path(output_dir, "report.md").read_text(encoding="utf-8")
    assert report.count("**Hiring Manager Deep-Dive**") == 1


def test_a_failed_briefing_creates_no_processing_error(workspace) -> None:
    deep_dive = MagicMock(side_effect=RuntimeError("boom"))

    _patched_run(workspace, 1, deep_dive=deep_dive)
    _, _, output_dir = workspace

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert data["errors"] == []
    assert {"parse", "extract", "score"} >= {
        e["stage"] for e in data["errors"]
    }


def test_no_scored_candidates_makes_no_deep_dive_calls(workspace) -> None:
    """Every resume fails at score — there is nobody to brief."""

    jd_path, resumes_dir, output_dir = workspace
    deep_dive = MagicMock(return_value=_REPORT)
    client = MagicMock()
    client.messages.create.return_value = _usage_response()

    with (
        patch.object(main_module, "Anthropic", return_value=client),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(
            main_module, "score_candidate", side_effect=RuntimeError("scoring down")
        ),
        patch.object(main_module, "generate_deep_dive", deep_dive),
    ):
        code = main_module.run(
            jd_path, resumes_dir, output_dir, "m", False, False, False, 3
        )

    assert code == 0
    deep_dive.assert_not_called()

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert len(data["errors"]) == 3


# ---------------------------------------------------------------------------
# Outputs stay compatible
# ---------------------------------------------------------------------------


def test_results_json_gains_no_deep_dive_key(workspace) -> None:
    _patched_run(workspace, 3)
    _, _, output_dir = workspace

    raw = Path(output_dir, "results.json").read_text(encoding="utf-8")
    assert "deep_dive" not in raw

    data = json.loads(raw)
    assert set(data) == {"candidates", "errors"}
    for candidate in data["candidates"]:
        assert set(candidate) == set(DIMENSION_KEYS) | {
            "profile",
            "reasoning",
            "gaps",
            "source_file",
        }


def test_csv_columns_are_unchanged_when_deep_dive_runs(workspace) -> None:
    jd_path, resumes_dir, output_dir = workspace
    client = MagicMock()
    client.messages.create.return_value = _usage_response()

    def _score(_client, _profile, _jd, source_file, model=None):
        return _candidate(source_file, 90, source_file)

    with (
        patch.object(main_module, "Anthropic", return_value=client),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(main_module, "score_candidate", side_effect=_score),
        patch.object(main_module, "generate_deep_dive", return_value=_REPORT),
    ):
        main_module.run(jd_path, resumes_dir, output_dir, "m", False, False, True, 2)

    from src.reporter import _COLUMNS

    header = Path(output_dir, "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(_COLUMNS)
    assert "deep_dive" not in header


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def test_flag_defaults_to_zero() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep-dive-top", type=main_module._non_negative_int, default=0
    )
    assert parser.parse_args([]).deep_dive_top == 0


@pytest.mark.parametrize("value,expected", [("0", 0), ("1", 1), ("3", 3), ("99", 99)])
def test_non_negative_integers_are_accepted(value: str, expected: int) -> None:
    assert main_module._non_negative_int(value) == expected


@pytest.mark.parametrize("value", ["-1", "-99"])
def test_negative_values_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="0 or greater"):
        main_module._non_negative_int(value)


@pytest.mark.parametrize("value", ["abc", "3.5", "", "two"])
def test_non_integer_values_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
        main_module._non_negative_int(value)
