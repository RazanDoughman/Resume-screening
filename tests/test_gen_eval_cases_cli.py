"""Tests for the Challenge 7 CLI — scripts/gen_eval_cases.py.

No real Anthropic client and no network: `Anthropic` and the generation
primitive are both patched on the CLI module, and every filesystem test runs
under `tmp_path`. Several tests additionally assert that `Anthropic` was never
even constructed, which is the property that makes a `--dry-run` safe to run
on a machine with no API key.

Five groups:
  - JD input (reading, encoding, failure)
  - dry-run (nothing called, nothing written, plan shown)
  - orchestration (what the CLI hands to which existing function)
  - usage integration (Challenge 8 infrastructure, reused not reinvented)
  - failure and the human-review boundary
"""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import gen_eval_cases as cli
from src.eval_gen import GENERATION_ORDER, EvalGenerationError
from src.models import GeneratedResumeProposal

_ROOT = Path(__file__).resolve().parent.parent

_JD_TEXT = "Senior Backend Engineer - Payments.\nPython, PostgreSQL, 5+ years.\n"


def _jd_file(tmp_path: Path, text: str = _JD_TEXT) -> Path:
    path = tmp_path / "jd.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _proposal(level: str) -> GeneratedResumeProposal:
    return GeneratedResumeProposal(
        resume_markdown=f"# Candidate {level}\n\nBackend engineer.\n",
        intended_level=level,
        expected_score_range=(40, 70),
        dimensions_expected_high=["skills_match"],
        dimensions_expected_low=[],
        rationale=f"Written at the {level} level.",
    )


def _usage_response() -> MagicMock:
    """A response shaped enough for APIUsage.from_response to read it."""

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 200
    usage.cache_read_input_tokens = 0

    response = MagicMock()
    response.model = "claude-sonnet-4-6"
    response.usage = usage
    return response


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the CLI's client constructor and generation primitive.

    Records what the CLI passed on, so the orchestration tests can assert the
    wiring without any of the underlying stages actually running.
    """

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")

    calls: dict = {"generate": [], "anthropic_constructed": 0}

    def _fake_anthropic(*args, **kwargs):
        calls["anthropic_constructed"] += 1
        return MagicMock()

    def _fake_generate(client, jd, model=cli.DEFAULT_MODEL):
        calls["generate"].append({"client": client, "jd": jd, "model": model})
        return [_proposal(level) for level in GENERATION_ORDER]

    monkeypatch.setattr(cli, "Anthropic", _fake_anthropic)
    monkeypatch.setattr(cli, "generate_proposal_set", _fake_generate)
    return calls


# ---------------------------------------------------------------------------
# Job description input
# ---------------------------------------------------------------------------


def test_reads_the_jd_file_and_passes_it_through(
    tmp_path: Path, stubbed: dict
) -> None:
    jd = _jd_file(tmp_path)

    assert cli.run(str(jd), out_dir=tmp_path / "out") == 0
    assert stubbed["generate"][0]["jd"] == _JD_TEXT


def test_utf8_jd_is_read_correctly(tmp_path: Path, stubbed: dict) -> None:
    """A JD pasted from a job board routinely has typographic punctuation."""

    text = "Senior Engineer — Zürich office\n\n• 日本語 a plus\n"
    jd = _jd_file(tmp_path, text)

    cli.run(str(jd), out_dir=tmp_path / "out")

    assert stubbed["generate"][0]["jd"] == text


def test_missing_jd_fails_clearly(tmp_path: Path, capsys) -> None:
    assert cli.run(str(tmp_path / "nope.txt")) == 1

    assert "JD file not found" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["", "   ", "\n\n\t\n"])
def test_empty_jd_fails_clearly(tmp_path: Path, content: str, capsys) -> None:
    jd = _jd_file(tmp_path, content)

    assert cli.run(str(jd)) == 1
    assert "is empty" in capsys.readouterr().out


def test_a_missing_jd_never_constructs_a_client(
    tmp_path: Path, stubbed: dict
) -> None:
    """Fail before spending anything, including on a client."""

    cli.run(str(tmp_path / "nope.txt"))

    assert stubbed["anthropic_constructed"] == 0
    assert stubbed["generate"] == []


def test_missing_api_key_fails_before_generating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    called: list = []
    monkeypatch.setattr(
        cli, "generate_proposal_set", lambda *a, **k: called.append(1)
    )

    assert cli.run(str(_jd_file(tmp_path))) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().out
    assert called == []


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_makes_no_api_call_and_builds_no_client(
    tmp_path: Path, stubbed: dict
) -> None:
    assert cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out", dry_run=True) == 0

    assert stubbed["anthropic_constructed"] == 0
    assert stubbed["generate"] == []


def test_dry_run_writes_nothing_at_all(tmp_path: Path, stubbed: dict) -> None:
    out = tmp_path / "out"
    jd = _jd_file(tmp_path)

    cli.run(str(jd), out_dir=out, dry_run=True)

    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["jd.txt"]


def test_dry_run_writes_no_usage_file(tmp_path: Path, stubbed: dict) -> None:
    out = tmp_path / "out"

    cli.run(str(_jd_file(tmp_path)), out_dir=out, dry_run=True)

    assert not (out / cli.USAGE_FILENAME).exists()


def test_dry_run_reports_no_token_usage_or_cost(
    tmp_path: Path, stubbed: dict, capsys
) -> None:
    """Dry run must not print fabricated numbers."""

    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out", dry_run=True)
    out = capsys.readouterr().out

    assert "API Usage" not in out
    assert "Estimated cost" not in out


def test_dry_run_invents_no_resume_content(
    tmp_path: Path, stubbed: dict, capsys
) -> None:
    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out", dry_run=True)
    out = capsys.readouterr().out

    assert "# Candidate" not in out
    assert "rationale" not in out.lower()


def test_dry_run_needs_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of checking wiring without credentials."""

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "o", dry_run=True) == 0


def test_dry_run_shows_the_plan(tmp_path: Path, stubbed: dict, capsys) -> None:
    jd = _jd_file(tmp_path)
    out = tmp_path / "out"

    cli.run(str(jd), out_dir=out, model="claude-haiku-4-5", dry_run=True)
    printed = capsys.readouterr().out

    assert str(jd) in printed              # resolved JD source
    assert "claude-haiku-4-5" in printed   # model selection
    assert "out" in printed                # output directory
    assert "4 separate LLM calls" in printed
    assert "no API calls made" in printed


def test_dry_run_shows_the_four_levels_in_order(
    tmp_path: Path, stubbed: dict, capsys
) -> None:
    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "o", dry_run=True)
    printed = capsys.readouterr().out

    assert "strong, partial, weak, adversarial" in printed
    assert list(GENERATION_ORDER) == ["strong", "partial", "weak", "adversarial"]


def test_dry_run_warns_the_output_is_unreviewed(
    tmp_path: Path, stubbed: dict, capsys
) -> None:
    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "o", dry_run=True)

    assert "UNREVIEWED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_calls_generate_proposal_set_exactly_once(
    tmp_path: Path, stubbed: dict
) -> None:
    """The set function owns the loop; the CLI must not loop over levels."""

    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out")

    assert len(stubbed["generate"]) == 1


def test_passes_the_metered_client_not_a_raw_one(
    tmp_path: Path, stubbed: dict
) -> None:
    """Challenge 8's seam: generation calls must land in usage.json."""

    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out")
    client = stubbed["generate"][0]["client"]

    assert isinstance(client, cli.MeteredClient)
    assert stubbed["anthropic_constructed"] == 1


def test_model_defaults_and_can_be_overridden(
    tmp_path: Path, stubbed: dict
) -> None:
    jd = str(_jd_file(tmp_path))

    cli.run(jd, out_dir=tmp_path / "a")
    assert stubbed["generate"][0]["model"] == cli.DEFAULT_MODEL

    cli.run(jd, out_dir=tmp_path / "b", model="claude-haiku-4-5")
    assert stubbed["generate"][1]["model"] == "claude-haiku-4-5"


def test_proposals_reach_the_stage_3_writer(tmp_path: Path, stubbed: dict) -> None:
    out = tmp_path / "out"

    assert cli.run(str(_jd_file(tmp_path)), out_dir=out) == 0

    # The layout Stage 3's write_proposals produces, unmodified.
    assert sorted(p.name for p in out.iterdir()) == [
        "01-strong.md",
        "02-partial.md",
        "03-weak.md",
        "04-adversarial.md",
        "proposals.json",
        cli.USAGE_FILENAME,
    ]


def test_the_output_directory_reaches_the_writer(
    tmp_path: Path, stubbed: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list = []
    monkeypatch.setattr(
        cli,
        "write_proposals",
        lambda proposals, directory: seen.append((proposals, Path(directory))) or [],
    )
    out = tmp_path / "nested" / "out"

    cli.run(str(_jd_file(tmp_path)), out_dir=out)

    assert seen[0][1] == out
    assert [p.intended_level for p in seen[0][0]] == list(GENERATION_ORDER)


def test_written_proposals_round_trip(tmp_path: Path, stubbed: dict) -> None:
    from src.eval_fixtures import read_proposals

    out = tmp_path / "out"
    cli.run(str(_jd_file(tmp_path)), out_dir=out)

    assert [p.intended_level for p in read_proposals(out)] == list(GENERATION_ORDER)


def test_reports_where_artifacts_went(tmp_path: Path, stubbed: dict, capsys) -> None:
    out = tmp_path / "out"

    cli.run(str(_jd_file(tmp_path)), out_dir=out)
    printed = capsys.readouterr().out

    assert "Wrote 4 proposals" in printed
    assert "proposals.json" in printed
    assert "01-strong.md" in printed


def test_out_dir_resolution_is_deterministic(tmp_path: Path, stubbed: dict) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    jd = str(_jd_file(tmp_path))

    cli.run(jd, out_dir=a)
    cli.run(jd, out_dir=b)

    for path in sorted(p for p in a.iterdir() if p.name != cli.USAGE_FILENAME):
        assert path.read_bytes() == (b / path.name).read_bytes()


# ---------------------------------------------------------------------------
# Usage integration — Challenge 8 infrastructure, reused
# ---------------------------------------------------------------------------


def test_writes_a_usage_report_with_the_existing_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same keys usage.py's UsageReport.to_dict() produces for main.py."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(cli, "Anthropic", lambda *a, **k: MagicMock())

    def _generate(client, jd, model=cli.DEFAULT_MODEL):
        # Drive the real tracker through the real MeteredClient seam.
        for _ in GENERATION_ORDER:
            client._tracker.record(_usage_response())
        return [_proposal(level) for level in GENERATION_ORDER]

    monkeypatch.setattr(cli, "generate_proposal_set", _generate)
    out = tmp_path / "out"

    cli.run(str(_jd_file(tmp_path)), out_dir=out)
    usage = json.loads((out / cli.USAGE_FILENAME).read_text(encoding="utf-8"))

    assert set(usage) == {
        "call_count",
        "models",
        "totals",
        "estimated_cost_usd",
        "unpriced_models",
        "pricing_note",
        "calls",
    }
    assert usage["call_count"] == 4
    assert usage["totals"]["output_tokens"] == 200
    assert usage["estimated_cost_usd"] > 0


def test_uses_the_shared_challenge_8_infrastructure() -> None:
    """Imported from usage.py/reporter.py, not reimplemented in the script."""

    source = (_ROOT / "scripts" / "gen_eval_cases.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(a.name for a in node.names)

    assert {"UsageTracker", "MeteredClient", "build_report", "format_summary"} <= imported
    assert "write_usage" in imported
    # No second cost table or token arithmetic in the script.
    assert "MODEL_PRICING" not in source
    assert "input_tokens" not in source


def test_usage_summary_is_printed(tmp_path: Path, stubbed: dict, capsys) -> None:
    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out")

    assert "API Usage" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_generation_failure_reports_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(cli, "Anthropic", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        cli,
        "generate_proposal_set",
        MagicMock(side_effect=EvalGenerationError("no tool_use block for 'weak'")),
    )
    out = tmp_path / "out"

    assert cli.run(str(_jd_file(tmp_path)), out_dir=out) == 1

    printed = capsys.readouterr().out
    assert "generation failed" in printed
    assert "no tool_use block for 'weak'" in printed  # not swallowed
    assert "No proposals were written" in printed


def test_generation_failure_writes_no_partial_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed run must not leave a half-written set that looks complete."""

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(cli, "Anthropic", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        cli,
        "generate_proposal_set",
        MagicMock(side_effect=EvalGenerationError("boom")),
    )
    out = tmp_path / "out"

    cli.run(str(_jd_file(tmp_path)), out_dir=out)

    assert not out.exists()


def test_writer_failure_is_not_swallowed(
    tmp_path: Path, stubbed: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "write_proposals",
        MagicMock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out")


# ---------------------------------------------------------------------------
# The human-review boundary
# ---------------------------------------------------------------------------


def test_cli_never_touches_the_review_layer() -> None:
    """It cannot promote or approve: it does not import the module that can."""

    source = (_ROOT / "scripts" / "gen_eval_cases.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
            names.update(a.name for a in node.names)

    assert "src.generated_cases" not in modules
    assert not {"promote_proposal", "write_generated_cases", "GeneratedEvalCase"} & names
    assert "reviewed" not in {n for n in names}


def test_cli_never_writes_reviewed_true(tmp_path: Path, stubbed: dict) -> None:
    """No artifact the CLI produces claims review."""

    out = tmp_path / "out"
    cli.run(str(_jd_file(tmp_path)), out_dir=out)

    for path in out.iterdir():
        assert "reviewed" not in path.read_text(encoding="utf-8").lower().replace(
            "unreviewed", ""
        )


def test_cli_does_not_modify_the_eval_case_files(
    tmp_path: Path, stubbed: dict
) -> None:
    cases = _ROOT / "tests" / "evals" / "cases.py"
    generated = _ROOT / "tests" / "evals" / "generated_cases.json"
    before = cases.read_bytes()
    # Compared rather than asserted absent: once a human has reviewed a set,
    # this file legitimately exists. What must stay true is that a generation
    # run never touches it — creating it, or editing one already there, would
    # both be the CLI crossing the review boundary.
    generated_before = generated.read_bytes() if generated.exists() else None

    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out")

    assert cases.read_bytes() == before
    generated_after = generated.read_bytes() if generated.exists() else None
    assert generated_after == generated_before


def test_generated_output_stays_out_of_all_cases(
    tmp_path: Path, stubbed: dict
) -> None:
    """Running the CLI activates nothing in the eval suite.

    Stated as "the case list is the same before and after" rather than "the
    generated list is empty". The empty version only passed while nothing had
    been reviewed; it would have gone green forever after without ever
    checking the thing it was named for. Comparing across the run catches a CLI
    that activates a case whether or not other cases are already active.
    """

    import importlib

    import tests.evals.cases as cases_module

    importlib.reload(cases_module)
    before = [c.name for c in cases_module.ALL_CASES]
    generated_before = [c.name for c in cases_module.GENERATED_CASES]

    cli.run(str(_jd_file(tmp_path)), out_dir=tmp_path / "out")
    importlib.reload(cases_module)

    assert [c.name for c in cases_module.ALL_CASES] == before
    assert [c.name for c in cases_module.GENERATED_CASES] == generated_before


def test_cli_adds_no_concurrency() -> None:
    """Sequential generation is deliberate — it is what shares the cache prefix."""

    source = (_ROOT / "scripts" / "gen_eval_cases.py").read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" not in source


def test_cli_builds_no_api_request_of_its_own() -> None:
    """Request construction belongs to eval_gen.py, or caching silently breaks."""

    source = (_ROOT / "scripts" / "gen_eval_cases.py").read_text(encoding="utf-8")

    assert "messages.create" not in source
    assert "tool_choice" not in source
    assert "cache_control" not in source


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_main_requires_a_jd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["gen_eval_cases.py"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2  # argparse usage error


def test_main_wires_arguments_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        cli,
        "run",
        lambda jd, out_dir, model, dry_run: seen.update(
            jd=jd, out_dir=out_dir, model=model, dry_run=dry_run
        )
        or 0,
    )
    jd = str(_jd_file(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "gen_eval_cases.py",
            "--jd", jd,
            "--out-dir", str(tmp_path / "out"),
            "--model", "claude-haiku-4-5",
            "--dry-run",
        ],
    )

    assert cli.main() == 0
    assert seen == {
        "jd": jd,
        "out_dir": str(tmp_path / "out"),
        "model": "claude-haiku-4-5",
        "dry_run": True,
    }


def test_defaults_are_the_documented_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(
        cli,
        "run",
        lambda jd, out_dir, model, dry_run: seen.update(
            out_dir=out_dir, model=model, dry_run=dry_run
        )
        or 0,
    )
    monkeypatch.setattr("sys.argv", ["gen_eval_cases.py", "--jd", "x.txt"])

    cli.main()

    assert seen["out_dir"] == str(cli.DEFAULT_OUT_DIR)
    assert seen["model"] == cli.DEFAULT_MODEL
    assert seen["dry_run"] is False
