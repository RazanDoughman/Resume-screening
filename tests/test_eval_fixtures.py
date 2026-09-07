"""Tests for Challenge 7 Stage 3 — set orchestration and proposal persistence.

Nothing here constructs an `Anthropic()` and nothing reaches the network: the
orchestration tests replace `generate_resume_proposal` with a recording stub,
and the persistence tests are pure filesystem work under `tmp_path`.

Three groups:
  - generate_proposal_set (one call per level, order, propagation, failures)
  - the writers (determinism, round-trip, encoding, layout)
  - the boundary that keeps generated proposals out of the eval suite
"""

import ast
import json
from pathlib import Path
from typing import get_args
from unittest.mock import MagicMock

import pytest

from src import eval_gen
from src.eval_fixtures import (
    PROPOSALS_FILENAME,
    proposal_filename,
    read_proposals,
    render_proposal_markdown,
    write_proposals,
)
from src.eval_gen import (
    GENERATION_ORDER,
    EvalGenerationError,
    generate_proposal_set,
)
from src.models import GeneratedResumeProposal, MatchLevel
from src.prompts import SYNTH_RESUME_LEVELS

_JD = "Senior Backend Engineer — Payments. Python, PostgreSQL, 5+ years."

_ROOT = Path(__file__).resolve().parent.parent


def _proposal(level: str = "strong", **overrides) -> GeneratedResumeProposal:
    payload = {
        "resume_markdown": f"# Candidate {level}\n\nBackend engineer.\n",
        "intended_level": level,
        "expected_score_range": (80, 100),
        "dimensions_expected_high": ["skills_match"],
        "dimensions_expected_low": [],
        "rationale": f"Written to sit at the {level} level.",
    }
    payload.update(overrides)
    return GeneratedResumeProposal(**payload)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the single-proposal primitive with a recording stub.

    Stage 2 already covers what one generation call sends; these tests are
    about how the set layer drives it, so the primitive is stubbed rather than
    re-exercised through a mock client.
    """

    calls: list[dict] = []

    def _fake(client, jd, level, model=eval_gen.DEFAULT_MODEL):
        calls.append(
            {"client": client, "jd": jd, "level": level, "model": model}
        )
        return _proposal(level)

    monkeypatch.setattr(eval_gen, "generate_resume_proposal", _fake)
    return calls


# ---------------------------------------------------------------------------
# Complete-set orchestration
# ---------------------------------------------------------------------------


def test_generates_exactly_four_proposals(recorded: list[dict]) -> None:
    proposals = generate_proposal_set(MagicMock(), _JD)

    assert len(proposals) == 4
    assert all(isinstance(p, GeneratedResumeProposal) for p in proposals)


def test_calls_the_primitive_exactly_once_per_level(recorded: list[dict]) -> None:
    generate_proposal_set(MagicMock(), _JD)

    assert len(recorded) == 4
    assert sorted(c["level"] for c in recorded) == sorted(SYNTH_RESUME_LEVELS)


def test_generation_order_is_deterministic(recorded: list[dict]) -> None:
    proposals = generate_proposal_set(MagicMock(), _JD)

    expected = ["strong", "partial", "weak", "adversarial"]
    assert [c["level"] for c in recorded] == expected
    assert [p.intended_level for p in proposals] == expected


def test_generation_order_constant_matches_the_match_level_type() -> None:
    """Pins the derived order so reordering MatchLevel fails loudly.

    GENERATION_ORDER is `get_args(MatchLevel)` rather than a second hardcoded
    list. That keeps one source of truth, but it also means a reorder in
    models.py would silently renumber every generated artifact — hence this
    explicit snapshot.
    """

    assert GENERATION_ORDER == ("strong", "partial", "weak", "adversarial")
    assert GENERATION_ORDER == get_args(MatchLevel)
    assert set(GENERATION_ORDER) == set(SYNTH_RESUME_LEVELS)


def test_the_same_job_description_reaches_every_call(recorded: list[dict]) -> None:
    generate_proposal_set(MagicMock(), _JD)

    assert [c["jd"] for c in recorded] == [_JD] * 4


def test_the_client_is_passed_through_untouched(recorded: list[dict]) -> None:
    """A MeteredClient handed in must be the one every call uses."""

    client = MagicMock()
    generate_proposal_set(client, _JD)

    assert [c["client"] for c in recorded] == [client] * 4


def test_model_defaults_and_propagates(recorded: list[dict]) -> None:
    generate_proposal_set(MagicMock(), _JD)
    assert [c["model"] for c in recorded] == [eval_gen.DEFAULT_MODEL] * 4

    recorded.clear()
    generate_proposal_set(MagicMock(), _JD, model="claude-haiku-4-5")
    assert [c["model"] for c in recorded] == ["claude-haiku-4-5"] * 4


def test_orchestration_constructs_no_client_of_its_own() -> None:
    source = (_ROOT / "src" / "eval_gen.py").read_text(encoding="utf-8")

    assert "Anthropic()" not in source


def test_a_failing_level_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad level aborts the set rather than returning a short list.

    A set silently missing its adversarial case is invisible in the output —
    the reviewer has no reason to count the files.
    """

    def _fake(client, jd, level, model=eval_gen.DEFAULT_MODEL):
        if level == "weak":
            raise EvalGenerationError("no tool_use block")
        return _proposal(level)

    monkeypatch.setattr(eval_gen, "generate_resume_proposal", _fake)

    with pytest.raises(EvalGenerationError, match="no tool_use block"):
        generate_proposal_set(MagicMock(), _JD)


def test_a_failing_level_stops_the_remaining_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No partial-success semantics, and no money spent after the failure."""

    seen: list[str] = []

    def _fake(client, jd, level, model=eval_gen.DEFAULT_MODEL):
        seen.append(level)
        if level == "partial":
            raise EvalGenerationError("boom")
        return _proposal(level)

    monkeypatch.setattr(eval_gen, "generate_resume_proposal", _fake)

    with pytest.raises(EvalGenerationError):
        generate_proposal_set(MagicMock(), _JD)

    assert seen == ["strong", "partial"]  # weak and adversarial never ran


def test_set_orchestration_writes_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: list[dict]
) -> None:
    """Persistence is eval_fixtures.py's job, not the generator's."""

    monkeypatch.chdir(tmp_path)
    generate_proposal_set(MagicMock(), _JD)

    assert list(tmp_path.iterdir()) == []


def test_set_orchestration_uses_no_thread_pool() -> None:
    """Sequential on purpose — the four calls share one cached prefix."""

    source = (_ROOT / "src" / "eval_gen.py").read_text(encoding="utf-8")

    assert "ThreadPoolExecutor" not in source


# ---------------------------------------------------------------------------
# Persistence — layout and determinism
# ---------------------------------------------------------------------------


def _set() -> list[GeneratedResumeProposal]:
    return [_proposal(level) for level in GENERATION_ORDER]


def test_writes_the_expected_file_layout(tmp_path: Path) -> None:
    written = write_proposals(_set(), tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "01-strong.md",
        "02-partial.md",
        "03-weak.md",
        "04-adversarial.md",
        PROPOSALS_FILENAME,
    ]
    assert [p.name for p in written][0] == PROPOSALS_FILENAME
    assert all(p.exists() for p in written)


def test_filenames_sort_into_generation_order(tmp_path: Path) -> None:
    """Zero-padded index first, so a listing reads strong to adversarial."""

    write_proposals(_set(), tmp_path)
    markdown = sorted(p.name for p in tmp_path.glob("*.md"))

    assert [name.split("-", 1)[1].removesuffix(".md") for name in markdown] == [
        "strong",
        "partial",
        "weak",
        "adversarial",
    ]


def test_filename_is_derived_only_from_index_and_level() -> None:
    assert proposal_filename(1, _proposal("strong")) == "01-strong.md"
    assert proposal_filename(4, _proposal("adversarial")) == "04-adversarial.md"
    assert proposal_filename(12, _proposal("weak")) == "12-weak.md"


def test_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    """The determinism requirement, checked the way git would see it."""

    first, second = tmp_path / "a", tmp_path / "b"
    write_proposals(_set(), first)
    write_proposals(_set(), second)

    for path in sorted(first.iterdir()):
        assert path.read_bytes() == (second / path.name).read_bytes()


def test_rewriting_the_same_set_changes_nothing(tmp_path: Path) -> None:
    write_proposals(_set(), tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    write_proposals(_set(), tmp_path)

    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_creates_a_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "generated"

    write_proposals(_set(), target)

    assert (target / PROPOSALS_FILENAME).exists()


def test_no_timestamp_or_absolute_path_leaks_into_the_output(
    tmp_path: Path,
) -> None:
    write_proposals(_set(), tmp_path)

    for path in tmp_path.iterdir():
        text = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "20" + "26-" not in text  # no ISO date stamped into content


# ---------------------------------------------------------------------------
# Persistence — the data contract
# ---------------------------------------------------------------------------


def test_every_field_survives_a_round_trip(tmp_path: Path) -> None:
    original = [
        _proposal(
            "strong",
            expected_score_range=(81, 97),
            dimensions_expected_high=["skills_match", "role_relevance"],
            dimensions_expected_low=["experience_match"],
            rationale="Deep payments background, thin on team leadership.",
        ),
        _proposal("adversarial", expected_score_range=(15, 45)),
    ]

    write_proposals(original, tmp_path)
    restored = read_proposals(tmp_path)

    assert restored == original
    for before, after in zip(original, restored):
        assert after.resume_markdown == before.resume_markdown
        assert after.intended_level == before.intended_level
        assert after.expected_score_range == before.expected_score_range
        assert after.dimensions_expected_high == before.dimensions_expected_high
        assert after.dimensions_expected_low == before.dimensions_expected_low
        assert after.rationale == before.rationale


def test_round_trip_preserves_order(tmp_path: Path) -> None:
    write_proposals(_set(), tmp_path)

    assert [p.intended_level for p in read_proposals(tmp_path)] == list(
        GENERATION_ORDER
    )


def test_json_payload_is_the_models_own_dump_not_a_second_schema(
    tmp_path: Path,
) -> None:
    """The writer serializes validated objects; it defines no schema itself."""

    proposals = _set()
    write_proposals(proposals, tmp_path)

    payload = json.loads(
        (tmp_path / PROPOSALS_FILENAME).read_text(encoding="utf-8")
    )

    assert payload["proposals"] == [p.model_dump(mode="json") for p in proposals]


def test_writer_rejects_anything_that_is_not_a_validated_proposal(
    tmp_path: Path,
) -> None:
    """A raw dict has no model_dump; it must not be silently accepted."""

    with pytest.raises(AttributeError):
        write_proposals([{"intended_level": "strong"}], tmp_path)


def test_hand_edited_json_that_breaks_the_contract_fails_on_read(
    tmp_path: Path,
) -> None:
    """Review happens by editing these files; a bad edit must not pass."""

    from pydantic import ValidationError

    write_proposals(_set(), tmp_path)
    path = tmp_path / PROPOSALS_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["proposals"][0]["expected_score_range"] = [90, 10]  # inverted
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError):
        read_proposals(tmp_path)


def test_utf8_content_survives_both_writers(tmp_path: Path) -> None:
    proposal = _proposal(
        "partial",
        resume_markdown="# Zoë Müller\n\nEngineer — Kraków. 日本語, 5 ans.\n",
        rationale="Accents, an em dash — and CJK, all through both writers.",
    )

    write_proposals([proposal], tmp_path)

    assert read_proposals(tmp_path)[0] == proposal
    markdown = (tmp_path / "01-partial.md").read_text(encoding="utf-8")
    assert "Zoë Müller" in markdown and "日本語" in markdown
    # ensure_ascii=False, so the JSON stays readable rather than \u-escaped.
    raw = (tmp_path / PROPOSALS_FILENAME).read_text(encoding="utf-8")
    assert "Zoë Müller" in raw


def test_markdown_reproduces_the_resume_verbatim(tmp_path: Path) -> None:
    """A reviewer must be judging the text the scorer will read."""

    resume = "# Ada Byrne\n\n## Experience\n\n- Built a ledger.\n- On-call.\n"
    write_proposals([_proposal("weak", resume_markdown=resume)], tmp_path)

    assert resume.strip() in (tmp_path / "01-weak.md").read_text(encoding="utf-8")


def test_markdown_shows_the_stated_intent(tmp_path: Path) -> None:
    proposal = _proposal(
        "adversarial",
        expected_score_range=(20, 45),
        dimensions_expected_high=["skills_match"],
        dimensions_expected_low=["role_relevance"],
        rationale="Keyword-dense but hollow.",
    )

    rendered = render_proposal_markdown(proposal)

    assert "adversarial" in rendered
    assert "20-45" in rendered
    assert "`skills_match`" in rendered
    assert "`role_relevance`" in rendered
    assert "Keyword-dense but hollow." in rendered


def test_empty_dimension_lists_render_readably() -> None:
    rendered = render_proposal_markdown(
        _proposal("weak", dimensions_expected_high=[], dimensions_expected_low=[])
    )

    assert "_(none)_" in rendered


# ---------------------------------------------------------------------------
# The boundary: generated != active eval case
# ---------------------------------------------------------------------------


def test_every_markdown_file_carries_the_unreviewed_banner(tmp_path: Path) -> None:
    """The one line that stops a reader trusting these numbers."""

    write_proposals(_set(), tmp_path)

    for path in tmp_path.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        assert "Unreviewed, machine-generated proposal — not an eval case" in text
        assert "not been checked against the scorer" in text


def test_persistence_does_not_touch_the_eval_cases_file(tmp_path: Path) -> None:
    cases = _ROOT / "tests" / "evals" / "cases.py"
    before = cases.read_bytes()

    write_proposals(_set(), tmp_path)
    read_proposals(tmp_path)

    assert cases.read_bytes() == before


def test_persistence_does_not_activate_anything_in_all_cases(tmp_path: Path) -> None:
    from tests.evals.cases import ALL_BIAS_CASES, ALL_CASES, ALL_DEEP_DIVE_CASES

    counts = (len(ALL_CASES), len(ALL_BIAS_CASES), len(ALL_DEEP_DIVE_CASES))
    names = [c.name for c in ALL_CASES]

    write_proposals(_set(), tmp_path)
    read_proposals(tmp_path)

    assert (len(ALL_CASES), len(ALL_BIAS_CASES), len(ALL_DEEP_DIVE_CASES)) == counts
    assert [c.name for c in ALL_CASES] == names


def _imported_names(module_filename: str) -> set[str]:
    """Every module and symbol name a source file actually imports.

    Parsed rather than grepped, the same reason tests/test_app_parity.py parses
    app.py: both modules discuss `ALL_CASES` and `anthropic` in prose explaining
    what they deliberately do *not* touch, and a string search would flag that
    prose as a dependency.
    """

    tree = ast.parse((_ROOT / "src" / module_filename).read_text(encoding="utf-8"))
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)

    return names


def test_generation_modules_never_import_the_eval_cases() -> None:
    """Promotion needs a human; neither module may reach the case list at all."""

    for name in ("eval_gen.py", "eval_fixtures.py"):
        imported = _imported_names(name)
        assert not {"ALL_CASES", "EvalCase"} & imported, name
        assert not any(n.startswith("tests") for n in imported), name


def test_persistence_module_makes_no_api_calls() -> None:
    """It is reporter.py's counterpart: files only, never an LLM."""

    imported = _imported_names("eval_fixtures.py")

    assert not any("anthropic" in n.lower() for n in imported)

    source = (_ROOT / "src" / "eval_fixtures.py").read_text(encoding="utf-8")
    calls = {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "create" not in calls
