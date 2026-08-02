"""Tests for reporter.py — pure transform, no API calls."""

import csv
import json
from pathlib import Path

from src.models import (
    CandidateProfile,
    DimensionScore,
    Gap,
    ProcessingError,
    Role,
    ScoredCandidate,
)
from src.reporter import write_csv, write_json, write_markdown


def _make_scored(name: str, overall: int, source: str) -> ScoredCandidate:
    return ScoredCandidate(
        profile=CandidateProfile(
            name=name,
            years_experience=5.0,
            skills=["Python", "SQL"],
            past_roles=[
                Role(
                    title="Backend Engineer",
                    company="Acme",
                    duration_months=24,
                    description="Worked on APIs.",
                )
            ],
            education=["BS Computer Science"],
            raw_summary="I build backend systems.",
        ),
        skills_match=DimensionScore(score=overall - 5, reasoning="Skills reason"),
        experience_match=DimensionScore(score=overall, reasoning="Experience reason"),
        role_relevance=DimensionScore(score=overall + 2, reasoning="Role reason"),
        overall_fit=DimensionScore(score=overall, reasoning="Overall reason"),
        reasoning="Two-sentence reasoning. Looks strong.",
        gaps=[Gap(category="skill", detail="No Kafka experience")],
        source_file=source,
    )


def test_json_output_sorts_by_overall_fit(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Carol", 80, "carol.pdf"),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)

    data = json.loads(out.read_text())
    names = [c["profile"]["name"] for c in data["candidates"]]
    assert names == ["Alice", "Carol", "Bob"]
    assert data["errors"] == []


def test_json_separates_errors_from_candidates(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="parse", message="PDF corrupt"
        ),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)

    data = json.loads(out.read_text())
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["profile"]["name"] == "Alice"
    assert len(data["errors"]) == 1
    assert data["errors"][0]["stage"] == "parse"


def test_markdown_contains_ranked_table_and_sections(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
    ]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "# Resume Match Report" in content
    assert "## Ranked candidates" in content
    assert "Alice" in content
    assert "Bob" in content
    # Alice should appear before Bob in the ranked table
    assert content.index("Alice") < content.index("Bob")


def test_markdown_includes_error_section_when_errors_present(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="extract", message="API timeout"
        ),
    ]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "## Could not process" in content
    assert "broken.pdf" in content
    assert "extract" in content
    assert "API timeout" in content


def test_markdown_omits_error_section_when_no_errors(tmp_path: Path) -> None:
    results = [_make_scored("Alice", 90, "alice.pdf")]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "## Could not process" not in content


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV back into dicts. newline="" mirrors how write_csv opened it,
    which is what keeps newlines embedded inside quoted cells intact."""

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_csv_header_lists_every_column_in_order(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    write_csv([_make_scored("Alice", 90, "alice.pdf")], out)

    with out.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert header == [
        "rank",
        "name",
        "source_file",
        "years_experience",
        "overall_fit_score",
        "overall_fit_reasoning",
        "skills_match_score",
        "skills_match_reasoning",
        "experience_match_score",
        "experience_match_reasoning",
        "role_relevance_score",
        "role_relevance_reasoning",
        "summary_reasoning",
        "skills",
        "education",
        "gaps",
    ]


def test_csv_rows_are_ranked_like_the_json_output(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Carol", 80, "carol.pdf"),
    ]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    rows = _read_csv(out)
    assert [r["name"] for r in rows] == ["Alice", "Carol", "Bob"]
    assert [r["rank"] for r in rows] == ["1", "2", "3"]


def test_csv_omits_processing_errors(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="parse", message="PDF corrupt"
        ),
    ]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert "broken.pdf" not in out.read_text(encoding="utf-8")


def test_csv_preserves_commas_quotes_and_newlines_in_reasoning(tmp_path: Path) -> None:
    tricky = 'Strong, but junior. She said "ship it".\nSecond line.'
    candidate = _make_scored("Alice", 90, "alice.pdf")
    candidate.reasoning = tricky

    out = tmp_path / "results.csv"
    write_csv([candidate], out)

    rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["summary_reasoning"] == tricky


def test_csv_flattens_list_fields_into_one_cell_each(tmp_path: Path) -> None:
    candidate = _make_scored("Alice", 90, "alice.pdf")
    candidate.profile.education = ["BS Computer Science", "MS Data Science"]
    candidate.gaps = [
        Gap(category="skill", detail="Missing Kubernetes"),
        Gap(category="experience", detail="Needs more backend work"),
    ]

    out = tmp_path / "results.csv"
    write_csv([candidate], out)

    row = _read_csv(out)[0]
    assert row["skills"] == "Python; SQL"
    assert row["education"] == "BS Computer Science; MS Data Science"
    assert row["gaps"] == "Skill: Missing Kubernetes; Experience: Needs more backend work"


def test_csv_writes_header_only_when_every_result_failed(tmp_path: Path) -> None:
    results = [
        ProcessingError(source_file="a.pdf", stage="parse", message="corrupt"),
        ProcessingError(source_file="b.pdf", stage="score", message="timeout"),
    ]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    assert out.exists()
    assert _read_csv(out) == []
    assert out.read_text(encoding="utf-8").startswith("rank,name,source_file")


def test_csv_has_no_blank_lines_between_rows(tmp_path: Path) -> None:
    results = [_make_scored("Alice", 90, "alice.pdf"), _make_scored("Bob", 70, "bob.pdf")]

    out = tmp_path / "results.csv"
    write_csv(results, out)

    # The csv module writes its own \r\n. If write_csv ever drops newline="",
    # Windows translates that again into \r\r\n — a blank line per row.
    assert b"\r\r\n" not in out.read_bytes()
