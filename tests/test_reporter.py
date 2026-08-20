"""Tests for reporter.py — pure transform, no API calls."""

import csv
import json
from pathlib import Path

from src.models import (
    BiasAuditReport,
    CandidateProfile,
    DeepDiveReport,
    DimensionScore,
    Gap,
    ProcessingError,
    Role,
    ScoredCandidate,
    ScoreReport,
)
from src.reporter import (
    _BIN_COUNT,
    _COLUMNS,
    _score_bins,
    rank_candidates,
    write_csv,
    write_json,
    write_markdown,
)


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
        # The derived dimensions are clamped because DimensionScore is bounded
        # 0-100: the histogram tests pass overall=0 and overall=100, which would
        # otherwise derive -5 and 102 and fail validation.
        skills_match=DimensionScore(score=max(0, overall - 5), reasoning="Skills reason"),
        experience_match=DimensionScore(score=overall, reasoning="Experience reason"),
        role_relevance=DimensionScore(score=min(100, overall + 2), reasoning="Role reason"),
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


def _bins(*scores: int) -> list[int]:
    """Bin a list of overall_fit scores, one throwaway candidate per score."""

    return _score_bins([_make_scored(f"C{i}", s, f"c{i}.pdf") for i, s in enumerate(scores)])


def test_score_bins_counts_a_known_set_of_scores() -> None:
    assert _bins(92, 88, 76, 71, 45) == [0, 0, 0, 0, 1, 0, 0, 2, 1, 1]


def test_score_bins_returns_all_zeros_for_empty_input() -> None:
    counts = _score_bins([])

    assert len(counts) == _BIN_COUNT
    assert counts == [0] * _BIN_COUNT


def test_score_bins_handles_a_single_candidate() -> None:
    counts = _bins(55)

    assert counts[5] == 1
    assert sum(counts) == 1


def test_score_bins_counts_zero_in_the_bottom_bin() -> None:
    # `if score:` is False for a legitimate 0 — that candidate must still count.
    counts = _bins(0)

    assert counts[0] == 1
    assert sum(counts) == 1


def test_score_bins_counts_one_hundred_in_the_top_bin() -> None:
    # 100 // 10 == 10, one past the last index, unless the bin is clamped.
    counts = _bins(100)

    assert counts[_BIN_COUNT - 1] == 1
    assert sum(counts) == 1


def test_score_bins_puts_89_and_90_in_different_bins() -> None:
    counts = _bins(89, 90)

    assert counts[8] == 1
    assert counts[9] == 1


def test_score_bins_puts_all_identical_scores_in_one_bin() -> None:
    # Score compression is a real signal about the scoring prompt, not a bug.
    counts = _bins(75, 75, 75, 75, 75)

    assert counts[7] == 5
    assert sum(counts) == 5


def _chart_line(content: str, label: str) -> str:
    """Return the histogram line for one bin, e.g. '90-100'."""

    for line in content.splitlines():
        if line.strip().startswith(f"{label} |"):
            return line
    raise AssertionError(f"no histogram line for bin {label!r}")


def _report(tmp_path: Path, results: list) -> str:
    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    return out.read_text(encoding="utf-8")


def test_markdown_includes_score_distribution_section(tmp_path: Path) -> None:
    content = _report(tmp_path, [_make_scored("Alice", 90, "alice.pdf")])

    assert "## Score distribution" in content
    assert content.index("## Score distribution") < content.index("## Ranked candidates")


def test_markdown_histogram_is_inside_a_code_fence(tmp_path: Path) -> None:
    # Without the fence, markdown collapses the padding and the bars misalign.
    content = _report(tmp_path, [_make_scored("Alice", 90, "alice.pdf")])

    chart = content.split("## Score distribution", 1)[1].split("## Ranked candidates", 1)[0]
    assert chart.count("```") == 2
    assert _chart_line(content, "90-100") in chart.split("```")[1]


def test_markdown_top_bin_is_labelled_inclusive_of_one_hundred(tmp_path: Path) -> None:
    # The top bin holds 90..100. "Simplifying" the label to 90-99 would lie.
    content = _report(tmp_path, [_make_scored("Alice", 100, "alice.pdf")])

    assert "90-100" in content
    assert "90-99" not in content


def test_markdown_histogram_prints_counts_for_empty_bins(tmp_path: Path) -> None:
    content = _report(tmp_path, [_make_scored("Alice", 55, "alice.pdf")])

    assert _chart_line(content, "0-9").endswith(" 0")
    assert _chart_line(content, "50-59").endswith(" 1")


def test_markdown_histogram_renders_a_bar_for_every_nonzero_bin(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 92, "alice.pdf"),
        _make_scored("Bob", 76, "bob.pdf"),
        _make_scored("Carol", 71, "carol.pdf"),
    ]

    content = _report(tmp_path, results)

    assert "#" in _chart_line(content, "90-100")
    assert "##" in _chart_line(content, "70-79")
    assert "#" not in _chart_line(content, "60-69")


def test_markdown_shows_placeholder_when_no_candidates_scored(tmp_path: Path) -> None:
    # Every resume failed: no max(), no division, no crash.
    results = [
        ProcessingError(source_file="a.pdf", stage="parse", message="corrupt"),
        ProcessingError(source_file="b.pdf", stage="score", message="timeout"),
    ]

    content = _report(tmp_path, results)

    assert "## Score distribution" in content
    assert "_No scored candidates to chart._" in content
    assert "```" not in content


def test_markdown_histogram_excludes_failed_resumes_from_counts(tmp_path: Path) -> None:
    # write_markdown must chart the split-out candidates, not the raw results.
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(source_file="a.pdf", stage="parse", message="corrupt"),
        ProcessingError(source_file="b.pdf", stage="score", message="timeout"),
    ]

    content = _report(tmp_path, results)

    assert "n = 1 scored candidate)" in content
    assert _chart_line(content, "90-100").endswith(" 1")


def test_markdown_notes_failed_resumes_beneath_the_histogram(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(source_file="a.pdf", stage="parse", message="corrupt"),
        ProcessingError(source_file="b.pdf", stage="score", message="timeout"),
    ]

    content = _report(tmp_path, results)

    assert "2 resumes failed to process" in content


def test_markdown_omits_failure_note_when_nothing_failed(tmp_path: Path) -> None:
    content = _report(tmp_path, [_make_scored("Alice", 90, "alice.pdf")])

    assert "failed to process" not in content


# ---------------------------------------------------------------------------
# Deep-dive rendering (Challenge 5)
#
# The sidecar defaults to None, so every test above this line exercises the
# "feature off" path. These cover the "feature on" one.
# ---------------------------------------------------------------------------


def _deep_dive(summary: str = "Alice owns the ledger at Brex.") -> DeepDiveReport:
    return DeepDiveReport(
        summary=summary,
        pros=["Owned a card ledger", "Nine years in payments"],
        cons=["No Go in production", "Lead scope unproven"],
        interview_questions=[
            "Walk through the replay logic.",
            "Describe a settlement mismatch you handled.",
            "What would you change about that schema?",
        ],
    )


def _bias_audit() -> BiasAuditReport:
    scores = ScoreReport(
        skills_match=DimensionScore(score=90, reasoning="s"),
        experience_match=DimensionScore(score=90, reasoning="e"),
        role_relevance=DimensionScore(score=90, reasoning="r"),
        overall_fit=DimensionScore(score=90, reasoning="o"),
        reasoning="Baseline reasoning.",
        gaps=[],
    )
    return BiasAuditReport(
        baseline=scores,
        variants=[],
        max_score_drift=0.0,
        flagged=False,
    )


def test_markdown_is_byte_identical_when_no_deep_dives_are_passed(
    tmp_path: Path,
) -> None:
    """The core no-regression check: the sidecar must be inert when absent."""

    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Bob", 70, "bob.pdf"),
        ProcessingError(source_file="broken.pdf", stage="parse", message="corrupt"),
    ]

    default = tmp_path / "default.md"
    explicit = tmp_path / "explicit.md"
    empty = tmp_path / "empty.md"

    write_markdown(results, default, Path("jd.txt"))
    write_markdown(results, explicit, Path("jd.txt"), deep_dives=None)
    write_markdown(results, empty, Path("jd.txt"), deep_dives={})

    baseline = default.read_text(encoding="utf-8")
    assert explicit.read_text(encoding="utf-8") == baseline
    assert empty.read_text(encoding="utf-8") == baseline
    assert "Deep-Dive" not in baseline


def test_markdown_renders_the_deep_dive_section(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    assert "**Hiring Manager Deep-Dive**" in content
    assert "Alice owns the ledger at Brex." in content
    assert "_Strengths_" in content
    assert "_Risks_" in content
    assert "_Suggested interview questions_" in content


def test_markdown_renders_every_pro_con_and_question_as_a_bullet(
    tmp_path: Path,
) -> None:
    report = _deep_dive()
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": report},
    )
    content = out.read_text(encoding="utf-8")

    for item in report.pros + report.cons + report.interview_questions:
        assert f"- {item}" in content


def test_markdown_deep_dives_only_the_matching_candidate(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [
            _make_scored("Alice", 90, "alice.pdf"),
            _make_scored("Bob", 70, "bob.pdf"),
        ],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    assert content.count("**Hiring Manager Deep-Dive**") == 1
    # The briefing sits inside Alice's section, above Bob's heading.
    assert content.index("**Hiring Manager Deep-Dive**") < content.index("### Bob")


def test_markdown_shows_no_placeholder_for_candidates_without_a_deep_dive(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [
            _make_scored("Alice", 90, "alice.pdf"),
            _make_scored("Bob", 70, "bob.pdf"),
        ],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    for phrase in ("not deep-dived", "No deep-dive", "not analyzed", "N/A"):
        assert phrase not in content


def test_markdown_renders_deep_dive_and_bias_audit_together(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        audits={"alice.pdf": _bias_audit()},
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    assert "**Hiring Manager Deep-Dive**" in content
    assert "**Bias Audit**" in content


def test_markdown_places_the_deep_dive_after_gaps_and_before_bias_audit(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        audits={"alice.pdf": _bias_audit()},
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    assert (
        content.index("**Gaps**")
        < content.index("**Hiring Manager Deep-Dive**")
        < content.index("**Bias Audit**")
    )


def test_markdown_keeps_the_histogram_first_when_deep_dives_render(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    assert (
        content.index("## Score distribution")
        < content.index("## Ranked candidates")
        < content.index("**Hiring Manager Deep-Dive**")
    )


def test_markdown_keeps_the_score_breakdown_when_deep_dives_render(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": _deep_dive()},
    )
    content = out.read_text(encoding="utf-8")

    assert "**Score breakdown**" in content
    assert "- Skills (85): Skills reason" in content


def test_markdown_handles_pipes_and_newlines_in_deep_dive_prose(
    tmp_path: Path,
) -> None:
    """Bullets, not a table - so a pipe in model prose cannot break the layout."""

    report = DeepDiveReport(
        summary="Ran the a|b split test.",
        pros=["Owns pipe | delimited parsing"],
        cons=["Unclear scope | ambiguous title"],
        interview_questions=["What does staff | principal mean at your company?"],
    )
    out = tmp_path / "report.md"
    write_markdown(
        [_make_scored("Alice", 90, "alice.pdf")],
        out,
        Path("jd.txt"),
        deep_dives={"alice.pdf": report},
    )
    content = out.read_text(encoding="utf-8")

    assert "- Owns pipe | delimited parsing" in content
    assert "Ran the a|b split test." in content


def test_json_and_csv_ignore_the_deep_dive_sidecar(tmp_path: Path) -> None:
    """Neither writer takes the parameter, so neither output can change."""

    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Bob", 70, "bob.pdf"),
    ]

    json_out = tmp_path / "results.json"
    csv_out = tmp_path / "results.csv"
    write_json(results, json_out)
    write_csv(results, csv_out)

    raw = json_out.read_text(encoding="utf-8")
    assert "deep_dive" not in raw
    assert set(json.loads(raw)) == {"candidates", "errors"}

    header = csv_out.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(_COLUMNS)
    assert "deep_dive" not in header


# ---------------------------------------------------------------------------
# rank_candidates - the single ranking definition (Challenge 5)
# ---------------------------------------------------------------------------


def test_rank_candidates_sorts_by_overall_fit_descending() -> None:
    ranked = rank_candidates(
        [
            _make_scored("Bob", 70, "bob.pdf"),
            _make_scored("Alice", 90, "alice.pdf"),
            _make_scored("Carol", 80, "carol.pdf"),
        ]
    )

    assert [c.profile.name for c in ranked] == ["Alice", "Carol", "Bob"]


def test_rank_candidates_drops_processing_errors() -> None:
    ranked = rank_candidates(
        [
            _make_scored("Alice", 90, "alice.pdf"),
            ProcessingError(source_file="x.pdf", stage="parse", message="corrupt"),
        ]
    )

    assert [c.profile.name for c in ranked] == ["Alice"]


def test_rank_candidates_breaks_ties_on_source_file() -> None:
    ranked = rank_candidates(
        [
            _make_scored("Zoe", 88, "zeta.pdf"),
            _make_scored("Amy", 88, "alpha.pdf"),
        ]
    )

    assert [c.source_file for c in ranked] == ["alpha.pdf", "zeta.pdf"]


def test_rank_candidates_is_independent_of_input_order() -> None:
    results = [
        _make_scored("Amy", 88, "alpha.pdf"),
        _make_scored("Zoe", 88, "zeta.pdf"),
        _make_scored("Bob", 70, "bob.pdf"),
    ]

    forward = [c.source_file for c in rank_candidates(results)]
    backward = [c.source_file for c in rank_candidates(list(reversed(results)))]
    assert forward == backward


def test_rank_candidates_returns_empty_for_no_input() -> None:
    assert rank_candidates([]) == []


def test_reports_rank_the_same_way_rank_candidates_does(tmp_path: Path) -> None:
    """The invariant deep-dive selection relies on."""

    results = [
        _make_scored("Zoe", 88, "zeta.pdf"),
        _make_scored("Alice", 94, "alice.pdf"),
        _make_scored("Amy", 88, "alpha.pdf"),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)
    reported = [c["source_file"] for c in json.loads(out.read_text())["candidates"]]

    assert reported == [c.source_file for c in rank_candidates(results)]
