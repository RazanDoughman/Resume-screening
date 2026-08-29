"""Tests for the Challenge 6 benchmark harness.

Everything here runs on hand-written usage data. Nothing constructs an
`Anthropic()`, nothing calls `main.run()`, and nothing sleeps — so no test can
reach the network or depend on a real duration.

What is deliberately NOT tested: how long anything takes. A test that asserts
`elapsed < 5.0` fails on a slow CI box and passes on a fast one while proving
nothing either way. The clock is a two-line `perf_counter()` bracket in
`run_once`; what is worth testing is everything that turns its output into a
number a reader can trust.
"""

import pytest

from scripts.benchmark import (
    REFERENCE_PREFIXES,
    build_run_record,
    cache_prefix_summary,
    expected_call_count,
    format_cache_report,
    format_table,
    median_elapsed,
    stability,
    uncached_call_count,
)


def _call(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> dict:
    return {
        "model": "claude-sonnet-4-6",
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "output_tokens": output_tokens,
    }


def _sequential_calls() -> list[dict]:
    """A healthy sequential run: 3 resumes, 2 deep-dives.

    extract/score interleaved, one scorer cache write then reads, then a
    deep-dive write followed by a read. Same shape as a real usage.json.
    """

    return [
        _call(),                                            # extract
        _call(cache_creation_input_tokens=1667),            # score  <- write
        _call(),                                            # extract
        _call(cache_read_input_tokens=1667),                # score  <- read
        _call(),                                            # extract
        _call(cache_read_input_tokens=1667),                # score  <- read
        _call(cache_creation_input_tokens=1863),            # deep-dive <- write
        _call(cache_read_input_tokens=1863),                # deep-dive <- read
    ]


def _usage(calls: list[dict] | None = None) -> dict:
    """A usage.json payload matching UsageReport.to_dict()'s schema."""

    calls = _sequential_calls() if calls is None else calls
    return {
        "call_count": len(calls),
        "models": ["claude-sonnet-4-6"],
        "totals": {
            "input_tokens": sum(c["input_tokens"] for c in calls),
            "cache_read_input_tokens": sum(c["cache_read_input_tokens"] for c in calls),
            "cache_creation_input_tokens": sum(
                c["cache_creation_input_tokens"] for c in calls
            ),
            "total_prompt_tokens": sum(
                c["input_tokens"]
                + c["cache_read_input_tokens"]
                + c["cache_creation_input_tokens"]
                for c in calls
            ),
            "output_tokens": sum(c["output_tokens"] for c in calls),
        },
        "estimated_cost_usd": 0.293629,
        "unpriced_models": [],
        "pricing_note": "Estimated ...",
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# Expected call count (the pre-run cost banner)
# ---------------------------------------------------------------------------


def test_expected_call_count_is_two_per_resume_plus_the_deep_dives() -> None:
    assert expected_call_count(10, deep_dive_top=3) == 23


def test_deep_dive_never_costs_more_than_the_field_size() -> None:
    """--deep-dive-top 99 over 3 resumes is 3 briefings, not 99."""

    assert expected_call_count(3, deep_dive_top=99) == 9


def test_self_critique_adds_one_call_per_resume() -> None:
    assert expected_call_count(10, deep_dive_top=0, self_critique=True) == 30


# ---------------------------------------------------------------------------
# Cache prefix grouping — derived from the data, not from constants
# ---------------------------------------------------------------------------


def test_sequential_run_shows_one_write_then_reads_per_prefix() -> None:
    summary = cache_prefix_summary(_sequential_calls())

    assert summary == {
        1667: {"writes": 1, "reads": 2},
        1863: {"writes": 1, "reads": 1},
    }


def test_concurrent_cold_start_shows_up_as_several_writes() -> None:
    """The failure this benchmark exists to detect: N writes, no reads."""

    calls = [_call(cache_creation_input_tokens=1667) for _ in range(5)]

    assert cache_prefix_summary(calls) == {1667: {"writes": 5, "reads": 0}}


def test_grouping_does_not_depend_on_the_reference_sizes() -> None:
    """A changed JD moves both prefixes; the grouping must still be right.

    Nothing in cache_prefix_summary knows 1667 or 1863 — that is the whole
    point, since those numbers are observations of today's prompts, not
    invariants of the code.
    """

    calls = [
        _call(cache_creation_input_tokens=999),
        _call(cache_read_input_tokens=999),
        _call(cache_creation_input_tokens=4242),
    ]

    assert cache_prefix_summary(calls) == {
        999: {"writes": 1, "reads": 1},
        4242: {"writes": 1, "reads": 0},
    }


def test_no_caching_at_all_groups_to_nothing() -> None:
    assert cache_prefix_summary([_call(), _call()]) == {}


def test_uncached_calls_are_counted_separately() -> None:
    """Extraction sends no cache_control, so it lands in neither group."""

    assert uncached_call_count(_sequential_calls()) == 3


def test_missing_cache_fields_are_treated_as_zero() -> None:
    """The SDK reports None, not 0, on an uncached call; usage.json may omit."""

    calls = [{"input_tokens": 10, "output_tokens": 5}]

    assert cache_prefix_summary(calls) == {}
    assert uncached_call_count(calls) == 1


# ---------------------------------------------------------------------------
# Run records
# ---------------------------------------------------------------------------


def test_run_record_flattens_usage_and_elapsed() -> None:
    record = build_run_record(1, 123.456, _usage())

    assert record["run"] == 1
    assert record["elapsed_seconds"] == 123.46
    assert record["call_count"] == 8
    assert record["input_tokens"] == 800
    assert record["output_tokens"] == 400
    assert record["cache_creation_input_tokens"] == 3530
    assert record["cache_read_input_tokens"] == 1667 * 2 + 1863
    assert record["estimated_cost_usd"] == 0.293629
    assert record["models"] == ["claude-sonnet-4-6"]


def test_run_record_keeps_every_call_for_later_inspection() -> None:
    """A paid run should never need re-running to answer a cache question."""

    record = build_run_record(1, 1.0, _usage())

    assert record["calls"] == _sequential_calls()
    assert record["cache_prefixes"] == {
        "1667": {"writes": 1, "reads": 2},
        "1863": {"writes": 1, "reads": 1},
    }
    assert record["uncached_calls"] == 3


def test_run_record_survives_an_unpriced_model() -> None:
    """estimate_cost returns None, never 0.0 — the record must carry that."""

    usage = _usage()
    usage["estimated_cost_usd"] = None

    assert build_run_record(1, 1.0, usage)["estimated_cost_usd"] is None


# ---------------------------------------------------------------------------
# Median and stability
# ---------------------------------------------------------------------------


def _records(*elapsed: float, **overrides) -> list[dict]:
    out = []
    for i, seconds in enumerate(elapsed, start=1):
        record = build_run_record(i, seconds, _usage())
        record.update(overrides.get(f"run{i}", {}))
        out.append(record)
    return out


def test_median_of_three_picks_the_middle_run() -> None:
    assert median_elapsed(_records(210.0, 180.0, 195.0)) == 195.0


def test_median_ignores_a_slow_outlier() -> None:
    """The reason it is a median: one retried call must not move the number."""

    assert median_elapsed(_records(181.0, 900.0, 179.0)) == 181.0


def test_median_of_an_even_number_of_runs_averages_the_middle_two() -> None:
    assert median_elapsed(_records(100.0, 200.0, 300.0, 400.0)) == 250.0


def test_median_of_a_single_run_is_that_run() -> None:
    assert median_elapsed(_records(42.5)) == 42.5


def test_identical_token_counts_report_as_stable() -> None:
    assert all(stability(_records(1.0, 2.0, 3.0)).values())


def test_a_drifting_token_count_reports_as_unstable() -> None:
    """Cache tokens moving between runs is a finding, not noise to smooth."""

    records = _records(1.0, 2.0, 3.0)
    records[1]["cache_creation_input_tokens"] = 10198

    result = stability(records)
    assert result["cache_creation_input_tokens"] is False
    assert result["call_count"] is True


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def test_table_has_one_column_per_run_and_a_row_per_metric() -> None:
    table = format_table(_records(180.0, 181.0, 179.0))
    lines = table.splitlines()

    assert lines[0] == "| Metric | Run 1 | Run 2 | Run 3 |"
    assert lines[1] == "| --- | ---: | ---: | ---: |"
    assert len(lines) == 2 + 8  # header, rule, eight metric rows
    assert "| Elapsed seconds | 180.00 | 181.00 | 179.00 |" in table
    assert "| Estimated cost (USD) | $0.293629 | $0.293629 | $0.293629 |" in table


def test_table_renders_an_unavailable_cost_without_crashing() -> None:
    usage = _usage()
    usage["estimated_cost_usd"] = None

    assert "n/a" in format_table([build_run_record(1, 1.0, usage)])


def test_cache_report_names_the_recognised_prefixes() -> None:
    report = format_cache_report(build_run_record(1, 1.0, _usage()))

    assert "scorer" in report
    assert "deep-dive" in report
    assert "1 write(s), 2 read(s)" in report
    assert "uncached calls (no prefix): 3" in report


def test_cache_report_flags_a_prefix_it_does_not_recognise() -> None:
    """A changed prompt moves the size; report it, never silently drop it."""

    usage = _usage([_call(cache_creation_input_tokens=1234)])
    report = format_cache_report(build_run_record(1, 1.0, usage))

    assert "1,234-token prefix" in report
    assert "unrecognised prefix" in report


def test_cache_report_says_when_a_reference_prefix_vanished() -> None:
    """Only the scorer prefix appears — the deep-dive one must be called out."""

    usage = _usage([_call(cache_creation_input_tokens=1667)])
    report = format_cache_report(build_run_record(1, 1.0, usage))

    assert "NOT OBSERVED" in report
    assert "1,863-token prefix (deep-dive)" in report


@pytest.mark.parametrize("size,name", sorted(REFERENCE_PREFIXES.items()))
def test_reference_prefixes_are_labels_only(size: int, name: str) -> None:
    """Guards the rule that these numbers stay out of the application.

    They exist to print "scorer" next to a group of records. If one of them ever
    turns into a threshold or a branch condition in src/, this test's premise is
    gone and the benchmark has started asserting production behaviour.
    """

    assert isinstance(size, int) and size > 0
    assert name in {"scorer", "deep-dive"}
