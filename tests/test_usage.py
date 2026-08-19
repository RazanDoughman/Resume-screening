"""Tests for usage.py — mocks the Anthropic client, no real API calls.

Every response here is a MagicMock shaped like a real one: `.model`, `.usage`
with the five token fields, and `.content`. Nothing in this file constructs an
`Anthropic()` client, so no test can reach the network.
"""

import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.extractor import extract_candidate
from src.reporter import write_usage
from src.usage import (
    APIUsage,
    MeteredClient,
    UsageTotals,
    UsageTracker,
    build_report,
    estimate_cost,
    format_summary,
)


def _usage_response(
    *,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_creation_input_tokens: int | None = 0,
    cache_read_input_tokens: int | None = 0,
    content: list | None = None,
) -> MagicMock:
    """Build a fake Anthropic response carrying usage.

    The two cache fields default to 0 but accept None, because that is what the
    SDK actually returns on an uncached call.
    """

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens

    response = MagicMock()
    response.model = model
    response.usage = usage
    response.content = [] if content is None else content
    return response


def _record(
    *,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> APIUsage:
    return APIUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


# ---------------------------------------------------------------------------
# Empty runs
# ---------------------------------------------------------------------------


def test_empty_tracker_totals_are_zero() -> None:
    totals = UsageTotals.from_records(UsageTracker().records)

    assert totals.call_count == 0
    assert totals.input_tokens == 0
    assert totals.cache_read_input_tokens == 0
    assert totals.cache_creation_input_tokens == 0
    assert totals.output_tokens == 0
    assert totals.total_prompt_tokens == 0


def test_empty_run_costs_zero_not_unavailable() -> None:
    # No records means no unpriced model — the run genuinely cost nothing.
    # That is a different answer from "we don't know", which is None.
    assert estimate_cost([]) == Decimal("0")


def test_empty_summary_renders_without_crashing() -> None:
    # No max() over an empty sequence, no division by a zero token count.
    summary = format_summary(build_report([]))

    assert "API Usage" in summary
    assert "API calls:" in summary


# ---------------------------------------------------------------------------
# Recording and aggregation
# ---------------------------------------------------------------------------


def test_records_one_response() -> None:
    tracker = UsageTracker()
    tracker.record(
        _usage_response(
            model="claude-sonnet-4-6",
            input_tokens=1204,
            output_tokens=315,
            cache_creation_input_tokens=40,
            cache_read_input_tokens=900,
        )
    )

    (record,) = tracker.records
    assert record.model == "claude-sonnet-4-6"
    assert record.input_tokens == 1204
    assert record.output_tokens == 315
    assert record.cache_creation_input_tokens == 40
    assert record.cache_read_input_tokens == 900


def test_aggregates_multiple_responses() -> None:
    tracker = UsageTracker()
    tracker.record(_usage_response(input_tokens=100, output_tokens=10))
    tracker.record(
        _usage_response(
            input_tokens=200,
            output_tokens=20,
            cache_creation_input_tokens=1000,
        )
    )
    tracker.record(
        _usage_response(
            input_tokens=300,
            output_tokens=30,
            cache_read_input_tokens=2000,
        )
    )

    totals = UsageTotals.from_records(tracker.records)
    assert totals.call_count == 3
    assert totals.input_tokens == 600
    assert totals.output_tokens == 60
    assert totals.cache_creation_input_tokens == 1000
    assert totals.cache_read_input_tokens == 2000


def test_none_cache_fields_normalize_to_zero() -> None:
    # This is the shape of every extraction response: the fields exist on the
    # SDK model but hold None. Summing them without normalizing is a TypeError.
    tracker = UsageTracker()
    tracker.record(
        _usage_response(
            input_tokens=500,
            output_tokens=25,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )
    )

    (record,) = tracker.records
    assert record.cache_creation_input_tokens == 0
    assert record.cache_read_input_tokens == 0
    assert record.total_prompt_tokens == 500

    totals = UsageTotals.from_records(tracker.records)
    assert totals.cache_creation_input_tokens == 0
    assert totals.cache_read_input_tokens == 0


def test_aggregates_cache_read_tokens() -> None:
    records = [
        _record(cache_read_input_tokens=0),
        _record(cache_read_input_tokens=1240),
        _record(cache_read_input_tokens=8660),
    ]

    assert UsageTotals.from_records(records).cache_read_input_tokens == 9900


def test_aggregates_cache_creation_tokens() -> None:
    records = [
        _record(cache_creation_input_tokens=1240),
        _record(cache_creation_input_tokens=0),
        _record(cache_creation_input_tokens=0),
    ]

    assert UsageTotals.from_records(records).cache_creation_input_tokens == 1240


def test_total_prompt_tokens_sums_all_three_input_categories() -> None:
    # input_tokens alone is the uncached remainder, never the prompt size.
    records = [
        _record(
            input_tokens=12480,
            cache_creation_input_tokens=1240,
            cache_read_input_tokens=9900,
        )
    ]

    totals = UsageTotals.from_records(records)
    assert totals.total_prompt_tokens == 23620
    assert totals.total_prompt_tokens != totals.input_tokens


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_for_known_model() -> None:
    # claude-sonnet-4-6: $3.00 input, $3.75 cache write, $0.30 cache read,
    # $15.00 output, all per million tokens.
    #   0.2M x  3.00 = 0.600
    #   0.1M x  3.75 = 0.375
    #   0.4M x  0.30 = 0.120
    #  0.05M x 15.00 = 0.750
    #                = 1.845
    records = [
        _record(
            model="claude-sonnet-4-6",
            input_tokens=200_000,
            cache_creation_input_tokens=100_000,
            cache_read_input_tokens=400_000,
            output_tokens=50_000,
        )
    ]

    assert estimate_cost(records) == Decimal("1.845")


def test_unknown_model_cost_is_unavailable_not_zero() -> None:
    records = [_record(model="claude-experimental-9", input_tokens=2410, output_tokens=640)]

    report = build_report(records)
    assert report.estimated_cost_usd is None
    assert report.estimated_cost_usd != 0
    assert report.unpriced_models == ("claude-experimental-9",)
    # Tokens are model-independent and stay correct.
    assert report.totals.input_tokens == 2410
    assert report.totals.output_tokens == 640


def test_mixed_model_run_prices_each_record_at_its_own_rate() -> None:
    # Sonnet record as above (1.845) plus a haiku record at $1.00 in / $5.00 out.
    records = [
        _record(
            model="claude-sonnet-4-6",
            input_tokens=200_000,
            cache_creation_input_tokens=100_000,
            cache_read_input_tokens=400_000,
            output_tokens=50_000,
        ),
        _record(
            model="claude-haiku-4-5",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        ),
    ]

    report = build_report(records)
    assert report.estimated_cost_usd == Decimal("7.845")
    assert report.models == ("claude-haiku-4-5", "claude-sonnet-4-6")


def test_mixed_model_with_one_unpriced_yields_unavailable() -> None:
    records = [
        _record(model="claude-sonnet-4-6", input_tokens=1000, output_tokens=100),
        _record(model="claude-experimental-9", input_tokens=1000, output_tokens=100),
    ]

    report = build_report(records)
    assert report.estimated_cost_usd is None
    assert report.unpriced_models == ("claude-experimental-9",)
    assert report.totals.input_tokens == 2000


# ---------------------------------------------------------------------------
# MeteredClient
# ---------------------------------------------------------------------------


def test_metered_client_returns_the_original_response() -> None:
    tracker = UsageTracker()
    inner = MagicMock()
    response = _usage_response()
    inner.messages.create.return_value = response

    returned = MeteredClient(inner, tracker).messages.create(model="m", max_tokens=1)

    # The same object, not a copy or a wrapper.
    assert returned is response


def test_metered_client_records_once_per_call_and_forwards_kwargs() -> None:
    tracker = UsageTracker()
    inner = MagicMock()
    inner.messages.create.return_value = _usage_response()
    client = MeteredClient(inner, tracker)

    client.messages.create(model="claude-test-model", max_tokens=4096, system="hi")
    assert len(tracker.records) == 1

    client.messages.create(model="claude-test-model", max_tokens=4096)
    client.messages.create(model="claude-test-model", max_tokens=4096)
    assert len(tracker.records) == 3

    kwargs = inner.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-test-model"
    assert kwargs["max_tokens"] == 4096


def test_usage_survives_a_downstream_validation_failure() -> None:
    """A billed call still counts even when Pydantic rejects what it returned.

    This drives the real extract_candidate, so it also proves the wrapper is a
    drop-in for a client in the untouched pipeline modules.
    """

    tracker = UsageTracker()
    inner = MagicMock()
    bad_block = MagicMock(type="tool_use", input={"name": "Alice"})  # missing fields
    inner.messages.create.return_value = _usage_response(
        input_tokens=1204, output_tokens=315, content=[bad_block]
    )

    client = MeteredClient(inner, tracker)

    with pytest.raises(ValidationError):
        extract_candidate(client, resume_text="anything")

    assert len(tracker.records) == 1
    assert tracker.records[0].input_tokens == 1204


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def test_summary_names_every_token_category_and_says_estimated() -> None:
    report = build_report(
        [
            _record(
                model="claude-sonnet-4-6",
                input_tokens=12480,
                cache_creation_input_tokens=1240,
                cache_read_input_tokens=9900,
                output_tokens=8150,
            )
        ]
    )

    summary = format_summary(report)

    assert "Uncached input tokens:" in summary
    assert "Cache read tokens:" in summary
    assert "Cache creation tokens:" in summary
    assert "Total prompt tokens:" in summary
    assert "Output tokens:" in summary
    assert "Estimated" in summary
    # The distinction the whole report exists to make.
    assert "12,480" in summary
    assert "23,620" in summary
    # No invented cache-hit event count.
    assert "Cache hits" not in summary


def test_summary_names_the_model_when_pricing_is_unavailable() -> None:
    report = build_report([_record(model="claude-experimental-9", input_tokens=10)])

    summary = format_summary(report)

    assert "unavailable" in summary
    assert "claude-experimental-9" in summary
    assert "$" not in summary


def test_summary_is_plain_ascii() -> None:
    # Printed to a terminal that may not be UTF-8.
    summary = format_summary(build_report([_record(input_tokens=1, output_tokens=1)]))

    summary.encode("ascii")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_write_usage_saves_the_expected_json_structure(tmp_path: Path) -> None:
    report = build_report(
        [
            _record(
                model="claude-sonnet-4-6",
                input_tokens=200_000,
                cache_creation_input_tokens=100_000,
                cache_read_input_tokens=400_000,
                output_tokens=50_000,
            ),
            _record(model="claude-sonnet-4-6", input_tokens=1000, output_tokens=100),
        ]
    )

    out = tmp_path / "usage.json"
    write_usage(report, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["call_count"] == 2
    assert data["models"] == ["claude-sonnet-4-6"]
    assert data["totals"] == {
        "input_tokens": 201_000,
        "cache_read_input_tokens": 400_000,
        "cache_creation_input_tokens": 100_000,
        "total_prompt_tokens": 701_000,
        "output_tokens": 50_100,
    }
    assert data["estimated_cost_usd"] == pytest.approx(1.8495)
    assert data["unpriced_models"] == []
    assert "Estimated" in data["pricing_note"]
    assert len(data["calls"]) == 2
    assert data["calls"][0] == {
        "model": "claude-sonnet-4-6",
        "input_tokens": 200_000,
        "cache_read_input_tokens": 400_000,
        "cache_creation_input_tokens": 100_000,
        "output_tokens": 50_000,
    }


def test_write_usage_serializes_unknown_pricing_as_null(tmp_path: Path) -> None:
    report = build_report([_record(model="claude-experimental-9", input_tokens=2410)])

    out = tmp_path / "usage.json"
    write_usage(report, out)

    data = json.loads(out.read_text(encoding="utf-8"))
    # Explicitly null, never 0.0 — a zero would read as "this run was free".
    assert data["estimated_cost_usd"] is None
    assert data["unpriced_models"] == ["claude-experimental-9"]
    assert data["totals"]["input_tokens"] == 2410
