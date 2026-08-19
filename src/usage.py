"""
Cost reporter: count the tokens every Claude call actually used, and estimate
what the run cost.

The pipeline modules (`extractor.py`, `scorer.py`, `critique.py`,
`bias_auditor.py`) all take an Anthropic client as their first argument and all
reach it the same way: `client.messages.create(...)`. That single uniform seam
is where this module plugs in. `MeteredClient` wraps a real client, records
`response.usage` the instant a response comes back, and returns that same
response unchanged. Nothing downstream knows it happened — no tuple returns, no
tracker parameters threaded through five modules, and no usage fields on any
candidate model in `src/models.py`.

Recording happens *before* the caller pulls the tool_use block out and hands it
to Pydantic. That ordering is the whole point: an API call that returns a
response you were billed for, and then fails `model_validate`, still shows up in
the report. The report describes API work observed during the run, not just the
candidates that came out the other end.

Three things about the numbers are easy to get wrong:

1. `usage.input_tokens` is the *uncached* input only. When prompt caching is
   working (scorer.py and critique.py cache the job description), the rest of
   the prompt is reported separately as cache reads and cache writes. The real
   prompt size is `total_prompt_tokens` — the sum of all three.
2. `cache_creation_input_tokens` and `cache_read_input_tokens` are `Optional`
   in the SDK and arrive as `None`, not `0`, whenever caching isn't involved.
   `APIUsage.from_response` is the one place that normalizes them.
3. The API reports cached *token counts*, not a count of cache-hit events. So
   this module reports token counts and does not invent a "cache hits" number.

Cost is an estimate, never an invoice — see MODEL_PRICING below.

Known limitation: a request that never returns a response (connection error, or
a timeout after the SDK's internal retries) has no `usage` to read, so it cannot
be counted here even though the server may have processed it.
"""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Sequence

from anthropic import Anthropic
from anthropic.types import Message

# ---------------------------------------------------------------------------
# Per-call usage record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class APIUsage:
    """Normalized token usage from one API response.

    Frozen on purpose: a record can never be edited after the fact, so
    combining records from several sources is always safe. `model` is the
    *resolved* model from `response.model`, not the model that was requested —
    that makes the record self-describing and is what pricing keys off.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    @property
    def total_prompt_tokens(self) -> int:
        """Every prompt token that was processed, cached or not."""

        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @classmethod
    def from_response(cls, response: Message) -> "APIUsage":
        """Read `response.usage` into a record.

        This is the only place in the codebase that touches `usage`, and the
        only place that has to think about `None`. The two cache fields are
        `Optional[int]` in the SDK and are `None` — not `0` — on any call that
        didn't involve the cache, which is every extraction call. `or 0` rather
        than a `getattr` default because the attributes exist; they just hold
        `None`.
        """

        usage = response.usage
        return cls(
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageTotals:
    """Token sums across a set of records. Pure arithmetic, no pricing."""

    call_count: int
    input_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    output_tokens: int

    @property
    def total_prompt_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @classmethod
    def from_records(cls, records: Sequence[APIUsage]) -> "UsageTotals":
        return cls(
            call_count=len(records),
            input_tokens=sum(r.input_tokens for r in records),
            cache_read_input_tokens=sum(r.cache_read_input_tokens for r in records),
            cache_creation_input_tokens=sum(
                r.cache_creation_input_tokens for r in records
            ),
            output_tokens=sum(r.output_tokens for r in records),
        )


class UsageTracker:
    """Collects one APIUsage record per API response.

    Deliberately append-only: totals are derived from the records on demand
    rather than maintained as running counters, so there is no read-modify-write
    state to get wrong. There is no module-level state anywhere in this file —
    a tracker is created per run, which is also what makes two runs in one
    process (and the tests below) work.

    Not synchronized. When concurrency arrives, either guard `record()` with a
    lock or give each worker its own tracker and pass the concatenated records
    to `build_report` — which already accepts any sequence.
    """

    def __init__(self) -> None:
        self._records: list[APIUsage] = []

    def record(self, response: Message) -> APIUsage:
        """Normalize and store the usage from one response."""

        usage = APIUsage.from_response(response)
        self._records.append(usage)
        return usage

    @property
    def records(self) -> tuple[APIUsage, ...]:
        """An immutable snapshot — callers can't mutate the tracker's state."""

        return tuple(self._records)


# ---------------------------------------------------------------------------
# The metered client
# ---------------------------------------------------------------------------


class _MeteredMessages:
    """Stands in for `client.messages`, recording usage on the way through."""

    def __init__(self, messages: Any, tracker: UsageTracker) -> None:
        self._messages = messages
        self._tracker = tracker

    def create(self, *args: Any, **kwargs: Any) -> Message:
        response = self._messages.create(*args, **kwargs)
        # Record first. Everything the caller does next — pulling the tool_use
        # block out, running model_validate — can raise, and this call was
        # billed either way.
        self._tracker.record(response)
        return response


class MeteredClient:
    """An Anthropic client that counts what it spends.

    Swap this in for `Anthropic()` at the entry point and every LLM call in the
    pipeline is metered, with no change to any pipeline module. The pipeline
    still annotates its parameter as `client: Anthropic`; annotations aren't
    enforced, and `.messages` is the only attribute any of those modules ever
    touch.

    Only `.messages` is exposed, on purpose. A catch-all passthrough would make
    a future `client.beta.messages.create(...)` work while silently escaping
    metering; here it raises AttributeError instead, which is the failure we'd
    rather have.
    """

    def __init__(self, client: Anthropic, tracker: UsageTracker) -> None:
        self._client = client
        self._tracker = tracker
        self.messages = _MeteredMessages(client.messages, tracker)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens, one rate per billing category."""

    input_per_mtok: Decimal
    cache_write_5m_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    output_per_mtok: Decimal


# Prices in USD per million tokens.
#
# Source:   https://platform.claude.com/docs/en/about-claude/pricing
# Verified: 2026-08-18
#
# Written out as four explicit rates rather than a base rate plus multipliers so
# this table can be diffed line-for-line against the published pricing page. For
# reference, the published multipliers relative to base input are: 5-minute
# cache write 1.25x, 1-hour cache write 2x, cache read 0.1x.
#
# Cache writes are priced at the 5-minute rate. This project only ever uses the
# default 5-minute TTL — see the `cache_control` blocks in scorer.py and
# critique.py, neither of which sets `ttl`. If a 1-hour TTL is ever introduced,
# split cache_creation_input_tokens using the SDK's
# `usage.cache_creation.ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`
# breakdown instead of pricing the whole field at one rate.
#
# Decimals are built from strings, never floats: Decimal(3.00) would inherit
# float imprecision at construction.
MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-6": ModelPricing(
        input_per_mtok=Decimal("3.00"),
        cache_write_5m_per_mtok=Decimal("3.75"),
        cache_read_per_mtok=Decimal("0.30"),
        output_per_mtok=Decimal("15.00"),
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_mtok=Decimal("1.00"),
        cache_write_5m_per_mtok=Decimal("1.25"),
        cache_read_per_mtok=Decimal("0.10"),
        output_per_mtok=Decimal("5.00"),
    ),
    "claude-opus-4-8": ModelPricing(
        input_per_mtok=Decimal("5.00"),
        cache_write_5m_per_mtok=Decimal("6.25"),
        cache_read_per_mtok=Decimal("0.50"),
        output_per_mtok=Decimal("25.00"),
    ),
}

_TOKENS_PER_MTOK = Decimal(1_000_000)

# Rounding applied only at the JSON boundary. Six places is micro-dollar
# resolution — enough that a single cheap call doesn't round away to zero.
_COST_QUANTUM = Decimal("0.000001")

_PRICING_NOTE = (
    "Estimated from a local pricing table (Anthropic list prices, standard "
    "tier, 5-minute cache TTL). Not an invoice."
)


def _record_cost(record: APIUsage, pricing: ModelPricing) -> Decimal:
    """Cost of one API call, using that call's own model's rates."""

    return (
        Decimal(record.input_tokens) * pricing.input_per_mtok
        + Decimal(record.cache_creation_input_tokens) * pricing.cache_write_5m_per_mtok
        + Decimal(record.cache_read_input_tokens) * pricing.cache_read_per_mtok
        + Decimal(record.output_tokens) * pricing.output_per_mtok
    ) / _TOKENS_PER_MTOK


def unpriced_models(records: Sequence[APIUsage]) -> tuple[str, ...]:
    """Distinct resolved models with no entry in MODEL_PRICING, sorted."""

    return tuple(sorted({r.model for r in records if r.model not in MODEL_PRICING}))


def estimate_cost(records: Sequence[APIUsage]) -> Decimal | None:
    """Estimated USD for these calls, or None if any model is unpriced.

    Each record is priced with its own model's rates and the results summed, so
    a run that used more than one model costs correctly with no extra work.

    All-or-nothing on unknown models is deliberate. A partial total printed next
    to a complete token count reads as the run's full cost and would be
    believed; `None` plus the model name is the honest answer. Token totals are
    model-independent and stay correct either way.
    """

    if unpriced_models(records):
        return None
    return sum(
        (_record_cost(r, MODEL_PRICING[r.model]) for r in records),
        start=Decimal("0"),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageReport:
    """Everything the terminal summary and usage.json need, already computed."""

    totals: UsageTotals
    models: tuple[str, ...]
    unpriced_models: tuple[str, ...]
    estimated_cost_usd: Decimal | None
    records: tuple[APIUsage, ...]

    def to_dict(self) -> dict[str, Any]:
        """The usage.json schema.

        Every key is always present, including `unpriced_models` (empty when
        everything priced) and `estimated_cost_usd` (null when it isn't), so a
        consumer never needs to guard for missing fields.

        This is also the one place Decimal leaves the module: json.dumps can't
        serialize Decimal, and a JSON number is what a benchmark script wants to
        diff, so the value is quantized and converted to float here and nowhere
        else.
        """

        cost = self.estimated_cost_usd
        return {
            "call_count": self.totals.call_count,
            "models": list(self.models),
            "totals": {
                "input_tokens": self.totals.input_tokens,
                "cache_read_input_tokens": self.totals.cache_read_input_tokens,
                "cache_creation_input_tokens": self.totals.cache_creation_input_tokens,
                "total_prompt_tokens": self.totals.total_prompt_tokens,
                "output_tokens": self.totals.output_tokens,
            },
            "estimated_cost_usd": (
                None
                if cost is None
                else float(cost.quantize(_COST_QUANTUM, rounding=ROUND_HALF_UP))
            ),
            "unpriced_models": list(self.unpriced_models),
            "pricing_note": _PRICING_NOTE,
            "calls": [
                {
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "cache_read_input_tokens": r.cache_read_input_tokens,
                    "cache_creation_input_tokens": r.cache_creation_input_tokens,
                    "output_tokens": r.output_tokens,
                }
                for r in self.records
            ],
        }


def build_report(records: Sequence[APIUsage]) -> UsageReport:
    """Turn raw records into totals, models, and an estimated cost.

    Takes a plain sequence rather than a tracker so a report can be built from
    hand-written records in a test — and so several trackers' records can be
    concatenated without any new API.
    """

    records = tuple(records)
    return UsageReport(
        totals=UsageTotals.from_records(records),
        models=tuple(sorted({r.model for r in records})),
        unpriced_models=unpriced_models(records),
        estimated_cost_usd=estimate_cost(records),
        records=records,
    )


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

_LABEL_WIDTH = 23
_MIN_VALUE_WIDTH = 8
_MIN_RULE_WIDTH = 32

# Plain ASCII for the rule, matching the histogram in reporter.py: this string
# is printed to a terminal, and a box-drawing character raises
# UnicodeEncodeError on a console that isn't UTF-8.
_RULE_CHAR = "-"


def format_summary(report: UsageReport) -> str:
    """Render a UsageReport as the block printed at the end of a run.

    Reads only the report — never the pricing table, never raw records — so the
    numbers can be tested without parsing any of this text.
    """

    totals = report.totals
    rows = [
        ("API calls:", f"{totals.call_count:,}"),
        ("Uncached input tokens:", f"{totals.input_tokens:,}"),
        ("Cache read tokens:", f"{totals.cache_read_input_tokens:,}"),
        ("Cache creation tokens:", f"{totals.cache_creation_input_tokens:,}"),
        ("Total prompt tokens:", f"{totals.total_prompt_tokens:,}"),
        ("Output tokens:", f"{totals.output_tokens:,}"),
    ]

    cost = report.estimated_cost_usd
    cost_value = None if cost is None else f"${cost:.4f}"

    widths = [len(value) for _, value in rows]
    if cost_value is not None:
        widths.append(len(cost_value))
    value_width = max(_MIN_VALUE_WIDTH, *widths)

    lines = ["API Usage", _RULE_CHAR * max(_MIN_RULE_WIDTH, _LABEL_WIDTH + value_width)]

    # "Model:" gets an extra space so both forms start in the same column.
    if len(report.models) == 1:
        lines.append(f"Model:  {report.models[0]}")
        lines.append("")
    elif report.models:
        lines.append(f"Models: {', '.join(report.models)}")
        lines.append("")

    for label, value in rows:
        lines.append(f"{label:<{_LABEL_WIDTH}}{value:>{value_width}}")
    lines.append("")

    if cost_value is not None:
        lines.append(f"{'Estimated cost:':<{_LABEL_WIDTH}}{cost_value:>{value_width}}")
    else:
        named = ", ".join(f"'{m}'" for m in report.unpriced_models)
        lines.append(f"Estimated cost:  unavailable - no pricing entry for {named}")

    return "\n".join(lines)
