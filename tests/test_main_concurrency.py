"""Concurrency tests for the resume loop in main.run().

Everything that would touch the network or a PDF is patched, so these run in
milliseconds with no API key. Fake stage functions use short sleeps to force a
specific interleaving; nothing here asserts on a duration, only on ordering and
on counts, so a slow machine changes nothing.

What these pin down is the part concurrency can silently break: that the pool
never runs wider than asked, that output stays byte-identical no matter what
order candidates finish in, that one bad resume still doesn't take the batch
down, and that every billed call still reaches the tracker.
"""

import argparse
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import main as main_module
from src.dimensions import DIMENSION_KEYS
from src.models import (
    CandidateProfile,
    DeepDiveReport,
    DimensionScore,
    Gap,
    Role,
    ScoredCandidate,
)
from src.usage import APIUsage, MeteredClient, UsageTracker, build_report

_RESUMES = ("a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf", "f.pdf")

# A real report, not a MagicMock: write_markdown renders these straight into
# report.md, so a mock would blow up in the reporter rather than in the code
# under test.
_REPORT = DeepDiveReport(
    summary="A briefing paragraph.",
    pros=["Owned the ledger"],
    cons=["No Go in production"],
    interview_questions=["Walk me through the replay logic.", "Why leave?"],
)


def _candidate(source: str, overall: int) -> ScoredCandidate:
    dimensions = {
        key: DimensionScore(score=overall, reasoning=f"{key} reasoning")
        for key in DIMENSION_KEYS
    }
    dimensions["overall_fit"] = DimensionScore(score=overall, reasoning="overall")
    return ScoredCandidate(
        profile=CandidateProfile(
            name=source.replace(".pdf", "").upper(),
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


def _usage_response() -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    response = MagicMock()
    response.model = "claude-sonnet-4-6"
    response.usage = usage
    response.content = []
    return response


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[str, str, str]:
    """A JD file, a resumes dir holding six PDFs, and an output dir."""

    jd = tmp_path / "jd.txt"
    jd.write_text("Senior Backend Engineer. Python, PostgreSQL.", encoding="utf-8")

    resumes = tmp_path / "resumes"
    resumes.mkdir()
    for name in _RESUMES:
        (resumes / name).write_bytes(b"%PDF-1.4 stub")

    return str(jd), str(resumes), str(tmp_path / "out")


def _run(
    workspace: tuple[str, str, str],
    concurrency: int,
    *,
    score=None,
    extract=None,
    deep_dive_top: int = 0,
    client: MagicMock | None = None,
) -> int:
    """Run main.run() with the pipeline mocked, at a given concurrency."""

    jd_path, resumes_dir, output_dir = workspace

    if score is None:

        def score(_client, _profile, _jd, source_file, model=None):
            return _candidate(source_file, 90)

    inner = client or MagicMock()
    if client is None:
        inner.messages.create.return_value = _usage_response()

    with (
        patch.object(main_module, "Anthropic", return_value=inner),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(
            main_module,
            "extract_candidate",
            side_effect=extract,
        )
        if extract
        else patch.object(main_module, "extract_candidate"),
        patch.object(main_module, "score_candidate", side_effect=score),
        patch.object(
            main_module, "generate_deep_dive", MagicMock(return_value=_REPORT)
        ),
    ):
        return main_module.run(
            jd_path,
            resumes_dir,
            output_dir,
            "claude-sonnet-4-6",
            False,
            False,
            False,
            deep_dive_top,
            concurrency,
        )


# ---------------------------------------------------------------------------
# A. The configured concurrency limit is never exceeded
# ---------------------------------------------------------------------------


class _Gauge:
    """Counts how many workers are inside the guarded region at once."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self.lock:
            self.current -= 1


def _gauged_score(gauge: _Gauge, hold: float = 0.05):
    def score(_client, _profile, _jd, source_file, model=None):
        gauge.enter()
        try:
            time.sleep(hold)
            return _candidate(source_file, 90)
        finally:
            gauge.leave()

    return score


@pytest.mark.parametrize("concurrency", [1, 2, 3])
def test_never_more_workers_in_flight_than_configured(workspace, concurrency) -> None:
    gauge = _Gauge()

    assert _run(workspace, concurrency, score=_gauged_score(gauge)) == 0
    assert gauge.peak <= concurrency


def test_concurrency_is_actually_used_not_just_capped(workspace) -> None:
    """The mirror of the test above: a cap of 3 must really run 3 at once.

    Without this, a pool that silently ran everything sequentially would pass
    every limit assertion in the file.
    """

    gauge = _Gauge()

    _run(workspace, 3, score=_gauged_score(gauge, hold=0.15))

    assert gauge.peak == 3


def test_concurrency_one_runs_strictly_one_at_a_time(workspace) -> None:
    gauge = _Gauge()

    _run(workspace, 1, score=_gauged_score(gauge))

    assert gauge.peak == 1


# ---------------------------------------------------------------------------
# B & C. Deterministic output regardless of concurrency or completion order
# ---------------------------------------------------------------------------


def _reversed_latency_score(_client, _profile, _jd, source_file, model=None):
    """Finish in reverse submission order: the last resume completes first."""

    delay = 0.02 * (len(_RESUMES) - _RESUMES.index(source_file))
    time.sleep(delay)
    return _candidate(source_file, 90)


def _outputs(output_dir: str) -> tuple[str, str]:
    return (
        Path(output_dir, "results.json").read_text(encoding="utf-8"),
        Path(output_dir, "report.md").read_text(encoding="utf-8"),
    )


def test_concurrency_one_matches_concurrent_output_byte_for_byte(workspace) -> None:
    """--concurrency 1 is the sequential path, not an approximation of it.

    Both runs use the same workspace so the JD path rendered into report.md is
    identical; the only variable is the concurrency argument.
    """

    _, _, output_dir = workspace

    _run(workspace, 1)
    sequential = _outputs(output_dir)

    _run(workspace, 5)
    concurrent = _outputs(output_dir)

    assert sequential == concurrent


def test_reversed_completion_order_leaves_outputs_unchanged(workspace) -> None:
    """Candidates finishing backwards must not reorder anything on disk."""

    _, _, output_dir = workspace

    _run(workspace, 5, score=_reversed_latency_score)
    scrambled = _outputs(output_dir)

    _run(workspace, 1)
    sequential = _outputs(output_dir)

    assert scrambled == sequential


def test_errors_keep_input_order_when_candidates_finish_backwards(workspace) -> None:
    """The reason collection is submission-ordered rather than as_completed().

    Scored candidates are re-sorted by rank_candidates() and would survive
    either way. ProcessingErrors are not sorted anywhere — the reporter emits
    them in list order — so completion-order collection would shuffle them.
    """

    def score(_client, _profile, _jd, source_file, model=None):
        # Every resume fails, slowest first, so completion order is reversed.
        time.sleep(0.02 * (len(_RESUMES) - _RESUMES.index(source_file)))
        raise RuntimeError(f"scoring down for {source_file}")

    _, _, output_dir = workspace

    assert _run(workspace, 5, score=score) == 0

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert [e["source_file"] for e in data["errors"]] == list(_RESUMES)
    assert {e["stage"] for e in data["errors"]} == {"score"}


# ---------------------------------------------------------------------------
# D. Failure isolation survives concurrency
# ---------------------------------------------------------------------------


def test_one_failed_candidate_does_not_abort_the_batch(workspace) -> None:
    def score(_client, _profile, _jd, source_file, model=None):
        if source_file == "c.pdf":
            raise RuntimeError("scoring blew up")
        return _candidate(source_file, 90)

    _, _, output_dir = workspace

    assert _run(workspace, 5, score=score) == 0

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert len(data["candidates"]) == len(_RESUMES) - 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["source_file"] == "c.pdf"
    assert data["errors"][0]["stage"] == "score"


def test_an_unexpected_worker_exception_still_surfaces(workspace) -> None:
    """Concurrency preserves the failure contract; it must not widen it.

    process_one turns expected parse/extract/score failures into a
    ProcessingError. Anything else — a real bug — killed the sequential run and
    must still kill this one rather than being laundered into a ProcessingError
    that hides it.
    """

    with patch.object(
        main_module, "process_one", side_effect=KeyError("a genuine bug")
    ):
        with pytest.raises(KeyError, match="a genuine bug"):
            _run(workspace, 5)


# ---------------------------------------------------------------------------
# E. CLI validation
# ---------------------------------------------------------------------------


def test_concurrency_flag_defaults_to_five() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=main_module._positive_int, default=5)

    assert parser.parse_args([]).concurrency == 5


@pytest.mark.parametrize("value,expected", [("1", 1), ("5", 5), ("64", 64)])
def test_positive_integers_are_accepted(value: str, expected: int) -> None:
    assert main_module._positive_int(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "-20"])
def test_zero_and_negatives_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="1 or greater"):
        main_module._positive_int(value)


@pytest.mark.parametrize("value", ["abc", "2.5", "", "five"])
def test_non_integers_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="not an integer"):
        main_module._positive_int(value)


def test_zero_is_valid_for_deep_dive_but_not_for_concurrency() -> None:
    """The asymmetry is deliberate: 0 briefings is a request, 0 workers is not."""

    assert main_module._non_negative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        main_module._positive_int("0")


# ---------------------------------------------------------------------------
# F. The shared UsageTracker under concurrent writers
# ---------------------------------------------------------------------------


def _record(tokens: int = 1) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = tokens
    usage.output_tokens = tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    response = MagicMock()
    response.model = "claude-sonnet-4-6"
    response.usage = usage
    return response


def test_tracker_loses_no_records_under_concurrent_writers() -> None:
    tracker = UsageTracker()
    threads = [
        threading.Thread(target=lambda: [tracker.record(_record()) for _ in range(200)])
        for _ in range(8)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(tracker.records) == 1600


def test_totals_are_correct_after_concurrent_recording() -> None:
    tracker = UsageTracker()
    threads = [
        threading.Thread(target=lambda: [tracker.record(_record(3)) for _ in range(100)])
        for _ in range(4)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    totals = build_report(tracker.records).totals
    assert totals.call_count == 400
    assert totals.input_tokens == 1200
    assert totals.output_tokens == 1200


def test_records_snapshot_is_immutable_and_consistent() -> None:
    """A snapshot taken mid-run must not change under the caller's feet."""

    tracker = UsageTracker()
    tracker.record(_record())
    snapshot = tracker.records

    tracker.record(_record())

    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1
    assert len(tracker.records) == 2


def test_snapshot_taken_while_writers_run_is_never_torn() -> None:
    """Not timing-sensitive: any snapshot must be a valid prefix-length tuple.

    The assertion is on structure, not on a particular count — a snapshot may
    legitimately catch any number of records. What it may never be is short of
    a complete tuple of well-formed APIUsage objects.
    """

    tracker = UsageTracker()
    stop = threading.Event()

    def write() -> None:
        while not stop.is_set():
            tracker.record(_record())

    writers = [threading.Thread(target=write) for _ in range(4)]
    for t in writers:
        t.start()

    try:
        for _ in range(200):
            snapshot = tracker.records
            assert isinstance(snapshot, tuple)
            assert all(isinstance(r, APIUsage) for r in snapshot)
    finally:
        stop.set()
        for t in writers:
            t.join()


# ---------------------------------------------------------------------------
# G. Call accounting end to end through the real MeteredClient
# ---------------------------------------------------------------------------


def test_every_concurrent_call_reaches_usage_json(workspace) -> None:
    """Concurrency changes when calls happen, never how many are recorded."""

    inner = MagicMock()
    inner.messages.create.side_effect = lambda **kwargs: _usage_response()

    def score(_client, _profile, _jd, source_file, model=None):
        # One real call through the metered client per candidate.
        _client.messages.create(model="claude-sonnet-4-6", max_tokens=1, messages=[])
        return _candidate(source_file, 90)

    _, _, output_dir = workspace

    assert _run(workspace, 5, score=score, client=inner) == 0

    usage = json.loads(Path(output_dir, "usage.json").read_text(encoding="utf-8"))
    assert usage["call_count"] == len(_RESUMES)
    assert usage["totals"]["input_tokens"] == 100 * len(_RESUMES)


def test_the_metered_client_is_shared_by_every_worker(workspace) -> None:
    """One client, one tracker — not one per candidate."""

    seen: list[int] = []

    def score(_client, _profile, _jd, source_file, model=None):
        assert isinstance(_client, MeteredClient)
        seen.append(id(_client))
        return _candidate(source_file, 90)

    _run(workspace, 5, score=score)

    assert len(seen) == len(_RESUMES)
    assert len(set(seen)) == 1


# ---------------------------------------------------------------------------
# H. Deep-dive stays sequential and stays after the pool
# ---------------------------------------------------------------------------


def test_deep_dive_runs_only_after_every_candidate_is_scored(workspace) -> None:
    """Concurrency must not let a briefing start mid-batch: top-N needs the field."""

    order: list[str] = []
    lock = threading.Lock()

    def score(_client, _profile, _jd, source_file, model=None):
        time.sleep(0.02)
        with lock:
            order.append("score")
        return _candidate(source_file, 90)

    def brief(_client, candidate, _jd, model=None):
        with lock:
            order.append("brief")
        return _REPORT

    jd_path, resumes_dir, output_dir = workspace
    inner = MagicMock()
    inner.messages.create.return_value = _usage_response()

    with (
        patch.object(main_module, "Anthropic", return_value=inner),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(main_module, "score_candidate", side_effect=score),
        patch.object(main_module, "generate_deep_dive", side_effect=brief),
    ):
        main_module.run(
            jd_path, resumes_dir, output_dir, "m", False, False, False, 2, 5
        )

    assert order.count("score") == len(_RESUMES)
    assert order.count("brief") == 2
    last_score = max(i for i, step in enumerate(order) if step == "score")
    first_brief = min(i for i, step in enumerate(order) if step == "brief")
    assert last_score < first_brief


def test_deep_dive_briefings_are_not_concurrent(workspace) -> None:
    """The briefing loop is sequential by design — no nested pool.

    Deep-dive has its own cached prefix; a pool around it would reintroduce the
    write race that priming fixes upstream.
    """

    gauge = _Gauge()

    def brief(_client, candidate, _jd, model=None):
        gauge.enter()
        try:
            time.sleep(0.05)
            return _REPORT
        finally:
            gauge.leave()

    jd_path, resumes_dir, output_dir = workspace
    inner = MagicMock()
    inner.messages.create.return_value = _usage_response()

    with (
        patch.object(main_module, "Anthropic", return_value=inner),
        patch.object(main_module, "parse_pdf", return_value="resume text"),
        patch.object(main_module, "extract_candidate"),
        patch.object(
            main_module,
            "score_candidate",
            side_effect=lambda _c, _p, _j, s, model=None: _candidate(s, 90),
        ),
        patch.object(main_module, "generate_deep_dive", side_effect=brief),
    ):
        main_module.run(
            jd_path, resumes_dir, output_dir, "m", False, False, False, 4, 5
        )

    assert gauge.peak == 1


# ---------------------------------------------------------------------------
# Cache priming: candidate one runs alone, then the rest fan out
#
# These replace an earlier guard, `test_no_candidate_is_run_alone_before_the_
# others`, which asserted the exact opposite. That guard existed to keep the
# naive all-at-once architecture in place long enough to benchmark it; the
# benchmark measured four scorer cache writes instead of one, so the guard had
# done its job and the assertion now runs the other way. Two contradictory
# tests must not coexist. See prompts/challenge6.md for the measurements.
# ---------------------------------------------------------------------------


def _traced_score(events: list[tuple[str, str]], lock: threading.Lock, hold=0.05):
    """A scorer that records when each candidate's scoring starts and ends."""

    def score(_client, _profile, _jd, source_file, model=None):
        with lock:
            events.append(("start", source_file))
        time.sleep(hold)
        with lock:
            events.append(("end", source_file))
        return _candidate(source_file, 90)

    return score


def test_first_candidate_scores_alone_before_any_other_starts(workspace) -> None:
    """The priming guarantee, stated as orchestration rather than as caching.

    What we can test offline is ordering: candidate one's scoring call must
    finish before any other candidate's begins. Whether the API then serves a
    cached prefix is Anthropic's business and the live benchmark's to measure.
    """

    events: list[tuple[str, str]] = []
    lock = threading.Lock()

    _run(workspace, 5, score=_traced_score(events, lock))

    first_end = events.index(("end", _RESUMES[0]))
    later_starts = [
        i
        for i, (kind, source) in enumerate(events)
        if kind == "start" and source != _RESUMES[0]
    ]

    assert events[0] == ("start", _RESUMES[0])
    assert first_end < min(later_starts)


def test_exactly_one_candidate_is_in_flight_during_priming(workspace) -> None:
    gauge = _Gauge()
    peaks: list[int] = []

    def score(_client, _profile, _jd, source_file, model=None):
        gauge.enter()
        try:
            time.sleep(0.05)
            if source_file == _RESUMES[0]:
                peaks.append(gauge.peak)
            return _candidate(source_file, 90)
        finally:
            gauge.leave()

    _run(workspace, 5, score=score)

    assert peaks == [1]


def test_the_rest_still_fan_out_after_the_prime(workspace) -> None:
    """Priming must not quietly serialize the whole batch.

    The failure mode this catches is a refactor that primes correctly and then
    forgets to submit the remainder concurrently — cache-healthy, and as slow
    as the sequential pipeline it replaced.
    """

    gauge = _Gauge()

    _run(workspace, 5, score=_gauged_score(gauge, hold=0.15))

    # 6 resumes: 1 primes alone, the other 5 fan out at concurrency 5.
    assert gauge.peak == len(_RESUMES) - 1


def test_priming_still_respects_the_concurrency_limit(workspace) -> None:
    gauge = _Gauge()

    _run(workspace, 2, score=_gauged_score(gauge, hold=0.05))

    assert gauge.peak <= 2


# ---------------------------------------------------------------------------
# Priming edge cases
# ---------------------------------------------------------------------------


def test_a_single_resume_is_processed_by_the_prime_alone(tmp_path: Path) -> None:
    """One resume means one candidate: the pool is created and submits nothing."""

    jd = tmp_path / "jd.txt"
    jd.write_text("Senior Backend Engineer.", encoding="utf-8")
    resumes = tmp_path / "resumes"
    resumes.mkdir()
    (resumes / "solo.pdf").write_bytes(b"%PDF-1.4 stub")
    output_dir = str(tmp_path / "out")

    assert _run((str(jd), str(resumes), output_dir), 5) == 0

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["source_file"] == "solo.pdf"
    assert data["errors"] == []


def test_no_resumes_still_exits_one_without_priming_anything(tmp_path: Path) -> None:
    """The empty-directory guard fires before pdfs[0] is ever indexed."""

    jd = tmp_path / "jd.txt"
    jd.write_text("Senior Backend Engineer.", encoding="utf-8")
    resumes = tmp_path / "resumes"
    resumes.mkdir()

    score = MagicMock()
    with (
        patch.object(main_module, "Anthropic", return_value=MagicMock()),
        patch.object(main_module, "score_candidate", score),
    ):
        code = main_module.run(
            str(jd), str(resumes), str(tmp_path / "out"), "m", False, False, False, 0, 5
        )

    assert code == 1
    score.assert_not_called()


def test_a_failed_primer_still_lets_the_rest_of_the_batch_run(workspace) -> None:
    """Candidate one fails; the remaining candidates fan out normally.

    The chosen behaviour: do not promote another candidate to primer. A failed
    prime means the cache may not have been written, so the run degrades to the
    naive cache profile — more cache writes, identical results. That is a cost,
    never a wrong answer, and the alternative is a retry loop that serializes
    the entire batch when every resume is bad.
    """

    def score(_client, _profile, _jd, source_file, model=None):
        if source_file == _RESUMES[0]:
            raise RuntimeError("primer scoring failed")
        return _candidate(source_file, 90)

    _, _, output_dir = workspace

    assert _run(workspace, 5, score=score) == 0

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert len(data["candidates"]) == len(_RESUMES) - 1
    assert len(data["errors"]) == 1
    assert data["errors"][0]["source_file"] == _RESUMES[0]
    assert data["errors"][0]["stage"] == "score"


def test_a_primer_that_fails_at_parse_does_not_stop_the_fan_out(workspace) -> None:
    """The earliest possible prime failure: no LLM call happened at all."""

    def parse(path):
        if path.endswith(_RESUMES[0]):
            raise ValueError("PDF contains no text.")
        return "resume text"

    jd_path, resumes_dir, output_dir = workspace
    inner = MagicMock()
    inner.messages.create.return_value = _usage_response()

    with (
        patch.object(main_module, "Anthropic", return_value=inner),
        patch.object(main_module, "parse_pdf", side_effect=parse),
        patch.object(main_module, "extract_candidate"),
        patch.object(
            main_module,
            "score_candidate",
            side_effect=lambda _c, _p, _j, s, model=None: _candidate(s, 90),
        ),
    ):
        code = main_module.run(
            jd_path, resumes_dir, output_dir, "m", False, False, False, 0, 5
        )

    assert code == 0

    data = json.loads(Path(output_dir, "results.json").read_text(encoding="utf-8"))
    assert len(data["candidates"]) == len(_RESUMES) - 1
    assert data["errors"][0]["stage"] == "parse"


def test_the_primer_is_the_first_file_in_sorted_order(workspace) -> None:
    """Which candidate primes must be deterministic, not incidental."""

    seen: list[str] = []
    lock = threading.Lock()

    def score(_client, _profile, _jd, source_file, model=None):
        with lock:
            seen.append(source_file)
        return _candidate(source_file, 90)

    _run(workspace, 5, score=score)

    assert seen[0] == _RESUMES[0]
