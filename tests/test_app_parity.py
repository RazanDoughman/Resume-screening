"""Parity checks between the CLI and the Streamlit entry point.

`app.py` is a Streamlit script: its orchestration lives at module level inside
`if st.button(...)`, so it cannot be imported and called without a Streamlit
runtime. Standing up that runtime to test six lines of thread-pool wiring would
cost far more than it proves, and the wiring itself is already covered — the
same architecture is exercised end to end against `main.run()` in
tests/test_main_concurrency.py.

So this file does two cheap, honest things instead:

1. Tests `_to_temp_pdf`, the one genuinely importable helper the concurrency
   change added.
2. Reads app.py as source and asserts the architectural decisions match the
   CLI's. Source inspection is an established pattern here — see
   tests/test_deep_dive.py, which asserts `"Anthropic()" not in source`. It
   cannot prove the code runs, but it does catch the failure that actually
   matters between two hand-maintained entry points: one of them drifting.

CLAUDE.md is explicit that main.py and app.py do not share a helper and that
both must be updated together. These tests are the tripwire for forgetting.
"""

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_APP = (_ROOT / "app.py").read_text(encoding="utf-8")
_MAIN = (_ROOT / "main.py").read_text(encoding="utf-8")


def _call_names(source: str) -> list[str]:
    """Every function name actually *called* in this source.

    Parsed rather than grepped. Both entry points discuss `as_completed()` and
    `ThreadPoolExecutor(max_workers=0)` in comments and docstrings explaining
    why they are not used — string matching would flag that prose as a call.
    """

    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


# ---------------------------------------------------------------------------
# The importable helper
# ---------------------------------------------------------------------------


class _FakeUpload:
    """Stands in for a Streamlit UploadedFile: a name and a read()."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.name = "resume.pdf"

    def read(self) -> bytes:
        return self._payload


def test_to_temp_pdf_writes_the_upload_and_returns_a_path() -> None:
    from app import _to_temp_pdf

    path = Path(_to_temp_pdf(_FakeUpload(b"%PDF-1.4 payload")))
    try:
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.read_bytes() == b"%PDF-1.4 payload"
    finally:
        path.unlink()


def test_each_upload_gets_its_own_temp_file() -> None:
    """Workers run concurrently, so two candidates must never share a path."""

    from app import _to_temp_pdf

    first = Path(_to_temp_pdf(_FakeUpload(b"one")))
    second = Path(_to_temp_pdf(_FakeUpload(b"two")))
    try:
        assert first != second
        assert first.read_bytes() == b"one"
        assert second.read_bytes() == b"two"
    finally:
        first.unlink()
        second.unlink()


# ---------------------------------------------------------------------------
# Architectural parity with main.py
# ---------------------------------------------------------------------------


def test_app_uses_a_thread_pool() -> None:
    assert "from concurrent.futures import ThreadPoolExecutor" in _APP
    assert "ThreadPoolExecutor(max_workers=" in _APP


def test_app_primes_with_the_first_candidate_then_fans_out_the_rest() -> None:
    """The Stage 2 architecture, mirrored: jobs[0] alone, jobs[1:] pooled."""

    assert "jobs[0]" in _APP
    assert "for name, path in jobs[1:]" in _APP

    prime_at = _APP.index("jobs[0]")
    pool_at = _APP.index("ThreadPoolExecutor(max_workers=")
    assert prime_at < pool_at, "the prime must come before the pool is opened"


def test_app_collects_in_submission_order() -> None:
    assert "zip(jobs[1:], futures)" in _APP


_ENTRY_POINTS = {"app.py": _APP, "main.py": _MAIN}


@pytest.mark.parametrize("label", sorted(_ENTRY_POINTS))
def test_neither_entry_point_uses_as_completed(label: str) -> None:
    """Comments may mention it; no entry point may call it.

    Completion-order collection would shuffle ProcessingError entries and
    bias-audit insertion order between runs of identical input.
    """

    assert "as_completed" not in _call_names(_ENTRY_POINTS[label]), (
        f"{label} calls as_completed"
    )


@pytest.mark.parametrize("label", sorted(_ENTRY_POINTS))
def test_neither_entry_point_nests_a_second_pool(label: str) -> None:
    assert _call_names(_ENTRY_POINTS[label]).count("ThreadPoolExecutor") == 1, label


def test_app_deep_dive_stays_sequential() -> None:
    """Deep-dive keeps its own cached prefix; a pool around it would race."""

    deep_dive_at = _APP.index("for c in selected:")
    pool_at = _APP.index("ThreadPoolExecutor(max_workers=")
    assert pool_at < deep_dive_at, "deep-dive must run after the pool, not inside it"

    # Only one pool exists at all (asserted above), so "after it" means the
    # briefing loop cannot be inside one.
    assert _call_names(_APP).count("ThreadPoolExecutor") == 1


def test_app_shares_one_client_and_one_tracker() -> None:
    """No per-worker metering: same seam the CLI uses."""

    assert _APP.count("UsageTracker()") == 1
    assert _APP.count("MeteredClient(Anthropic(), tracker)") == 1


def test_concurrency_defaults_match_between_entry_points() -> None:
    """A user switching between the CLI and the UI should get the same speed."""

    cli = re.search(r'"--concurrency".*?default=(\d+)', _MAIN, re.DOTALL)
    assert cli, "could not find the CLI --concurrency default"

    # (?<![_a-z]) so this does not match the `value=` inside `min_value=1`.
    app = re.search(
        r'"Resumes processed at once".*?(?<![_a-z])value=(\d+)', _APP, re.DOTALL
    )
    assert app, "could not find the Streamlit concurrency default"

    assert cli.group(1) == app.group(1) == "5"


def test_app_concurrency_control_has_a_floor_of_one() -> None:
    control = _APP[_APP.index('"Resumes processed at once"') :][:400]
    assert "min_value=1" in control


def test_app_does_not_import_the_cli_orchestration() -> None:
    """Both entry points own their own loop, by design (see CLAUDE.md).

    Importing main.run() into the UI would drag argparse-era assumptions —
    a resumes *directory*, an output directory, exit codes — into a path whose
    input is a list of uploads.
    """

    assert "import main" not in _APP
    assert "from main import" not in _APP
