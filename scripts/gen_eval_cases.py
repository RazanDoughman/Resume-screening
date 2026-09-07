"""
Challenge 7 CLI: generate a set of synthetic resume proposals from a job
description.

Run: `uv run python -m scripts.gen_eval_cases --jd sample_data/sample_jd.txt`
     `uv run python -m scripts.gen_eval_cases --jd ... --dry-run`  (spends nothing)

This script is thin on purpose. Every piece of real work already exists and is
tested; the CLI's job is to read a file, build a client, call two functions in
order, and say where the output went. It reimplements nothing:

    src/eval_gen.py        builds and sends the four generation requests
    src/eval_fixtures.py   writes the proposal artifacts
    src/usage.py           counts what the run cost
    src/reporter.py        serializes the usage report

In particular it does **not** construct an API request. `generate_proposal_set`
does, one level at a time, with the job description first in a cached prefix
that all four calls share. Rebuilding any of that here would silently drop the
caching Stage 2 was designed around.

**What this produces is unreviewed.** Four proposals, written to disk, none of
them an eval case and none of them trusted. The CLI cannot mark anything
reviewed and cannot touch `tests/evals/cases.py` or `generated_cases.json` —
promoting a proposal takes a human and `src/generated_cases.py`, which this
script does not import. Generating is where it stops.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

from src.eval_fixtures import write_proposals
from src.eval_gen import (
    DEFAULT_MODEL,
    GENERATION_ORDER,
    EvalGenerationError,
    generate_proposal_set,
)
from src.reporter import write_usage
from src.usage import MeteredClient, UsageTracker, build_report, format_summary

ROOT = Path(__file__).resolve().parent.parent

# Beside the hand-written eval cases and Stage 4's generated_cases.json, so a
# reviewer finds everything in one directory — and, unlike `output/`, not
# gitignored. These artifacts exist to be read in a diff by whoever reviews
# them, which cannot happen if git never sees them.
DEFAULT_OUT_DIR = ROOT / "tests" / "evals" / "generated"

# Written beside the proposals by the same convention main.py uses: one run,
# one output directory, usage.json in it. Separate file, separate schema, and
# built by usage.py — the proposals never carry token counts.
USAGE_FILENAME = "usage.json"


def _read_jd(jd_path: str) -> str:
    """Read the job description, or raise with a message a user can act on.

    UTF-8 explicitly: a JD pasted from a job board routinely contains typographic
    dashes and quotes, and on Windows the default encoding is not UTF-8.
    """

    path = Path(jd_path)

    if not path.is_file():
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"JD file {jd_path} is not valid UTF-8: {exc}") from exc

    if not text.strip():
        raise ValueError(f"JD file {jd_path} is empty.")

    return text


def _display(path: Path) -> str:
    """A path relative to the repo when it is inside it, else absolute.

    Same courtesy scripts/benchmark.py extends when it prints its inputs:
    a reader recognises `tests/evals/generated` instantly and has to parse
    an absolute path with someone else's home directory in it.
    """

    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _print_plan(jd_path: str, jd: str, out_dir: Path, model: str) -> None:
    """Everything the run is about to do, before it does any of it.

    Printed on every invocation, not just `--dry-run`, so a real run announces
    its cost before spending it — the same courtesy scripts/benchmark.py
    extends.
    """

    print("Challenge 7 - synthetic eval-case generation")
    print("-" * 52)
    print(f"  jd               {jd_path}")
    print(f"  jd characters    {len(jd):,}")
    print(f"  model            {model}")
    print(f"  out-dir          {_display(out_dir)}")
    print(f"  levels           {', '.join(GENERATION_ORDER)}")
    print()
    print(
        f"  THIS COSTS MONEY: {len(GENERATION_ORDER)} separate LLM calls, one "
        f"per level,\n  issued sequentially so they share one cached prompt "
        f"prefix."
    )
    print()
    print(
        "  Output is UNREVIEWED. Nothing generated here becomes an eval case\n"
        "  until a human reviews it and marks it reviewed."
    )
    print()


def run(
    jd_path: str,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> int:
    """Generate and persist one proposal set. Returns an exit code.

    Split from `main()` the way `main.py` splits its own, so the orchestration
    can be tested without going through argparse.
    """

    out_dir = Path(out_dir)

    try:
        jd = _read_jd(jd_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    _print_plan(jd_path, jd, out_dir, model)

    # Everything above this line is reading and printing. The dry run stops
    # here, before the API-key check and before a client exists, so it works on
    # a machine with no credentials at all — which is the point of being able
    # to check the wiring without an account.
    if dry_run:
        print("--dry-run: no API calls made, no files written.")
        return 0

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.")
        return 1

    # The CLI owns bootstrapping. MeteredClient wraps the real client and
    # records the `usage` on every response, so the generation modules stay
    # unaware that anything is being counted — the Challenge 8 seam, reused
    # exactly as main.py and app.py use it.
    tracker = UsageTracker()
    client = MeteredClient(Anthropic(), tracker)

    try:
        proposals = generate_proposal_set(client, jd, model=model)
    except EvalGenerationError as exc:
        # Reported, not swallowed: the message names the level that failed and
        # the chained cause is intact for anyone who wants the traceback.
        print(f"\nERROR: generation failed - {exc}")
        # Calls before the failure were still billed, so say what they cost
        # rather than letting the spend go unreported. Nothing is written: the
        # set is generated in full before anything reaches disk, so a failure
        # leaves no partial set behind.
        print()
        print(format_summary(build_report(tracker.records)))
        print("\nNo proposals were written.")
        return 1

    # Created here, after generation has succeeded, rather than leaning on
    # write_proposals() to do it as a side effect: this function writes
    # usage.json into the same directory, and depending on another function's
    # mkdir for that is the kind of coupling that breaks the moment either one
    # moves. Same explicit makedirs main.py does before its own writes. After
    # generation, so a failed run leaves no empty directory behind.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Any failure here (a bad path, a full disk) propagates. A write that
    # half-succeeded is not something this script can meaningfully paper over,
    # and the traceback names the file.
    written = write_proposals(proposals, out_dir)

    usage_report = build_report(tracker.records)
    usage_path = out_dir / USAGE_FILENAME
    write_usage(usage_report, usage_path)

    print(f"Wrote {len(proposals)} proposals to {_display(out_dir)}/")
    for path in written:
        print(f"  {path.name}")
    print(f"  {usage_path.name}")

    print()
    print(format_summary(usage_report))

    print()
    print("These are UNREVIEWED proposals, not eval cases.")
    print("Read each one, then promote the ones you accept with")
    print("src.generated_cases.promote_proposal(..., reviewed=True).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic resume proposals at each match level for a "
            "job description. Output is unreviewed and does not enter the "
            "eval suite."
        ),
    )
    parser.add_argument(
        "--jd",
        type=str,
        required=True,
        help="Path to the job description (plain text, UTF-8).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=(
            f"Directory for the generated proposal artifacts "
            f"(default: {DEFAULT_OUT_DIR.relative_to(ROOT)})."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without making a single API call.",
    )
    args = parser.parse_args()

    return run(args.jd, args.out_dir, args.model, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
