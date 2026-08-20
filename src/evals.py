"""
Eval harness for the extract + score pipeline.

Runs every case in `tests/evals/cases.py` through the real extractor and
scorer, compares each dimension to its expected score range, and prints
a pass/fail table. Exits nonzero on any failure.

Run: `uv run python -m src.evals`
Run bias cases only: `uv run python -m src.evals --bias`
Run deep-dive cases only: `uv run python -m src.evals --deep-dive`

Why ranges instead of exact scores: the LLM gives slightly different
numbers each time it's called. A strong match should *always* score 80+,
but 82 vs 88 is indistinguishable noise. Ranges test a property of the
output, not a frozen value — the right way to test an LLM pipeline.
"""

import argparse
import os
import sys

from anthropic import Anthropic

from src.bias_auditor import NAME_SWAPS, run_bias_audit
from src.deep_dive import generate_deep_dive
from src.extractor import DEFAULT_MODEL, extract_candidate
from src.scorer import score_candidate
from tests.evals.cases import (
    ALL_BIAS_CASES,
    ALL_CASES,
    ALL_DEEP_DIVE_CASES,
    BiasAuditEvalCase,
    DeepDiveEvalCase,
    EvalCase,
)


def run_case(client: Anthropic, case: EvalCase, model: str) -> bool:
    """Run one scoring eval case. Return True if every dimension passed."""
    try:
        profile = extract_candidate(client, case.resume_text, model=model)
        scored = score_candidate(
            client, profile, case.jd, source_file=f"{case.name}.eval", model=model
        )
    except Exception as e:
        print(f"\n[FAIL] {case.name}")
        print(f"       {case.description}")
        print(f"       ERROR: {e}")
        return False

    dimensions = [
        ("skills_match", scored.skills_match.score, case.expected_skills_match),
        ("experience_match", scored.experience_match.score, case.expected_experience_match),
        ("role_relevance", scored.role_relevance.score, case.expected_role_relevance),
        ("overall_fit", scored.overall_fit.score, case.expected_overall_fit),
    ]

    all_passed = all(lo <= score <= hi for _, score, (lo, hi) in dimensions)
    banner = "PASS" if all_passed else "FAIL"

    print(f"\n[{banner}] {case.name}")
    print(f"       {case.description}")
    for name, score, (lo, hi) in dimensions:
        mark = "✓" if lo <= score <= hi else "✗"
        print(f"  {mark} {name:<20} {score:>3}  expected {lo}–{hi}")

    return all_passed


def run_bias_case(client: Anthropic, case: BiasAuditEvalCase, model: str) -> bool:
    """Run one bias audit eval case. Return True if drift and flag checks passed.

    Uses only NAME_SWAPS (4 variants) to keep eval cost manageable — name is
    the most common source of demographic bias and the fastest to probe.
    """
    try:
        profile = extract_candidate(client, case.resume_text, model=model)
        audit = run_bias_audit(
            client, profile, case.jd, model=model, swaps=NAME_SWAPS
        )
    except Exception as e:
        print(f"\n[FAIL] {case.name}")
        print(f"       {case.description}")
        print(f"       ERROR: {e}")
        return False

    drift_ok = audit.max_score_drift <= case.expected_max_drift
    flag_ok = audit.flagged == case.expected_flagged

    all_passed = drift_ok and flag_ok
    banner = "PASS" if all_passed else "FAIL"

    print(f"\n[{banner}] {case.name} (bias audit)")
    print(f"       {case.description}")

    drift_mark = "✓" if drift_ok else "✗"
    print(
        f"  {drift_mark} max_score_drift      "
        f"{audit.max_score_drift:.0f} pts  "
        f"expected ≤{case.expected_max_drift}"
    )

    flag_mark = "✓" if flag_ok else "✗"
    print(
        f"  {flag_mark} flagged              "
        f"{audit.flagged}  "
        f"expected {case.expected_flagged}"
    )

    if audit.flag_reason:
        print(f"       reason: {audit.flag_reason}")

    for v in audit.variants:
        drift = max(
            abs(getattr(audit.baseline, d).score - getattr(v.scores, d).score)
            for d in ("skills_match", "experience_match", "role_relevance", "overall_fit")
        )
        print(f"       {v.label:<28} max Δ {drift} pts")

    return all_passed


def run_deep_dive_case(client: Anthropic, case: DeepDiveEvalCase, model: str) -> bool:
    """Run one deep-dive eval case. Return True if every structural check passed.

    Costs three calls: extract, score, then the briefing itself — the briefing
    reads a ScoredCandidate, so the first two are setup rather than the thing
    under test.

    The checks are floors and a keyword probe, never exact prose. What they
    catch is a briefing that has gone generic or dropped its risks; what they
    tolerate is every legitimate way a paragraph can be worded.
    """

    try:
        profile = extract_candidate(client, case.resume_text, model=model)
        scored = score_candidate(
            client, profile, case.jd, source_file=f"{case.name}.eval", model=model
        )
        report = generate_deep_dive(client, scored, case.jd, model=model)
    except Exception as e:
        print(f"\n[FAIL] {case.name}")
        print(f"       {case.description}")
        print(f"       ERROR: {e}")
        return False

    summary_words = len(report.summary.split())
    briefing = " ".join(
        [report.summary, *report.pros, *report.cons, *report.interview_questions]
    ).lower()
    mentioned = [t for t in case.must_mention_any if t.lower() in briefing]

    checks = [
        ("summary words", summary_words, f">={case.min_summary_words}",
         summary_words >= case.min_summary_words),
        ("pros", len(report.pros), f">={case.min_pros}",
         len(report.pros) >= case.min_pros),
        ("cons", len(report.cons), f">={case.min_cons}",
         len(report.cons) >= case.min_cons),
        ("interview questions", len(report.interview_questions),
         f">={case.min_questions}",
         len(report.interview_questions) >= case.min_questions),
        ("candidate-specific terms", len(mentioned), ">=1", bool(mentioned)),
    ]

    all_passed = all(ok for *_, ok in checks)
    banner = "PASS" if all_passed else "FAIL"

    print(f"\n[{banner}] {case.name} (deep dive)")
    print(f"       {case.description}")
    for label, value, expected, ok in checks:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label:<26} {value:>3}  expected {expected}")

    if mentioned:
        print(f"       mentioned: {', '.join(mentioned)}")

    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline evals.")
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Run only bias audit eval cases (skips scoring cases).",
    )
    parser.add_argument(
        "--deep-dive",
        action="store_true",
        help="Run only deep-dive eval cases (skips scoring and bias cases).",
    )
    args = parser.parse_args()

    # A selector flag narrows the run to its own group. With none given, every
    # group runs.
    run_all = not (args.bias or args.deep_dive)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.")
        return 1

    client = Anthropic()
    model = DEFAULT_MODEL
    total_passed = 0
    total_cases = 0

    if run_all:
        print(f"Running {len(ALL_CASES)} scoring eval cases against {model}...")
        passed = sum(run_case(client, c, model) for c in ALL_CASES)
        print(f"\n{passed}/{len(ALL_CASES)} scoring cases passed")
        total_passed += passed
        total_cases += len(ALL_CASES)

    if run_all or args.bias:
        print(f"\nRunning {len(ALL_BIAS_CASES)} bias audit eval cases against {model}...")
        passed = sum(run_bias_case(client, c, model) for c in ALL_BIAS_CASES)
        print(f"\n{passed}/{len(ALL_BIAS_CASES)} bias cases passed")
        total_passed += passed
        total_cases += len(ALL_BIAS_CASES)

    if run_all or args.deep_dive:
        print(
            f"\nRunning {len(ALL_DEEP_DIVE_CASES)} deep-dive eval cases "
            f"against {model}..."
        )
        passed = sum(run_deep_dive_case(client, c, model) for c in ALL_DEEP_DIVE_CASES)
        print(f"\n{passed}/{len(ALL_DEEP_DIVE_CASES)} deep-dive cases passed")
        total_passed += passed
        total_cases += len(ALL_DEEP_DIVE_CASES)

    return 0 if total_passed == total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
