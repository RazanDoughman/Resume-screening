"""
CLI entry point. Processes resumes one at a time through the pipeline:
parse -> extract -> score -> (optional critique) -> (optional bias audit).

If any stage fails for a given resume, we record a ProcessingError and
keep going — one bad PDF shouldn't crash the whole run.

Optional quality knobs (opt-in so students don't burn their API budget):
- `--self-critique`: add one more LLM call that reviews the scores.
- `--bias-audit`: re-score each candidate with demographic signals swapped
  (name, graduation year, location) and report score drift.
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

from src.bias_auditor import run_bias_audit
from src.critique import critique_and_maybe_revise
from src.extractor import DEFAULT_MODEL, extract_candidate
from src.models import BiasAuditReport, ProcessingError, ScoredCandidate
from src.pdf_parser import parse_pdf
from src.reporter import write_json, write_markdown
from src.scorer import score_candidate

Result = ScoredCandidate | ProcessingError


def process_one(
    client: Anthropic,
    pdf_path: str,
    jd: str,
    model: str,
    self_critique: bool,
    bias_audit: bool,
) -> tuple[Result, BiasAuditReport | None]:
    """Run one resume through parse -> extract -> score (-> critique) (-> bias audit).

    If any stage fails, return a ProcessingError tagged with which stage.
    The second element of the tuple is the BiasAuditReport, or None if the
    audit was not requested or failed.
    """
    name = os.path.basename(pdf_path)

    # 1. Parse the PDF
    try:
        text = parse_pdf(pdf_path)
    except Exception as e:
        return ProcessingError(source_file=name, stage="parse", message=str(e)), None

    # 2. Extract candidate info (LLM call #1)
    try:
        profile = extract_candidate(client, text, model=model)
    except Exception as e:
        print(f"  ERROR in extract: {e}")
        return ProcessingError(source_file=name, stage="extract", message=str(e)), None

    # 3. Score against the JD (LLM call #2)
    try:
        scored = score_candidate(client, profile, jd, name, model=model)
    except Exception as e:
        return ProcessingError(source_file=name, stage="score", message=str(e)), None

    # 4. Optional self-critique pass (LLM call #3)
    if self_critique:
        try:
            scored = critique_and_maybe_revise(client, scored, jd, model=model)
        except Exception as e:
            # If the critique itself fails, keep the original scores rather
            # than throwing away valid work.
            print(f"  {name}: critique failed ({e}) — keeping original scores")

    # 5. Optional bias audit — re-scores with swapped demographic signals.
    audit: BiasAuditReport | None = None
    if bias_audit:
        try:
            audit = run_bias_audit(client, profile, jd, model=model)
            if audit.flagged:
                print(f"  {name}: bias audit FLAGGED — {audit.flag_reason}")
            else:
                print(f"  {name}: bias audit OK (max drift {audit.max_score_drift:.0f} pts)")
        except Exception as e:
            print(f"  {name}: bias audit failed ({e}) — skipping")

    return scored, audit


def run(
    jd_path: str,
    resumes_dir: str,
    output_dir: str,
    model: str,
    self_critique: bool,
    bias_audit: bool,
) -> int:
    import json

    with open(jd_path, "r") as f:
        jd = f.read()
    pdfs = sorted(
        os.path.join(resumes_dir, f)
        for f in os.listdir(resumes_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdfs:
        print(f"No PDFs found in {resumes_dir}")
        return 1

    # Count how many LLM calls we'll make for each resume:
    #   1 extract + 1 score + 1 critique (opt) + N swaps for audit (opt)
    calls_per_resume = 2
    if self_critique:
        calls_per_resume += 1
    if bias_audit:
        from src.bias_auditor import ALL_SWAPS
        calls_per_resume += len(ALL_SWAPS)
    print(
        f"Found {len(pdfs)} resumes. "
        f"~{calls_per_resume} LLM calls per resume "
        f"(extract=1, score=1"
        + (", critique=1" if self_critique else "")
        + (f", bias-audit={calls_per_resume - 2 - self_critique}" if bias_audit else "")
        + ")."
    )

    client = Anthropic()
    results: list[Result] = []
    audits: dict[str, BiasAuditReport] = {}

    for i, pdf in enumerate(pdfs, start=1):
        pdf_name = os.path.basename(pdf)
        print(f"[{i}/{len(pdfs)}] {pdf_name}: processing...")
        result, audit = process_one(client, pdf, jd, model, self_critique, bias_audit)

        if isinstance(result, ScoredCandidate):
            print(
                f"[{i}/{len(pdfs)}] {pdf_name}: "
                f"scored {result.overall_fit.score} ({result.profile.name})"
            )
        else:
            print(f"[{i}/{len(pdfs)}] {pdf_name}: failed at {result.stage}")

        results.append(result)
        if audit is not None:
            audits[pdf_name] = audit

    os.makedirs(output_dir, exist_ok=True)
    write_json(results, os.path.join(output_dir, "results.json"))
    write_markdown(results, os.path.join(output_dir, "report.md"), jd_path, audits=audits or None)

    if audits:
        audit_path = os.path.join(output_dir, "bias_audit.json")
        with open(audit_path, "w") as f:
            json.dump(
                [{"source_file": k, **v.model_dump()} for k, v in audits.items()],
                f,
                indent=2,
            )
        flagged = sum(1 for v in audits.values() if v.flagged)
        print(f"Bias audit: {flagged}/{len(audits)} candidates flagged. See {audit_path}")

    print(f"\nDone. Output written to {output_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score resume PDFs against a job description.",
    )
    parser.add_argument(
        "--jd",
        type=str,
        required=True,
        help="Path to the job description (plain text).",
    )
    parser.add_argument(
        "--resumes",
        type=str,
        required=True,
        help="Directory containing resume PDFs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory to write results.json and report.md (default: ./output).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--self-critique",
        action="store_true",
        help="Add a second-pass LLM call that reviews and may revise scores.",
    )
    parser.add_argument(
        "--bias-audit",
        action="store_true",
        help=(
            "Re-score each candidate with demographic signals swapped "
            "(name, graduation year, location) and report score drift. "
            "Adds one LLM call per swap variant per resume."
        ),
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.")
        return 1

    if not os.path.isfile(args.jd):
        print(f"JD file not found: {args.jd}")
        return 1

    if not os.path.isdir(args.resumes):
        print(f"Resumes directory not found: {args.resumes}")
        return 1

    return run(
        args.jd,
        args.resumes,
        args.output,
        args.model,
        args.self_critique,
        args.bias_audit,
    )


if __name__ == "__main__":
    sys.exit(main())

"""


main.py -> args (jd, resumes, output, model, self critique ) -> run -
for each pdfs:
1. parse_pdf -> text
2. extract_candidate (LLM call #1) -> CandidateProfile
3. score_candidate (LLM call #2) -> ScoredCandidate
4. optional critique_and_maybe_revise (LLM call #3) -> maybe revised

"""