"""
CLI entry point. Runs each resume through the pipeline:
parse -> extract -> score -> (optional critique) -> (optional bias audit).

Resumes are processed concurrently — `--concurrency N`, default 5 — because
they are independent of one another. The first candidate runs alone so that
its scoring call writes the job description's cached prefix before the rest
fan out; see the comment in `run()` for what that buys. Everything that needs
the whole field (ranking, top-N selection, the deep-dive stage, the usage
report, and every file written) stays sequential, after the pool has drained.

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
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

from src.bias_auditor import run_bias_audit
from src.critique import critique_and_maybe_revise
from src.deep_dive import generate_deep_dive
from src.extractor import DEFAULT_MODEL, extract_candidate
from src.models import (
    BiasAuditReport,
    DeepDiveReport,
    ProcessingError,
    ScoredCandidate,
)
from src.pdf_parser import parse_pdf
from src.reporter import (
    rank_candidates,
    write_csv,
    write_json,
    write_markdown,
    write_usage,
)
from src.scorer import score_candidate
from src.usage import MeteredClient, UsageTracker, build_report, format_summary

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
    csv: bool,
    deep_dive_top: int = 0,
    concurrency: int = 1,
) -> int:
    """Score every resume in `resumes_dir` against `jd_path`. Returns an exit code.

    `concurrency` is how many resumes are processed at once. It defaults to 1 —
    the original sequential loop — so every existing caller that omits it keeps
    exactly the behaviour it had. The CLI's own default is 5; see `main()`.
    """

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
    # Deep-dive is priced separately because it is the one stage that doesn't
    # multiply by the resume count: it runs once per selected candidate.
    if deep_dive_top > 0:
        print(
            f"Deep-dive: up to {deep_dive_top} extra LLM call"
            f"{'' if deep_dive_top == 1 else 's'} total (top-{deep_dive_top} "
            f"candidates only, not per resume)."
        )

    # MeteredClient wraps the real client and records the `usage` field of every
    # response. The pipeline modules are untouched — they still just call
    # client.messages.create(...).
    tracker = UsageTracker()
    client = MeteredClient(Anthropic(), tracker)
    results: list[Result] = []
    audits: dict[str, BiasAuditReport] = {}

    # Resumes are independent of each other, so `process_one` is the unit of
    # concurrency: one thread owns one candidate from PDF to score. Threads, not
    # asyncio — every function in the pipeline is synchronous, pdfplumber blocks,
    # and the work is pure I/O wait on the API.
    #
    # EXECUTION is concurrent. COLLECTION is deterministic.
    #
    # Every future is submitted up front so they all run at once, but results are
    # read back by iterating `futures` in submission order — deliberately NOT
    # `as_completed()`. Scored candidates would survive either way, since
    # `rank_candidates()` re-sorts them; ProcessingError entries would not. They
    # are emitted in list order by the reporter, so completion-order collection
    # would shuffle the "Could not process" table and the errors array in
    # results.json between runs of identical input. Same for `audits` insertion
    # order, which bias_audit.json iterates. `as_completed()` buys nothing here —
    # nothing downstream streams — so it would trade reproducible output for no
    # gain. Leave this as it is.
    #
    # A slow candidate therefore holds back the progress lines of faster ones
    # behind it. That is the cost of printing in a fixed order, and it is worth it.
    print(f"Processing {len(pdfs)} resumes (concurrency {concurrency})...")

    def accept(index: int, pdf: str, outcome: tuple[Result, BiasAuditReport | None]) -> None:
        """Record and announce one finished candidate. Main thread only.

        Workers just return their outcome; every mutation of `results` and
        `audits`, and every line printed, happens here. That is why neither
        collection needs a lock and why progress lines never interleave.
        """

        result, audit = outcome
        pdf_name = os.path.basename(pdf)

        if isinstance(result, ScoredCandidate):
            print(
                f"[{index}/{len(pdfs)}] {pdf_name}: "
                f"scored {result.overall_fit.score} ({result.profile.name})"
            )
        else:
            print(f"[{index}/{len(pdfs)}] {pdf_name}: failed at {result.stage}")

        results.append(result)
        if audit is not None:
            audits[pdf_name] = audit

    # Prime the scorer's prompt cache with the first *real* candidate, alone.
    #
    # scorer.py sends the job description as a cached prefix, so the first
    # scoring call of a run writes that entry and every later one reads it.
    # Sequentially that ordering came for free. Fanning out from cold does not:
    # measured at concurrency 5, four of the first five scoring calls raced,
    # each wrote its own copy of the prefix, and the run paid the 1.25x write
    # premium four times instead of once (benchmarks/parallel-naive.json).
    #
    # Running candidate one to completion first fixes that with ordering alone —
    # no warm-up request, no latch, no change to a single prompt byte. Its own
    # extraction is serialized as a side effect, which is the whole cost of this;
    # the work itself is real work that had to happen anyway.
    accept(1, pdfs[0], process_one(client, pdfs[0], jd, model, self_critique, bias_audit))

    # If candidate one failed, the cache may not have been written and the rest
    # of the batch simply behaves as it did before priming: more cache writes,
    # same results. That degradation is a cost, never a wrong answer, so it is
    # left alone deliberately. Promoting the next candidate to primer would mean
    # a retry loop that serializes the whole batch when every resume is bad, and
    # "did the cache get written?" is not even answerable here without reading
    # token counts — which would make this function cache-aware and break the
    # boundary that keeps metering out of the pipeline.
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                process_one, client, pdf, jd, model, self_critique, bias_audit
            )
            for pdf in pdfs[1:]
        ]

        # An unexpected exception inside a worker surfaces here at .result(),
        # ending the run exactly as it would have in the sequential loop —
        # process_one already converts every *expected* failure into a
        # ProcessingError, and widening that here would swallow real bugs.
        for i, (pdf, future) in enumerate(zip(pdfs[1:], futures), start=2):
            accept(i, pdf, future.result())

    # Deep-dive the top N. A post-scoring batch stage, not part of process_one:
    # "top N" is undefined until the whole field has been scored and ranked.
    #
    # This must stay above build_report — these calls go through the same
    # MeteredClient, but only the responses recorded before the report is built
    # can appear in it.
    deep_dives: dict[str, DeepDiveReport] = {}
    if deep_dive_top > 0:
        ranked = rank_candidates(results)
        selected = ranked[:deep_dive_top]

        if selected:
            print(f"\nDeep-dive: briefing top {len(selected)} of {len(ranked)}...")

            # Strict N: a candidate tied with the last one picked is still cut,
            # so the cost stays exactly what the flag promised. Say so rather
            # than letting a filename quietly decide it.
            cutoff = selected[-1].overall_fit.score
            tied = [c for c in ranked[len(selected):] if c.overall_fit.score == cutoff]
            if tied:
                print(
                    f"  note: {', '.join(c.source_file for c in tied)} also "
                    f"scored {cutoff} but was not included "
                    f"(raise --deep-dive-top to {len(selected) + len(tied)})."
                )
        else:
            print("\nDeep-dive: no scored candidates to brief.")

        for c in selected:
            try:
                deep_dives[c.source_file] = generate_deep_dive(
                    client, c, jd, model=model
                )
                print(f"  {c.source_file}: briefed ({c.profile.name})")
            except Exception as e:
                # Enrichment only. A failed briefing leaves the scored
                # candidate untouched and never becomes a ProcessingError.
                print(f"  {c.source_file}: deep-dive failed ({e}) — skipping")

    # Covers every API response that came back, including calls for resumes that
    # later failed — the report is a record of API work, not of successes.
    usage_report = build_report(tracker.records)

    os.makedirs(output_dir, exist_ok=True)
    write_json(results, os.path.join(output_dir, "results.json"))
    write_markdown(
        results,
        os.path.join(output_dir, "report.md"),
        jd_path,
        audits=audits or None,
        deep_dives=deep_dives or None,
    )
    write_usage(usage_report, os.path.join(output_dir, "usage.json"))
    if csv:
        write_csv(results, os.path.join(output_dir, "results.csv"))

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

    print()
    print(format_summary(usage_report))

    print(f"\nDone. Output written to {output_dir}/")
    return 0


def _non_negative_int(value: str) -> int:
    """argparse type for --deep-dive-top.

    Rejects negatives rather than passing them to a list slice, where `[:-1]`
    silently means "all but the last" — a wrong answer that would spend money
    before anyone noticed. Zero is allowed and means off.
    """

    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.")
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or greater, got {n}.")
    return n


def _positive_int(value: str) -> int:
    """argparse type for --concurrency.

    Rejects 0, unlike `_non_negative_int` above. The asymmetry is real, not an
    oversight: "brief nobody" is a coherent request, so `--deep-dive-top 0` means
    off, but "use zero workers" is a typo, not an instruction — there is no run
    it could describe. ThreadPoolExecutor(max_workers=0) raises anyway; failing
    here gives a readable message instead of a traceback after the API-key check.

    `--concurrency 1` is the sequential path, not a special case of it.
    """

    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer.")
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {n}.")
    return n


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
        help="Directory to write the report files (default: ./output).",
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
    parser.add_argument(
        "--deep-dive-top",
        type=_non_negative_int,
        default=0,
        metavar="N",
        help=(
            "Write a hiring-manager briefing (summary, pros, cons, interview "
            "questions) for the top N ranked candidates, in report.md. Adds N "
            "LLM calls in total, not N per resume. Default 0 (off)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=5,
        metavar="N",
        help=(
            "How many resumes to process at once (default: 5). Each resume is "
            "independent, so this is wall-clock only — it does not change how "
            "many LLM calls the run makes. Use 1 for the sequential path."
        ),
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help=(
            "Also write results.csv — one flat row per scored candidate, "
            "for spreadsheets. No extra LLM calls."
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
        args.csv,
        args.deep_dive_top,
        args.concurrency,
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