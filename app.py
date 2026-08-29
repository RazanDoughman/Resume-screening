import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from src.bias_auditor import VARIANT_DISPLAY_LABELS, run_bias_audit
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
from src.reporter import rank_candidates
from src.scorer import score_candidate
from src.usage import MeteredClient, UsageTracker, build_report, format_summary

load_dotenv()

Result = ScoredCandidate | ProcessingError


def _to_temp_pdf(uploaded) -> str:
    """Write one Streamlit upload to a temp file and return its path.

    pdfplumber reads from a path, not a buffer. The caller owns deletion.
    """

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.read())
        return tmp.name


def process_one(
    client, pdf_path, filename, jd_text, model, self_critique, bias_audit
) -> tuple[Result, BiasAuditReport | None]:
    """Run one resume through parse -> extract -> score (-> critique) (-> bias audit)."""

    # 1. Parse the PDF
    try:
        resume_text = parse_pdf(pdf_path)
    except Exception as e:
        return ProcessingError(source_file=filename, stage="parse", message=str(e)), None

    # 2. Extract candidate info
    try:
        profile = extract_candidate(client, resume_text, model=model)
    except Exception as e:
        return ProcessingError(source_file=filename, stage="extract", message=str(e)), None

    # 3. Score against the JD
    try:
        scored = score_candidate(client, profile, jd_text, filename, model=model)
    except Exception as e:
        return ProcessingError(source_file=filename, stage="score", message=str(e)), None

    # 4. Optional self-critique
    if self_critique:
        try:
            scored = critique_and_maybe_revise(client, scored, jd_text, model=model)
        except Exception:
            pass  # keep original scores if critique fails

    # 5. Optional bias audit
    audit: BiasAuditReport | None = None
    if bias_audit:
        try:
            audit = run_bias_audit(client, profile, jd_text, model=model)
        except Exception:
            pass  # scoring result is still valid if audit fails

    return scored, audit


# ── Page config ──
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Resume Matcher")
st.write("Upload a job description and resumes to score candidates.")

# ── Sidebar: settings ──
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Claude model", value=DEFAULT_MODEL)
    self_critique = st.checkbox("Enable self-critique (extra LLM call per resume)")
    bias_audit = st.checkbox(
        "Enable bias audit (5 extra LLM calls per resume)",
        help=(
            "Re-scores each candidate with demographic signals swapped "
            "(name, graduation year) and reports score drift per dimension."
        ),
    )
    deep_dive_top = st.number_input(
        "Deep-dive top N candidates",
        min_value=0,
        value=0,
        step=1,
        help=(
            "Write a hiring-manager briefing (summary, pros, cons, interview "
            "questions) for the top N ranked candidates. Adds N LLM calls in "
            "total, not N per resume. 0 disables it."
        ),
    )
    concurrency = st.number_input(
        "Resumes processed at once",
        min_value=1,
        value=5,
        step=1,
        help=(
            "How many resumes may be processed simultaneously. Higher is "
            "faster and makes no extra LLM calls; 1 processes them one at a "
            "time. Matches the CLI's --concurrency default of 5."
        ),
    )

# ── File uploads ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload JD (TXT)", type=["txt"])

with col2:
    st.subheader("Resumes")
    resume_files = st.file_uploader(
        "Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True
    )

# ── Run button ──
if st.button("Match Resumes", type="primary"):
    # Validate inputs
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        st.stop()

    if not jd_file:
        st.error("Please upload a job description PDF.")
        st.stop()

    if not resume_files:
        st.error("Please upload at least one resume PDF.")
        st.stop()

    # Read the JD text file
    jd_text = jd_file.read().decode("utf-8")

    # Show the parsed JD
    with st.expander("Parsed Job Description"):
        st.text(jd_text)

    # Process each resume. MeteredClient records the `usage` field of every
    # response so we can show the run's token counts and estimated cost below.
    # One client and one tracker for the whole run, shared by every worker —
    # the tracker is lock-protected, so concurrency needs nothing extra here.
    tracker = UsageTracker()
    client = MeteredClient(Anthropic(), tracker)
    pairs: list[tuple[Result, BiasAuditReport | None]] = []
    progress = st.progress(0, text="Processing resumes...")

    # Save every upload to a temp file first, on this thread. Reading an
    # UploadedFile is Streamlit's business, not a worker's, and pdfplumber needs
    # a real path either way. They are all removed in the `finally` below, which
    # also covers the case where something raises mid-batch.
    jobs = [(f.name, _to_temp_pdf(f)) for f in resume_files]

    def accept(index: int, name: str, outcome: tuple[Result, BiasAuditReport | None]) -> None:
        """Record and announce one finished candidate. Main thread only.

        Workers return their outcome and touch nothing shared; `pairs` and the
        progress bar are only ever mutated here. That matters twice over in
        Streamlit: it keeps `pairs` lock-free, and it keeps every `st.*` call on
        the thread that owns the script's run context.
        """

        pairs.append(outcome)
        progress.progress(index / len(jobs), text=f"Processed {name} ({index}/{len(jobs)})")

    try:
        # Same architecture as main.py, and for the same measured reason: the
        # first candidate runs alone so its scoring call writes the job
        # description's cached prefix before the rest fan out. Starting cold at
        # concurrency 5 had four of the first five scoring calls race and each
        # write its own copy. See prompts/challenge6.md.
        first_name, first_path = jobs[0]
        accept(
            1,
            first_name,
            process_one(
                client, first_path, first_name, jd_text, model, self_critique, bias_audit
            ),
        )

        # Concurrent execution, deterministic collection: futures are submitted
        # together but read back in submission order, never `as_completed()`,
        # so errors and bias-audit entries keep upload order between runs.
        with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
            futures = [
                executor.submit(
                    process_one, client, path, name, jd_text, model, self_critique, bias_audit
                )
                for name, path in jobs[1:]
            ]
            for i, ((name, _), future) in enumerate(zip(jobs[1:], futures), start=2):
                accept(i, name, future.result())
    finally:
        for _, path in jobs:
            os.unlink(path)

    progress.progress(1.0, text="Done!")

    # ── Split results into scored candidates and errors ──
    # Ranking comes from reporter.rank_candidates so the UI, the CLI and every
    # written report order candidates by the same rule, ties included.
    results = [r for r, _ in pairs]
    audits_by_file = {
        r.source_file: a
        for r, a in pairs
        if isinstance(r, ScoredCandidate) and a is not None
    }
    errors = [r for r in results if isinstance(r, ProcessingError)]
    scored = rank_candidates(results)
    scored_pairs = [(c, audits_by_file.get(c.source_file)) for c in scored]

    # ── Deep-dive the top N (post-scoring, same client, so it's metered) ──
    deep_dives: dict[str, DeepDiveReport] = {}
    if deep_dive_top > 0 and scored:
        selected = scored[: int(deep_dive_top)]
        with st.spinner(f"Writing deep-dive briefings for the top {len(selected)}..."):
            for c in selected:
                try:
                    deep_dives[c.source_file] = generate_deep_dive(
                        client, c, jd_text, model=model
                    )
                except Exception as e:
                    # Enrichment only — the scored candidate stays valid.
                    st.warning(f"Deep-dive failed for {c.source_file}: {e}")

    # ── Display ranked candidates ──
    if scored:
        st.subheader("Ranked Candidates")

        # Summary table
        table_data = []
        for rank, c in enumerate(scored, 1):
            table_data.append({
                "Rank": rank,
                "Candidate": c.profile.name,
                "Overall": c.overall_fit.score,
                "Skills": c.skills_match.score,
                "Experience": c.experience_match.score,
                "Role Relevance": c.role_relevance.score,
            })
        st.table(table_data)

        # Detailed view for each candidate
        for c, audit in scored_pairs:
            with st.expander(f"{c.profile.name} — {c.overall_fit.score}/100"):
                st.write(f"**Source:** {c.source_file}")
                st.write(f"**Years of experience:** {c.profile.years_experience}")
                st.write(f"**Skills:** {', '.join(c.profile.skills)}")

                st.write("---")
                st.write(f"**Overall reasoning:** {c.reasoning}")

                st.write("**Scores:**")
                st.write(f"- Skills match ({c.skills_match.score}): {c.skills_match.reasoning}")
                st.write(f"- Experience match ({c.experience_match.score}): {c.experience_match.reasoning}")
                st.write(f"- Role relevance ({c.role_relevance.score}): {c.role_relevance.reasoning}")
                st.write(f"- Overall fit ({c.overall_fit.score}): {c.overall_fit.reasoning}")

                if c.gaps:
                    st.write("**Gaps:**")
                    for gap in c.gaps:
                        st.write(f"- _{gap.category}_: {gap.detail}")

                if c.source_file in deep_dives:
                    dd = deep_dives[c.source_file]
                    st.write("---")
                    st.write("**Hiring Manager Deep-Dive**")
                    st.write(dd.summary)
                    for heading, items in (
                        ("Strengths", dd.pros),
                        ("Risks", dd.cons),
                        ("Suggested interview questions", dd.interview_questions),
                    ):
                        if items:
                            st.write(f"_{heading}_")
                            for item in items:
                                st.write(f"- {item}")

                if audit:
                    st.write("---")
                    st.write("**Bias Audit**")
                    st.caption(
                        f"Each row re-scores the same candidate with one demographic "
                        f"signal swapped. Threshold: {audit.drift_threshold} pts."
                    )

                    if audit.flagged:
                        st.warning(
                            f"Score drift detected — largest shift: "
                            f"{audit.max_score_drift:.0f} pts. "
                            + (audit.flag_reason or "")
                        )
                    else:
                        st.success(
                            f"Stable — max drift {audit.max_score_drift:.0f} pts "
                            f"(within {audit.drift_threshold} pt threshold)"
                        )

                    _DIMS = [
                        ("skills_match",     "Skills"),
                        ("experience_match", "Experience"),
                        ("role_relevance",   "Role"),
                        ("overall_fit",      "Overall"),
                    ]
                    base = {k: getattr(audit.baseline, k).score for k, _ in _DIMS}
                    rows = [
                        {
                            "Signal swapped": "Baseline (original resume)",
                            **{label: base[k] for k, label in _DIMS},
                            "Max Δ": 0,
                        }
                    ]
                    for v in audit.variants:
                        var_scores = {k: getattr(v.scores, k).score for k, _ in _DIMS}
                        max_delta = max(abs(var_scores[k] - base[k]) for k, _ in _DIMS)
                        rows.append({
                            "Signal swapped": VARIANT_DISPLAY_LABELS.get(v.label, v.label),
                            **{label: var_scores[k] for k, label in _DIMS},
                            "Max Δ": max_delta,
                        })
                    st.dataframe(
                        rows,
                        hide_index=True,
                        column_config={
                            "Max Δ": st.column_config.NumberColumn(
                                help="Largest score change vs baseline across all dimensions"
                            ),
                        },
                    )

    # ── Display errors ──
    if errors:
        st.subheader("Errors")
        for e in errors:
            st.error(f"**{e.source_file}** failed at `{e.stage}`: {e.message}")

    # ── API usage and estimated cost ──
    # Reuses the CLI's format_summary verbatim so the two entry points can't
    # drift. The app writes no files, so there is no usage.json here.
    st.subheader("API Usage")
    st.code(format_summary(build_report(tracker.records)))
