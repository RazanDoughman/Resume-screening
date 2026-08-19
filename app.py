import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic

from src.bias_auditor import VARIANT_DISPLAY_LABELS, run_bias_audit
from src.critique import critique_and_maybe_revise
from src.extractor import DEFAULT_MODEL, extract_candidate
from src.models import BiasAuditReport, ProcessingError, ScoredCandidate
from src.pdf_parser import parse_pdf
from src.scorer import score_candidate
from src.usage import MeteredClient, UsageTracker, build_report, format_summary

load_dotenv()

Result = ScoredCandidate | ProcessingError


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
    tracker = UsageTracker()
    client = MeteredClient(Anthropic(), tracker)
    pairs: list[tuple[Result, BiasAuditReport | None]] = []
    progress = st.progress(0, text="Processing resumes...")

    for i, resume_file in enumerate(resume_files):
        progress.progress(
            i / len(resume_files),
            text=f"Processing {resume_file.name} ({i + 1}/{len(resume_files)})...",
        )

        # Save uploaded PDF to a temp file so pdfplumber can read it
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resume_file.read())
            tmp_path = tmp.name

        result, audit = process_one(
            client, tmp_path, resume_file.name, jd_text, model, self_critique, bias_audit
        )
        os.unlink(tmp_path)
        pairs.append((result, audit))

    progress.progress(1.0, text="Done!")

    # ── Split results into scored candidates and errors ──
    scored_pairs = [(r, a) for r, a in pairs if isinstance(r, ScoredCandidate)]
    errors = [r for r, _ in pairs if isinstance(r, ProcessingError)]

    # Sort by overall score, highest first
    scored_pairs.sort(key=lambda x: x[0].overall_fit.score, reverse=True)
    scored = [r for r, _ in scored_pairs]

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
