"""
Challenge 7: write generated resume proposals to disk, and read them back.

This module is to `eval_gen.py` what `reporter.py` is to the scoring pipeline:
it turns validated objects into files and does nothing else. No LLM calls, no
Anthropic import, no generation, no ranking, no judgment about whether a
proposal is any good. It exists as its own module for the same reason
`reporter.py` does — a stage that makes API calls should not also own the
filesystem, and `tests/test_eval_gen.py` pins that `generate_resume_proposal()`
writes nothing.

**What is written here is not an eval case.** It is an unreviewed proposal an
LLM produced, and the layout says so at every level a reader might look at:
the directory is named `generated/`, every Markdown file opens with a banner
saying the numbers below are unverified, and nothing here imports, reads or
touches `tests/evals/cases.py` or `ALL_CASES`. Promotion to a trusted case
requires a human, and that step is deliberately not implemented in this
module.

Two outputs per run, both written from the same already-validated objects:

  proposals.json   The contract. An exact `model_dump(mode="json")` of every
                   proposal, in generation order, and the only file read back
                   by `read_proposals()`.
  NN-<level>.md    One per proposal, for the human doing the review: the
                   resume as readable Markdown under a header carrying the
                   proposal's stated intent.

That split is `reporter.py`'s exactly — `results.json` for machines,
`report.md` for people, both rendered from one in-memory source so they cannot
disagree, and only the JSON is ever parsed back. The Markdown is a view.

Everything is deterministic: identical proposals produce byte-identical files.
Ordering comes from the input list, filenames from that list's index and the
proposal's own level, and nothing embeds a timestamp, a hostname, an absolute
path, or a random value. Two runs over the same proposals produce an empty
`git diff`.
"""

import json
from pathlib import Path

from src.models import GeneratedResumeProposal

# The machine-readable record, and the only file `read_proposals()` reads.
PROPOSALS_FILENAME = "proposals.json"

# Wrapped in an object rather than written as a bare JSON array, matching
# results.json's `{"candidates": [...], "errors": [...]}`. A top-level array
# would have to change shape to ever carry anything else.
_PROPOSALS_KEY = "proposals"

# Opens every generated Markdown file. The single most important line in the
# artifact: whoever opens one of these must not mistake the numbers under it
# for something the scorer produced or a human agreed with.
_REVIEW_BANNER = (
    "> **Unreviewed, machine-generated proposal — not an eval case.**\n"
    ">\n"
    "> An LLM wrote this resume and then stated what it was aiming for. The\n"
    "> level, score range and dimension expectations below are that stated\n"
    "> intent: they have not been checked against the scorer and no human has\n"
    "> agreed with them. Nothing here runs in the eval suite. Review this\n"
    "> proposal before any part of it is trusted."
)


def proposal_filename(index: int, proposal: GeneratedResumeProposal) -> str:
    """Filename for one proposal's Markdown view, e.g. `01-strong.md`.

    The zero-padded index comes first so a directory listing sorts into
    generation order rather than alphabetically by level, which would put
    `adversarial` first and scramble the strong-to-weak reading order. The
    level follows so a reviewer can tell the files apart without opening them.

    Derived entirely from the position and the proposal's own validated
    `intended_level` — no counter, no clock, no uuid — so the same proposals
    always land in the same filenames.
    """

    return f"{index:02d}-{proposal.intended_level}.md"


def render_proposal_markdown(proposal: GeneratedResumeProposal) -> str:
    """Render one proposal as the Markdown a reviewer reads.

    A view, not a serialization format: this output is never parsed back, which
    is what lets it be laid out for a human instead of for a parser.
    `proposals.json` holds the round-trippable copy.

    The proposal's stated intent goes in a table above the resume, and the
    resume itself is reproduced verbatim below a rule. Nothing is reformatted
    or re-wrapped — a reviewer is judging the document the scorer will see, so
    it has to be that document.
    """

    def _keys(keys: list[str]) -> str:
        return ", ".join(f"`{key}`" for key in keys) if keys else "_(none)_"

    low, high = proposal.expected_score_range

    return "\n".join(
        [
            f"# Proposal: {proposal.intended_level}",
            "",
            _REVIEW_BANNER,
            "",
            "## Stated generation intent",
            "",
            "| Field | Proposed value |",
            "| --- | --- |",
            f"| Intended level | `{proposal.intended_level}` |",
            f"| Expected `overall_fit` | {low}-{high} (inclusive) |",
            f"| Expected high | {_keys(proposal.dimensions_expected_high)} |",
            f"| Expected low | {_keys(proposal.dimensions_expected_low)} |",
            "",
            "**Rationale (the generator's own words):**",
            "",
            proposal.rationale,
            "",
            "## Resume",
            "",
            "<!-- Verbatim resume_markdown. This is the text the scorer would",
            "     read; do not reformat it when reviewing. -->",
            "",
            proposal.resume_markdown.strip(),
            "",
        ]
    )


def write_proposals(
    proposals: list[GeneratedResumeProposal],
    directory: str | Path,
) -> list[Path]:
    """Write a set of proposals to `directory`. Returns every path written.

    Takes already-validated `GeneratedResumeProposal` objects and serializes
    them with `model_dump(mode="json")`. This module defines no schema of its
    own and re-validates nothing: the model is the contract, and a second
    description of it here would be a second thing to keep in step.

    The directory is created if it does not exist. Files from a previous run
    are overwritten, not merged — a run's output is the whole set — but no file
    is deleted, so a stale `03-weak.md` from a larger earlier run would remain.
    That is deliberate: silently deleting files under a caller-supplied path is
    a worse failure than leaving one behind for `git status` to show.
    """

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # sort_keys is off on purpose: model_dump() yields fields in the model's
    # declaration order, which is the order a reviewer expects to read them,
    # and it is already deterministic. A trailing newline keeps the file
    # POSIX-clean and diff-friendly.
    payload = {_PROPOSALS_KEY: [p.model_dump(mode="json") for p in proposals]}
    json_path = out / PROPOSALS_FILENAME
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    written.append(json_path)

    # Numbered from 1: these filenames are read by people, and `01-strong.md`
    # beside `00-strong.md` is the kind of detail that costs a reviewer a
    # moment every time.
    for index, proposal in enumerate(proposals, start=1):
        path = out / proposal_filename(index, proposal)
        path.write_text(render_proposal_markdown(proposal), encoding="utf-8")
        written.append(path)

    return written


def read_proposals(directory: str | Path) -> list[GeneratedResumeProposal]:
    """Read back the proposals written to `directory`, in written order.

    Reads `proposals.json` only. The Markdown files are a view for humans and
    are never parsed — round-tripping through rendered prose would make the
    layout of that prose part of the data contract, and it is not.

    Validation on the way back in is `GeneratedResumeProposal`'s, not this
    module's: a hand-edited file that no longer fits the model raises
    `ValidationError` here rather than becoming a malformed proposal that
    something downstream trusts.
    """

    path = Path(directory) / PROPOSALS_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))

    return [
        GeneratedResumeProposal.model_validate(entry)
        for entry in payload[_PROPOSALS_KEY]
    ]
