"""
Optional post-scoring LLM call: a hiring-manager briefing for one shortlisted
candidate.

Same tool-use + Pydantic pattern as extractor.py, scorer.py and critique.py —
see extractor.py for the three-beat explanation. Two things are specific to
this stage:

1. It runs *after* the whole field has been scored and ranked, on the top N
   candidates only. That makes it the one stage whose cost is O(N selected),
   not O(resumes): you can screen a hundred resumes and brief the five you'd
   actually interview. `main.py` owns the selection; this module briefs
   whichever candidate it is handed.

2. It reads a ScoredCandidate, not a resume. Everything it needs — the
   profile, every dimension's score and reasoning, and the gaps — is already
   on that object, so no PDF is re-parsed and no resume is re-extracted. The
   raw resume text is deliberately not required: it has gone out of scope by
   this point in both entry points, and the screening results are the more
   useful input anyway, since a briefing's risks and interview questions are
   built from the gaps the scorer already found.

The dimension scores are serialized from `DIMENSIONS`, never named literally,
so a fifth dimension added to dimensions.yaml reaches the briefing with no
change to this file.

Prompts live in src/prompts.py, never inlined here.
"""

import json

from anthropic import Anthropic

from src.dimensions import DIMENSIONS
from src.models import DeepDiveReport, ScoredCandidate
from src.prompts import DEEP_DIVE_SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-4-6"
DEEP_DIVE_TOOL_NAME = "record_deep_dive"


def _screening_results(candidate: ScoredCandidate) -> str:
    """Serialize a candidate's scores, reasoning and gaps as JSON for the prompt.

    Dimensions come from `DIMENSIONS` rather than being listed by name, so this
    stays correct when dimensions.yaml changes. `label` rides along with each
    entry so the model reads "Skills match" rather than inferring meaning from
    a field name.
    """

    return json.dumps(
        {
            "dimension_scores": {
                dimension.key: {
                    "label": dimension.label,
                    "score": getattr(candidate, dimension.key).score,
                    "reasoning": getattr(candidate, dimension.key).reasoning,
                }
                for dimension in DIMENSIONS
            },
            "overall_reasoning": candidate.reasoning,
            "gaps": [gap.model_dump() for gap in candidate.gaps],
        },
        indent=2,
    )


def generate_deep_dive(
    client: Anthropic,
    candidate: ScoredCandidate,
    jd: str,
    model: str = DEFAULT_MODEL,
) -> DeepDiveReport:
    """Write a hiring-manager briefing for one scored candidate. Single LLM call.

    `client` is whatever the caller is already using — in both entry points
    that is the MeteredClient built at startup, which is how these calls end up
    in usage.json without this module knowing anything about metering.
    """

    tool = {
        "name": DEEP_DIVE_TOOL_NAME,
        "description": (
            "Record a hiring-manager briefing for a shortlisted candidate."
        ),
        "input_schema": DeepDiveReport.model_json_schema(),
    }

    # Three blocks, JD first. The cache breakpoint sits at the end of the JD
    # block, so the cached prefix is the tool schema, the system prompt and the
    # JD — all identical across candidates. Everything candidate-specific comes
    # after it and therefore can't contaminate the prefix. Briefing the second
    # candidate of a run reads that prefix from cache instead of paying for it
    # again.
    #
    # This is its own cache entry, not the scorer's: a prefix covers the tools
    # and system prompt too, and both differ here. No TTL is set, so it uses
    # the project-wide default 5-minute ephemeral cache that usage.py prices.
    content = [
        {
            "type": "text",
            "text": f"<job_description>\n{jd}\n</job_description>",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"<candidate_profile>\n"
                f"{candidate.profile.model_dump_json(indent=2)}\n"
                f"</candidate_profile>"
            ),
        },
        {
            "type": "text",
            "text": (
                f"<screening_results>\n"
                f"{_screening_results(candidate)}\n"
                f"</screening_results>"
            ),
        },
    ]

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=DEEP_DIVE_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": DEEP_DIVE_TOOL_NAME},
        messages=[{"role": "user", "content": content}],
    )

    tool_use = next(block for block in response.content if block.type == "tool_use")
    return DeepDiveReport.model_validate(tool_use.input)
