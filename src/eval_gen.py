"""
Challenge 7: generate synthetic resume proposals from a job description.

Two functions, one layered on the other. `generate_resume_proposal()` is the
unit of work — one LLM call, one match level, one validated proposal.
`generate_proposal_set()` sits above it and asks for each level in turn, which
is the whole of the orchestration: the levels are independent of each other and
nothing has to be reconciled between them.

Same tool-use + Pydantic pattern as extractor.py, scorer.py, critique.py and
deep_dive.py — see extractor.py for the three-beat explanation. Three things
are specific to this stage:

1. **It runs the pipeline backwards.** Every other module reads a resume and
   produces judgments about it. This one is handed a judgment — "write me a
   partial match" — and produces the resume. Nothing here scores anything, and
   nothing here decides whether the resume it got back actually is a partial
   match. That question belongs to the scorer, and after it to a human.

2. **What it returns is a proposal, not a result.** A `GeneratedResumeProposal`
   carries the level, score range and dimension expectations the generator was
   *aiming* for. They are design intent awaiting review, never ground truth,
   and no caller may promote one to a trusted eval case without a human in
   between. The prompt says this to the model; the model's docstring says it to
   the next reader; this module never treats those fields as facts.

3. **One level, one call.** Asking for all four levels in a single response
   would make them contend for the same output budget and let the model tune
   them against each other. The caller loops; this function generates exactly
   one.

This module writes no files, reads no eval cases, ranks nothing, and never
constructs a client. It takes the client it is handed — which in the eventual
CLI is the `MeteredClient` both existing entry points already build — so its
calls land in usage.json by construction, exactly as deep_dive.py's do.

Prompts live in src/prompts.py, never inlined here.
"""

from typing import get_args

from anthropic import Anthropic

from src.models import GeneratedResumeProposal, MatchLevel
from src.prompts import SYNTH_RESUME_LEVELS, SYNTH_RESUME_SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-4-6"
GENERATION_TOOL_NAME = "record_generated_resume"


class EvalGenerationError(RuntimeError):
    """Raised when a generation call did not produce a usable proposal.

    One error type for every way a single generation can fail — an unknown
    level, a response with no tool_use block, the wrong tool, or a payload that
    fails `GeneratedResumeProposal` validation. A caller generating several
    levels wants to catch one thing, report which level failed, and keep the
    others; it does not want to know which of four SDK- and Pydantic-level
    exceptions it should be listing.

    The underlying exception is always chained (`raise ... from exc`), so the
    Pydantic `ValidationError` behind a bad payload is still there in
    `__cause__` for anyone debugging a prompt regression.

    This is deliberately *not* a `ProcessingError`. That model describes a
    resume that failed the screening pipeline and is part of the results.json
    contract; a generation failure is neither.
    """


def _proposal_from_response(response: object, level: str) -> GeneratedResumeProposal:
    """Pull the forced tool call out of a response and validate it.

    Split out from the request so the failure modes are readable in one place.
    Every branch here raises rather than returning a default: a generation that
    half-worked is worse than one that stopped, because its output would be
    saved as a fixture and reviewed by a human who had no reason to doubt it.
    """

    # Deliberately not `next(block for block in ...)` — that is what the older
    # stage modules do, and on an empty match it raises a bare StopIteration
    # with no message. Fine inside process_one(), which relabels it as a
    # ProcessingError; useless to someone debugging a generation prompt.
    tool_use = next(
        (
            block
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "tool_use"
        ),
        None,
    )

    if tool_use is None:
        types = ", ".join(
            sorted({str(getattr(b, "type", "?")) for b in getattr(response, "content", [])})
        )
        raise EvalGenerationError(
            f"Generating a {level!r} resume: the model returned no tool_use "
            f"block, so there is no proposal to read. Content block types "
            f"present: {types or '(none)'}."
        )

    # tool_choice forces this exact tool, so a different name means the request
    # was built wrong or the API behaved unexpectedly. Either way the payload
    # underneath is not a GeneratedResumeProposal and must not be validated as
    # one.
    name = getattr(tool_use, "name", None)
    if name != GENERATION_TOOL_NAME:
        raise EvalGenerationError(
            f"Generating a {level!r} resume: expected the model to call "
            f"{GENERATION_TOOL_NAME!r}, but it called {name!r}."
        )

    try:
        return GeneratedResumeProposal.model_validate(tool_use.input)
    except Exception as exc:
        raise EvalGenerationError(
            f"Generating a {level!r} resume: the model's tool input did not "
            f"validate as a GeneratedResumeProposal ({exc})."
        ) from exc


def generate_resume_proposal(
    client: Anthropic,
    jd: str,
    level: MatchLevel,
    model: str = DEFAULT_MODEL,
) -> GeneratedResumeProposal:
    """Generate one synthetic resume proposal at `level`. Single LLM call.

    `client` is whatever the caller is already using; pass the `MeteredClient`
    the entry points build and this call is metered with no work here.

    Returns a validated `GeneratedResumeProposal` — a *proposal*, whose
    expectations are the generator's stated intent and are not to be trusted
    until a human has reviewed them. Raises `EvalGenerationError` if no usable
    proposal came back.
    """

    # Checked before the request, not after: `level` is annotated `MatchLevel`
    # but annotations aren't enforced at runtime, and a typo would otherwise
    # surface as a KeyError *after* a paid call had already been made.
    definition = SYNTH_RESUME_LEVELS.get(level)
    if definition is None:
        known = ", ".join(repr(k) for k in SYNTH_RESUME_LEVELS)
        raise EvalGenerationError(
            f"Unknown match level {level!r}; expected one of {known}."
        )

    tool = {
        "name": GENERATION_TOOL_NAME,
        "description": (
            "Record one synthetic resume and the generation intent behind it."
        ),
        "input_schema": GeneratedResumeProposal.model_json_schema(),
    }

    # Two blocks, JD first. The cache breakpoint sits at the end of the JD
    # block, so the cached prefix is the tool schema, the system prompt and the
    # JD — all identical across the four levels of one run. The requested level
    # comes after it and therefore can't contaminate the prefix, so generating
    # the second level of a run reads that prefix from cache.
    #
    # Its own cache entry, not the scorer's or the deep-dive's: a prefix covers
    # the tools and system prompt too, and both differ here. No TTL is set, so
    # it uses the project-wide default 5-minute ephemeral cache that usage.py
    # prices.
    content = [
        {
            "type": "text",
            "text": f"<job_description>\n{jd}\n</job_description>",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                f"<requested_match_level>\n"
                f"level: {level}\n\n"
                f"{definition}\n"
                f"</requested_match_level>\n\n"
                f"Write one {level} resume for the job description above and "
                f"record it by calling {GENERATION_TOOL_NAME} exactly once."
            ),
        },
    ]

    response = client.messages.create(
        model=model,
        # Roomier than the scoring stages: this call writes a whole resume plus
        # a rationale, where they write a handful of sentences.
        max_tokens=4096,
        system=SYNTH_RESUME_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": GENERATION_TOOL_NAME},
        messages=[{"role": "user", "content": content}],
    )

    return _proposal_from_response(response, level)


# ---------------------------------------------------------------------------
# Complete-set orchestration
# ---------------------------------------------------------------------------

# The levels to generate, in the order they are generated and reported.
#
# Derived from MatchLevel rather than retyped: `Literal` preserves its
# declaration order, and src/models.py already declares the four levels in the
# intended order — strongest to weakest, then the adversarial probe. Writing the
# list again here would be a second source of truth that could drift from the
# type the proposals are validated against.
#
# tests/test_eval_gen.py pins the resulting tuple against the literal four
# names, so reordering MatchLevel is a deliberate change with a failing test
# attached rather than a silent reshuffle of every generated artifact.
GENERATION_ORDER: tuple[MatchLevel, ...] = get_args(MatchLevel)


def generate_proposal_set(
    client: Anthropic,
    jd: str,
    model: str = DEFAULT_MODEL,
) -> list[GeneratedResumeProposal]:
    """Generate one proposal per match level for `jd`. One LLM call per level.

    Returns the proposals in `GENERATION_ORDER` — strong, partial, weak,
    adversarial — regardless of how long any individual call took, because the
    calls are sequential and the list is built in order.

    Sequential on purpose. The four calls share one cached prefix (the tool
    schema, the system prompt and the JD, in that order), and running them one
    after another means call one writes that prefix and the other three read
    it. This is the same ordering argument `main.py` makes when it primes the
    scorer's cache with a single candidate before fanning out — except that
    here there is nothing to fan out to afterwards, so there is no pool at all.
    Four calls is not a batch worth a thread pool, and adding one would trade a
    cache prefix for wall-clock the caller did not ask for.

    A failure in any level aborts the set. `EvalGenerationError` propagates
    untouched: a partial set silently missing its adversarial case is the one
    outcome a generated eval set must not produce, because the gap is invisible
    in the output and the reviewer has no reason to look for it. The caller
    sees which level failed in the message and reruns.

    Writes nothing. Persisting these proposals is `src/eval_fixtures.py`'s job.
    """

    return [
        generate_resume_proposal(client, jd, level, model=model)
        for level in GENERATION_ORDER
    ]
