"""LLM client factory with model tiering and a rate-limit failsafe.

Centralizing model construction here means every agent gets its model the
same way, and swapping providers later (Groq -> OpenRouter -> local Ollama)
is a one-file change instead of a find-and-replace across agents/.

Model tiering: the four content-generating agents (job analysis, CV
matching, interview prep, report writing) get the strongest model since
output quality matters most there. The supervisor only makes a cheap
classification-style routing decision, so it defaults to a much
faster/cheaper model instead.

Rate-limit failsafe: Groq's free/dev-tier rate limits are easy to hit
during a normal session (each pipeline run makes 5+ calls). Every
structured-output call gets a short retry-with-backoff, and if a model is
still failing (rate-limited or otherwise erroring) after that, the chain
falls through to the next model in its tier instead of failing the whole
pipeline run.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel

load_dotenv()

# NOTE (2026-08-17): llama-3.1-8b-instant and llama-3.3-70b-versatile were
# both deprecated by Groq on 2026-08-16 (confirmed live — one of them 404s
# with "model_not_found" on a real account) and are no longer used as
# defaults anywhere below. Groq's lineup moves fast; if you hit another
# model_not_found error, check console.groq.com/docs/models and
# console.groq.com/docs/deprecations before assuming the code is wrong.

# openai/gpt-oss-120b: strong open-weight reasoning model on Groq (MoE,
# ~120B total params), native JSON-schema structured-output support,
# ~500 tok/s, $0.15 / $0.60 per 1M input/output tokens, 131K context,
# 65,536 max completion tokens (room for long reports). Falls back to
# openai/gpt-oss-20b (smaller but still solid, and confirmed not
# deprecated) if rate-limited. Used by job_analyzer, cv_matcher,
# interview_prep, report_writer.
CONTENT_MODEL_CHAIN = [
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    os.getenv("GROQ_MODEL_FALLBACK", "openai/gpt-oss-20b"),
]

# openai/gpt-oss-20b: fastest/cheapest non-deprecated model on Groq,
# ~1000 tok/s, $0.075 / $0.30 per 1M tokens, native JSON-schema structured
# output. Falls back to the bigger gpt-oss-120b (costs more, but only ever
# triggered when gpt-oss-20b itself is rate-limited/erroring) for the
# supervisor's routing decision.
FAST_MODEL_CHAIN = [
    os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b"),
    os.getenv("GROQ_FAST_MODEL_FALLBACK", "openai/gpt-oss-120b"),
]

# Reasoning tier (2026-08-17): used only by agents/scorer.py's dedicated
# scoring pass. Groq's gpt-oss models are already reasoning models (visible
# chain-of-thought), and Groq exposes a `reasoning_effort` request param
# (confirmed via console.groq.com/docs/reasoning — "low"/"medium"/"high",
# supported on both gpt-oss-120b and gpt-oss-20b) that controls how much of
# that reasoning the model actually does before answering. Reusing
# CONTENT_MODEL_CHAIN's models rather than picking a different model family
# was a deliberate choice: Groq's other "reasoning-branded" models change
# availability often (see the llama-3.1/3.3 deprecations above), and
# gpt-oss-120b is already Groq's strongest generally-available model — so
# "another reasoning model" is implemented as the same model reasoning
# harder (reasoning_effort="high"), not a different, less-proven one. Ask
# for a specific alternate model/provider if you'd rather have that
# instead.
REASONING_MODEL_CHAIN = [
    os.getenv("GROQ_REASONING_MODEL", "openai/gpt-oss-120b"),
    os.getenv("GROQ_REASONING_MODEL_FALLBACK", "openai/gpt-oss-20b"),
]

# Kept for anything that just wants "the" default model (e.g. file_parsing's
# vision calls override this explicitly anyway).
DEFAULT_MODEL = CONTENT_MODEL_CHAIN[0]
FAST_MODEL = FAST_MODEL_CHAIN[0]


def _require_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "Groq API key (https://console.groq.com/keys)."
        )
    return api_key


def get_llm(temperature: float = 0.2, model: str | None = None) -> ChatGroq:
    """Return a single configured Groq chat model, no retry/fallback chain.

    Used where structured output isn't needed (e.g. report_writer's plain
    text path, vision OCR calls).
    """
    return ChatGroq(
        model=model or DEFAULT_MODEL, temperature=temperature, api_key=_require_api_key()
    )


def get_structured_llm(
    schema: type[BaseModel],
    *,
    tier: Literal["content", "fast", "reasoning"] = "content",
    temperature: float = 0.2,
) -> Runnable:
    """Return a structured-output Runnable with retry + model fallback.

    tier="content" walks CONTENT_MODEL_CHAIN (job_analyzer, cv_matcher,
    interview_prep, report_writer). tier="fast" walks FAST_MODEL_CHAIN
    (supervisor's routing decision, skill_verifier's evaluation).
    tier="reasoning" walks REASONING_MODEL_CHAIN with reasoning_effort="high"
    (agents/scorer.py's dedicated scoring pass only). Each model gets a
    couple of quick retries with backoff before the chain falls through to
    the next model.

    method="json_schema" (2026-08-17): langchain-groq's with_structured_output
    defaults to method="function_calling", which forces a tool call. In
    practice both gpt-oss-120b and gpt-oss-20b occasionally answer with
    plain text instead of actually invoking the forced tool — surfaced live
    as `Error code: 400 ... 'Tool choice is required, but model did not
    call a tool' ... 'code': 'tool_use_failed'`. That's not just a hard
    failure: with_retry()/with_fallbacks() then burn through retries and a
    second model before giving up, which is why routing (supervisor) and
    the skill-flagging evaluator (skill_verifier, same tier="fast" chain)
    both got slow, not just occasionally wrong. Groq's docs confirm both
    gpt-oss models support method="json_schema" — their dedicated
    Structured Outputs API — which doesn't go through tool-calling at all,
    so this failure mode doesn't apply to it. Switching here fixes the
    root cause for every agent at once instead of patching each call site.

    strict=True (2026-08-17, round 7): switching to method="json_schema"
    alone wasn't the full fix — it surfaced a second bug, live, on
    agents/scorer.py's schema: `Error code: 400 ... 'Failed to validate
    JSON' ... 'code': 'json_validate_failed'`. Traced to
    langchain-groq's own source (ChatGroq.with_structured_output): with
    strict left at its default (None), the "strict" key is never even
    included in the request Groq receives, AND the JSON Schema sent for any
    Pydantic field with a Python default (e.g. ScoringResult.scoring_notes,
    JobRequirements.soft_skills) omits that field from "required" and never
    sets "additionalProperties": false — confirmed directly by calling
    convert_to_json_schema(schema, strict=None) vs strict=True and diffing
    the output. Groq's own docs describe non-strict structured outputs as
    best-effort and warn it "may occasionally 400 error or produce
    syntactically valid but schema-invalid JSON" — exactly what we hit.
    strict=True asks langchain-groq to build the fully-compliant schema
    instead (every property in "required", additionalProperties: false at
    every nesting level — verified for every schema in schemas.py before
    shipping this) and engages Groq's actual constrained decoding, which is
    supposed to guarantee schema compliance rather than best-effort it.
    Both gpt-oss-120b and gpt-oss-20b are in langchain-groq's
    _STRICT_STRUCTURED_OUTPUT_MODELS allowlist, so this applies cleanly
    across every tier.
    """
    model_chain = {
        "content": CONTENT_MODEL_CHAIN,
        "fast": FAST_MODEL_CHAIN,
        "reasoning": REASONING_MODEL_CHAIN,
    }[tier]
    api_key = _require_api_key()
    extra_kwargs = {"reasoning_effort": "high"} if tier == "reasoning" else {}

    runnables = [
        ChatGroq(model=model_id, temperature=temperature, api_key=api_key, **extra_kwargs)
        .with_structured_output(schema, method="json_schema", strict=True)
        .with_retry(stop_after_attempt=2, wait_exponential_jitter=True)
        for model_id in model_chain
    ]

    primary, *fallbacks = runnables
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
