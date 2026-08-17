"""Scorer agent: a dedicated, reasoning-focused second pass that produces a
weighted, per-requirement score — not just the single eyeballed match_score
CV Matcher derives as a byproduct of listing matched/missing skills.

Runs after cv_matcher, using tier="reasoning" (Groq's reasoning_effort="high"
on the same gpt-oss models — deeper chain-of-thought before answering, see
llm.py for why this is "another reasoning model" rather than a different
model family). Every hard requirement gets an explicit weight (how much it
matters to this specific role) and a yes/partial/no verdict grounded in the
CV, and overall_score is computed FROM those — so a candidate missing one
critical (weight 5) requirement scores meaningfully lower than one missing
a minor (weight 1) one, which a single holistic 0-100 guess tends to blur
together.

This agent's overall_score becomes the new source of truth: it overwrites
match_analysis["match_score"] (see run() below) so every downstream
consumer — the UI badge, the DB's match_score column, interview_prep,
report_writer — automatically uses the rigorous score without needing to
change how they read it.
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import ScoringResult
from state import AgentState

SYSTEM_PROMPT = """You are a meticulous technical hiring assessor doing a
rigorous, auditable scoring pass — not a quick impression.

For EVERY hard requirement in required_skills:
1. Assign a weight 1-5 for how critical it is to this specific role (use
   seniority_level and company_context as signals — a core language or
   framework requirement for a "Senior" role is a 5; a minor/peripheral
   tool mention is a 1-2).
2. Judge meets_requirement as "yes" (clearly demonstrated in the CV),
   "partial" (adjacent/transferable experience but not a clean match), or
   "no" (no evidence in the CV at all) — ground every verdict in the
   actual CV text, never assume.
3. Give one sentence of evidence for that verdict.

Then compute overall_score (0-100) AS A FUNCTION of those weighted
verdicts — roughly: sum(weight * verdict_value) / sum(weight) * 100, where
yes=1.0, partial=0.5, no=0.0 — rounded to a whole number. You may adjust
slightly for context (e.g. several "partial" verdicts clustering around
one missing prerequisite technology), but explain any such adjustment in
methodology. Never just restate cv_matcher's preliminary match_score —
derive your own number from the weighted breakdown, independently.

If years_of_experience_required is "Not specified in the posting", don't
penalize or credit the candidate for it either way — note that in
scoring_notes instead of factoring it into a requirement's score."""

USER_PROMPT = """Job's hard/technical requirements (JSON):
{job_requirements}

CV Matcher's preliminary read (matched/missing skills, strengths,
weaknesses — for context only; re-derive your own verdicts from the CV,
don't just copy these):
{match_analysis}

Candidate CV:
{cv_text}"""


def _fallback_score_breakdown(state: AgentState, error: Exception) -> dict:
    """If the dedicated scoring pass itself fails (structured-output
    quirks have already bitten this agent once live — see llm.py's
    strict=True note), don't crash the whole analysis run over it. Fall
    back to a simple, clearly-labeled score derived from cv_matcher's
    preliminary matched/missing counts instead of losing the run entirely.
    """
    match_analysis = state.get("match_analysis") or {}
    matched = len(match_analysis.get("matched_skills") or [])
    missing = len(match_analysis.get("missing_skills") or [])
    total = matched + missing
    if total:
        fallback_score = round((matched / total) * 100)
    else:
        fallback_score = match_analysis.get("match_score", 0)

    return {
        "overall_score": fallback_score,
        "requirement_scores": [],
        "methodology": (
            f"Fallback score — the dedicated weighted scoring pass failed "
            f"({type(error).__name__}), so this is a simple matched/missing "
            "skill ratio from CV Matcher's preliminary read, not the full "
            "per-requirement breakdown."
        ),
        "scoring_notes": "Re-run the analysis to get the full weighted breakdown.",
    }


def run(state: AgentState) -> dict:
    llm = get_structured_llm(ScoringResult, tier="reasoning", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm

    # Same structural exclusion as cv_matcher — never send soft skills into
    # a scoring pass either.
    hard_requirements = {
        k: v for k, v in state["job_requirements"].items() if k != "soft_skills"
    }

    try:
        result: ScoringResult = chain.invoke(
            {
                "job_requirements": json.dumps(hard_requirements, ensure_ascii=False),
                "match_analysis": json.dumps(state["match_analysis"], ensure_ascii=False),
                "cv_text": state["cv_text"],
            }
        )
        score_breakdown = result.model_dump()
        log_line = "scorer: computed a weighted per-requirement score."
    except Exception as exc:  # noqa: BLE001 — see _fallback_score_breakdown docstring
        score_breakdown = _fallback_score_breakdown(state, exc)
        log_line = f"scorer: weighted scoring failed ({exc}); used fallback score."

    # match_analysis["match_score"] is the single field every other
    # consumer (UI badge, db.match_score, interview_prep, report_writer)
    # already reads — overwrite it here so the (rigorous or fallback) score
    # propagates everywhere without those call sites needing to change.
    updated_match_analysis = dict(state["match_analysis"])
    updated_match_analysis["match_score"] = score_breakdown["overall_score"]

    return {
        "match_analysis": updated_match_analysis,
        "score_breakdown": score_breakdown,
        "messages": [*state.get("messages", []), log_line],
    }
