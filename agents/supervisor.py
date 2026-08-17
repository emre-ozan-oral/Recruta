"""Supervisor agent: decides which worker agent runs next.

Per the project README this uses LLM-based routing rather than a fixed
pipeline order, so the graph reads as genuinely agentic rather than a
disguised linear script. A deterministic safety net still prevents
infinite loops if the LLM ever misroutes (e.g. it can't send us back to
job_analyzer once report_writer has already produced a final_report).
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import RoutingDecision
from state import AgentState

SYSTEM_PROMPT = """You are the supervisor of a job-application-assistant
pipeline with five worker agents, each producing one piece of state:

- job_analyzer   -> fills `job_requirements` (needs: job_posting_text)
- cv_matcher     -> fills `match_analysis`   (needs: job_requirements, cv_text)
- scorer         -> fills `score_breakdown`  (needs: job_requirements, match_analysis, cv_text) —
                     also overwrites match_analysis.match_score with its rigorous weighted score
- interview_prep -> fills `interview_prep`   (needs: job_requirements, match_analysis)
- report_writer  -> fills `final_report`     (needs: job_requirements, match_analysis, interview_prep)

Given the current state (which fields are already filled), choose the
single next agent to run. Follow the natural dependency order above —
never pick an agent whose inputs aren't ready yet. Once `final_report` is
already filled, respond with "END"."""

USER_PROMPT = """Current state:
- job_requirements filled: {has_job_requirements}
- match_analysis filled: {has_match_analysis}
- score_breakdown filled: {has_score_breakdown}
- interview_prep filled: {has_interview_prep}
- final_report filled: {has_final_report}

Recent progress log:
{messages}"""

# Deterministic fallback order, used only if the LLM returns something that
# would violate a dependency or loop forever.
_PIPELINE_ORDER = ["job_analyzer", "cv_matcher", "scorer", "interview_prep", "report_writer"]


def _next_by_dependency(state: AgentState) -> str:
    if not state.get("job_requirements"):
        return "job_analyzer"
    if not state.get("match_analysis"):
        return "cv_matcher"
    if not state.get("score_breakdown"):
        return "scorer"
    if not state.get("interview_prep"):
        return "interview_prep"
    if not state.get("final_report"):
        return "report_writer"
    return "END"


def run(state: AgentState) -> dict:
    fallback = _next_by_dependency(state)

    llm = get_structured_llm(RoutingDecision, tier="fast", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm

    try:
        decision: RoutingDecision = chain.invoke(
            {
                "has_job_requirements": bool(state.get("job_requirements")),
                "has_match_analysis": bool(state.get("match_analysis")),
                "has_score_breakdown": bool(state.get("score_breakdown")),
                "has_interview_prep": bool(state.get("interview_prep")),
                "has_final_report": bool(state.get("final_report")),
                "messages": "\n".join(state.get("messages", [])) or "(none yet)",
            }
        )
        next_agent = decision.next_agent
        reasoning = decision.reasoning
    except Exception as exc:  # LLM/parsing failure — fall back deterministically.
        next_agent = fallback
        reasoning = f"LLM routing failed ({exc}); used dependency-order fallback."

    # Safety net: never let the LLM pick an agent whose prerequisites aren't
    # met, and never let it loop past END.
    if fallback == "END":
        next_agent = "END"
    elif next_agent not in _PIPELINE_ORDER:
        next_agent = fallback
    elif _PIPELINE_ORDER.index(next_agent) > _PIPELINE_ORDER.index(fallback):
        next_agent = fallback

    return {
        "next_agent": next_agent,
        "messages": [*state.get("messages", []), f"supervisor: routing to {next_agent} ({reasoning})"],
    }
