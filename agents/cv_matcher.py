"""CV Matcher agent: compares the CV against the job requirements."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import MatchAnalysis
from state import AgentState

SYSTEM_PROMPT = """You are an expert career coach who compares candidate CVs
against job requirements. Be honest, not flattering: a candidate who pays
for this tool wants an accurate gap analysis, not encouragement. Ground
every claim in the actual CV text — don't assume skills that aren't
mentioned.

You are given ONLY the hard/technical requirements below (soft skills and
passion/motivation statements have already been filtered out — you won't
see them, and you must not invent any). match_score, matched_skills, and
missing_skills must be based solely on these hard requirements. Never
penalize or guess at personality traits, culture fit, or "passion" for
anything — that's not something a CV can prove or disprove."""

USER_PROMPT = """Job's hard/technical requirements (JSON):
{job_requirements}

Candidate CV:
{cv_text}"""


def run(state: AgentState) -> dict:
    llm = get_structured_llm(MatchAnalysis, tier="content")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm

    # Structurally exclude soft skills — don't just rely on the prompt.
    # Job overview / report_writer still get the full job_requirements
    # (including soft_skills) separately for informational display.
    hard_requirements = {
        k: v for k, v in state["job_requirements"].items() if k != "soft_skills"
    }

    result: MatchAnalysis = chain.invoke(
        {
            "job_requirements": json.dumps(hard_requirements, ensure_ascii=False),
            "cv_text": state["cv_text"],
        }
    )

    return {
        "match_analysis": result.model_dump(),
        "messages": [*state.get("messages", []), "cv_matcher: compared CV against requirements."],
    }
