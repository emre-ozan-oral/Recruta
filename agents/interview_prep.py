"""Interview Prep agent: generates likely questions and talking points."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import InterviewPrep
from state import AgentState

SYSTEM_PROMPT = """You are a senior hiring manager helping a candidate
prepare for an interview. Use the job requirements and the gap analysis to
generate realistic interview questions, especially ones probing the
candidate's weak spots. Also suggest talking points that let the candidate
proactively address gaps, and a few smart questions to ask the
interviewer."""

USER_PROMPT = """Job requirements (JSON):
{job_requirements}

Match analysis (JSON):
{match_analysis}"""


def run(state: AgentState) -> dict:
    llm = get_structured_llm(InterviewPrep, tier="content")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm
    result: InterviewPrep = chain.invoke(
        {
            "job_requirements": json.dumps(state["job_requirements"], ensure_ascii=False),
            "match_analysis": json.dumps(state["match_analysis"], ensure_ascii=False),
        }
    )

    return {
        "interview_prep": result.model_dump(),
        "messages": [*state.get("messages", []), "interview_prep: generated interview prep."],
    }


PARALLEL_USER_PROMPT = """Job requirements (JSON):
{job_requirements}

Candidate CV (read it yourself to find the gaps — no separate gap analysis is
provided):
{cv_text}"""


def run_parallel(state: AgentState) -> dict:
    """Same output as run(), but derives the candidate's weak spots straight
    from the CV instead of waiting for cv_matcher's match_analysis. That
    removes the data dependency, so graph_fast.py can run it concurrently
    with cv_matcher and the scorer instead of after them."""
    llm = get_structured_llm(InterviewPrep, tier="content")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", PARALLEL_USER_PROMPT)]
    )
    hard_requirements = {
        k: v for k, v in state["job_requirements"].items() if k != "soft_skills"
    }
    result: InterviewPrep = (prompt | llm).invoke(
        {
            "job_requirements": json.dumps(hard_requirements, ensure_ascii=False),
            "cv_text": state["cv_text"],
        }
    )
    return {"interview_prep": result.model_dump()}
