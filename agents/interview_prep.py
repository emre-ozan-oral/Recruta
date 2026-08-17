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
