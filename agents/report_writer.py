"""Report Writer agent: merges every agent's output into one Markdown report
plus a short TL;DR summary."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import FinalReportOutput
from state import AgentState

SYSTEM_PROMPT = """You are a technical writer producing a final candidate
report. Combine the job analysis, match analysis, and interview prep into
one clear, well-organized Markdown document with headings, plus a short
standalone summary. Do not invent new information — only reorganize and
clearly present what's given.

`final_report` must follow this structure:

# Job Application Report

## Job Overview
Include years_of_experience_required explicitly (e.g. "Experience required:
3-5 years"). If it's "Not specified in the posting", say that plainly
rather than omitting the line — the candidate should know the posting
didn't give a number, not wonder why it's missing.
## Match Analysis (include the match score)
## Strengths & Projects to Highlight
## Gaps to Address
## Interview Preparation
### Likely Questions
### Talking Points
### Questions to Ask the Interviewer

`short_summary` is a separate 2-3 sentence TL;DR: the fit verdict and the
single most important next step for the candidate. It should stand alone
(the reader may see only this, not the full report)."""

USER_PROMPT = """Job requirements (JSON):
{job_requirements}

Match analysis (JSON):
{match_analysis}

Interview prep (JSON):
{interview_prep}"""


def run(state: AgentState) -> dict:
    llm = get_structured_llm(FinalReportOutput, tier="content", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm
    result: FinalReportOutput = chain.invoke(
        {
            "job_requirements": json.dumps(state["job_requirements"], ensure_ascii=False),
            "match_analysis": json.dumps(state["match_analysis"], ensure_ascii=False),
            "interview_prep": json.dumps(state["interview_prep"], ensure_ascii=False),
        }
    )

    return {
        "final_report": result.final_report,
        "short_summary": result.short_summary,
        "messages": [*state.get("messages", []), "report_writer: assembled final report."],
    }
