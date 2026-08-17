"""Shared LangGraph state schema for the job application assistant."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State object threaded through the StateGraph.

    Only ``cv_text`` and ``job_posting_text`` are required inputs; every
    other field is populated progressively as agents run.
    """

    # Raw inputs
    cv_text: str
    job_posting_text: str

    # Agent outputs (dict form of the pydantic schemas in schemas.py)
    job_requirements: dict[str, Any]
    match_analysis: dict[str, Any]
    score_breakdown: dict[str, Any]
    interview_prep: dict[str, Any]
    final_report: str
    short_summary: str

    # Supervisor bookkeeping
    messages: list[str]
    next_agent: str
