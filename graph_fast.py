"""Low-latency variant of the pipeline (graph.py stays the "agentic" reference).

Why a second graph: the reference graph runs 11 LLM calls strictly one after
another — a supervisor call before *and* after each of the 5 workers — and
the supervisor never actually decides anything the dependency-order safety
net wouldn't (see agents/supervisor.py). Wall time is the sum of all of them.

This graph keeps the same agents' prompts and outputs but changes the shape:

    job_analyzer ──┬── cv_matcher ───────┐
                   ├── scorer (indep.) ──┼── finalize (pure Python)
                   └── interview_prep ───┘

  * no supervisor calls (static edges)          -> -6 sequential LLM calls
  * three independent calls run concurrently     -> wall time = slowest, not sum
  * report + summary built in code               -> -1 long generation
Result: 1 serial LLM call, then 3 in parallel, then microseconds. That is
4 model calls total instead of 11, and the critical path is
job_analyzer + max(cv_matcher, scorer, interview_prep).

Failure isolation: a failure in scorer or interview_prep no longer loses the
run — the scorer falls back to the matched/missing ratio, interview_prep to
an empty section — and the problem is recorded in state["errors"].
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

import report_builder
from agents import cv_matcher, interview_prep, job_analyzer, scorer
from llm import get_structured_llm
from schemas import InterviewPrep, JobRequirements, MatchAnalysis, ScoringResult


def _merge_dicts(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class FastState(TypedDict, total=False):
    cv_text: str
    job_posting_text: str
    job_requirements: dict[str, Any]
    match_analysis: dict[str, Any]
    score_breakdown: dict[str, Any] | None
    interview_prep: dict[str, Any]
    final_report: str
    short_summary: str
    # Parallel branches append/merge instead of clobbering each other.
    messages: Annotated[list[str], operator.add]
    errors: Annotated[dict[str, str], _merge_dicts]


def _only_new_messages(state: FastState, out: dict) -> dict:
    """Existing agents return the *whole* messages list; with a reducer we
    must return only the delta or every line would be duplicated."""
    if "messages" in out:
        out = dict(out)
        out["messages"] = out["messages"][len(state.get("messages", [])):]
    return out


def _job_analyzer(state: FastState) -> dict:
    return _only_new_messages(state, job_analyzer.run(state))


def _cv_matcher(state: FastState) -> dict:
    return _only_new_messages(state, cv_matcher.run(state))


def _scorer(state: FastState) -> dict:
    out = scorer.run_independent(state)
    out["messages"] = ["scorer: weighted scoring " + ("failed, fallback pending." if out["score_breakdown"] is None else "done.")]
    return out


def _interview_prep(state: FastState) -> dict:
    try:
        out = interview_prep.run_parallel(state)
        out["messages"] = ["interview_prep: generated interview prep."]
        return out
    except Exception as exc:  # noqa: BLE001 — never lose the whole run over this
        return {
            "interview_prep": {"likely_questions": [], "talking_points": [], "questions_to_ask_interviewer": []},
            "errors": {"interview_prep": f"{type(exc).__name__}: {exc}"},
            "messages": ["interview_prep: FAILED — section left empty."],
        }


def _finalize(state: FastState) -> dict:
    """Join point: reconcile the score, then build summary + report in code."""
    breakdown = state.get("score_breakdown")
    if not breakdown:
        breakdown = scorer._fallback_score_breakdown(
            state, RuntimeError((state.get("errors") or {}).get("scorer", "scorer produced no result"))
        )
    match = dict(state["match_analysis"])
    match["match_score"] = breakdown["overall_score"]  # same contract as scorer.run()

    merged = {**state, "match_analysis": match, "score_breakdown": breakdown}
    return {
        "match_analysis": match,
        "score_breakdown": breakdown,
        "short_summary": report_builder.build_short_summary(merged),
        "final_report": report_builder.build_report(merged),
        "messages": ["finalize: merged score and assembled report."],
    }


def warm_up() -> None:
    """Pre-build every LLM chain this graph uses (call once at process start).

    The arguments must mirror what each agent passes to get_structured_llm —
    a mismatch is only a cache miss (slower first request), not a bug, and
    tests/test_graph_fast.py asserts that no chain is built after warm-up.
    """
    get_structured_llm(JobRequirements, tier="content")  # job_analyzer
    get_structured_llm(MatchAnalysis, tier="content")  # cv_matcher
    get_structured_llm(InterviewPrep, tier="content")  # interview_prep.run_parallel
    get_structured_llm(ScoringResult, tier="reasoning", temperature=0)  # scorer.run_independent


def build_fast_graph():
    wf = StateGraph(FastState)
    wf.add_node("job_analyzer", _job_analyzer)
    wf.add_node("cv_matcher", _cv_matcher)
    wf.add_node("scorer", _scorer)
    wf.add_node("interview_prep", _interview_prep)
    wf.add_node("finalize", _finalize)

    wf.add_edge(START, "job_analyzer")
    for branch in ("cv_matcher", "scorer", "interview_prep"):
        wf.add_edge("job_analyzer", branch)
    wf.add_edge(["cv_matcher", "scorer", "interview_prep"], "finalize")  # wait for all three
    wf.add_edge("finalize", END)
    return wf.compile()


def run_fast_pipeline(cv_text: str, job_posting_text: str, callbacks: list | None = None) -> FastState:
    config = {"callbacks": callbacks} if callbacks else {}
    return build_fast_graph().invoke(
        {"cv_text": cv_text, "job_posting_text": job_posting_text, "messages": [], "errors": {}},
        config=config,
    )
