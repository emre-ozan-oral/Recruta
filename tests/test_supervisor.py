"""Tests for the supervisor's routing logic, including the safety net that
stops the LLM from skipping ahead of unmet dependencies."""

from __future__ import annotations

from agents import supervisor
from schemas import RoutingDecision


def _fake_structured_llm(decision: RoutingDecision):
    return lambda _formatted_prompt: decision


def test_next_by_dependency_order():
    assert supervisor._next_by_dependency({}) == "job_analyzer"
    assert supervisor._next_by_dependency({"job_requirements": {"a": 1}}) == "cv_matcher"
    assert (
        supervisor._next_by_dependency(
            {"job_requirements": {"a": 1}, "match_analysis": {"b": 2}}
        )
        == "scorer"
    )
    assert (
        supervisor._next_by_dependency(
            {
                "job_requirements": {"a": 1},
                "match_analysis": {"b": 2},
                "score_breakdown": {"s": 4},
            }
        )
        == "interview_prep"
    )
    assert (
        supervisor._next_by_dependency(
            {
                "job_requirements": {"a": 1},
                "match_analysis": {"b": 2},
                "score_breakdown": {"s": 4},
                "interview_prep": {"c": 3},
            }
        )
        == "report_writer"
    )
    assert (
        supervisor._next_by_dependency(
            {
                "job_requirements": {"a": 1},
                "match_analysis": {"b": 2},
                "score_breakdown": {"s": 4},
                "interview_prep": {"c": 3},
                "final_report": "done",
            }
        )
        == "END"
    )


def test_run_follows_valid_llm_decision(monkeypatch):
    decision = RoutingDecision(next_agent="job_analyzer", reasoning="nothing done yet")
    monkeypatch.setattr(
        supervisor, "get_structured_llm", lambda *a, **k: _fake_structured_llm(decision)
    )

    out = supervisor.run({"messages": []})

    assert out["next_agent"] == "job_analyzer"


def test_run_overrides_llm_decision_that_skips_a_dependency(monkeypatch):
    # LLM tries to jump straight to report_writer before anything is ready.
    decision = RoutingDecision(next_agent="report_writer", reasoning="skip ahead")
    monkeypatch.setattr(
        supervisor, "get_structured_llm", lambda *a, **k: _fake_structured_llm(decision)
    )

    out = supervisor.run({"messages": []})

    assert out["next_agent"] == "job_analyzer"


def test_run_forces_end_once_report_is_final(monkeypatch):
    decision = RoutingDecision(next_agent="cv_matcher", reasoning="LLM mistake")
    monkeypatch.setattr(
        supervisor, "get_structured_llm", lambda *a, **k: _fake_structured_llm(decision)
    )

    state = {
        "job_requirements": {"a": 1},
        "match_analysis": {"b": 2},
        "score_breakdown": {"s": 4},
        "interview_prep": {"c": 3},
        "final_report": "done",
        "messages": [],
    }
    out = supervisor.run(state)

    assert out["next_agent"] == "END"
