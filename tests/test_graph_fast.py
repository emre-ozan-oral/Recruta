"""graph_fast: correctness, parallelism, and failure isolation (fake LLMs)."""

from __future__ import annotations

import threading
import time

import pytest
from langchain_core.runnables import RunnableLambda

import graph_fast
import schemas
from agents import cv_matcher, interview_prep, job_analyzer, scorer

CV = "Ada. Python, LangGraph."
JOB = "Backend engineer. Python, LangGraph, Docker."


def _fake_factory(delay=0.0, fail=(), log=None):
    def factory(schema, *, tier="content", temperature=0.2):
        name = schema.__name__

        def call(_prompt):
            if log is not None:
                log.append((name, threading.current_thread().name, time.perf_counter()))
            time.sleep(delay)
            if name in fail:
                raise RuntimeError(f"{name} boom")
            return {
                "JobRequirements": lambda: schemas.JobRequirements(
                    job_title="Backend Eng", company_name="Acme",
                    required_skills=["Python", "LangGraph", "Docker"], soft_skills=["team player"],
                    seniority_level="Mid", years_of_experience_required="Not specified in the posting",
                    keywords=[], company_context="c"),
                "MatchAnalysis": lambda: schemas.MatchAnalysis(
                    match_score=60, matched_skills=["Python", "LangGraph"], missing_skills=["Docker"],
                    strengths=["s"], weaknesses=["w"], projects_to_highlight=["p"]),
                "ScoringResult": lambda: schemas.ScoringResult(
                    overall_score=80,
                    requirement_scores=[
                        schemas.RequirementScore(requirement="Docker", weight=4, meets_requirement="no", evidence="none"),
                        schemas.RequirementScore(requirement="Python", weight=5, meets_requirement="yes", evidence="e"),
                    ],
                    methodology="m"),
                "InterviewPrep": lambda: schemas.InterviewPrep(likely_questions=["q"], talking_points=["t"]),
            }[name]()

        return RunnableLambda(call)

    return factory


def _patch(monkeypatch, **kw):
    fac = _fake_factory(**kw)
    for mod in (job_analyzer, cv_matcher, scorer, interview_prep):
        monkeypatch.setattr(mod, "get_structured_llm", fac)


def test_fast_graph_produces_complete_result(monkeypatch):
    _patch(monkeypatch)
    out = graph_fast.run_fast_pipeline(CV, JOB)

    assert out["job_requirements"]["job_title"] == "Backend Eng"
    assert out["score_breakdown"]["overall_score"] == 80
    assert out["match_analysis"]["match_score"] == 80  # scorer's score is the source of truth
    assert out["interview_prep"]["likely_questions"] == ["q"]
    assert "# Job Application Report" in out["final_report"]
    assert "80/100" in out["final_report"]
    assert "Strong match" in out["short_summary"] and "Docker" in out["short_summary"]
    assert not out["errors"]
    # messages from the parallel branches are all present exactly once
    assert len([m for m in out["messages"] if m.startswith("finalize")]) == 1
    assert len(out["messages"]) == len(set(out["messages"]))


def test_only_four_llm_calls_and_three_run_concurrently(monkeypatch):
    log: list = []
    _patch(monkeypatch, delay=0.3, log=log)
    t0 = time.perf_counter()
    graph_fast.run_fast_pipeline(CV, JOB)
    wall = time.perf_counter() - t0

    assert [n for n, *_ in log][0] == "JobRequirements"
    assert sorted(n for n, *_ in log) == sorted(["JobRequirements", "MatchAnalysis", "ScoringResult", "InterviewPrep"])
    starts = {n: t for n, _, t in log}
    parallel = [starts["MatchAnalysis"], starts["ScoringResult"], starts["InterviewPrep"]]
    assert max(parallel) - min(parallel) < 0.15  # started together, not one after another
    assert wall < 0.3 * 2 + 0.25  # ~2 serial stages, not 4


def test_scorer_failure_falls_back_and_run_survives(monkeypatch):
    _patch(monkeypatch, fail=("ScoringResult",))
    out = graph_fast.run_fast_pipeline(CV, JOB)
    assert "Fallback" in out["score_breakdown"]["methodology"]
    assert out["match_analysis"]["match_score"] == 67  # 2 matched / 3 total
    assert "scorer" in out["errors"]
    assert out["final_report"]


def test_interview_prep_failure_leaves_empty_section_not_a_crash(monkeypatch):
    _patch(monkeypatch, fail=("InterviewPrep",))
    out = graph_fast.run_fast_pipeline(CV, JOB)
    assert out["interview_prep"]["likely_questions"] == []
    assert "interview_prep" in out["errors"]
    assert out["score_breakdown"]["overall_score"] == 80


def test_job_analyzer_failure_is_fatal(monkeypatch):
    _patch(monkeypatch, fail=("JobRequirements",))
    with pytest.raises(Exception):
        graph_fast.run_fast_pipeline(CV, JOB)


def test_bench_call_recorder_captures_tokens_and_429s():
    """bench.py's recorder must see per-call tokens and rate-limit errors."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    import bench

    rec = bench.CallRecorder(time.perf_counter())
    ok = GenericFakeChatModel(messages=iter([AIMessage(
        content="hi", usage_metadata={"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500})]))
    ok.invoke("x", config={"callbacks": [rec]})

    class Boom(GenericFakeChatModel):
        def _generate(self, *a, **k):
            raise RuntimeError("Error code: 429 - rate_limit_exceeded")

    with pytest.raises(RuntimeError):
        Boom(messages=iter([])).invoke("x", config={"callbacks": [rec]})

    assert rec.calls[0]["in_tokens"] == 1200 and rec.calls[0]["out_tokens"] == 300
    assert "429" in rec.calls[1]["error"]


def test_warm_up_prebuilds_every_chain_the_fast_graph_uses(monkeypatch):
    """After warm_up(), running the pipeline must not construct any new client."""
    import llm

    monkeypatch.setenv("GROQ_API_KEY", "x")
    llm._build_chain.cache_clear()
    canned = _fake_factory()
    built = {"n": 0}

    class FakeChat:
        def __init__(self, **kw):
            built["n"] += 1

        def with_structured_output(self, schema, **kw):
            return canned(schema)

    monkeypatch.setattr(llm, "ChatGroq", FakeChat)
    graph_fast.warm_up()
    after_warm_up = built["n"]
    assert after_warm_up > 0

    out = graph_fast.run_fast_pipeline(CV, JOB)

    assert out["final_report"]
    assert built["n"] == after_warm_up  # no client construction on the hot path
