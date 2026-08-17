"""Unit tests for individual worker agents, using a fake LLM (no API key needed)."""

from __future__ import annotations

from agents import cv_matcher, interview_prep, job_analyzer, report_writer
from schemas import FinalReportOutput, InterviewPrep, JobRequirements, MatchAnalysis


def _fake_structured_llm(result):
    """Mimic what get_structured_llm(schema) returns: a Runnable-ish callable
    that `prompt | llm` can compose with and that ignores its input."""
    return lambda _formatted_prompt: result


def test_job_analyzer(monkeypatch):
    fake_result = JobRequirements(
        job_title="Backend Engineer",
        company_name="Acme Corp",
        required_skills=["Python"],
        soft_skills=["passion for mobile games"],
        seniority_level="Senior",
        years_of_experience_required="3-5 years",
        keywords=["backend"],
        company_context="Fintech",
    )
    monkeypatch.setattr(
        job_analyzer, "get_structured_llm", lambda *a, **k: _fake_structured_llm(fake_result)
    )

    out = job_analyzer.run({"job_posting_text": "some posting", "messages": []})

    assert out["job_requirements"]["required_skills"] == ["Python"]
    assert out["job_requirements"]["soft_skills"] == ["passion for mobile games"]
    assert out["job_requirements"]["job_title"] == "Backend Engineer"
    assert out["job_requirements"]["seniority_level"] == "Senior"
    assert out["job_requirements"]["years_of_experience_required"] == "3-5 years"
    assert "job_analyzer" in out["messages"][-1]


def test_job_analyzer_years_of_experience_when_not_specified(monkeypatch):
    """The posting not mentioning a number of years is a valid, expected
    outcome — it must be reported explicitly, never silently blank or
    guessed at."""
    fake_result = JobRequirements(
        job_title="Backend Engineer",
        company_name="",
        required_skills=["Python"],
        seniority_level="Mid",
        years_of_experience_required="Not specified in the posting",
        keywords=[],
        company_context="",
    )
    monkeypatch.setattr(
        job_analyzer, "get_structured_llm", lambda *a, **k: _fake_structured_llm(fake_result)
    )

    out = job_analyzer.run({"job_posting_text": "some posting", "messages": []})

    assert out["job_requirements"]["years_of_experience_required"] == "Not specified in the posting"


def test_cv_matcher(monkeypatch):
    fake_result = MatchAnalysis(
        match_score=80,
        matched_skills=["Python"],
        missing_skills=["Kubernetes"],
        strengths=["APIs"],
        weaknesses=["K8s"],
        projects_to_highlight=["Payments migration"],
    )
    monkeypatch.setattr(
        cv_matcher, "get_structured_llm", lambda *a, **k: _fake_structured_llm(fake_result)
    )

    state = {
        "job_requirements": {"required_skills": ["Python"]},
        "cv_text": "some cv",
        "messages": [],
    }
    out = cv_matcher.run(state)

    assert out["match_analysis"]["match_score"] == 80
    assert "cv_matcher" in out["messages"][-1]


def test_cv_matcher_never_sees_soft_skills(monkeypatch):
    """Structural guard: soft_skills must be stripped before the LLM call,
    not just discouraged in the prompt — this is what stops things like
    'passion for mobile puzzle games' from ever being scored as a missing
    requirement."""
    captured = {}

    def fake_get_structured_llm(*a, **k):
        def _capture(formatted_prompt):
            captured["human_message"] = formatted_prompt.to_messages()[-1].content
            return MatchAnalysis(
                match_score=50,
                matched_skills=[],
                missing_skills=[],
                strengths=[],
                weaknesses=[],
                projects_to_highlight=[],
            )

        return _capture

    monkeypatch.setattr(cv_matcher, "get_structured_llm", fake_get_structured_llm)

    state = {
        "job_requirements": {
            "required_skills": ["Python"],
            "soft_skills": ["passion for mobile puzzle games", "team player"],
        },
        "cv_text": "some cv",
        "messages": [],
    }
    cv_matcher.run(state)

    sent = captured["human_message"]
    assert "soft_skills" not in sent
    assert "passion for mobile puzzle games" not in sent
    assert "Python" in sent


def test_interview_prep(monkeypatch):
    fake_result = InterviewPrep(
        likely_questions=["Tell me about a time you scaled a service."],
        talking_points=["Highlight Helm/K8s exposure."],
    )
    monkeypatch.setattr(
        interview_prep, "get_structured_llm", lambda *a, **k: _fake_structured_llm(fake_result)
    )

    state = {"job_requirements": {}, "match_analysis": {}, "messages": []}
    out = interview_prep.run(state)

    assert out["interview_prep"]["likely_questions"]
    assert "interview_prep" in out["messages"][-1]


def test_report_writer(monkeypatch):
    fake_result = FinalReportOutput(
        short_summary="Strong match (85/100) — apply and highlight the payments migration.",
        final_report="# Job Application Report\n\nAll good.",
    )
    monkeypatch.setattr(
        report_writer, "get_structured_llm", lambda *a, **k: _fake_structured_llm(fake_result)
    )

    state = {
        "job_requirements": {},
        "match_analysis": {},
        "interview_prep": {},
        "messages": [],
    }
    out = report_writer.run(state)

    assert out["final_report"].startswith("# Job Application Report")
    assert out["short_summary"].startswith("Strong match")
    assert "report_writer" in out["messages"][-1]
