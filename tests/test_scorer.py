"""Unit tests for the Scorer agent (agents/scorer.py), using a fake LLM."""

from __future__ import annotations

from agents import scorer
from schemas import RequirementScore, ScoringResult


def _fake_structured_llm(result):
    return lambda _formatted_prompt: result


def test_scorer_overwrites_match_score_with_its_own(monkeypatch):
    fake_result = ScoringResult(
        overall_score=62,
        requirement_scores=[
            RequirementScore(
                requirement="Python",
                weight=5,
                meets_requirement="yes",
                evidence="5 years of Python listed under Experience.",
            ),
            RequirementScore(
                requirement="Kubernetes",
                weight=3,
                meets_requirement="no",
                evidence="No mention of Kubernetes or container orchestration anywhere.",
            ),
        ],
        methodology="Weighted average of yes=1.0/no=0.0 by requirement weight.",
        scoring_notes="",
    )
    monkeypatch.setattr(scorer, "get_structured_llm", lambda *a, **k: _fake_structured_llm(fake_result))

    state = {
        "job_requirements": {
            "required_skills": ["Python", "Kubernetes"],
            "soft_skills": ["team player"],
        },
        # CV Matcher's preliminary (cruder) score — scorer must not just echo it.
        "match_analysis": {"match_score": 80, "matched_skills": ["Python"], "missing_skills": ["Kubernetes"]},
        "cv_text": "some cv",
        "messages": [],
    }
    out = scorer.run(state)

    assert out["score_breakdown"]["overall_score"] == 62
    assert out["match_analysis"]["match_score"] == 62  # overwritten, not left at 80
    assert out["match_analysis"]["matched_skills"] == ["Python"]  # rest of match_analysis preserved
    assert "scorer" in out["messages"][-1]


def test_scorer_never_sees_soft_skills(monkeypatch):
    """Same structural guard as cv_matcher — soft skills must never reach
    the scoring prompt, so they can never be judged 'missing' there either."""
    captured = {}

    def fake_get_structured_llm(*a, **k):
        def _capture(formatted_prompt):
            captured["human_message"] = formatted_prompt.to_messages()[-1].content
            return ScoringResult(
                overall_score=100,
                requirement_scores=[],
                methodology="n/a",
            )

        return _capture

    monkeypatch.setattr(scorer, "get_structured_llm", fake_get_structured_llm)

    state = {
        "job_requirements": {
            "required_skills": ["Python"],
            "soft_skills": ["passion for mobile puzzle games"],
        },
        "match_analysis": {"match_score": 50},
        "cv_text": "some cv",
        "messages": [],
    }
    scorer.run(state)

    sent = captured["human_message"]
    assert "soft_skills" not in sent
    assert "passion for mobile puzzle games" not in sent
    assert "Python" in sent


def test_scorer_falls_back_instead_of_crashing_when_llm_call_fails(monkeypatch):
    """The scoring pass has already failed live once (a Groq structured-
    output validation error) — this is the safety net so that a repeat
    (or any other LLM/parsing failure) degrades to a simple score instead
    of losing the whole analysis run."""

    def _raise(*a, **k):
        raise RuntimeError("Error code: 400 - json_validate_failed")

    monkeypatch.setattr(scorer, "get_structured_llm", lambda *a, **k: _raise)

    state = {
        "job_requirements": {"required_skills": ["Python", "Kubernetes", "AWS"]},
        "match_analysis": {
            "match_score": 80,
            "matched_skills": ["Python", "AWS"],
            "missing_skills": ["Kubernetes"],
        },
        "cv_text": "some cv",
        "messages": [],
    }
    out = scorer.run(state)

    # 2 matched / 3 total = 67 (rounded), not a crash and not just echoing
    # cv_matcher's original 80.
    assert out["score_breakdown"]["overall_score"] == 67
    assert out["score_breakdown"]["requirement_scores"] == []
    assert "failed" in out["score_breakdown"]["methodology"].lower()
    assert out["match_analysis"]["match_score"] == 67
    assert "fallback" in out["messages"][-1].lower()
