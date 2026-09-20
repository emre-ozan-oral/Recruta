"""Regression tests for the "🚩 I have this" skill-flagging button:
(1) an LLM failure must show an inline error, not an uncaught traceback;
(2) the result message must survive the st.rerun() that follows a click."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import db
import graph
import schemas
import ui_helpers

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

RESULT = {
    "job_requirements": {
        "job_title": "Backend Eng", "company_name": "Acme", "required_skills": ["Docker"],
        "soft_skills": [], "nice_to_have_skills": [], "seniority_level": "Mid",
        "years_of_experience_required": "3", "keywords": [], "company_context": "c",
    },
    "match_analysis": {
        "match_score": 50, "matched_skills": [], "missing_skills": ["Docker"],
        "strengths": [], "weaknesses": [], "projects_to_highlight": [],
    },
    "score_breakdown": {"overall_score": 50, "requirement_scores": [], "methodology": "m", "scoring_notes": ""},
    "interview_prep": {"likely_questions": [], "talking_points": []},
    "final_report": "r", "short_summary": "s", "messages": [],
}


class _FakeGraph:
    def invoke(self, state, config=None):
        return dict(RESULT)


def _app_with_result(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "flag.db")
    monkeypatch.setattr(graph, "build_graph", lambda: _FakeGraph())
    at = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    at.radio(key="cv_input_mode").set_value("Paste text").run()
    at.text_input(key="new_cv_name").set_value("CV").run()
    at.text_area(key="cv_paste_text").set_value("Ada, Python dev").run()
    next(b for b in at.button if b.label == "Save CV profile").click().run()
    next(t for t in at.text_area if t.key != "cv_paste_text").set_value("job").run()
    next(b for b in at.button if b.label == "Run analysis").click().run()
    return at


def _flag_button(at):
    return next(b for b in at.button if "I have this" in b.label)


def test_flag_llm_failure_shows_error_not_traceback(tmp_path, monkeypatch):
    at = _app_with_result(tmp_path, monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("429 rate_limit_exceeded")

    monkeypatch.setattr(ui_helpers.skill_verifier, "evaluate_skill", boom)
    _flag_button(at).click().run()

    assert not at.exception
    assert any("Couldn't check this skill" in e.value for e in at.error)
    assert db.list_user_skills() == []


def test_flag_success_saves_skill_and_shows_toast(tmp_path, monkeypatch):
    at = _app_with_result(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ui_helpers.skill_verifier,
        "evaluate_skill",
        lambda skill, ctx="", **kw: schemas.SkillVerification(is_plausible=True, normalized_skill="Docker"),
    )
    _flag_button(at).click().run()

    assert not at.exception
    assert [s["skill_text"] for s in db.list_user_skills()] == ["Docker"]
    assert any("Added 'Docker'" in t.value for t in at.toast)


def test_flag_works_end_to_end_while_llm_is_down(tmp_path, monkeypatch):
    """The real fix, not just an error message: with the checker LLM fully
    unavailable, flagging a requirement still succeeds and is saved."""
    at = _app_with_result(tmp_path, monkeypatch)

    def boom(*a, **kw):
        raise RuntimeError("429 rate_limit_exceeded")

    monkeypatch.setattr(ui_helpers.skill_verifier, "get_structured_llm", boom)
    _flag_button(at).click().run()

    assert not at.exception
    assert not at.error
    assert [s["skill_text"] for s in db.list_user_skills()] == ["Docker"]
    assert any("Added 'Docker'" in t.value for t in at.toast)
