"""Headless Streamlit smoke test for app.py: does it load without raising,
and does the CV-profile save flow (paste-text path) actually work end to
end against a scratch DB? No live Groq call is involved."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import db

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_loads_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "apptest.db")

    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.run()

    assert not at.exception
    assert at.title[0].value == "📄 Job Application Assistant"


def test_save_cv_profile_via_paste_text_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "apptest.db")

    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.run()

    # Switch the sidebar's CV input mode from the "Upload file" default to
    # "Paste text" so the paste text_area renders.
    at.radio(key="cv_input_mode").set_value("Paste text").run()
    assert not at.exception

    at.text_input(key="new_cv_name").set_value("Test CV").run()
    at.text_area(key="cv_paste_text").set_value("Ada Yilmaz, backend engineer.").run()
    assert not at.exception

    save_button = next(b for b in at.button if b.label == "Save CV profile")
    assert not save_button.disabled
    save_button.click().run()

    assert not at.exception
    profiles = db.list_cv_profiles()
    assert len(profiles) == 1
    assert profiles[0]["name"] == "Test CV"

    # After the rerun, the new profile should be selectable in the sidebar.
    cv_selectbox = at.sidebar.selectbox[0]
    assert "Test CV" in cv_selectbox.options
