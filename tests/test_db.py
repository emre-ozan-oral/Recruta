"""Tests for the SQLite storage layer, isolated to a temp DB file per test."""

from __future__ import annotations

import sqlite3

import db


def test_cv_profile_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    pid = db.save_cv_profile("Backend CV", "some cv text", "cv.pdf")
    profiles = db.list_cv_profiles()

    assert len(profiles) == 1
    assert profiles[0]["id"] == pid
    assert profiles[0]["name"] == "Backend CV"

    fetched = db.get_cv_profile(pid)
    assert fetched["raw_text"] == "some cv text"

    db.delete_cv_profile(pid)
    assert db.list_cv_profiles() == []


def test_cv_profile_stores_original_file_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    pid = db.save_cv_profile(
        "Backend CV", "extracted text", "cv.pdf", file_bytes=b"%PDF-fake-bytes", mime_type="application/pdf"
    )
    fetched = db.get_cv_profile(pid)

    assert fetched["file_bytes"] == b"%PDF-fake-bytes"
    assert fetched["mime_type"] == "application/pdf"


def test_init_db_migrates_older_schema_without_file_bytes(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)

    # Simulate a pre-migration DB: cv_profiles without file_bytes/mime_type.
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE cv_profiles (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               raw_text TEXT NOT NULL,
               source_filename TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now'))
           )"""
    )
    conn.execute(
        "INSERT INTO cv_profiles (name, raw_text) VALUES ('Old CV', 'old text')"
    )
    conn.commit()
    conn.close()

    db.init_db()  # should migrate in place without raising

    profiles = db.list_cv_profiles()
    assert len(profiles) == 1
    fetched = db.get_cv_profile(profiles[0]["id"])
    assert fetched["raw_text"] == "old text"
    assert fetched["file_bytes"] is None


def test_application_save_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    pid = db.save_cv_profile("Backend CV", "some cv text")
    db.save_application(
        cv_profile_id=pid,
        job_posting_text="Senior backend role",
        job_requirements={"required_skills": ["Python"]},
        match_analysis={"match_score": 77},
        score_breakdown={"overall_score": 77, "requirement_scores": [], "methodology": "weighted avg"},
        interview_prep={"likely_questions": ["Q1"]},
        short_summary="Good match.",
        final_report="# Report",
    )

    applications = db.list_applications()
    assert len(applications) == 1
    record = applications[0]
    assert record["match_score"] == 77
    assert record["job_requirements"] == {"required_skills": ["Python"]}
    assert record["score_breakdown"]["overall_score"] == 77
    assert record["short_summary"] == "Good match."

    scoped = db.list_applications(cv_profile_id=pid)
    assert len(scoped) == 1

    db.delete_application(record["id"])
    assert db.list_applications() == []


def test_init_db_migrates_older_applications_table_without_score_breakdown(tmp_path, monkeypatch):
    """score_breakdown is a new column on an EXISTING table (applications),
    unlike user_skills which is a brand-new table — this needs the explicit
    ALTER TABLE path in _migrate(), not just CREATE TABLE IF NOT EXISTS."""
    db_path = tmp_path / "old.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE applications (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               cv_profile_id INTEGER,
               job_posting_text TEXT NOT NULL,
               job_requirements TEXT,
               match_analysis TEXT,
               interview_prep TEXT,
               short_summary TEXT,
               final_report TEXT,
               match_score INTEGER,
               created_at TEXT NOT NULL DEFAULT (datetime('now'))
           )"""
    )
    conn.execute(
        "INSERT INTO applications (job_posting_text, match_score) VALUES ('An old posting', 55)"
    )
    conn.commit()
    conn.close()

    db.init_db()  # should migrate in place without raising

    applications = db.list_applications()
    assert len(applications) == 1
    assert applications[0]["match_score"] == 55
    assert applications[0]["score_breakdown"] is None

    # And saving a new application post-migration should work normally.
    # (Look up by job_posting_text rather than assuming list order — both
    # rows can share the same created_at second-resolution timestamp.)
    db.save_application(
        cv_profile_id=None,
        job_posting_text="A new posting",
        score_breakdown={"overall_score": 90, "requirement_scores": []},
    )
    newest = next(a for a in db.list_applications() if a["job_posting_text"] == "A new posting")
    assert newest["score_breakdown"]["overall_score"] == 90


def test_application_survives_cv_profile_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    pid = db.save_cv_profile("Backend CV", "some cv text")
    db.save_application(cv_profile_id=pid, job_posting_text="A role")
    db.delete_cv_profile(pid)

    applications = db.list_applications()
    assert len(applications) == 1
    assert applications[0]["cv_profile_id"] is None


def test_user_skills_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    sid = db.add_user_skill("Kubernetes", note="confirmed via job posting flag")
    skills = db.list_user_skills()

    assert len(skills) == 1
    assert skills[0]["id"] == sid
    assert skills[0]["skill_text"] == "Kubernetes"
    assert skills[0]["note"] == "confirmed via job posting flag"

    db.delete_user_skill(sid)
    assert db.list_user_skills() == []


def test_init_db_creates_user_skills_table_on_older_db(tmp_path, monkeypatch):
    """user_skills is a brand-new table (not a new column on an existing
    one), so CREATE TABLE IF NOT EXISTS in init_db() should be enough to
    add it to a pre-existing recruta.db without any extra migration step."""
    db_path = tmp_path / "old.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE cv_profiles (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               name TEXT NOT NULL,
               raw_text TEXT NOT NULL,
               source_filename TEXT,
               created_at TEXT NOT NULL DEFAULT (datetime('now'))
           )"""
    )
    conn.commit()
    conn.close()

    db.init_db()  # should create user_skills without raising

    sid = db.add_user_skill("Docker")
    assert db.list_user_skills()[0]["id"] == sid
