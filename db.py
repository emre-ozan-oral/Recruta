"""SQLite-backed storage for CV profiles and past applications.

Deliberately plain CRUD, not a LangGraph agent: saving/loading records is
pure I/O with no reasoning involved, so routing it through an LLM would
only add latency and cost. Kept as a small service module that app.py
(and, if useful later, a non-LLM graph node) can call directly.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "recruta.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cv_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    source_filename TEXT,
    file_bytes BLOB,
    mime_type TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cv_profile_id INTEGER REFERENCES cv_profiles(id) ON DELETE SET NULL,
    job_posting_text TEXT NOT NULL,
    job_requirements TEXT,
    match_analysis TEXT,
    score_breakdown TEXT,
    interview_prep TEXT,
    short_summary TEXT,
    final_report TEXT,
    match_score INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- A persistent "skills I actually have but don't always write in my CV"
-- profile. Populated when the user flags a missing requirement from an
-- analysis (see agents/skill_verifier.py + ui_helpers.py), and read back
-- by app.py to augment the CV text on every future pipeline run — so a
-- flagged skill gets credited automatically next time, without having to
-- re-edit the actual CV file/text.
CREATE TABLE IF NOT EXISTS user_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_text TEXT NOT NULL,
    note TEXT,
    source_application_id INTEGER REFERENCES applications(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a user's DB already existed.

    SQLite has no "ADD COLUMN IF NOT EXISTS", so check first — this keeps
    init_db() safe to call on every startup without erroring on a
    pre-existing recruta.db from an older version of the app.
    """
    existing_cv_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(cv_profiles)").fetchall()
    }
    if "file_bytes" not in existing_cv_columns:
        conn.execute("ALTER TABLE cv_profiles ADD COLUMN file_bytes BLOB")
    if "mime_type" not in existing_cv_columns:
        conn.execute("ALTER TABLE cv_profiles ADD COLUMN mime_type TEXT")

    existing_application_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(applications)").fetchall()
    }
    if "score_breakdown" not in existing_application_columns:
        conn.execute("ALTER TABLE applications ADD COLUMN score_breakdown TEXT")


def init_db() -> None:
    """Create tables if they don't exist yet, and migrate older ones. Safe
    to call on every startup."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


# --- CV profiles -----------------------------------------------------------


def save_cv_profile(
    name: str,
    raw_text: str,
    source_filename: str | None = None,
    file_bytes: bytes | None = None,
    mime_type: str | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO cv_profiles (name, raw_text, source_filename, file_bytes, mime_type)
               VALUES (?, ?, ?, ?, ?)""",
            (name, raw_text, source_filename, file_bytes, mime_type),
        )
        return cur.lastrowid


def list_cv_profiles() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, source_filename, created_at FROM cv_profiles "
            "ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_cv_profile(profile_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM cv_profiles WHERE id = ?", (profile_id,)).fetchone()
        return dict(row) if row else None


def delete_cv_profile(profile_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM cv_profiles WHERE id = ?", (profile_id,))


# --- Applications ------------------------------------------------------------


def save_application(
    *,
    cv_profile_id: int | None,
    job_posting_text: str,
    job_requirements: dict | None = None,
    match_analysis: dict | None = None,
    score_breakdown: dict | None = None,
    interview_prep: dict | None = None,
    short_summary: str | None = None,
    final_report: str | None = None,
) -> int:
    # match_analysis["match_score"] is already the scorer agent's rigorous
    # weighted score by the time this is called (agents/scorer.py
    # overwrites it) — this column just mirrors it for cheap SQL sorting/
    # filtering without needing to json.loads match_analysis every time.
    match_score = (match_analysis or {}).get("match_score")
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO applications
               (cv_profile_id, job_posting_text, job_requirements, match_analysis,
                score_breakdown, interview_prep, short_summary, final_report, match_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cv_profile_id,
                job_posting_text,
                json.dumps(job_requirements, ensure_ascii=False) if job_requirements else None,
                json.dumps(match_analysis, ensure_ascii=False) if match_analysis else None,
                json.dumps(score_breakdown, ensure_ascii=False) if score_breakdown else None,
                json.dumps(interview_prep, ensure_ascii=False) if interview_prep else None,
                short_summary,
                final_report,
                match_score,
            ),
        )
        return cur.lastrowid


def list_applications(cv_profile_id: int | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        if cv_profile_id is not None:
            rows = conn.execute(
                "SELECT * FROM applications WHERE cv_profile_id = ? ORDER BY created_at DESC",
                (cv_profile_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM applications ORDER BY created_at DESC"
            ).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            for field in ("job_requirements", "match_analysis", "score_breakdown", "interview_prep"):
                if record.get(field):
                    record[field] = json.loads(record[field])
            results.append(record)
        return results


def delete_application(application_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM applications WHERE id = ?", (application_id,))


# --- User skills profile ------------------------------------------------------
# Deliberately plain CRUD too — the "is this a real skill?" reasoning happens
# once, in agents/skill_verifier.py, before add_user_skill() is ever called.


def add_user_skill(
    skill_text: str, note: str | None = None, source_application_id: int | None = None
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO user_skills (skill_text, note, source_application_id) VALUES (?, ?, ?)",
            (skill_text, note, source_application_id),
        )
        return cur.lastrowid


def list_user_skills() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM user_skills ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def delete_user_skill(skill_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM user_skills WHERE id = ?", (skill_id,))
