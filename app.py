"""Streamlit UI for the job application assistant.

Two tabs: Analyze (pick/save a CV profile, feed in a job posting, run the
pipeline, optionally save the result) and History (a scannable list of
everything you've chosen to save to recruta.db).
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import db
import ui_helpers
from file_parsing import UnsupportedFileType, extract_text
from graph import build_graph
from langfuse_utils import get_langfuse_handler

load_dotenv()
db.init_db()

st.set_page_config(page_title="Job Application Assistant", page_icon="📄", layout="wide")

st.title("📄 Job Application Assistant")
st.caption(
    "Multi-agent LangGraph pipeline: analyzes a job posting, matches it against your "
    "CV, and generates interview prep — all in one report."
)

if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is not set. Copy `.env.example` to `.env`, add your key from "
        "https://console.groq.com/keys, and restart the app.",
        icon="⚠️",
    )

UPLOAD_TYPES = ["pdf", "docx", "png", "jpg", "jpeg", "webp", "txt", "md"]


def _handle_upload(uploaded_file):
    """Extract text from a Streamlit UploadedFile, surfacing errors inline."""
    try:
        return extract_text(uploaded_file.getvalue(), uploaded_file.name)
    except (UnsupportedFileType, ValueError) as exc:
        st.error(str(exc))
        return None


def _history_label(app_record: dict) -> str:
    score = app_record.get("match_score")
    score_str = f"{score}/100" if score is not None else "no score"
    job_req = app_record.get("job_requirements") or {}
    title = (job_req.get("job_title") or "Untitled role").strip()
    company = (job_req.get("company_name") or "").strip()
    # If the posting genuinely didn't name a company (or this record
    # predates the company_name field), say so explicitly instead of just
    # dropping it silently — that reads as a bug otherwise.
    heading = f"{title} @ {company}" if company else f"{title} (company not listed)"
    return f"{heading} — {score_str}"


def _augment_cv_with_user_skills(cv_text: str) -> str:
    """Append the user's flagged/confirmed skills profile (db.user_skills)
    onto the CV text sent to the pipeline, so anything flagged from a past
    analysis gets credited automatically — without touching the actual
    saved CV profile text."""
    skills = db.list_user_skills()
    if not skills:
        return cv_text
    bullet_list = "\n".join(f"- {s['skill_text']}" for s in skills)
    return (
        cv_text
        + "\n\nAdditional skills the candidate has confirmed they have "
        "(not necessarily written elsewhere in this CV):\n"
        + bullet_list
    )


def _render_result(result: dict, *, key_prefix: str = "live") -> None:
    """Structured, highlighted results view — not just a markdown dump."""
    if result.get("short_summary"):
        st.info(result["short_summary"], icon="📝")

    job_requirements = result.get("job_requirements") or {}
    match_analysis = result.get("match_analysis") or {}
    score_breakdown = result.get("score_breakdown") or {}
    interview_prep = result.get("interview_prep") or {}

    title = job_requirements.get("job_title") or ""
    company = job_requirements.get("company_name") or ""
    job_context = f"{title} at {company}".strip() if (title or company) else ""

    if job_requirements:
        st.markdown("#### Job overview")
        ui_helpers.render_job_overview(job_requirements)

    if match_analysis:
        st.markdown("#### Match analysis")
        ui_helpers.render_match_analysis(
            match_analysis, key_prefix=key_prefix, job_context=job_context
        )

    if score_breakdown:
        with st.expander("📊 How this score was calculated"):
            ui_helpers.render_score_breakdown(score_breakdown)

    if interview_prep:
        st.markdown("#### Interview prep")
        ui_helpers.render_interview_prep(interview_prep)

    with st.expander("Full written report"):
        st.markdown(result.get("final_report", "_No report was generated._"))

    with st.expander("Agent trace"):
        for line in result.get("messages", []):
            st.text(line)

    with st.expander("Raw intermediate outputs (JSON)"):
        st.json(
            {
                "job_requirements": job_requirements,
                "match_analysis": match_analysis,
                "score_breakdown": score_breakdown,
                "interview_prep": interview_prep,
            }
        )


# --- Sidebar: CV profile management -----------------------------------------

st.sidebar.header("Your CV")
profiles = db.list_cv_profiles()
profile_options = {p["id"]: p["name"] for p in profiles}

selected_cv_id = None
if profiles:
    selected_cv_id = st.sidebar.selectbox(
        "Active CV profile",
        options=list(profile_options.keys()),
        format_func=lambda pid: profile_options[pid],
    )
else:
    st.sidebar.caption("No saved CV yet — add one below.")

with st.sidebar.expander("+ Add new CV", expanded=not profiles):
    new_cv_name = st.text_input(
        "Profile name", key="new_cv_name", placeholder="e.g. Backend CV 2026"
    )
    cv_input_mode = st.radio(
        "Input method", ["Upload file", "Paste text"], key="cv_input_mode", horizontal=True
    )

    cv_upload_text = None
    cv_source_filename = None
    cv_file_bytes = None
    cv_mime_type = None
    if cv_input_mode == "Upload file":
        cv_file = st.file_uploader("CV file", type=UPLOAD_TYPES, key="cv_uploader")
        if cv_file is not None:
            cv_upload_text = _handle_upload(cv_file)
            cv_source_filename = cv_file.name
            cv_file_bytes = cv_file.getvalue()
            cv_mime_type = cv_file.type
            if cv_upload_text:
                st.success(f"Extracted {len(cv_upload_text)} characters from {cv_file.name}.")
    else:
        cv_upload_text = st.text_area("Paste CV text", height=200, key="cv_paste_text")

    if st.button("Save CV profile", disabled=not (new_cv_name and cv_upload_text)):
        db.save_cv_profile(
            new_cv_name, cv_upload_text, cv_source_filename, cv_file_bytes, cv_mime_type
        )
        st.success("Saved.")
        st.rerun()

active_cv_text = None
if selected_cv_id is not None:
    profile = db.get_cv_profile(selected_cv_id)
    if profile:
        active_cv_text = profile["raw_text"]
        with st.sidebar:
            st.caption(f"📄 {profile.get('source_filename') or 'Pasted text'}")
            ui_helpers.render_file_open_button(
                file_bytes=profile.get("file_bytes"),
                mime_type=profile.get("mime_type"),
                filename=profile.get("source_filename"),
                raw_text=active_cv_text,
                key=f"cv_open_{selected_cv_id}",
            )
        if st.sidebar.button("Delete this CV profile"):
            db.delete_cv_profile(selected_cv_id)
            st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🎯 Your flagged skills")
st.sidebar.caption(
    "Skills you've confirmed you have but don't always write in your CV — "
    "these get factored into every future analysis automatically."
)
user_skills = db.list_user_skills()
if not user_skills:
    st.sidebar.caption("None yet — flag a missing skill from your results below.")
else:
    for skill in user_skills:
        row = st.sidebar.columns([4, 1])
        row[0].markdown(f"- {skill['skill_text']}")
        if row[1].button("🗑️", key=f"delete_user_skill_{skill['id']}", help="Remove this skill"):
            db.delete_user_skill(skill["id"])
            st.rerun()


# --- Main area ---------------------------------------------------------------

tab_analyze, tab_history = st.tabs(["Analyze", "History"])

with tab_analyze:
    st.subheader("Job posting")
    job_input_mode = st.radio(
        "Input method", ["Paste text", "Upload file"], key="job_input_mode", horizontal=True
    )

    job_posting_text = None
    if job_input_mode == "Paste text":
        job_posting_text = st.text_area(
            "Job posting text", height=300, placeholder="Paste the job posting here…"
        )
    else:
        job_file = st.file_uploader("Job posting file", type=UPLOAD_TYPES, key="job_uploader")
        if job_file is not None:
            job_posting_text = _handle_upload(job_file)
            if job_posting_text:
                st.success(f"Extracted {len(job_posting_text)} characters from {job_file.name}.")
                with st.expander("Preview extracted text"):
                    st.text(job_posting_text[:3000])

    run_clicked = st.button(
        "Run analysis",
        type="primary",
        disabled=not (active_cv_text and job_posting_text),
        help=None if active_cv_text else "Select or add a CV profile in the sidebar first.",
    )

    if run_clicked:
        with st.spinner("Running the agent pipeline… this can take a minute."):
            try:
                handler = get_langfuse_handler()
                graph = build_graph()
                config = {"callbacks": [handler]} if handler else {}
                cv_text_for_pipeline = _augment_cv_with_user_skills(active_cv_text)
                result = graph.invoke(
                    {
                        "cv_text": cv_text_for_pipeline,
                        "job_posting_text": job_posting_text,
                        "messages": [],
                    },
                    config=config,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()

        # Stash in session_state rather than saving straight to the DB —
        # saving is an explicit user choice (see the button below).
        st.session_state["last_result"] = result
        st.session_state["last_job_posting_text"] = job_posting_text
        st.session_state["last_cv_id"] = selected_cv_id
        st.session_state["last_saved"] = False

    if "last_result" in st.session_state:
        st.divider()
        _render_result(st.session_state["last_result"])

        if st.session_state.get("last_saved"):
            st.success("Saved to your application history.", icon="💾")
        else:
            if st.button("💾 Save this application to history"):
                result = st.session_state["last_result"]
                db.save_application(
                    cv_profile_id=st.session_state.get("last_cv_id"),
                    job_posting_text=st.session_state.get("last_job_posting_text", ""),
                    job_requirements=result.get("job_requirements"),
                    match_analysis=result.get("match_analysis"),
                    score_breakdown=result.get("score_breakdown"),
                    interview_prep=result.get("interview_prep"),
                    short_summary=result.get("short_summary"),
                    final_report=result.get("final_report"),
                )
                st.session_state["last_saved"] = True
                st.rerun()

with tab_history:
    st.subheader("Past applications")
    applications = db.list_applications(cv_profile_id=None)

    if not applications:
        st.caption("No applications analyzed yet — run one from the Analyze tab and save it.")
    else:
        for record in applications:
            score = record.get("match_score")
            date = (record.get("created_at") or "").split(" ")[0]
            with st.expander(f"{date} — {_history_label(record)}"):
                st.caption("Job posting (excerpt):")
                st.text((record.get("job_posting_text") or "")[:500])

                result_view = {
                    "short_summary": record.get("short_summary"),
                    "job_requirements": record.get("job_requirements"),
                    "match_analysis": record.get("match_analysis"),
                    "score_breakdown": record.get("score_breakdown"),
                    "interview_prep": record.get("interview_prep"),
                    "final_report": record.get("final_report"),
                    "messages": [],
                }
                _render_result(result_view, key_prefix=f"hist_{record['id']}")

                if st.button("🗑️ Delete this record", key=f"delete_application_{record['id']}"):
                    db.delete_application(record["id"])
                    st.rerun()
