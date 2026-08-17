"""Streamlit rendering helpers: an "open the original file" button, visual
highlighting for analysis results, and the missing-skill flagging widget —
so the UI isn't just walls of text and missing skills aren't a dead end."""

from __future__ import annotations

import base64

import streamlit as st
import streamlit.components.v1 as components

import db
from agents import skill_verifier

SCORE_GREEN = "#1a7f37"
SCORE_AMBER = "#9a6700"
SCORE_RED = "#cf222e"


def _score_color(score: int) -> str:
    if score >= 75:
        return SCORE_GREEN
    if score >= 50:
        return SCORE_AMBER
    return SCORE_RED


def render_score_badge(score: int | None) -> None:
    """Big colored match-score readout instead of a number buried in text."""
    if score is None:
        return
    color = _score_color(score)
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:16px;margin:8px 0 16px 0;">
            <div style="font-size:2.4em;font-weight:700;color:{color};line-height:1;">
                {score}<span style="font-size:0.4em;font-weight:500;">/100</span>
            </div>
            <div style="flex:1;">
                <div style="background:#e1e4e8;border-radius:8px;height:12px;overflow:hidden;">
                    <div style="background:{color};width:{max(score, 2)}%;height:100%;"></div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_chips(items: list[str], color: str) -> None:
    """Small colored pills — used for matched skills, job overview, etc."""
    if not items:
        st.caption("(none)")
        return
    chips_html = "".join(
        f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;'
        f'margin:3px;display:inline-block;font-size:0.85em;">{_escape(item)}</span>'
        for item in items
    )
    st.markdown(chips_html, unsafe_allow_html=True)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_missing_skills_with_flagging(
    missing_skills: list[str], *, key_prefix: str, job_context: str = ""
) -> None:
    """Missing-skills chips, each with a "🚩 I have this" action.

    Clicking it runs the skill through agents/skill_verifier.py (a real,
    lightweight LLM check — not just a raw save) and, if it holds up,
    stores it in the user's persistent skills profile (db.user_skills).
    From then on it's appended to the CV text on every future pipeline run
    (see app.py's _augment_cv_with_user_skills), so it gets credited
    automatically instead of showing up as "missing" again.
    """
    if not missing_skills:
        st.caption("(none)")
        return

    already_have = [s["skill_text"].lower() for s in db.list_user_skills()]

    for i, skill in enumerate(missing_skills):
        row = st.columns([5, 2])
        with row[0]:
            st.markdown(
                f'<span style="background:{SCORE_RED};color:white;padding:3px 10px;'
                f'border-radius:12px;margin:3px 0;display:inline-block;font-size:0.85em;">'
                f"{_escape(skill)}</span>",
                unsafe_allow_html=True,
            )
        with row[1]:
            is_flagged = any(
                skill.lower() in known or known in skill.lower() for known in already_have
            )
            if is_flagged:
                st.caption("✅ in your profile")
            elif st.button(
                "🚩 I have this",
                key=f"{key_prefix}_flag_{i}",
                help="Flag this as a skill you actually have but didn't write in your CV.",
            ):
                with st.spinner("Checking…"):
                    verdict = skill_verifier.evaluate_skill(skill, job_context)
                if verdict.is_plausible:
                    db.add_user_skill(verdict.normalized_skill, note=verdict.note)
                    st.success(
                        f"Added '{verdict.normalized_skill}' to your skills profile — "
                        "future analyses will consider it automatically."
                    )
                else:
                    st.warning(verdict.note or "Couldn't confirm this as a specific skill.")
                st.rerun()


def render_match_analysis(match_analysis: dict, *, key_prefix: str = "match", job_context: str = "") -> None:
    render_score_badge(match_analysis.get("match_score"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Matched skills**")
        render_chips(match_analysis.get("matched_skills", []), SCORE_GREEN)
    with col2:
        st.markdown("**⚠️ Missing skills**")
        render_missing_skills_with_flagging(
            match_analysis.get("missing_skills", []), key_prefix=key_prefix, job_context=job_context
        )

    if match_analysis.get("strengths"):
        with st.container(border=True):
            st.markdown("**💪 Strengths**")
            for item in match_analysis["strengths"]:
                st.markdown(f"- {item}")

    if match_analysis.get("weaknesses"):
        with st.container(border=True):
            st.markdown("**🔧 Gaps to address**")
            for item in match_analysis["weaknesses"]:
                st.markdown(f"- {item}")

    if match_analysis.get("projects_to_highlight"):
        st.markdown("**⭐ Projects to highlight**")
        for item in match_analysis["projects_to_highlight"]:
            st.markdown(f"- {item}")


_VERDICT_STYLE = {
    "yes": (SCORE_GREEN, "✅"),
    "partial": (SCORE_AMBER, "🟡"),
    "no": (SCORE_RED, "❌"),
}


def render_score_breakdown(score_breakdown: dict) -> None:
    """Per-requirement weighted breakdown behind the match score — the
    output of agents/scorer.py's dedicated reasoning pass, not just the
    matched/missing skill chips. Shows how the score was actually computed:
    each requirement's importance (weight) and verdict (yes/partial/no),
    not just a single opaque number."""
    if not score_breakdown:
        return

    if score_breakdown.get("methodology"):
        st.caption(score_breakdown["methodology"])

    for item in score_breakdown.get("requirement_scores", []):
        verdict = item.get("meets_requirement", "")
        color, icon = _VERDICT_STYLE.get(verdict, ("#57606a", "•"))
        weight = item.get("weight") or 0
        stars = "★" * weight + "☆" * (5 - weight)
        st.markdown(
            f'<div style="padding:8px 0;border-bottom:1px solid #e1e4e8;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
            f'<span>{icon} <b>{_escape(item.get("requirement", ""))}</b></span>'
            f'<span style="color:{color};font-size:0.85em;white-space:nowrap;">{stars}</span>'
            f"</div>"
            f'<div style="font-size:0.85em;color:#57606a;margin-top:2px;">{_escape(item.get("evidence", ""))}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    if score_breakdown.get("scoring_notes"):
        st.caption(f"Note: {score_breakdown['scoring_notes']}")


def render_interview_prep(interview_prep: dict) -> None:
    if interview_prep.get("likely_questions"):
        st.markdown("**❓ Likely questions**")
        for i, q in enumerate(interview_prep["likely_questions"], 1):
            st.markdown(f"{i}. {q}")

    if interview_prep.get("talking_points"):
        with st.container(border=True):
            st.markdown("**🗣️ Talking points to raise proactively**")
            for item in interview_prep["talking_points"]:
                st.markdown(f"- {item}")

    if interview_prep.get("questions_to_ask_interviewer"):
        st.markdown("**🙋 Questions to ask them**")
        for item in interview_prep["questions_to_ask_interviewer"]:
            st.markdown(f"- {item}")


def render_job_overview(job_requirements: dict) -> None:
    title = job_requirements.get("job_title")
    company = job_requirements.get("company_name")
    if title or company:
        heading = title or "Role"
        if company:
            heading += f" @ {company}"
        st.markdown(f"##### {heading}")

    if job_requirements.get("company_context"):
        st.markdown(f"*{job_requirements['company_context']}*")
    if job_requirements.get("seniority_level"):
        st.markdown(f"**Seniority:** {job_requirements['seniority_level']}")
    # Always show this line when present in the record at all — including
    # the "Not specified" case — so it reads as "we checked and the
    # posting didn't say" rather than a missing field.
    if job_requirements.get("years_of_experience_required"):
        st.markdown(f"**Experience required:** {job_requirements['years_of_experience_required']}")
    if job_requirements.get("required_skills"):
        st.markdown("**Required skills**")
        render_chips(job_requirements["required_skills"], "#57606a")
    if job_requirements.get("nice_to_have_skills"):
        st.markdown("**Nice to have**")
        render_chips(job_requirements["nice_to_have_skills"], "#8250df")
    if job_requirements.get("soft_skills"):
        st.markdown("**Also valued (soft skills — not scored)**")
        render_chips(job_requirements["soft_skills"], "#6e7781")


def render_file_open_button(
    *,
    file_bytes: bytes | None,
    mime_type: str | None,
    filename: str | None,
    raw_text: str,
    key: str,
) -> None:
    """Instead of an embedded preview (too small to be useful — and PDF
    rendering inline turned out to be unreliable across browsers), offer a
    button that opens the original uploaded file in a new tab. The
    browser's own PDF/image viewer gives a far better reading experience
    than anything renderable inline in a Streamlit sidebar.

    IMPORTANT (2026-08-17): a plain `<a href="data:...;base64,..."
    target="_blank">` (via st.markdown) looked right but didn't actually
    work — modern Chrome/Edge silently block top-level navigation to a
    data: URL triggered by a link click (a phishing-prevention measure
    that's been in Chrome since ~2021; typing the URL directly still
    works, which is why it's easy to miss in manual testing). The fix is
    to build a Blob URL client-side instead (`URL.createObjectURL`) and
    open that with `window.open()` — Blob URLs aren't subject to the same
    restriction. That requires real, executing JavaScript, which
    st.markdown(unsafe_allow_html=True) can't do (script tags inserted via
    innerHTML are inert by spec) — st.components.v1.html() renders in a
    real iframe document instead, so its <script> tags do run. Verified
    with a real Playwright click-and-check-for-new-tab test before
    shipping this, since the previous data-URL approach silently "worked"
    (no exception) while doing nothing when actually clicked.
    """
    name = filename or "your CV"
    lower = name.lower()

    if not file_bytes:
        # No original file on record (e.g. a profile saved by pasting
        # text) — nothing to open, so fall back to the extracted text.
        with st.expander("View extracted text"):
            preview = raw_text[:2000]
            st.text(preview + ("…" if len(raw_text) > 2000 else ""))
        return

    is_pdf = mime_type == "application/pdf" or lower.endswith(".pdf")
    is_image = (mime_type or "").startswith("image/") or lower.endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
    )

    if is_pdf or is_image:
        mime = mime_type or ("application/pdf" if is_pdf else "image/png")
        b64 = base64.b64encode(file_bytes).decode()
        safe_name = _escape(name)
        element_id = f"open-file-btn-{key}".replace(" ", "-")
        html = f"""
        <div style="font-family:'Source Sans Pro',sans-serif;">
          <button id="{element_id}" style="display:inline-block;padding:8px 16px;
            background:#1a7f37;color:white;border:none;border-radius:6px;
            font-weight:600;font-size:0.9em;cursor:pointer;">
            🔗 Open {safe_name} in a new tab
          </button>
        </div>
        <script>
        (function() {{
            const b64Data = "{b64}";
            const mimeType = "{mime}";
            const btn = document.getElementById("{element_id}");
            btn.addEventListener("click", function() {{
                const byteChars = atob(b64Data);
                const byteNumbers = new Array(byteChars.length);
                for (let i = 0; i < byteChars.length; i++) {{
                    byteNumbers[i] = byteChars.charCodeAt(i);
                }}
                const byteArray = new Uint8Array(byteNumbers);
                const blob = new Blob([byteArray], {{ type: mimeType }});
                const blobUrl = URL.createObjectURL(blob);
                window.open(blobUrl, "_blank");
            }});
        }})();
        </script>
        """
        components.html(html, height=50)
    else:
        # Browsers can't render Word docs inline — a "new tab" link would
        # just confuse people, so offer a normal download instead.
        st.download_button(
            f"⬇️ Download {name} to view",
            data=file_bytes,
            file_name=name,
            mime=mime_type or "application/octet-stream",
            key=key,
        )
