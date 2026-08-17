"""Headless Streamlit smoke tests for the custom HTML rendering helpers —
catches runtime errors (KeyError/AttributeError/bad HTML calls) that a
plain py_compile wouldn't."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

_SCRIPT = """
# Isolate DB state for this smoke test — db.list_user_skills() is called
# by render_missing_skills_with_flagging(), so the user_skills table needs
# to exist (init_db()) and shouldn't touch the real recruta.db.
import pathlib
import tempfile

import db

db.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "ui_helpers_smoke.db"
db.init_db()

import ui_helpers

ui_helpers.render_score_badge(82)
ui_helpers.render_score_badge(None)  # should no-op, not raise

ui_helpers.render_chips(["Python", "FastAPI"], "#1a7f37")
ui_helpers.render_chips([], "#cf222e")  # empty list path

ui_helpers.render_match_analysis({
    "match_score": 70,
    "matched_skills": ["Python"],
    "missing_skills": ["Kubernetes"],
    "strengths": ["Strong API design experience"],
    "weaknesses": ["Limited K8s exposure"],
    "projects_to_highlight": ["Payments migration"],
}, key_prefix="smoke_match", job_context="Senior Backend Engineer at Acme")

# Missing-skills flagging widget: one already-flagged skill (should show
# the "in your profile" state, not a button) and one not-yet-flagged.
db.add_user_skill("Kubernetes")
ui_helpers.render_missing_skills_with_flagging(
    ["Kubernetes", "Terraform"], key_prefix="smoke_flagging", job_context="DevOps role"
)
ui_helpers.render_missing_skills_with_flagging([], key_prefix="smoke_flagging_empty")

ui_helpers.render_score_breakdown({
    "overall_score": 68,
    "requirement_scores": [
        {"requirement": "Python", "weight": 5, "meets_requirement": "yes", "evidence": "5 years listed."},
        {"requirement": "Kubernetes", "weight": 3, "meets_requirement": "no", "evidence": "No mention anywhere."},
        {"requirement": "GraphQL", "weight": 2, "meets_requirement": "partial", "evidence": "REST APIs only, no GraphQL."},
    ],
    "methodology": "Weighted average of yes=1.0/partial=0.5/no=0.0 by requirement weight.",
    "scoring_notes": "Years of experience wasn't specified in the posting, so it wasn't scored.",
})
ui_helpers.render_score_breakdown({})  # empty dict should no-op, not raise

ui_helpers.render_interview_prep({
    "likely_questions": ["Tell me about a time you scaled a service."],
    "talking_points": ["Highlight Helm/K8s exposure."],
    "questions_to_ask_interviewer": ["What does the on-call rotation look like?"],
})

ui_helpers.render_job_overview({
    "job_title": "Senior Backend Engineer",
    "company_name": "Acme Corp",
    "company_context": "Mid-size fintech, AI platform team",
    "seniority_level": "Senior",
    "years_of_experience_required": "5+ years",
    "required_skills": ["Python", "PostgreSQL"],
    "nice_to_have_skills": ["Kubernetes"],
    "soft_skills": ["passion for mobile puzzle games", "team player"],
})

# years_of_experience_required should render even in the "not specified" case
ui_helpers.render_job_overview({
    "job_title": "Junior Support Engineer",
    "required_skills": ["SQL"],
    "years_of_experience_required": "Not specified in the posting",
})

# render_job_overview should also tolerate a record with no title/company
# (e.g. legacy history rows saved before job_title/company_name existed)
ui_helpers.render_job_overview({"required_skills": ["Python"]})

# Open-file button: text-only fallback (no file_bytes)
ui_helpers.render_file_open_button(
    file_bytes=None, mime_type=None, filename=None, raw_text="Plain pasted CV text " * 20,
    key="open_text",
)

# Open-file button: image path
import io
from PIL import Image
img = Image.new("RGB", (10, 10), color="red")
buf = io.BytesIO()
img.save(buf, format="PNG")
ui_helpers.render_file_open_button(
    file_bytes=buf.getvalue(), mime_type="image/png", filename="cv.png", raw_text="ocr text",
    key="open_image",
)

# Open-file button: PDF path
from pypdf import PdfWriter
pdf_buf = io.BytesIO()
writer = PdfWriter()
writer.add_blank_page(width=200, height=200)
writer.write(pdf_buf)
ui_helpers.render_file_open_button(
    file_bytes=pdf_buf.getvalue(), mime_type="application/pdf", filename="cv.pdf", raw_text="pdf text",
    key="open_pdf",
)

# Open-file button: docx path (falls through to a download button — browsers
# can't render Word docs inline)
from docx import Document
doc = Document()
doc.add_paragraph("Ada Yilmaz")
docx_buf = io.BytesIO()
doc.save(docx_buf)
ui_helpers.render_file_open_button(
    file_bytes=docx_buf.getvalue(), mime_type=None, filename="cv.docx", raw_text="Ada Yilmaz",
    key="open_docx",
)
"""


def test_ui_helpers_render_without_raising():
    at = AppTest.from_string(_SCRIPT, default_timeout=10)
    at.run()

    assert not at.exception, f"ui_helpers raised: {[e.value for e in at.exception]}"
