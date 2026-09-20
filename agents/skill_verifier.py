"""Ad-hoc skill-flagging evaluator.

NOT wired into graph.py's StateGraph — unlike the four worker agents, this
one isn't part of the CV-analysis pipeline. It's invoked directly from the
UI (ui_helpers.render_missing_skills_with_flagging) the moment a user flags
a "missing" requirement as something they actually have but forgot to
write in their CV. It normalizes the claim and sanity-checks it before it's
stored in the user's persistent skills profile (db.user_skills), so it can
be automatically credited in future analyses — see app.py's
_augment_cv_with_user_skills, which appends the stored skills onto the CV
text before it's ever sent to the pipeline.
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import SkillVerification

SYSTEM_PROMPT = """You help a job seeker maintain a personal skills profile.
They are flagging a requirement from a job posting as something they
genuinely have but forgot to write in their CV. Normalize it into a short,
clean skill phrase, and sanity-check that it's a real, specific, statable
skill or qualification — not spam, a joke, or something too vague to ever
verify (e.g. "being a good person"). Only judge plausibility as a
*specific claim*, never how impressive or common it is — a beginner-level
skill is just as plausible as an expert one."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Flagged skill/requirement: {skill_text}\n\n"
            "Job context (for reference only — the skill doesn't need to be "
            "mentioned there): {job_context}",
        ),
    ]
)


MAX_SKILL_CHARS = 120
MAX_SKILL_WORDS = 15
UNVERIFIED_NOTE = "Saved without the AI check (the checker was unavailable)."


def _clean(text: str) -> str:
    """Whitespace/bullet/quote cleanup — no judgment involved."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^[\-\*\u2022\u00b7\d\.\)\s]+(?=[A-Za-z(])", "", text)
    return text.strip(" \t\"'`.;,:")


def _deterministic_verdict(skill_text: str) -> SkillVerification | None:
    """Rule-based verdict for input that is already a clean requirement
    phrase. Returns None when the text is too long/sentence-like to judge by
    rules alone (=> an LLM should look at it)."""
    cleaned = _clean(skill_text)
    if len(cleaned) < 2 or not re.search(r"[A-Za-z0-9]", cleaned):
        return SkillVerification(
            is_plausible=False, normalized_skill="", note="Too short to be a skill."
        )
    if len(cleaned) > MAX_SKILL_CHARS or len(cleaned.split()) > MAX_SKILL_WORDS:
        return None
    return SkillVerification(is_plausible=True, normalized_skill=cleaned, note="")


def evaluate_skill(
    skill_text: str, job_context: str = "", *, trust_input: bool = False
) -> SkillVerification:
    """Normalize + sanity-check a flagged skill.

    trust_input=True is for text that was itself extracted from a job posting
    by the Job Analyzer (which is how the UI's "I have this" button calls it):
    that text is already a specific, clean requirement phrase, so an LLM
    round-trip would add latency, cost and a failure mode for no benefit —
    rules decide instantly. Free-form user text (trust_input=False), or
    anything too long for rules, still goes to the LLM.

    If the LLM call fails (rate limit, outage), this degrades instead of
    raising: the claim is stored in its cleaned/shortened form and the note
    says it was not AI-checked — the user's flag is never lost or blocked.
    """
    if trust_input:
        verdict = _deterministic_verdict(skill_text)
        if verdict is not None:
            return verdict

    try:
        llm = get_structured_llm(SkillVerification, tier="fast", temperature=0)
        chain = PROMPT | llm
        return chain.invoke({"skill_text": skill_text, "job_context": job_context})
    except Exception:  # noqa: BLE001 — degrade, don't crash the click
        cleaned = _clean(skill_text)
        if len(cleaned) < 2 or not re.search(r"[A-Za-z0-9]", cleaned):
            return SkillVerification(
                is_plausible=False, normalized_skill="", note="Too short to be a skill."
            )
        return SkillVerification(
            is_plausible=True,
            normalized_skill=cleaned[:MAX_SKILL_CHARS].rstrip(),
            note=UNVERIFIED_NOTE,
        )
