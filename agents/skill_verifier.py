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


def evaluate_skill(skill_text: str, job_context: str = "") -> SkillVerification:
    """Normalize + sanity-check a flagged skill. One quick LLM call, run
    synchronously from a button click — cheap enough not to need the fast/
    content tiering the main pipeline uses, so it always uses tier="fast"."""
    llm = get_structured_llm(SkillVerification, tier="fast", temperature=0)
    chain = PROMPT | llm
    return chain.invoke({"skill_text": skill_text, "job_context": job_context})
