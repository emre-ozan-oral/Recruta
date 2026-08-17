"""Structured output schemas produced by each worker agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JobRequirements(BaseModel):
    """Output of the Job Analyzer agent."""

    job_title: str = Field(description="The job title as stated in the posting, e.g. 'Senior Backend Engineer'.")
    company_name: str = Field(
        description="The company's name if stated in the posting. Empty string if not mentioned."
    )
    required_skills: list[str] = Field(
        description=(
            "HARD/technical requirements only — measurable skills, tools, languages, "
            "years of experience, certifications. Do NOT include soft skills, personality "
            "traits, culture-fit language, or passion/motivation statements here — those "
            "go in soft_skills instead."
        )
    )
    soft_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Interpersonal traits, culture-fit language, and passion/motivation statements "
            "from the posting (e.g. 'team player', 'passion for mobile games', 'self-starter'). "
            "These are NOT verifiable from a CV and must never be scored or judged as "
            "'missing' — they're informational only."
        ),
    )
    nice_to_have_skills: list[str] = Field(default_factory=list)
    seniority_level: str = Field(description="e.g. Junior, Mid, Senior, Staff.")
    years_of_experience_required: str = Field(
        description=(
            "The years of professional experience the posting asks for, written as "
            "stated (e.g. '3-5 years', '5+ years', 'At least 2 years of backend "
            "experience'). If the posting does not state a number of years anywhere, "
            "set this to exactly 'Not specified in the posting' — never guess or "
            "infer a figure that isn't actually written there."
        )
    )
    keywords: list[str] = Field(description="ATS-relevant keywords found in the posting.")
    company_context: str = Field(description="Short summary of the company/team/domain.")


class MatchAnalysis(BaseModel):
    """Output of the CV Matcher agent."""

    match_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Overall fit score, 0-100, based ONLY on required_skills (hard/technical). "
            "Never factor in soft_skills or nice_to_have_skills."
        ),
    )
    matched_skills: list[str] = Field(
        description="Hard skills from required_skills that the CV demonstrates."
    )
    missing_skills: list[str] = Field(
        description=(
            "Hard skills from required_skills that the CV does NOT demonstrate. Must be a "
            "subset of required_skills — never include soft skills, traits, or anything "
            "unverifiable from a CV (e.g. never claim a candidate lacks 'passion' for something)."
        )
    )
    strengths: list[str]
    weaknesses: list[str]
    projects_to_highlight: list[str] = Field(
        description="Specific CV projects/experience the candidate should emphasize."
    )


class InterviewPrep(BaseModel):
    """Output of the Interview Prep agent."""

    likely_questions: list[str]
    talking_points: list[str] = Field(
        description="Points the candidate should proactively raise, tied to weak spots."
    )
    questions_to_ask_interviewer: list[str] = Field(default_factory=list)


class RequirementScore(BaseModel):
    """One line item within ScoringResult — one hard requirement, scored."""

    requirement: str = Field(description="The exact hard requirement being scored.")
    weight: int = Field(
        ge=1,
        le=5,
        description="How critical this requirement is to the role, 1 (minor) to 5 (critical).",
    )
    meets_requirement: Literal["yes", "partial", "no"] = Field(
        description=(
            "Whether the CV demonstrates this: 'yes' (clearly shown), 'partial' "
            "(adjacent/transferable experience, not a clean match), or 'no' (no "
            "evidence in the CV at all)."
        )
    )
    evidence: str = Field(
        description="One sentence citing what in the CV supports this verdict (or its absence)."
    )


class ScoringResult(BaseModel):
    """Output of the Scorer agent (agents/scorer.py) — a dedicated,
    reasoning-focused second pass that computes match_score from an
    explicit, weighted, per-requirement breakdown instead of a single
    holistic number CV Matcher estimates as a byproduct of listing
    matched/missing skills."""

    overall_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Weighted overall fit score, 0-100, computed FROM requirement_scores' "
            "weights and verdicts (e.g. weighted-average of yes=1.0/partial=0.5/no=0.0 "
            "by weight) — not an independently guessed number."
        ),
    )
    requirement_scores: list[RequirementScore] = Field(
        description="One entry per hard requirement from job_requirements.required_skills."
    )
    methodology: str = Field(
        description="1-2 sentences on how the weights and overall_score were derived."
    )
    scoring_notes: str = Field(
        default="",
        description=(
            "Optional caveats — e.g. requirements the posting left vague, or how an "
            "unspecified years-of-experience requirement factored (or didn't factor) in."
        ),
    )


class RoutingDecision(BaseModel):
    """Output of the Supervisor agent."""

    next_agent: Literal[
        "job_analyzer", "cv_matcher", "scorer", "interview_prep", "report_writer", "END"
    ]
    reasoning: str = Field(description="One sentence explaining the routing decision.")


class FinalReportOutput(BaseModel):
    """Output of the Report Writer agent."""

    short_summary: str = Field(
        description=(
            "2-3 sentence TL;DR: the fit verdict (e.g. strong/moderate/weak match) "
            "and the single most important thing the candidate should do next."
        )
    )
    final_report: str = Field(description="Full Markdown report, per the required structure.")


class SkillVerification(BaseModel):
    """Output of the ad-hoc skill-flagging evaluator (agents/skill_verifier.py).

    Not part of the main graph — invoked directly from the UI when a user
    flags a "missing" requirement as something they actually have.
    """

    is_plausible: bool = Field(
        description=(
            "Whether this reads like a genuine, specific skill/qualification claim — "
            "not spam, a joke, or something too vague/broad to ever state as a skill "
            "(e.g. 'being a good person'). Judge only specificity/genuineness, never "
            "how impressive or common the skill is."
        )
    )
    normalized_skill: str = Field(
        description="A clean, concise phrasing of the skill, suitable for a skills profile."
    )
    note: str = Field(
        default="",
        description="Optional short note — e.g. why it was rejected, or a normalization caveat.",
    )
