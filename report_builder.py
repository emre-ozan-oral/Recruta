"""Deterministic report + summary assembly (no LLM).

The LLM Report Writer (agents/report_writer.py) re-generates ~1k tokens of
Markdown out of JSON that other agents already produced — the slowest,
most token-expensive step for the least new information. graph_fast.py uses
these pure functions instead: identical section structure, zero latency,
zero tokens, and the numbers in the report can never disagree with the
structured data because they're copied, not re-generated.
"""

from __future__ import annotations

from typing import Any


def verdict_for(score: int) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    return "weak"


def _biggest_gap(score_breakdown: dict[str, Any], match_analysis: dict[str, Any]) -> str | None:
    """The highest-weight requirement the CV doesn't (fully) meet."""
    rank = {"no": 0, "partial": 1}
    gaps = [
        r for r in (score_breakdown or {}).get("requirement_scores", [])
        if r.get("meets_requirement") in rank
    ]
    if gaps:
        gaps.sort(key=lambda r: (-int(r.get("weight", 0)), rank[r["meets_requirement"]]))
        return gaps[0]["requirement"]
    missing = (match_analysis or {}).get("missing_skills") or []
    return missing[0] if missing else None


def build_short_summary(state: dict[str, Any]) -> str:
    match = state.get("match_analysis") or {}
    score = int(match.get("match_score", 0))
    job = state.get("job_requirements") or {}
    title = job.get("job_title") or "this role"
    company = job.get("company_name")
    where = f"{title} at {company}" if company else title
    verdict = verdict_for(score)
    gap = _biggest_gap(state.get("score_breakdown") or {}, match)
    if gap:
        nxt = f"Most important next step: address '{gap}' (the highest-weighted gap) in your CV or cover letter."
    else:
        nxt = "No significant gaps found — tailor your CV to the posting's keywords and apply."
    return f"{verdict.capitalize()} match for {where} ({score}/100). {nxt}"


def _bullets(items: list[str] | None) -> str:
    return "\n".join(f"- {i}" for i in (items or [])) or "_None._"


def build_report(state: dict[str, Any]) -> str:
    job = state.get("job_requirements") or {}
    match = state.get("match_analysis") or {}
    prep = state.get("interview_prep") or {}
    breakdown = state.get("score_breakdown") or {}
    score = match.get("match_score", "n/a")
    years = job.get("years_of_experience_required") or "Not specified in the posting"

    lines = [
        "# Job Application Report",
        "",
        "## Job Overview",
        f"- **Role:** {job.get('job_title', 'n/a')}"
        + (f" at {job['company_name']}" if job.get("company_name") else ""),
        f"- **Seniority:** {job.get('seniority_level', 'n/a')}",
        f"- **Experience required:** {years}",
        f"- **Required skills:** {', '.join(job.get('required_skills') or []) or 'n/a'}",
        f"- **Nice to have:** {', '.join(job.get('nice_to_have_skills') or []) or 'n/a'}",
        f"- **Company context:** {job.get('company_context') or 'n/a'}",
        "",
        f"## Match Analysis — {score}/100",
        f"**Matched:** {', '.join(match.get('matched_skills') or []) or 'none'}",
        "",
        f"**Missing:** {', '.join(match.get('missing_skills') or []) or 'none'}",
    ]
    if breakdown.get("methodology"):
        lines += ["", f"_How the score was calculated: {breakdown['methodology']}_"]
    lines += [
        "",
        "## Strengths & Projects to Highlight",
        _bullets(match.get("strengths")),
        "",
        "**Projects to highlight**",
        _bullets(match.get("projects_to_highlight")),
        "",
        "## Gaps to Address",
        _bullets(match.get("weaknesses")),
        "",
        "## Interview Preparation",
        "### Likely Questions",
        _bullets(prep.get("likely_questions")),
        "",
        "### Talking Points",
        _bullets(prep.get("talking_points")),
        "",
        "### Questions to Ask the Interviewer",
        _bullets(prep.get("questions_to_ask_interviewer")),
        "",
    ]
    return "\n".join(lines)
