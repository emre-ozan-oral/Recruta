"""Unit test for the ad-hoc skill-flagging evaluator (agents/skill_verifier.py).

Not part of the LangGraph pipeline — see the module docstring — so this is
a plain fake-LLM unit test, same pattern as tests/test_agents.py."""

from __future__ import annotations

from agents import skill_verifier
from schemas import SkillVerification


def test_evaluate_skill_returns_normalized_result(monkeypatch):
    fake_result = SkillVerification(
        is_plausible=True,
        normalized_skill="Kubernetes",
        note="normalized from 'yeah ive used k8s stuff'",
    )
    monkeypatch.setattr(
        skill_verifier, "get_structured_llm", lambda *a, **k: (lambda _prompt: fake_result)
    )

    result = skill_verifier.evaluate_skill("yeah ive used k8s stuff", job_context="DevOps role")

    assert result.is_plausible is True
    assert result.normalized_skill == "Kubernetes"


def test_evaluate_skill_can_reject_implausible_claims(monkeypatch):
    fake_result = SkillVerification(
        is_plausible=False, normalized_skill="", note="too vague to state as a skill"
    )
    monkeypatch.setattr(
        skill_verifier, "get_structured_llm", lambda *a, **k: (lambda _prompt: fake_result)
    )

    result = skill_verifier.evaluate_skill("being a good person")

    assert result.is_plausible is False
