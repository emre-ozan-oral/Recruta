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


# --- degradation paths: the click must never fail or block the user ---------


def _llm_that_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("429 rate_limit_exceeded")

    monkeypatch.setattr(skill_verifier, "get_structured_llm", boom)


def test_trusted_requirement_needs_no_llm(monkeypatch):
    _llm_that_raises(monkeypatch)  # would explode if the LLM were consulted
    result = skill_verifier.evaluate_skill("- Docker (containers).", trust_input=True)
    assert result.is_plausible is True
    assert result.normalized_skill == "Docker (containers)"
    assert result.note == ""


def test_trusted_but_too_short_is_rejected_without_llm(monkeypatch):
    _llm_that_raises(monkeypatch)
    result = skill_verifier.evaluate_skill(" . ", trust_input=True)
    assert result.is_plausible is False


def test_long_sentence_still_goes_to_llm(monkeypatch):
    fake = SkillVerification(is_plausible=True, normalized_skill="Microservices")
    monkeypatch.setattr(skill_verifier, "get_structured_llm", lambda *a, **k: (lambda _p: fake))
    long_text = "Deep hands-on experience designing and operating " + "distributed " * 12 + "systems"
    assert skill_verifier.evaluate_skill(long_text, trust_input=True).normalized_skill == "Microservices"


def test_llm_outage_degrades_to_unverified_save(monkeypatch):
    _llm_that_raises(monkeypatch)
    result = skill_verifier.evaluate_skill("yeah ive used k8s stuff")  # free text => LLM path
    assert result.is_plausible is True
    assert result.normalized_skill == "yeah ive used k8s stuff"
    assert result.note == skill_verifier.UNVERIFIED_NOTE


def test_llm_outage_truncates_overlong_text(monkeypatch):
    _llm_that_raises(monkeypatch)
    result = skill_verifier.evaluate_skill("x" * 500)
    assert len(result.normalized_skill) <= skill_verifier.MAX_SKILL_CHARS
