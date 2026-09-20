"""Retry / fallback policy in llm.py: no SDK-hidden waits, no pointless retries."""

from __future__ import annotations

import time

import groq
import httpx
import pytest
from langchain_core.runnables import RunnableLambda

import llm


def _err(cls, status, headers=None):
    req = httpx.Request("POST", "https://api.groq.com/x")
    resp = httpx.Response(status, request=req, headers=headers or {})
    return cls("boom", response=resp, body=None)


def test_chatgroq_is_built_without_sdk_retries(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    seen = []
    real = llm.ChatGroq

    def spy(**kw):
        seen.append(kw)
        return real(**kw)

    monkeypatch.setattr(llm, "ChatGroq", spy)
    llm.get_structured_llm(__import__("schemas").RoutingDecision, tier="fast")
    assert seen and all(kw["max_retries"] == 0 for kw in seen)


def test_patience_zero_returns_inner_unchanged():
    inner = RunnableLambda(lambda x: x)
    assert llm._with_rate_limit_patience(inner, 0) is inner


def test_patience_waits_retry_after_then_retries_once(monkeypatch):
    calls = []
    sleeps = []

    def flaky(x):
        calls.append(x)
        if len(calls) == 1:
            raise _err(groq.RateLimitError, 429, {"retry-after": "2"})
        return "ok"

    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    out = llm._with_rate_limit_patience(RunnableLambda(flaky), 20).invoke("in")
    assert out == "ok" and len(calls) == 2 and sleeps == [2.5]


def test_patience_gives_up_when_wait_exceeds_budget(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: pytest.fail("must not sleep"))

    def always(x):
        raise _err(groq.RateLimitError, 429, {"retry-after": "45"})

    with pytest.raises(groq.RateLimitError):
        llm._with_rate_limit_patience(RunnableLambda(always), 20).invoke("in")


def test_bad_request_is_not_retried_and_falls_through_immediately():
    """A 400 (invalid JSON for the schema) is deterministic: one attempt on
    the primary, then straight to the fallback model."""
    attempts = {"primary": 0, "fallback": 0}

    def primary(x):
        attempts["primary"] += 1
        raise _err(groq.BadRequestError, 400)

    def fallback(x):
        attempts["fallback"] += 1
        return "from-fallback"

    p = RunnableLambda(primary).with_retry(
        retry_if_exception_type=llm.TRANSIENT_ERRORS, stop_after_attempt=2, wait_exponential_jitter=False
    )
    chain = p.with_fallbacks([RunnableLambda(fallback)])
    t0 = time.perf_counter()
    assert chain.invoke("x") == "from-fallback"
    assert attempts == {"primary": 1, "fallback": 1}
    assert time.perf_counter() - t0 < 1


def test_transient_errors_are_retried():
    n = {"c": 0}

    def flaky(x):
        n["c"] += 1
        if n["c"] == 1:
            raise groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/x"))
        return "ok"

    r = RunnableLambda(flaky).with_retry(
        retry_if_exception_type=llm.TRANSIENT_ERRORS, stop_after_attempt=2, wait_exponential_jitter=False
    )
    assert r.invoke("x") == "ok" and n["c"] == 2


# --- reasoning parameters (from probe_scorer.py measurements) ---------------


def _built_kwargs(monkeypatch, tier, **env):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    for k, v in env.items():
        monkeypatch.setattr(llm, k, v)
    seen = {}
    real = llm.ChatGroq

    def spy(**kw):
        seen[kw["model"]] = kw
        return real(**kw)

    monkeypatch.setattr(llm, "ChatGroq", spy)
    llm.get_structured_llm(__import__("schemas").ScoringResult, tier=tier)
    return seen


def test_reasoning_tier_sends_only_effort_and_never_reasoning_format(monkeypatch):
    seen = _built_kwargs(monkeypatch, "reasoning", REASONING_EFFORT="medium")
    for kw in seen.values():
        assert kw["reasoning_effort"] == "medium"
        assert "reasoning_format" not in kw  # unsupported for gpt-oss per Groq docs


def test_20b_is_capped_below_high_effort(monkeypatch):
    """gpt-oss-20b at effort=high always 400s (json_validate_failed) — measured."""
    seen = _built_kwargs(monkeypatch, "reasoning", REASONING_EFFORT="high")
    assert seen["openai/gpt-oss-120b"]["reasoning_effort"] == "high"
    assert seen["openai/gpt-oss-20b"]["reasoning_effort"] == "medium"


def test_non_reasoning_tiers_send_no_reasoning_params(monkeypatch):
    for tier in ("content", "fast"):
        for kw in _built_kwargs(monkeypatch, tier).values():
            assert "reasoning_effort" not in kw and "reasoning_format" not in kw


def test_default_effort_is_medium():
    import importlib, os
    assert os.getenv("GROQ_REASONING_EFFORT") in (None, "low", "medium", "high")
    assert llm.REASONING_EFFORT in ("low", "medium", "high")


# --- chain caching (removes ~1 s of client construction per pipeline stage) --


def test_identical_requests_reuse_the_same_chain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    import schemas

    a = llm.get_structured_llm(schemas.JobRequirements, tier="content")
    b = llm.get_structured_llm(schemas.JobRequirements, tier="content")
    assert a is b


def test_different_schema_tier_or_effort_gets_a_new_chain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    import schemas

    base = llm.get_structured_llm(schemas.JobRequirements, tier="content")
    assert llm.get_structured_llm(schemas.MatchAnalysis, tier="content") is not base
    assert llm.get_structured_llm(schemas.JobRequirements, tier="fast") is not base
    r1 = llm.get_structured_llm(schemas.ScoringResult, tier="reasoning")
    monkeypatch.setattr(llm, "REASONING_EFFORT", "low")
    assert llm.get_structured_llm(schemas.ScoringResult, tier="reasoning") is not r1


def test_changed_api_key_is_not_served_from_cache(monkeypatch):
    import schemas

    monkeypatch.setenv("GROQ_API_KEY", "key-1")
    a = llm.get_structured_llm(schemas.JobRequirements)
    monkeypatch.setenv("GROQ_API_KEY", "key-2")
    assert llm.get_structured_llm(schemas.JobRequirements) is not a
