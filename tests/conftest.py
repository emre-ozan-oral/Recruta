"""Shared fixtures."""

import pytest

import llm


@pytest.fixture(autouse=True)
def _fresh_llm_chain_cache():
    """llm.get_structured_llm caches built chains; tests that monkeypatch
    ChatGroq / env must never see a chain cached by an earlier test."""
    llm._build_chain.cache_clear()
    yield
    llm._build_chain.cache_clear()
