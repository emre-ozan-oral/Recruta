"""Smoke test: the StateGraph must compile without hitting the network."""

from __future__ import annotations

from graph import build_graph


def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_includes_scorer_node():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "scorer" in node_names
    assert {"job_analyzer", "cv_matcher", "scorer", "interview_prep", "report_writer"} <= node_names
