"""LangGraph StateGraph wiring: supervisor pattern over 5 worker agents."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents import cv_matcher, interview_prep, job_analyzer, report_writer, scorer, supervisor
from state import AgentState


def build_graph():
    """Compile the job-application-assistant graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor.run)
    workflow.add_node("job_analyzer", job_analyzer.run)
    workflow.add_node("cv_matcher", cv_matcher.run)
    workflow.add_node("scorer", scorer.run)
    workflow.add_node("interview_prep", interview_prep.run)
    workflow.add_node("report_writer", report_writer.run)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "job_analyzer": "job_analyzer",
            "cv_matcher": "cv_matcher",
            "scorer": "scorer",
            "interview_prep": "interview_prep",
            "report_writer": "report_writer",
            "END": END,
        },
    )

    # Every worker agent hands control back to the supervisor.
    for node in ("job_analyzer", "cv_matcher", "scorer", "interview_prep", "report_writer"):
        workflow.add_edge(node, "supervisor")

    return workflow.compile()


def run_pipeline(cv_text: str, job_posting_text: str, callbacks: list | None = None) -> AgentState:
    """Convenience entry point: run the full graph for a given CV + posting.

    Pass callbacks=[langfuse_utils.get_langfuse_handler()] (filtering out
    None) to trace the run in Langfuse, if configured.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "cv_text": cv_text,
        "job_posting_text": job_posting_text,
        "messages": [],
    }
    config = {"callbacks": callbacks} if callbacks else {}
    return graph.invoke(initial_state, config=config)
