"""Parent LangGraph: supervisor -> existing specialist subgraphs -> synthesis.

Each child graph is compiled independently and invoked INSIDE an adapter node.
This is the official different-state-schemas subgraph composition pattern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ...case import CaseSnapshot, load_case
from ...models import create_model
from ..rag_agent.graph import build_graph as build_rag_graph
from ..tools_agent.graph import build_graph as build_tools_graph
from .nodes import (
    make_finalize_node,
    make_rag_subgraph_node,
    make_supervisor_node,
    make_tools_subgraph_node,
    preflight,
)
from .settings import HybridAgentSettings
from .state import HybridAgentState


@dataclass(frozen=True)
class HybridAgentResult:
    question: str
    architecture: str
    scenario: str
    period: str
    answer: str
    tool_calls_used: int
    max_tool_calls: int
    finance_calls_used: int
    rag_calls_used: int
    visited_agents: list[str]
    stop_reason: str
    trace: list[dict]
    finance_tool_results: list[dict]
    retrieved_passages: list[dict]
    evidence_review: dict
    specialist_status: dict[str, str]
    configuration: dict

    def to_dict(self) -> dict:
        return asdict(self)


def build_graph(
    settings: HybridAgentSettings,
    case: CaseSnapshot,
    model: Any | None = None,
    tools_graph: Any | None = None,
    rag_graph: Any | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compose actual compiled specialist graphs; optional injection enables offline tests."""
    model = (
        model
        if model is not None
        else create_model(settings.to_run_config(case.scenario))
    )
    tools_graph = (
        tools_graph
        if tools_graph is not None
        else build_tools_graph(settings.finance_settings(), case, model=model)
    )
    rag_graph = (
        rag_graph
        if rag_graph is not None
        else build_rag_graph(settings.rag_settings(), case, model=model)
    )

    builder = StateGraph(HybridAgentState)
    builder.add_node("preflight", preflight)
    builder.add_node("supervisor", make_supervisor_node(model, settings))
    builder.add_node("tools_subgraph", make_tools_subgraph_node(tools_graph))
    builder.add_node("rag_subgraph", make_rag_subgraph_node(rag_graph))
    builder.add_node(
        "finalize", make_finalize_node(model, settings.student_instructions)
    )

    builder.add_edge(START, "preflight")
    builder.add_conditional_edges(
        "preflight",
        lambda state: "end" if state.get("refused") else "supervisor",
        {"end": END, "supervisor": "supervisor"},
    )
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("route", "finalize"),
        {
            "tools_agent": "tools_subgraph",
            "rag_agent": "rag_subgraph",
            "finalize": "finalize",
        },
    )
    builder.add_edge("tools_subgraph", "supervisor")
    builder.add_edge("rag_subgraph", "supervisor")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def run_hybrid_agent(
    question: str,
    settings: HybridAgentSettings | None = None,
    case: CaseSnapshot | None = None,
    model: Any | None = None,
    tools_graph: Any | None = None,
    rag_graph: Any | None = None,
) -> HybridAgentResult:
    """Headless entry point; Streamlit integration deliberately not included."""
    if not question.strip():
        raise ValueError("The treasury investigation question cannot be empty.")
    settings = settings or HybridAgentSettings()
    case = case if case is not None else load_case()
    graph = build_graph(
        settings, case, model=model, tools_graph=tools_graph, rag_graph=rag_graph
    )
    state: HybridAgentState = graph.invoke(
        {
            "question": question.strip(),
            "period": case.period,
            "route": "",
            "delegation_task": "",
            "visited_agents": [],
            "supervisor_turns": 0,
            "tool_calls_used": 0,
            "finance_calls_used": 0,
            "rag_calls_used": 0,
            "finance_answer": "",
            "rag_answer": "",
            "finance_tool_results": [],
            "retrieved": [],
            "evidence_review": {},
            "specialist_status": {},
            "trace": [],
            "answer": "",
            "stop_reason": "",
            "refused": False,
        },
        config={"recursion_limit": 20},
    )

    if state.get("tool_calls_used", 0) > settings.max_tool_calls:
        raise RuntimeError("Global tool budget invariant violated")
    return HybridAgentResult(
        question=question,
        architecture="hybrid_agent",
        scenario=case.scenario.value,
        period=case.period,
        answer=state.get("answer", ""),
        tool_calls_used=state.get("tool_calls_used", 0),
        max_tool_calls=settings.max_tool_calls,
        finance_calls_used=state.get("finance_calls_used", 0),
        rag_calls_used=state.get("rag_calls_used", 0),
        visited_agents=list(state.get("visited_agents", [])),
        stop_reason=state.get("stop_reason", "unspecified"),
        trace=state.get("trace", []),
        finance_tool_results=state.get("finance_tool_results", []),
        retrieved_passages=state.get("retrieved", []),
        evidence_review=state.get("evidence_review", {}),
        specialist_status=state.get("specialist_status", {}),
        configuration={
            "model": settings.model,
            "backend": settings.backend,
            "finance_enabled_tools": list(settings.finance_enabled_tools),
            "selected_documents": (
                list(settings.selected_documents)
                if settings.selected_documents is not None
                else None
            ),
            "retrieval_k": settings.retrieval_k,
            "max_tool_calls": settings.max_tool_calls,
            "finance_tool_budget": settings.finance_tool_budget,
            "rag_search_budget": settings.rag_search_budget,
            "require_both_agents": settings.require_both_agents,
            # API keys are never exported.
        },
    )
