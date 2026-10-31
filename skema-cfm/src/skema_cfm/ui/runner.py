"""Framework-independent bridge between Streamlit and the three agent graphs.

No Streamlit calls here. Each agent keeps its own StateGraph, settings, nodes,
permissions and tools; the runner only chooses one builder and invokes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from ..agents.tools_agent.graph import build_graph as build_tools_graph
from ..agents.tools_agent.settings import FINANCE_TOOL_ALLOWLIST, ToolsAgentSettings
from ..agents.rag_agent.graph import build_graph as build_rag_graph
from ..agents.rag_agent.settings import RAGAgentSettings
from ..agents.hybrid_agent.graph import build_graph as build_hybrid_graph
from ..agents.hybrid_agent.settings import HybridAgentSettings
from .memory import hybrid_followup_question, optional_checkpointer

AgentKind = Literal["tools", "rag", "hybrid"]


@dataclass(frozen=True, slots=True)
class AgentRunOptions:
    agent: AgentKind
    model: str
    api_key: str | None = None
    max_tool_calls: int = 6
    enabled_tools: tuple[str, ...] = FINANCE_TOOL_ALLOWLIST
    selected_documents: tuple[str, ...] | None = None
    retrieval_k: int = 4
    finance_tool_budget: int = 3
    student_instructions: str = ""


def make_settings(options: AgentRunOptions) -> Any:
    """Keep agent-specific settings construction outside the user interface."""
    common = {
        "model": options.model,
        "api_key": options.api_key,
        "max_tool_calls": options.max_tool_calls,
        "student_instructions": options.student_instructions,
    }
    if options.agent == "tools":
        return ToolsAgentSettings(**common, enabled_tools=options.enabled_tools)
    if options.agent == "rag":
        return RAGAgentSettings(
            **common,
            selected_documents=options.selected_documents,
            retrieval_k=options.retrieval_k,
        )
    if options.agent == "hybrid":
        return HybridAgentSettings(
            **common,
            finance_enabled_tools=options.enabled_tools,
            selected_documents=options.selected_documents,
            retrieval_k=options.retrieval_k,
            finance_tool_budget=options.finance_tool_budget,
            rag_search_budget=options.max_tool_calls - options.finance_tool_budget,
        )
    raise ValueError(f"Unknown agent: {options.agent}")


def compile_selected_graph(
    agent: AgentKind, settings: Any, case: Any, checkpointer: Any | None,
) -> Any:
    """Call the original build_graph of the selected agent; no graph duplication."""
    builders = {
        "tools": build_tools_graph,
        "rag": build_rag_graph,
        "hybrid": build_hybrid_graph,
    }
    builder = builders[agent]
    if checkpointer is None:
        return builder(settings, case)
    try:
        return builder(settings, case, checkpointer=checkpointer)
    except TypeError as exc:
        if "checkpointer" in str(exc):
            raise RuntimeError(
                "The agent build_graph() must accept checkpointer=None and call "
                "builder.compile(checkpointer=checkpointer)."
            ) from exc
        raise


def initial_state(agent: AgentKind, question: str, period: str) -> dict[str, Any]:
    """Reset per-question budgets/evidence; `messages` stays append-only in a thread."""
    if agent == "tools":
        return {
            "messages": [HumanMessage(content=question)],
            "trace": [], "calls_used": 0, "model_turns": 0,
            "stop_reason": "", "refused": False,
        }
    if agent == "rag":
        return {
            "messages": [HumanMessage(content=question)],
            "trace": [], "retrieved": [], "calls_used": 0, "model_turns": 0,
            "search_failures": 0, "stale_searches": 0,
            "last_new_passages": 0, "evidence_review": {},
            "stop_reason": "", "refused": False,
        }
    if agent == "hybrid":
        return {
            "question": question, "period": period,
            "route": "", "delegation_task": "", "visited_agents": [],
            "supervisor_turns": 0, "tool_calls_used": 0,
            "finance_calls_used": 0, "rag_calls_used": 0,
            "finance_answer": "", "rag_answer": "",
            "finance_tool_results": [], "retrieved": [],
            "evidence_review": {}, "specialist_status": {},
            "trace": [], "answer": "", "stop_reason": "", "refused": False,
        }
    raise ValueError(f"Unknown agent: {agent}")


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def _financial_events(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"tool": event.get("tool"), "arguments": event.get("arguments"),
         "result": event.get("result")}
        for event in trace
        if event.get("kind") == "tool_call" and event.get("status") == "ok"
    ]


def run_turn(
    *, options: AgentRunOptions, case: Any, user_question: str,
    use_memory: bool, thread_id: str,
) -> dict[str, Any]:
    """Execute exactly one turn, returning only this turn's observable results.

    SQLite opens only when enabled. The graph and `get_state`/`invoke` all run
    while the SQLite connection is open. `trace` has an operator.add reducer,
    so remove the prefix from earlier turns before showing/exporting it.
    """
    if not user_question.strip():
        raise ValueError("The question cannot be empty.")
    settings = make_settings(options)
    with optional_checkpointer(use_memory) as checkpointer:
        graph = compile_selected_graph(options.agent, settings, case, checkpointer)
        config = {
            "recursion_limit": 80,
            "configurable": {"thread_id": thread_id},
        }
        prior: dict[str, Any] = {}
        if use_memory:
            snapshot = graph.get_state(config)
            prior = dict(snapshot.values or {})
        prior_trace_count = len(prior.get("trace", []))
        effective_question = (
            hybrid_followup_question(user_question, prior)
            if options.agent == "hybrid" and use_memory else user_question
        )
        result = graph.invoke(
            initial_state(options.agent, effective_question, case.period),
            config=config,
        )

    # The trace list is append-only; the other evidence fields were overwritten
    # by initial_state, so they refer only to this question.
    trace = list(result.get("trace", []))[prior_trace_count:]
    if options.agent == "hybrid":
        answer = str(result.get("answer", ""))
        used = int(result.get("tool_calls_used", 0))
        financial = result.get("finance_tool_results", [])
    else:
        answer = _text_content(result["messages"][-1].content)
        used = int(result.get("calls_used", 0))
        financial = _financial_events(trace)

    if used > options.max_tool_calls:
        raise RuntimeError("The agent exceeded the configured per-turn tool budget.")
    return {
        "answer": answer,
        "trace": trace,
        "retrieved": result.get("retrieved", []) if options.agent != "tools" else [],
        "finance": financial,
        "review": result.get("evidence_review", {}),
        "status": result.get("specialist_status", {}),
        "used": used,
        "stop_reason": result.get("stop_reason", "unspecified"),
    }
