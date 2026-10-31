"""Adapters invoke the *existing compiled* specialist subgraphs with private states.

This follows LangGraph's "call a subgraph inside a node" pattern for different
state schemas. Never blend ToolMessage histories across specialist graphs.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from ..state import HybridAgentState
from ._utils import append_specialist_trace, plain_text


def _subtask(state: HybridAgentState, specialist: str) -> str:
    focus = ("Perform only read-only finance calculations on instructor CSV records. "
             "Validate totals using reconciliation, inspect uncertain remittance references."
             if specialist == "tools_agent" else
             "Perform only approved TXT document retrieval. Find dated explanations, "
             "uncertainties, and conflicting accounts; cite sources.")
    return (f"Original treasury question: {state['question']}\n"
            f"Reporting period: {state['period']}\n"
            f"Assigned scope: {focus}\n"
            f"Focused task suggested by supervisor (not additional permissions): "
            f"{state.get('delegation_task', '')[:320]}")


def make_tools_subgraph_node(tools_graph: Any):
    """Private child state -> parent finance report; no RAG access at this node."""
    def tools_subgraph(state: HybridAgentState) -> dict:
        delegation = len(state.get("visited_agents", [])) + 1
        events = [{"kind": "subgraph_start", "agent": "tools_agent", "delegation": delegation}]
        summary = ""
        used = 0
        status = "error"
        verified: list[dict] = []
        try:
            result = tools_graph.invoke({
                "messages": [HumanMessage(content=_subtask(state, "tools_agent"))],
                "trace": [], "calls_used": 0, "model_turns": 0,
                "stop_reason": "", "refused": False,
            }, config={"recursion_limit": 40})
            summary = plain_text(result["messages"][-1].content)
            used = int(result.get("calls_used", 0))
            status = result.get("stop_reason") or "completed"
            events.extend(append_specialist_trace("tools_agent", result.get("trace", []), delegation))
            verified = [
                {"tool": e.get("tool"), "arguments": e.get("arguments", {}),
                 "result": e.get("result")}
                for e in result.get("trace", [])
                if e.get("kind") == "tool_call" and e.get("status") == "ok"
            ]
        except Exception as exc:
            summary = "Financial subgraph failed. No result is certified from this invocation."
            status = "subgraph_error"
            events.append({"kind": "subgraph_error", "agent": "tools_agent", "error_type": type(exc).__name__})
        events.append({"kind": "subgraph_end", "agent": "tools_agent", "calls_used": used, "status": status})
        return {
            "finance_answer": summary,
            "finance_tool_results": verified,
            "finance_calls_used": used,
            "tool_calls_used": state.get("tool_calls_used", 0) + used,
            "visited_agents": [*state.get("visited_agents", []), "tools_agent"],
            "specialist_status": {**state.get("specialist_status", {}), "tools_agent": status},
            "trace": events,
        }
    return tools_subgraph


def make_rag_subgraph_node(rag_graph: Any):
    """Private child state -> parent evidence ledger; no finance tools at this node."""
    def rag_subgraph(state: HybridAgentState) -> dict:
        delegation = len(state.get("visited_agents", [])) + 1
        events = [{"kind": "subgraph_start", "agent": "rag_agent", "delegation": delegation}]
        summary = ""
        used = 0
        status = "error"
        retrieved: list[dict] = []
        review: dict = {}
        try:
            result = rag_graph.invoke({
                "messages": [HumanMessage(content=_subtask(state, "rag_agent"))],
                "trace": [], "retrieved": [], "calls_used": 0,
                "model_turns": 0, "search_failures": 0, "stale_searches": 0,
                "last_new_passages": 0, "evidence_review": {},
                "stop_reason": "", "refused": False,
            }, config={"recursion_limit": 50})
            summary = plain_text(result["messages"][-1].content)
            used = int(result.get("calls_used", 0))
            status = result.get("stop_reason") or "completed"
            events.extend(append_specialist_trace("rag_agent", result.get("trace", []), delegation))
            retrieved = list(result.get("retrieved", []))
            review = dict(result.get("evidence_review", {}))
        except Exception as exc:
            summary = "RAG subgraph failed. Documentary claims could not be verified."
            status = "subgraph_error"
            events.append({"kind": "subgraph_error", "agent": "rag_agent", "error_type": type(exc).__name__})
        events.append({"kind": "subgraph_end", "agent": "rag_agent", "calls_used": used, "status": status})
        return {
            "rag_answer": summary,
            "retrieved": retrieved,
            "evidence_review": review,
            "rag_calls_used": used,
            "tool_calls_used": state.get("tool_calls_used", 0) + used,
            "visited_agents": [*state.get("visited_agents", []), "rag_agent"],
            "specialist_status": {**state.get("specialist_status", {}), "rag_agent": status},
            "trace": events,
        }
    return rag_subgraph
