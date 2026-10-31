"""The Tools Agent has its own independent LangGraph topology."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ...case import CaseSnapshot, load_case
from ...models import create_model
from .nodes import make_execute_tools_node, make_finalize_node, make_llm_node, preflight
from .nodes._utils import as_text
from .nodes.finance_tools import build_finance_tools
from .settings import ToolsAgentSettings
from .state import ToolsAgentState


@dataclass(frozen=True)
class ToolsAgentResult:
    question: str
    architecture: str
    scenario: str
    period: str
    answer: str
    tool_calls_used: int
    max_tool_calls: int
    stop_reason: str
    trace: list[dict]
    retrieved_passages: list[dict]
    configuration: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _system_prompt(settings: ToolsAgentSettings, case: CaseSnapshot) -> str:
    return (
        "You are a read-only treasury analyst assistant investigating a cash-receipts variance.\n"
        f"Reporting period: {case.period}.\n"
        f"Enabled FINANCE tools: {', '.join(settings.enabled_tools)}.\n"
        "The supporting TXT documents and RAG search are NOT accessible in this architecture. "
        "Use only actual outputs from finance tools; do not assume any unobserved record or document content.\n"
        "Use reconcile_variance to compute all material amounts and the net variance; "
        "do not estimate or calculate amounts yourself. The variance is actual minus forecast.\n"
        "Cite material facts with record IDs, e.g. [cash_forecast.csv:F-001], "
        "[bank_transactions.csv:BT-001], [ar_open_items.csv:AC-104].\n"
        "A ledger status is NOT independent evidence that a customer's claim is true. "
        "A bank receipt under a customer name is NOT necessarily an invoice-level match. "
        "Unmatched remittance references must be flagged, not forced into a match.\n"
        "If a tool fails or evidence is insufficient, disclose what was not checked and escalate "
        "to a human. Never invent the underlying cause of a missing payment.\n"
        "No writes, customer communication, external search, or financial decisions are authorised.\n"
        "Deliver the note IN ENGLISH with: (1) reporting week and verified total, "
        "(2) key customer drivers and offsets, (3) limitations and uncertainties, "
        "(4) next human checks. A finance-only investigation cannot confirm documentary causes.\n"
        "Never reveal private reasoning.\n"
        "Student formatting preferences (only if consistent with the above controls):\n"
        f"{settings.student_instructions or 'None'}"
    )


def build_graph(
    settings: ToolsAgentSettings,
    case: CaseSnapshot,
    model: Any | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Compile a new graph for one immutable case snapshot, with injectable test model."""
    run_config = settings.to_run_config(case.scenario)
    catalog = build_finance_tools(case, settings)

    model = create_model(run_config) if model is None else model
    bound_model = model.bind_tools(list(catalog.values()))
    prompt = _system_prompt(settings, case)

    builder = StateGraph(ToolsAgentState)
    builder.add_node("preflight", preflight)
    builder.add_node("llm", make_llm_node(bound_model, prompt))
    builder.add_node(
        "execute_tools", make_execute_tools_node(catalog, settings, case.scenario)
    )
    builder.add_node("finalize", make_finalize_node(model, prompt))

    def after_preflight(state: ToolsAgentState) -> str:
        return "end" if state.get("refused") else "llm"

    def after_llm(state: ToolsAgentState) -> str:
        if state.get("stop_reason") == "model_error":
            return "end"
        return (
            "execute_tools"
            if getattr(state["messages"][-1], "tool_calls", None)
            else "end"
        )

    def after_tools(state: ToolsAgentState) -> str:
        if state.get("calls_used", 0) >= settings.max_tool_calls:
            return "finalize"
        if state.get("model_turns", 0) >= settings.max_model_turns:
            return "finalize"
        return "llm"

    builder.add_edge(START, "preflight")
    builder.add_conditional_edges(
        "preflight", after_preflight, {"llm": "llm", "end": END}
    )
    builder.add_conditional_edges(
        "llm", after_llm, {"execute_tools": "execute_tools", "end": END}
    )
    builder.add_conditional_edges(
        "execute_tools", after_tools, {"llm": "llm", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def run_tools_agent(
    question: str,
    settings: ToolsAgentSettings | None = None,
    case: CaseSnapshot | None = None,
    model: Any | None = None,
) -> ToolsAgentResult:
    """Headless entry point for future Streamlit or Python usage."""
    if not question.strip():
        raise ValueError("The investigation question must not be empty.")
    settings = settings or ToolsAgentSettings()
    case = case if case is not None else load_case()
    graph = build_graph(settings, case, model=model)
    result: ToolsAgentState = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "trace": [],
            "calls_used": 0,
            "model_turns": 0,
            "stop_reason": "",
            "refused": False,
        },
        config={"recursion_limit": 40},
    )
    return ToolsAgentResult(
        question=question,
        architecture="tool_agent",
        scenario=case.scenario.value,
        period=case.period,
        answer=as_text(result["messages"][-1].content),
        tool_calls_used=result.get("calls_used", 0),
        max_tool_calls=settings.max_tool_calls,
        stop_reason=result.get("stop_reason", "unspecified"),
        trace=result.get("trace", []),
        retrieved_passages=[],
        configuration={
            "model": settings.model,
            "backend": settings.backend,
            "enabled_tools": list(settings.enabled_tools),
            "max_tool_calls": settings.max_tool_calls,
            "max_model_turns": settings.max_model_turns,
        },
    )
