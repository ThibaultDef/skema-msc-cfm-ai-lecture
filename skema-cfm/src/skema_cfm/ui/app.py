"""Streamlit presentation layer for the three independent SKEMA CFM agents.

No graph construction, model initialization, SQLite connection, or tool logic
is implemented here. Those belong to ui.runner/ui.memory and agents/*.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from ..agents.tools_agent.settings import FINANCE_TOOL_ALLOWLIST
from ..case import list_case_document_ids, load_case
from ..config import Scenario
from .memory import configuration_fingerprint, new_thread_id
from .runner import AgentRunOptions, run_turn

AGENT_LABELS = {
    "Tools Agent": "tools",
    "RAG Agent": "rag",
    "Hybrid Agent": "hybrid",
}
SCENARIO_LABELS = {
    "Standard investigation": Scenario.STANDARD,
    "Missing Acme remittance": Scenario.MISSING_ACME,
    "Conflicting Cobalt evidence": Scenario.COBALT_CONFLICT,
    "Unmatched bank transaction": Scenario.UNMATCHED_BANK,
    "Tool failure": Scenario.TOOL_FAILURE,
}
DEFAULT_QUESTION = (
    "Investigate the cash receipts variance for the reporting week. "
    "Identify the main drivers, available evidence, uncertainties and next human checks."
)


@dataclass(frozen=True, slots=True)
class UISelection:
    options: AgentRunOptions
    scenario: Scenario
    use_memory: bool
    agent_label: str


def sidebar_configuration() -> UISelection:
    """Render inputs only. Agent settings are instantiated in ui.runner."""
    with st.sidebar:
        st.header("Agent configuration")
        agent_label = st.selectbox("Architecture", options=list(AGENT_LABELS))
        agent = AGENT_LABELS[agent_label]
        scenario_label = st.selectbox("Test scenario", options=list(SCENARIO_LABELS))
        scenario = SCENARIO_LABELS[scenario_label]
        use_memory = st.toggle("Conversation memory (SQLite)", value=False)
        model = st.text_input(
            "OpenRouter model",
            value=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        entered_key = st.text_input(
            "OpenRouter API key",
            value="",
            type="password",
            help="Leave blank to use OPENROUTER_API_KEY from .env",
        )
        api_key = entered_key.strip() or os.getenv("OPENROUTER_API_KEY") or None
        max_calls = st.slider(
            "Max tool/search calls per question",
            min_value=2 if agent == "hybrid" else 1,
            max_value=6,
            value=6,
        )
        enabled_tools: tuple[str, ...] = FINANCE_TOOL_ALLOWLIST
        selected_documents: tuple[str, ...] | None = None
        retrieval_k = 4
        finance_budget = 3

        if agent in ("tools", "hybrid"):
            enabled_tools = tuple(
                st.multiselect(
                    "Finance tools",
                    options=FINANCE_TOOL_ALLOWLIST,
                    default=FINANCE_TOOL_ALLOWLIST,
                )
            )
        if agent in ("rag", "hybrid"):
            doc_ids = list_case_document_ids(scenario)
            selected_documents = tuple(
                st.multiselect(
                    "Approved documents",
                    options=doc_ids,
                    default=doc_ids,
                    key=f"documents_{scenario.value}_{agent}",
                )
            )
            retrieval_k = st.slider("Passages per search", 1, 8, 4)
        if agent == "hybrid":
            finance_budget = st.slider(
                "Finance-call quota",
                min_value=1,
                max_value=max_calls - 1,
                value=min(3, max_calls - 1),
            )
            st.caption(f"RAG-search quota: {max_calls - finance_budget}")

        instructions = st.text_area(
            "Additional student instructions",
            value="",
            help="Formatting preferences only; these cannot override tool permissions.",
        )
        if use_memory:
            st.info(
                "SQLite checkpoints persist by conversation ID. New settings start a new thread."
            )
        else:
            st.info(
                "Memory off: prompts are independent even though chat remains on screen."
            )

    return UISelection(
        options=AgentRunOptions(
            agent=agent,
            model=model.strip(),
            api_key=api_key,
            max_tool_calls=max_calls,
            enabled_tools=enabled_tools,
            selected_documents=selected_documents,
            retrieval_k=retrieval_k,
            finance_tool_budget=finance_budget,
            student_instructions=instructions,
        ),
        scenario=scenario,
        use_memory=use_memory,
        agent_label=agent_label,
    )


def ensure_conversation(selection: UISelection) -> None:
    """Reset visible chat and checkpoint thread when meaningful settings change."""
    opts = selection.options
    fingerprint = configuration_fingerprint(
        agent=opts.agent,
        scenario=selection.scenario.value,
        use_memory=selection.use_memory,
        model=opts.model,
        max_tool_calls=opts.max_tool_calls,
        enabled_tools=opts.enabled_tools,
        selected_documents=opts.selected_documents,
        retrieval_k=opts.retrieval_k,
        finance_tool_budget=opts.finance_tool_budget,
        student_instructions=opts.student_instructions,
    )
    if st.session_state.get("configuration_fingerprint") != fingerprint:
        st.session_state.configuration_fingerprint = fingerprint
        st.session_state.turns = []
        st.session_state.thread_id = new_thread_id(opts.agent)


def sidebar_conversation_actions(selection: UISelection) -> None:
    with st.sidebar:
        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.turns = []
            st.session_state.thread_id = new_thread_id(selection.options.agent)
            st.rerun()
        if selection.use_memory:
            st.caption("Current SQLite thread ID")
            st.code(st.session_state.thread_id)


def render_trace(details: dict[str, Any]) -> None:
    """Show only the latest turn, as returned by the non-UI runner."""
    st.caption(
        f"Tool/search calls: {details['used']} · Stop reason: {details['stop_reason']}"
    )
    trace_tab, sources_tab, finance_tab = st.tabs(
        [
            "Prompts & execution trace",
            "Retrieved passages",
            "Financial tool outputs",
        ]
    )
    with trace_tab:
        st.json(details["trace"], expanded=False)
    with sources_tab:
        if not details["retrieved"]:
            st.info("No documentary passages retrieved for this turn.")
        else:
            for index, item in enumerate(details["retrieved"]):
                label = f"{item.get('source_id', 'Unknown')} · {item.get('title', '')}"
                with st.expander(label, expanded=index == 0):
                    st.caption(
                        f"Date: {item.get('date', 'unknown')} · "
                        f"File: {item.get('filename', 'unknown')}"
                    )
                    st.write(item.get("passage", ""))
        if details["review"]:
            st.markdown("**Evidence review**")
            st.json(details["review"], expanded=False)
    with finance_tab:
        if details["finance"]:
            st.json(details["finance"], expanded=False)
        else:
            st.info("No successful financial tool calls for this turn.")
        if details["status"]:
            st.markdown("**Specialist status**")
            st.json(details["status"], expanded=False)


def render_chat_history() -> None:
    for turn in st.session_state.turns:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and "details" in turn:
                with st.expander("Inspect this execution"):
                    render_trace(turn["details"])


def render_case_records(case: Any) -> None:
    with st.expander("Inspect original case records (not an agent tool call)"):
        forecast, bank, ar = st.tabs(
            [
                "Forecast CSV",
                "Bank transactions CSV",
                "Open AR items CSV",
            ]
        )
        with forecast:
            st.dataframe(list(case.forecast), use_container_width=True)
        with bank:
            st.dataframe(list(case.bank), use_container_width=True)
        with ar:
            st.dataframe(list(case.ar), use_container_width=True)


def render_documents(case: Any) -> None:
    with st.expander("Inspect original documents (not a rag agent)"):
        selected_document_id = st.selectbox(
            "Available documents",
            tuple(doc.document_id for doc in case.documents),
            index=None,
            placeholder="Select a document...",
            disabled=(not case.documents),
        )
        if selected_document_id:
            filtered_documents = [
                doc for doc in case.documents if doc.document_id == selected_document_id
            ]
            selected_document = filtered_documents[0]
            st.markdown(selected_document.text, width="stretch")


def render_download(selection: UISelection) -> None:
    if not st.session_state.turns:
        return
    st.download_button(
        "Download current conversation & traces (JSON)",
        data=json.dumps(
            {
                "agent": selection.options.agent,
                "scenario": selection.scenario.value,
                "memory": selection.use_memory,
                "thread_id": st.session_state.thread_id,
                "turns": st.session_state.turns,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        file_name=f"cfm_{selection.options.agent}_{selection.scenario.value}_conversation.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="SKEMA | AI Treasury Agents",
        page_icon="📊",
        layout="wide",
    )
    load_dotenv()
    st.title("AI for Corporate Financial Management")
    st.caption("SKEMA MSc CFM · Fictional treasury case · Read-only LangGraph agents")

    selection = sidebar_configuration()
    ensure_conversation(selection)
    sidebar_conversation_actions(selection)

    if (
        selection.options.agent in ("tools", "hybrid")
        and not selection.options.enabled_tools
    ):
        st.warning("Enable at least one financial tool.")
        st.stop()
    if not selection.options.api_key:
        st.warning("Provide an OpenRouter API key in the sidebar or .env.")

    case = load_case(scenario=selection.scenario)
    a, b, c = st.columns(3)
    a.metric("Reporting week", case.period)
    b.metric("Architecture", selection.agent_label)
    c.metric("Memory", "SQLite" if selection.use_memory else "Disabled")
    if case.annotations:
        with st.expander("Scenario notes"):
            for annotation in case.annotations:
                st.write(annotation)

    if not st.session_state.turns:
        st.info(
            "Ask about the cash forecast variance; compare architectures and "
            "failure scenarios using the sidebar."
        )
        st.caption("Example: " + DEFAULT_QUESTION)

    render_case_records(case)
    render_documents(case)
    render_chat_history()

    question = st.chat_input(
        "Ask a treasury investigation question (English)",
        disabled=not bool(selection.options.api_key),
    )
    if question and question.strip():
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner(f"Running {selection.agent_label} …"):
                try:
                    details = run_turn(
                        options=selection.options,
                        case=case,
                        user_question=question,
                        use_memory=selection.use_memory,
                        thread_id=st.session_state.thread_id,
                    )
                except Exception as exc:
                    # In a controlled classroom prototype report the failure;
                    # do not claim a successful answer or silently export it.
                    st.error(f"Agent execution failed ({type(exc).__name__}): {exc}")
                else:
                    st.markdown(details["answer"])
                    with st.expander("Inspect this execution"):
                        render_trace(details)
                    st.session_state.turns.extend(
                        [
                            {"role": "user", "content": question},
                            {
                                "role": "assistant",
                                "content": details["answer"],
                                "details": details,
                            },
                        ]
                    )

    render_download(selection)


if __name__ == "__main__":
    main()
