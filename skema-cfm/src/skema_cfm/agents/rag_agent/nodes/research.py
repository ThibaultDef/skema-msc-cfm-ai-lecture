"""LLM-controlled research: select a new query or stop searching."""
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from ..state import RAGAgentState
from ._utils import as_text, serialize_messages


def make_research_node(bound_model, system_prompt: str):
    def research(state: RAGAgentState) -> dict:
        system = SystemMessage(content=system_prompt)
        messages = [system, *state["messages"]]
        trace = [{"kind": "model_prompt", "phase": "research", "messages": serialize_messages(messages)}]
        turns = state.get("model_turns", 0) + 1
        try:
            response = bound_model.invoke(messages)
            if getattr(response, "invalid_tool_calls", None):
                raise ValueError("Invalid tool-call request")
            if not getattr(response, "tool_calls", None) and not as_text(response.content).strip():
                raise ValueError("Empty response")
            trace.append({
                "kind": "model_response", "phase": "research", "content": as_text(response.content),
                "tool_calls": getattr(response, "tool_calls", []),
            })
            return {
                "messages": [response],
                "model_turns": turns,
                "trace": trace,
                "stop_reason": "" if getattr(response, "tool_calls", None) else "research_completed",
            }
        except Exception as exc:
            trace.append({"kind": "model_error", "phase": "research", "error_type": type(exc).__name__})
            return {
                "messages": [AIMessage(content=(
                    "The model could not complete the document investigation. "
                    "Refer the retrieved evidence to a treasury analyst."
                ))],
                "model_turns": turns,
                "trace": trace,
                "stop_reason": "model_error",
            }

    return research
