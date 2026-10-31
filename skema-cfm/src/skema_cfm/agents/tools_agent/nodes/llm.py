"""The model is asked to select finance tools; it cannot execute them directly."""
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage

from ..state import ToolsAgentState
from ._utils import as_text, serialize_messages


def make_llm_node(bound_model, system_prompt: str):
    def call_llm(state: ToolsAgentState) -> dict:
        prompt = [SystemMessage(content=system_prompt), *state["messages"]]
        trace = [{"kind": "model_prompt", "phase": "agent", "messages": serialize_messages(prompt)}]
        next_turn = state.get("model_turns", 0) + 1
        try:
            response = bound_model.invoke(prompt)
            if getattr(response, "invalid_tool_calls", None):
                raise ValueError("Malformed tool-call arguments")
            if not getattr(response, "tool_calls", None) and not as_text(response.content).strip():
                raise ValueError("Empty model response")
            trace.append({
                "kind": "model_response",
                "phase": "agent",
                "content": as_text(response.content),
                "tool_calls": getattr(response, "tool_calls", []),
            })
            return {
                "messages": [response],
                "model_turns": next_turn,
                "trace": trace,
                "stop_reason": "" if response.tool_calls else "model_finished",
            }
        except Exception as exc:
            # Never expose raw provider errors (which could contain credentials).
            trace.append({"kind": "model_error", "error_type": type(exc).__name__})
            return {
                "messages": [AIMessage(content=(
                    "The model could not complete a reliable investigation. "
                    "Please refer the case to a treasury analyst."
                ))],
                "model_turns": next_turn,
                "trace": trace,
                "stop_reason": "model_error",
            }

    return call_llm
