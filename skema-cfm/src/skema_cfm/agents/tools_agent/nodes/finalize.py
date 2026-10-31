"""A last, tool-free model invocation after a hard stop."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..state import ToolsAgentState
from ._utils import as_text, serialize_messages


def make_finalize_node(model, system_prompt: str):
    def finalize(state: ToolsAgentState) -> dict:
        instruction = HumanMessage(
            content=(
                "No more tool calls may be executed. Draft the investigation note using ONLY the "
                "tool results already present. If a required reconciliation was not performed, "
                "report it as unverified. Identify missing evidence and analyst follow-up."
            )
        )
        prompt = [SystemMessage(content=system_prompt), *state["messages"], instruction]
        trace = [
            {
                "kind": "model_prompt",
                "phase": "finalize",
                "messages": serialize_messages(prompt),
            }
        ]
        try:
            response = model.invoke(prompt)  # Deliberately unbound: no callable tools.
            if (
                getattr(response, "tool_calls", None)
                or not as_text(response.content).strip()
            ):
                raise ValueError("Invalid tool-free final response")
            trace.append(
                {
                    "kind": "model_response",
                    "phase": "finalize",
                    "content": as_text(response.content),
                }
            )
            return {
                "messages": [AIMessage(content=as_text(response.content))],
                "trace": trace,
                "stop_reason": "execution_limit_reached",
            }
        except Exception as exc:
            trace.append(
                {
                    "kind": "model_error",
                    "phase": "finalize",
                    "error_type": type(exc).__name__,
                }
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "The investigation reached its execution limit and could not be safely completed. "
                            "Please review the recorded financial tool outputs manually."
                        )
                    )
                ],
                "trace": trace,
                "stop_reason": "finalization_failed",
            }

    return finalize
