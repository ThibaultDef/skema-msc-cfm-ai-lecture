"""Reject explicit mutation requests before contacting the language model."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from ..state import ToolsAgentState
from ._utils import as_text

_WRITE_REQUEST = re.compile(
    r"\b(?:edit|delete|modify|update|amend|issue|release|send|contact|write|create|post)\b"
    r".{0,100}\b(?:invoice|ledger|forecast|payment|credit note|transaction|customer|bank)\b",
    re.IGNORECASE | re.DOTALL,
)


def preflight(state: ToolsAgentState) -> dict:
    last_question = next(
        (as_text(msg.content) for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        "",
    )
    if _WRITE_REQUEST.search(last_question):
        return {
            "messages": [AIMessage(content=(
                "This classroom agent is read-only. I cannot modify accounts, release payments, "
                "send customer messages or change forecasts. A human analyst must authorise follow-up."
            ))],
            "refused": True,
            "stop_reason": "out_of_scope_request",
            "trace": [{"kind": "permission_guard", "decision": "refused"}],
        }
    return {"refused": False, "trace": [{"kind": "permission_guard", "decision": "allowed"}]}
