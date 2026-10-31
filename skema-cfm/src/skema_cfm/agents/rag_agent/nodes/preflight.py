"""Deny explicit write actions; the actual boundary is the read-only tool allowlist."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage

from ..state import RAGAgentState
from ._utils import as_text

# Heuristic UX safeguard only, not a complete authorization policy.
_WRITE_REQUEST = re.compile(
    r"\b(?:edit|delete|modify|update|amend|issue|release|send|contact|post)\b"
    r".{0,100}\b(?:invoice|ledger|forecast|payment|credit note|transaction|customer|bank)\b",
    re.IGNORECASE | re.DOTALL,
)


def preflight(state: RAGAgentState) -> dict:
    question = next(
        (as_text(m.content) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    if _WRITE_REQUEST.search(question):
        return {
            "messages": [AIMessage(content=(
                "This classroom RAG agent is read-only. It cannot change accounting data, "
                "contact customers or authorise payments; refer the action to treasury."
            ))],
            "refused": True,
            "stop_reason": "out_of_scope_request",
            "trace": [{"kind": "permission_guard", "decision": "refused"}],
        }
    return {"refused": False, "trace": [{"kind": "permission_guard", "decision": "allowed"}]}
