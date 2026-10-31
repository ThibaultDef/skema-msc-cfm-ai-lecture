"""Coarse guard; the specialist tool allowlists are the actual authorization boundary."""

from __future__ import annotations

import re

from ..state import HybridAgentState

_WRITE = re.compile(
    r"\b(?:edit|delete|modify|update|amend|issue|release|send|contact|write|create|post)\b"
    r".{0,100}\b(?:invoice|ledger|forecast|payment|credit note|transaction|customer|bank)\b",
    re.IGNORECASE | re.DOTALL,
)


def preflight(state: HybridAgentState) -> dict:
    if _WRITE.search(state.get("question", "")):
        return {
            "refused": True,
            "answer": "This educational treasury agent is read-only. No account modifications, payments, or customer communications are permitted. Refer the request to a human analyst.",
            "stop_reason": "out_of_scope_request",
            "trace": [
                {
                    "agent": "hybrid_agent",
                    "kind": "permission_guard",
                    "decision": "refused",
                }
            ],
        }
    return {
        "refused": False,
        "trace": [
            {"agent": "hybrid_agent", "kind": "permission_guard", "decision": "allowed"}
        ],
    }
