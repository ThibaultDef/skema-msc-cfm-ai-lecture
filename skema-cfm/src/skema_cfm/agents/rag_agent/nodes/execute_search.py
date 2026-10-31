"""Execute only approved document searches; record source IDs, dates and limits."""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from ....config import Scenario
from ..settings import RAGAgentSettings
from ..state import RAGAgentState


def make_execute_search_node(
    search_tool, settings: RAGAgentSettings, scenario: Scenario
):
    def execute_search(state: RAGAgentState) -> dict:
        requests = state["messages"][-1].tool_calls
        used = state.get("calls_used", 0)
        failures = state.get("search_failures", 0)
        ledger = list(state.get("retrieved", []))
        seen = {doc["source_id"] for doc in ledger}
        messages: list[ToolMessage] = []
        trace: list[dict] = []
        new_count = 0
        for position, request in enumerate(requests):
            name = str(request.get("name") or "")
            args = request.get("args", {})
            tool_call_id = request.get("id") or f"missing-{used}-{position}"

            if used >= settings.max_tool_calls:
                status = "blocked_budget"
                payload = {
                    "status": status,
                    "error": "Search budget exhausted; not executed.",
                }
            else:
                used += 1  # Failed, denied and invalid requests consume the budget.
                if name != "search_documents" or not isinstance(args, dict):
                    status = "denied"
                    failures += 1
                    payload = {
                        "status": status,
                        "error": "Only search_documents is permitted.",
                    }
                else:
                    try:
                        if scenario == Scenario.TOOL_FAILURE and used == 1:
                            raise RuntimeError("Simulated retrieval outage")
                        payload = search_tool.invoke(args)
                        status = "ok"
                        for passage in payload["passages"]:
                            if passage["source_id"] not in seen:
                                seen.add(passage["source_id"])
                                ledger.append(passage)
                                new_count += 1
                    except Exception as exc:
                        status = "error"
                        failures += 1
                        # Provider secrets or local paths must not leak into trace/tool replies.
                        payload = {
                            "status": status,
                            "error_type": type(exc).__name__,
                            "next_step": "Retry with a valid search within budget or ask a human.",
                        }

            messages.append(
                ToolMessage(
                    content=json.dumps(payload, ensure_ascii=False, default=str),
                    tool_call_id=tool_call_id,
                    name=name or "unknown_tool",
                )
            )
            trace.append(
                {
                    "kind": "tool_call",
                    "tool": name,
                    "arguments": args,
                    "status": status,
                    "result": payload,
                    "call_index": used,
                    "max_tool_calls": settings.max_tool_calls,
                }
            )
        return {
            "messages": messages,
            "trace": trace,
            "retrieved": ledger,
            "last_new_passages": new_count,
            "calls_used": used,
            "search_failures": failures,
        }

    return execute_search
