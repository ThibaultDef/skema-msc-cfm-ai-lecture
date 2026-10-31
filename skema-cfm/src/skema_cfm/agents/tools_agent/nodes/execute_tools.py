"""Hard-enforced allowlist, call budget and safe tool execution."""
from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from ....config import Scenario
from ..settings import ToolsAgentSettings
from ..state import ToolsAgentState


def make_execute_tools_node(catalog: dict, settings: ToolsAgentSettings, scenario: Scenario):
    def execute_tools(state: ToolsAgentState) -> dict:
        requests = state["messages"][-1].tool_calls
        used = state.get("calls_used", 0)
        messages: list[ToolMessage] = []
        trace: list[dict] = []

        for index, request in enumerate(requests):
            name = request.get("name", "")
            args = request.get("args", {})
            call_id = request.get("id") or f"unknown-{used}-{index}"

            if used >= settings.max_tool_calls:
                status = "blocked_budget"
                result = {"status": status, "error": "Tool budget exhausted; not executed."}
            else:
                used += 1  # Charge attempts BEFORE checking permissions or running code.
                if name not in settings.enabled_tools or name not in catalog:
                    status = "denied"
                    result = {"status": status, "error": "Tool is not on the agent allowlist."}
                else:
                    try:
                        if scenario == Scenario.TOOL_FAILURE and used == 1:
                            raise RuntimeError("Simulated tool outage")
                        result = catalog[name].invoke(args)
                        status = "ok"
                    except Exception as exc:
                        status = "error"
                        # No raw error messages or secrets are exported.
                        result = {"status": status, "error_type": type(exc).__name__,
                                  "next_step": "Retry within remaining budget or escalate."}

            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=call_id,
                name=name or "unknown_tool",
            ))
            trace.append({
                "kind": "tool_call",
                "tool": name,
                "arguments": args,
                "status": status,
                "result": result,
                "call_index": used,
                "max_tool_calls": settings.max_tool_calls,
            })

        return {"messages": messages, "calls_used": used, "trace": trace}

    return execute_tools
