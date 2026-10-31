"""Only execution state belongs here; policy and dependencies belong in settings/graph."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ToolsAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    trace: Annotated[list[dict[str, Any]], operator.add]
    calls_used: int
    model_turns: int
    stop_reason: str
    refused: bool
