"""Private state for this RAG Agent; no reliance on the Tools Agent state."""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RAGAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    trace: Annotated[list[dict[str, Any]], operator.add]
    retrieved: list[dict[str, Any]]  # Replaced with a deduplicated evidence ledger.
    calls_used: int
    model_turns: int
    search_failures: int
    stale_searches: int
    last_new_passages: int
    evidence_review: dict[str, Any]
    stop_reason: str
    refused: bool
