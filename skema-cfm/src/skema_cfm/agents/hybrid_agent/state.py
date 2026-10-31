"""Parent state: private specialist message histories never mix with supervisor state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class HybridAgentState(TypedDict, total=False):
    question: str
    period: str
    route: str
    delegation_task: str
    visited_agents: list[str]
    supervisor_turns: int
    tool_calls_used: int
    finance_calls_used: int
    rag_calls_used: int
    finance_answer: str
    rag_answer: str
    finance_tool_results: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    evidence_review: dict[str, Any]
    specialist_status: dict[str, str]
    trace: Annotated[list[dict[str, Any]], operator.add]
    answer: str
    stop_reason: str
    refused: bool
