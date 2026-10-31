"""Conversation identifiers and optional SQLite checkpoints for Streamlit.

This module does not import Streamlit or any agent backend. A checkpointer is
created only for a memory-enabled invocation, and its connection is kept open
for the complete lifetime of the LangGraph invoke/get_state operations.

SQLite checkpoints persist *entire graph state*, not just chat messages.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4


def new_thread_id(agent: str) -> str:
    """Use a fresh, unpredictable thread ID for a new conversation."""
    if agent not in {"tools", "rag", "hybrid"}:
        raise ValueError(f"Unknown agent: {agent}")
    return f"cfm-{agent}-{uuid4().hex}"


def configuration_fingerprint(
    *,
    agent: str,
    scenario: str,
    use_memory: bool,
    model: str,
    max_tool_calls: int,
    enabled_tools: tuple[str, ...],
    selected_documents: tuple[str, ...] | None,
    retrieval_k: int,
    finance_tool_budget: int | None,
    student_instructions: str,
) -> str:
    """Identify settings that must not share the same SQLite conversation.

    Credentials are deliberately excluded. They must never reach traces or
    downloaded conversation metadata.
    """
    return json.dumps(
        {
            "agent": agent,
            "scenario": scenario,
            "memory": use_memory,
            "model": model,
            "max_tool_calls": max_tool_calls,
            "tools": list(enabled_tools),
            "docs": list(selected_documents)
            if selected_documents is not None
            else None,
            "retrieval_k": retrieval_k,
            "finance_tool_budget": finance_tool_budget if agent == "hybrid" else None,
            "student_instructions": student_instructions,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def database_path() -> Path:
    """By default, SQLite lives in .local/ relative to the launch directory.

    Set SKEMA_CFM_CHECKPOINT_DB to use a different writable file path.
    """
    supplied = os.getenv("SKEMA_CFM_CHECKPOINT_DB", "").strip()
    path = (
        Path(supplied).expanduser()
        if supplied
        else Path(".local/cfm_checkpoints.sqlite")
    )
    return path.resolve()


@contextmanager
def optional_checkpointer(enabled: bool) -> Iterator[Any | None]:
    """Yield None when off; otherwise yield an open SQLite saver.

    Do not cache this context-managed saver across Streamlit reruns: its
    connection closes at the end of this context.
    """
    if not enabled:
        yield None
        return
    # Lazy import: a memory-disabled execution does not need sqlite extras.
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "SQLite memory requires: pip install langgraph-checkpoint-sqlite"
        ) from exc
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer


# The existing Hybrid agent has a `question`/`answer` state instead of a
# `messages` reducer. This is a narrow compatibility adapter, NOT a replacement
# for a native Hybrid conversational state.
_WRITE_REQUEST = re.compile(
    r"\b(?:edit|delete|modify|update|amend|issue|release|send|contact|write|create|post)\b"
    r".{0,100}\b(?:invoice|ledger|forecast|payment|credit note|transaction|customer|bank)\b",
    re.IGNORECASE | re.DOTALL,
)


def hybrid_followup_question(current_question: str, prior_state: dict[str, Any]) -> str:
    """Supply only prior Q&A as conversational context; never as source evidence."""
    old_question = str(prior_state.get("question", "")).strip()
    old_answer = str(prior_state.get("answer", "")).strip()
    if not old_question or not old_answer:
        return current_question
    context = (
        f"Previous user question:\n{old_question}\n\n"
        f"Previous assistant answer:\n{old_answer}"
    )[-5500:]
    # Earlier denied write requests can cause the existing Hybrid preflight to
    # reject an unrelated follow-up. In that case carry only topic keywords.
    if _WRITE_REQUEST.search(context):
        customers = [
            name
            for name in ("Acme", "Beta", "Cobalt", "Delta")
            if re.search(rf"\b{name}\b", context, flags=re.IGNORECASE)
        ]
        context = "Earlier exchange discussed: " + (
            ", ".join(customers) or "cash receipts"
        )
    return (
        "PRIOR CONVERSATION (for resolving references only; NOT verified financial "
        "or documentary evidence):\n" + context + "\n\n"
        "CURRENT USER QUESTION (recheck underlying evidence):\n" + current_question
    )
