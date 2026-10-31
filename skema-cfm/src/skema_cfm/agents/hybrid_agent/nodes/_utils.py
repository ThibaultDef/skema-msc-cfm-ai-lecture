"""Pure helpers for trace formatting, citations, and safe graph adapters."""
from __future__ import annotations

import json
import re
from typing import Any

_DOC = re.compile(r"\bDOC-\d{2}\b")
_RECORD = re.compile(r"(?:cash_forecast|bank_transactions|ar_open_items)\.csv:[A-Za-z0-9-]+")


def plain_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item.get("text", "")) if isinstance(item, dict) else str(item)
                        for item in content)
    return str(content)


def trusted_record_ids(finance_tool_results: list[dict]) -> set[str]:
    """Collect ONLY source refs actually returned by successful tools."""
    refs: set[str] = set()
    def traverse(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"source_ref", "source_refs"}:
                    refs.update(_RECORD.findall(json.dumps(value, default=str)))
                else:
                    traverse(value)
        elif isinstance(obj, list):
            for value in obj:
                traverse(value)
    for item in finance_tool_results:
        traverse(item.get("result"))
    return refs


def cited_ids(text: str) -> tuple[set[str], set[str]]:
    return set(_DOC.findall(text)), set(_RECORD.findall(text))


def append_specialist_trace(agent: str, events: list[dict], number: int) -> list[dict]:
    return [{"agent": agent, "delegation": number, **event} for event in events]
