"""Deterministic evidence inventory, conflict flags, and explicit stopping conditions."""
from __future__ import annotations

from ..settings import RAGAgentSettings
from ..state import RAGAgentState


def make_review_evidence_node(settings: RAGAgentSettings):
    def review_evidence(state: RAGAgentState) -> dict:
        ids = {item["source_id"] for item in state.get("retrieved", [])}
        flags: list[str] = []
        if "DOC-08" in ids:
            flags.append(
                "DOC-08 is explicitly UNVERIFIED: it suggests a different withheld amount "
                "but cannot establish a revised ledger balance. Escalate for verification."
            )
            if "DOC-04" in ids or "DOC-05" in ids:
                flags.append("Conflicting Cobalt source accounts are present: cite BOTH before escalation.")
        if {"DOC-02", "DOC-03"} <= ids:
            flags.append(
                "Acme's earlier expected payment date was only an estimate; "
                "the later remittance provides no confirmed date for the unpaid balance."
            )
        stale = 0 if state.get("last_new_passages", 0) else state.get("stale_searches", 0) + 1
        review = {
            "source_ids": sorted(ids),
            "source_count": len(ids),
            "new_sources_last_search": state.get("last_new_passages", 0),
            "flags": flags,
        }
        if state.get("calls_used", 0) >= settings.max_tool_calls:
            reason = "search_budget_exhausted"
        elif state.get("model_turns", 0) >= settings.max_model_turns:
            reason = "model_turn_limit"
        elif state.get("search_failures", 0) >= settings.max_search_failures:
            reason = "search_failures_exceeded"
        elif stale >= settings.max_stale_searches:
            reason = "no_new_evidence"
        else:
            reason = ""
        return {
            "evidence_review": review,
            "stale_searches": stale,
            "stop_reason": reason,
            "trace": [{
                "kind": "evidence_review", **review,
                "stale_searches": stale, "stop_reason": reason,
            }],
        }

    return review_evidence
