"""Configuration types; permissions and limits are *not* model-controlled."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Architecture(str, Enum):
    RAG_WORKFLOW = "rag_workflow"
    TOOL_AGENT = "tool_agent"
    RAG_AGENT = "rag_agent"
    HYBRID_AGENT = "hybrid_agent"


class Scenario(str, Enum):
    STANDARD = "standard"
    MISSING_ACME = "missing_acme"
    COBALT_CONFLICT = "cobalt_conflict"
    UNMATCHED_BANK = "unmatched_bank"
    TOOL_FAILURE = "tool_failure"


FINANCE_TOOLS = (
    "query_forecast",
    "query_bank_receipts",
    "lookup_open_invoices",
    "reconcile_variance",
)
RETRIEVAL_TOOL = "search_documents"
ALL_TOOLS = FINANCE_TOOLS + (RETRIEVAL_TOOL,)

ARCHITECTURE_TOOLS: dict[Architecture, frozenset[str]] = {
    Architecture.RAG_WORKFLOW: frozenset(),
    Architecture.TOOL_AGENT: frozenset(FINANCE_TOOLS),
    Architecture.RAG_AGENT: frozenset((RETRIEVAL_TOOL,)),
    Architecture.HYBRID_AGENT: frozenset(ALL_TOOLS),
}


@dataclass(frozen=True)
class RunConfig:
    architecture: Architecture = Architecture.HYBRID_AGENT
    scenario: Scenario = Scenario.STANDARD
    model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    )
    backend: str = "openrouter"
    api_key: str | None = None
    max_tool_calls: int = 6
    retrieval_k: int = 4
    selected_documents: tuple[str, ...] | None = None
    enabled_tools: tuple[str, ...] | None = None
    student_instructions: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.max_tool_calls <= 6:
            raise ValueError(
                "max_tool_calls must be in [1, 6] (teaching harness control)."
            )
        if not 1 <= self.retrieval_k <= 8:
            raise ValueError("retrieval_k must be in [1, 8].")
        if self.backend not in {"openrouter", "openai_compatible"}:
            raise ValueError("Unknown model backend.")
        allowed = ARCHITECTURE_TOOLS[self.architecture]
        if self.enabled_tools is not None:
            unknown = set(self.enabled_tools) - allowed
            if unknown:
                raise ValueError(
                    f"Tools not permitted for {self.architecture.value}: {sorted(unknown)}"
                )
        if self.selected_documents is not None and any(
            not doc.startswith("DOC-") or len(doc) != 6
            for doc in self.selected_documents
        ):
            raise ValueError("Document identifiers must use the DOC-NN format.")

    @property
    def permitted_tools(self) -> tuple[str, ...]:
        allowed = ARCHITECTURE_TOOLS[self.architecture]
        return tuple(
            name
            for name in (
                self.enabled_tools or (ALL_TOOLS if self.enabled_tools is None else ())
            )
            if name in allowed
        )

    @property
    def credential(self) -> str | None:
        return self.api_key or os.getenv("OPENROUTER_API_KEY")
