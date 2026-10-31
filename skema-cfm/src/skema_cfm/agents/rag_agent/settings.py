"""Independent RAG Agent settings. All permissions are enforced in Python."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ...config import Architecture, RunConfig, Scenario

RAG_TOOL_ALLOWLIST = ("search_documents",)


@dataclass(frozen=True, slots=True)
class RAGAgentSettings:
    model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    )
    backend: str = "openrouter"
    api_key: str | None = None
    enabled_tools: tuple[str, ...] = RAG_TOOL_ALLOWLIST
    selected_documents: tuple[str, ...] | None = None
    retrieval_k: int = 4
    max_tool_calls: int = 6
    max_model_turns: int = 8
    max_search_failures: int = 2
    max_stale_searches: int = 2
    student_instructions: str = ""

    def __post_init__(self) -> None:
        if self.enabled_tools != RAG_TOOL_ALLOWLIST:
            raise ValueError(
                "RAG Agent permits only search_documents (no finance tools)."
            )
        if self.backend not in {"openrouter", "openai_compatible"}:
            raise ValueError("Unsupported backend.")
        if not 1 <= self.retrieval_k <= 8:
            raise ValueError("retrieval_k must be between 1 and 8.")
        if not 1 <= self.max_tool_calls <= 6:
            raise ValueError("max_tool_calls must be between 1 and 6.")
        if not 1 <= self.max_model_turns <= 12:
            raise ValueError("max_model_turns must be between 1 and 12.")
        if not 1 <= self.max_search_failures <= 6:
            raise ValueError("max_search_failures must be between 1 and 6.")
        if not 1 <= self.max_stale_searches <= 6:
            raise ValueError("max_stale_searches must be between 1 and 6.")
        if self.selected_documents is not None:
            if len(self.selected_documents) != len(set(self.selected_documents)):
                raise ValueError("Duplicate source identifiers are not allowed.")
            if any(
                not d.startswith("DOC-") or len(d) != 6 or not d[4:].isdigit()
                for d in self.selected_documents
            ):
                raise ValueError("Source identifiers must follow DOC-NN.")

    def to_run_config(self, scenario: Scenario) -> RunConfig:
        """Use the shared init_chat_model/OpenRouter adapter; no UI dependency."""
        return RunConfig(
            architecture=Architecture.RAG_AGENT,
            scenario=scenario,
            model=self.model,
            backend=self.backend,
            api_key=self.api_key,
            enabled_tools=self.enabled_tools,
            selected_documents=self.selected_documents,
            retrieval_k=self.retrieval_k,
            max_tool_calls=self.max_tool_calls,
            student_instructions=self.student_instructions,
        )

    @classmethod
    def from_run_config(cls, config: RunConfig) -> RAGAgentSettings:
        if config.architecture != Architecture.RAG_AGENT:
            raise ValueError("Expected RAG_AGENT architecture.")
        return cls(
            model=config.model,
            backend=config.backend,
            api_key=config.api_key,
            enabled_tools=config.permitted_tools,
            selected_documents=config.selected_documents,
            retrieval_k=config.retrieval_k,
            max_tool_calls=config.max_tool_calls,
            student_instructions=config.student_instructions,
        )
