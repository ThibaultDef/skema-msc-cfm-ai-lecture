"""Hybrid supervisor settings: compose specialist graphs without widening their permissions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ...config import Architecture, RunConfig, Scenario
from ..rag_agent.settings import RAGAgentSettings
from ..tools_agent.settings import FINANCE_TOOL_ALLOWLIST, ToolsAgentSettings


@dataclass(frozen=True, slots=True)
class HybridAgentSettings:
    model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    )
    backend: str = "openrouter"
    api_key: str | None = None
    finance_enabled_tools: tuple[str, ...] = FINANCE_TOOL_ALLOWLIST
    selected_documents: tuple[str, ...] | None = None
    retrieval_k: int = 4
    max_tool_calls: int = 6
    finance_tool_budget: int = 3
    rag_search_budget: int = 3
    max_supervisor_turns: int = 3
    require_both_agents: bool = True
    student_instructions: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.max_tool_calls <= 6:
            raise ValueError("The shared tool-call budget must be between 1 and 6.")
        if (
            not 1 <= self.finance_tool_budget <= 6
            or not 1 <= self.rag_search_budget <= 6
        ):
            raise ValueError("Both specialists require at least one reserved call.")
        if self.finance_tool_budget + self.rag_search_budget > self.max_tool_calls:
            raise ValueError("Specialist quotas must fit the global tool-call budget.")
        if not 2 <= self.max_supervisor_turns <= 6:
            raise ValueError(
                "Supervisor must have at least two opportunities to delegate."
            )
        if not self.finance_enabled_tools or len(
            set(self.finance_enabled_tools)
        ) != len(self.finance_enabled_tools):
            raise ValueError("Finance tools must be nonempty and unique.")
        unknown = set(self.finance_enabled_tools) - set(FINANCE_TOOL_ALLOWLIST)
        if unknown:
            raise ValueError(f"Unknown or disallowed finance tools: {sorted(unknown)}")
        if not 1 <= self.retrieval_k <= 8:
            raise ValueError("retrieval_k must be 1..8.")
        if self.backend not in {"openrouter", "openai_compatible"}:
            raise ValueError("Unsupported model backend.")
        if self.selected_documents is not None:
            if len(set(self.selected_documents)) != len(self.selected_documents):
                raise ValueError("Duplicate selected document IDs.")
            if any(
                not doc.startswith("DOC-") or len(doc) != 6 or not doc[4:].isdigit()
                for doc in self.selected_documents
            ):
                raise ValueError("Documents must use DOC-NN identifiers.")

    def finance_settings(self) -> ToolsAgentSettings:
        return ToolsAgentSettings(
            model=self.model,
            backend=self.backend,
            api_key=self.api_key,
            enabled_tools=self.finance_enabled_tools,
            max_tool_calls=self.finance_tool_budget,
            student_instructions=self.student_instructions,
        )

    def rag_settings(self) -> RAGAgentSettings:
        return RAGAgentSettings(
            model=self.model,
            backend=self.backend,
            api_key=self.api_key,
            selected_documents=self.selected_documents,
            retrieval_k=self.retrieval_k,
            max_tool_calls=self.rag_search_budget,
            student_instructions=self.student_instructions,
        )

    def to_run_config(self, scenario: Scenario) -> RunConfig:
        """Use the existing init_chat_model/OpenRouter adapter for the supervisor."""
        return RunConfig(
            architecture=Architecture.HYBRID_AGENT,
            scenario=scenario,
            model=self.model,
            backend=self.backend,
            api_key=self.api_key,
            enabled_tools=(*self.finance_enabled_tools, "search_documents"),
            selected_documents=self.selected_documents,
            retrieval_k=self.retrieval_k,
            max_tool_calls=self.max_tool_calls,
            student_instructions=self.student_instructions,
        )
