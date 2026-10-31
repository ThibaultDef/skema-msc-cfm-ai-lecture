"""Tool Agent settings: independent, strict and finance-only."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ...config import Architecture, RunConfig, Scenario

FINANCE_TOOL_ALLOWLIST: tuple[str, ...] = (
    "query_forecast",
    "query_bank_receipts",
    "lookup_open_invoices",
    "reconcile_variance",
)


@dataclass(frozen=True, slots=True)
class ToolsAgentSettings:
    model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    )
    backend: str = "openrouter"
    api_key: str | None = None
    enabled_tools: tuple[str, ...] = FINANCE_TOOL_ALLOWLIST
    max_tool_calls: int = 6
    max_model_turns: int = 8
    student_instructions: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.max_tool_calls <= 6:
            raise ValueError("max_tool_calls must be between 1 and 6.")
        if not 1 <= self.max_model_turns <= 12:
            raise ValueError("max_model_turns must be between 1 and 12.")
        if not self.enabled_tools:
            raise ValueError(
                "A Tools Agent requires at least one enabled finance tool."
            )
        if len(self.enabled_tools) != len(set(self.enabled_tools)):
            raise ValueError("Duplicate tool names are not permitted.")
        unknown = set(self.enabled_tools) - set(FINANCE_TOOL_ALLOWLIST)
        if unknown:
            raise ValueError(f"Tools Agent cannot access: {sorted(unknown)}")
        if self.backend not in ("openrouter", "openai_compatible"):
            raise ValueError("Unsupported model backend.")

    def to_run_config(self, scenario: Scenario) -> RunConfig:
        """Adapter to existing shared model and tool factories."""
        return RunConfig(
            architecture=Architecture.TOOL_AGENT,
            scenario=scenario,
            model=self.model,
            backend=self.backend,
            api_key=self.api_key,
            enabled_tools=self.enabled_tools,
            max_tool_calls=self.max_tool_calls,
            student_instructions=self.student_instructions,
        )

    @classmethod
    def from_run_config(cls, config: RunConfig) -> ToolsAgentSettings:
        """Adapter for future Streamlit integration; no UI changes required now."""
        if config.architecture != Architecture.TOOL_AGENT:
            raise ValueError("Expected TOOL_AGENT architecture.")
        return cls(
            model=config.model,
            backend=config.backend,
            api_key=config.api_key,
            enabled_tools=config.permitted_tools,
            max_tool_calls=config.max_tool_calls,
            student_instructions=config.student_instructions,
        )
