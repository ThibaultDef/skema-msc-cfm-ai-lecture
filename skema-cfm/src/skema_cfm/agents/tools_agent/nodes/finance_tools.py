"""Bind only the four shared deterministic finance functions; never build a RAG index."""
from __future__ import annotations

from langchain_core.tools import tool

from ....case import CaseSnapshot
from ....tools.finance import (
    lookup_open_invoices,
    query_bank_receipts,
    query_forecast,
    reconcile_variance,
)
from ..settings import ToolsAgentSettings


def build_finance_tools(case: CaseSnapshot, settings: ToolsAgentSettings) -> dict:
    @tool("query_forecast")
    def forecast(customer: str = "") -> dict:
        """Read forecast amounts and source record IDs for a customer, or everyone."""
        return query_forecast(case, customer or None)

    @tool("query_bank_receipts")
    def bank(customer: str = "") -> dict:
        """Read posted bank receipts and identify absent remittance references."""
        return query_bank_receipts(case, customer or None)

    @tool("lookup_open_invoices")
    def receivables(customer: str = "", invoice_id: str = "") -> dict:
        """Read open invoice balances and statuses by customer and/or invoice."""
        return lookup_open_invoices(case, customer or None, invoice_id or None)

    @tool("reconcile_variance")
    def variance(customer: str = "") -> dict:
        """Compute actual less forecast and customer breakdown using source records."""
        return reconcile_variance(case, customer or None)

    all_tools = {t.name: t for t in (forecast, bank, receivables, variance)}
    return {name: all_tools[name] for name in settings.enabled_tools}
