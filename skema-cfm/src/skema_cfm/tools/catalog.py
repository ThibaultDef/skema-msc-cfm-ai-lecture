"""LangChain tool definitions wrap deterministic, read-only functions only."""

from __future__ import annotations

from langchain_core.tools import tool

from ..case import CaseSnapshot
from ..config import RunConfig
from ..retrieval import LexicalDocumentIndex
from .finance import (
    lookup_open_invoices,
    query_bank_receipts,
    query_forecast,
    reconcile_variance,
)


def build_tools(case: CaseSnapshot, config: RunConfig):
    """Instantiated per run, so no mutable tool objects or scenario state are shared."""
    index = LexicalDocumentIndex(case.documents)

    @tool
    def query_forecast_tool(customer: str = "") -> dict:
        """Read forecast rows/amounts for one customer or all customers. Returns record IDs."""
        return query_forecast(case, customer or None)

    @tool
    def query_bank_receipts_tool(customer: str = "") -> dict:
        """Read bank receipts in the reporting week; disclose missing remittance references."""
        return query_bank_receipts(case, customer or None)

    @tool
    def lookup_open_invoices_tool(customer: str = "", invoice_id: str = "") -> dict:
        """Look up open AR balances, invoice status, and reference IDs without editing records."""
        return lookup_open_invoices(case, customer or None, invoice_id or None)

    @tool
    def reconcile_variance_tool(customer: str = "") -> dict:
        """Deterministically calculate forecast, actual and variance by customer and in total."""
        return reconcile_variance(case, customer or None)

    @tool
    def search_documents_tool(query: str) -> dict:
        """Search approved notes by query and return dated passages and stable DOC IDs."""
        return {"query": query, "passages": index.search(query, k=config.retrieval_k)}

    # Explicit stable names used by the permissions matrix and UI.
    candidates = {
        "query_forecast": query_forecast_tool,
        "query_bank_receipts": query_bank_receipts_tool,
        "lookup_open_invoices": lookup_open_invoices_tool,
        "reconcile_variance": reconcile_variance_tool,
        "search_documents": search_documents_tool,
    }
    for name, function in candidates.items():
        function.name = name
    return {name: candidates[name] for name in config.permitted_tools}
