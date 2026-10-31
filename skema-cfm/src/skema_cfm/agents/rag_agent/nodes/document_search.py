"""Build precisely one document-only retrieval tool against approved case documents."""

from __future__ import annotations

from langchain_core.tools import tool

from ....case import CaseSnapshot
from ....retrieval import LexicalDocumentIndex
from ..settings import RAGAgentSettings


def build_document_search_tool(case: CaseSnapshot, settings: RAGAgentSettings):
    available = {doc.document_id for doc in case.documents}
    selected = settings.selected_documents
    if selected is not None:
        missing = set(selected) - available
        if missing:
            raise ValueError(
                f"Document(s) unavailable in this case/scenario: {sorted(missing)}"
            )
    permitted = (
        tuple(doc for doc in case.documents if doc.document_id in selected)
        if selected is not None
        else case.documents
    )
    index = LexicalDocumentIndex(permitted)

    @tool("search_documents")
    def search_documents(query: str) -> dict:
        """Search approved case TXT documents by keywords; return dated, citable passages.

        Args:
            query: Specific customer, issue, invoice reference or payment-status search.
        """
        cleaned = query.strip()
        if not cleaned or len(cleaned) > 240:
            raise ValueError("Query must contain 1 to 240 non-blank characters.")
        passages = index.search(cleaned, k=settings.retrieval_k)
        return {
            "query": cleaned,
            "count": len(passages),
            "passages": passages,
            "retrieval_method": "local_lexical_tfidf",
        }

    return search_documents
