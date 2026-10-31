"""Deterministic, local, auditable lexical RAG (no external embeddings or network)."""

from __future__ import annotations

import math
import re
from collections import Counter

from .case import EvidenceDocument

STOPWORDS = frozenset(
    "a an and are as at be by can date did do does for from has have in is it of on or our "
    "the their this to was were what when where which who why will with would week cash payment "
    "payments eur document evidence find search explain investigate".split()
)


def tokenize(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in STOPWORDS and len(word) > 1
    ]


class LexicalDocumentIndex:
    """TF-IDF weighted document ranking, full text returned with stable IDs.

    Explicit lexical baseline for pedagogy. Replacing this class with a semantic
    retriever does not change the LangGraph or finance tools API.
    """

    def __init__(self, documents: tuple[EvidenceDocument, ...]):
        self.documents = documents
        self.counts = [
            Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents
        ]
        self.df = Counter(token for counts in self.counts for token in counts)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not 1 <= k <= 8:
            raise ValueError("k must be 1-8.")
        words = set(tokenize(query))
        if not words:
            return []
        n = len(self.documents)
        scored = []
        for doc, freq in zip(self.documents, self.counts):
            score = 0.0
            for term in words:
                if term in freq:
                    idf = math.log(1 + (n + 1) / (1 + self.df[term]))
                    score += (1.0 + math.log(freq[term])) * idf
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: (-x[0], x[1].document_id))
        return [
            {**doc.to_dict(), "retrieval_score": round(score, 3)}
            for score, doc in scored[:k]
        ]
