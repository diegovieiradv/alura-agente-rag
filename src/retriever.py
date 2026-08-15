from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.vector_store import load_vector_store

logger = logging.getLogger(__name__)

DEFAULT_K = 4
DEFAULT_MIN_RELEVANCE = 0.35


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict
    relevance: float


class Retriever:
    """Semantic search over the vector store, returning evidence chunks.

    Results below ``min_relevance`` are discarded so downstream answers are
    only grounded on genuinely related content.
    """

    def __init__(
        self,
        store: Chroma,
        k: int = DEFAULT_K,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
    ):
        self._store = store
        self.k = k
        self.min_relevance = min_relevance

    @classmethod
    def from_persisted(
        cls,
        persist_dir: str = "data/chromadb",
        collection_name: str = "alura_rag",
        k: int = DEFAULT_K,
        min_relevance: float = DEFAULT_MIN_RELEVANCE,
        embedding_function: Optional[Embeddings] = None,
    ) -> "Retriever":
        return cls(
            load_vector_store(
                persist_dir,
                collection_name=collection_name,
                embedding_function=embedding_function,
            ),
            k=k,
            min_relevance=min_relevance,
        )

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        results = self._store.similarity_search_with_relevance_scores(query, k=self.k)
        evidence: List[RetrievedChunk] = []
        for doc, score in results:
            relevance = float(score)
            if relevance < self.min_relevance:
                continue
            evidence.append(
                RetrievedChunk(
                    text=doc.page_content,
                    metadata=dict(doc.metadata),
                    relevance=round(relevance, 4),
                )
            )
        evidence.sort(key=lambda c: c.relevance, reverse=True)
        return evidence

    def has_evidence(self, query: str) -> bool:
        return bool(self.retrieve(query))