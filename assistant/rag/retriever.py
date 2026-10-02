"""
assistant/rag/retriever.py — Semantic search interface for RAG.

Single entry point for any knowledge-base search. Orchestrates
Embedder + FAISSIndexer behind a simple API.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from assistant.rag.embedder import get_embedder, Embedder
from assistant.rag.indexer import get_indexer, FAISSIndexer
from assistant.rag.document_loader import Chunk, DocumentLoader

logger = logging.getLogger(__name__)

# Cosine relevance threshold (post L2-normalization, scores are in [0, 1])
MIN_RELEVANCE_SCORE = 0.28


@dataclass
class RetrievalResult:
    """A retrieved chunk with its relevance score."""
    text: str
    source: str
    score: float
    chunk_id: int
    page: Optional[int]
    section: Optional[str]
    doc_type: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "text": self.text, "source": self.source, "score": self.score,
            "chunk_id": self.chunk_id, "page": self.page, "section": self.section,
            "doc_type": self.doc_type, "metadata": self.metadata,
        }

    def format_for_prompt(self) -> str:
        location = ""
        if self.page:
            location = f" [p.{self.page}]"
        elif self.section:
            location = f" [{self.section}]"
        return f"[{self.source}{location}] {self.text}"


class RAGRetriever:
    """
    Orchestrates Embedder + FAISSIndexer for semantic search.

    Lifecycle:
      1. On startup: call load_from_disk() (or build_from_docs() if no
         index exists yet)
      2. On every message: call retrieve(query)
    """

    def __init__(self, embedder: Optional[Embedder] = None, indexer: Optional[FAISSIndexer] = None) -> None:
        self._embedder = embedder or get_embedder()
        self._indexer = indexer or get_indexer()

    def load_from_disk(self) -> bool:
        return self._indexer.load(embedder=self._embedder)

    def build_from_docs(self) -> int:
        chunks = DocumentLoader.load_all()
        if not chunks:
            logger.warning("No documents loaded - index is empty.")
            return 0
        self._indexer.build(chunks, embedder=self._embedder)
        self._indexer.save()
        return len(chunks)

    def rebuild(self) -> int:
        logger.info("Rebuilding RAG index from scratch")
        self._indexer.clear()
        return self.build_from_docs()

    def retrieve(self, query: str, top_k: int = 5, threshold: float = MIN_RELEVANCE_SCORE) -> List[RetrievalResult]:
        """Semantic search over the knowledge base. Returns results sorted
        by descending relevance score, deduplicated on near-identical text."""
        if not self._indexer.is_ready:
            logger.debug("Index not ready - RAG disabled for this query.")
            return []
        if not query or not query.strip():
            return []

        q_vec = self._embedder.embed([query.strip()])
        raw_scores, raw_indices = self._indexer.search(q_vec, top_k)
        chunks = self._indexer.chunks
        results: List[RetrievalResult] = []
        seen_texts = set()

        for score, idx in zip(raw_scores, raw_indices):
            if idx < 0 or idx >= len(chunks):
                continue
            if score < threshold:
                continue
            chunk = chunks[idx]
            normalized = chunk.text[:200].strip().lower()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            results.append(RetrievalResult(
                text=chunk.text, source=chunk.source, score=round(float(score), 4),
                chunk_id=chunk.chunk_id, page=chunk.page, section=chunk.section,
                doc_type=chunk.doc_type, metadata=chunk.metadata,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        # The query contains the client's message: log its size, not its text.
        logger.debug("RAG query (len=%d) -> %d results (threshold=%.2f)", len(query), len(results), threshold)
        return results

    def retrieve_as_context(self, query: str, top_k: int = 4, sep: str = "\n\n---\n\n") -> str:
        """Returns retrieved results as a block of text ready to inject
        into an LLM prompt. Returns a placeholder string if nothing matched."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant reference document found for this query."
        blocks = [r.format_for_prompt() for r in results]
        return sep.join(blocks)

    @property
    def is_ready(self) -> bool:
        return self._indexer.is_ready

    @property
    def index_size(self) -> int:
        return self._indexer.size


_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever
