"""
assistant/rag/indexer.py — FAISS index construction, persistence, and reload.

Single responsibility: manage the vector index lifecycle. Vectorization
lives in embedder.py; search-and-format lives in retriever.py.

Index strategy:
  - IndexFlatIP  (exact, fast) when nb_chunks < 10,000
  - IndexIVFFlat (approximate, scalable) when nb_chunks >= 10,000
"""
from __future__ import annotations

import os
import pickle
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from assistant.rag.embedder import Embedder, get_embedder
from assistant.rag.document_loader import Chunk

logger = logging.getLogger(__name__)


def _faiss():
    """Imports faiss lazily. Keeps the API server bootable without the
    heavy vector-search stack installed — RAG then simply stays
    unavailable and the orchestrator uses its knowledge-base fallback."""
    import faiss
    return faiss

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss.index"
CHUNKS_PATH = BASE_DIR / "chunks.pkl"

IVF_THRESHOLD = 10_000


class FAISSIndexer:
    """Builds and persists a FAISS index from vectorized chunks."""

    def __init__(self) -> None:
        self._index = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk], embedder: Optional[Embedder] = None) -> None:
        if not chunks:
            logger.warning("build() - no chunks provided, index not created.")
            return

        emb = embedder or get_embedder()
        self._chunks = chunks

        texts = [c.text for c in chunks]
        logger.info("Vectorizing %d chunks...", len(texts))
        vecs = emb.embed(texts)
        dim = vecs.shape[1]

        if len(chunks) >= IVF_THRESHOLD:
            nlist = min(int(len(chunks) ** 0.5), 256)
            quantizer = _faiss().IndexFlatIP(dim)
            self._index = _faiss().IndexIVFFlat(quantizer, dim, nlist, _faiss().METRIC_INNER_PRODUCT)
            self._index.train(vecs)
            logger.info("IVFFlat index trained (nlist=%d)", nlist)
        else:
            self._index = _faiss().IndexFlatIP(dim)

        self._index.add(vecs)
        logger.info("FAISS index built - %d vectors, dim=%d", self._index.ntotal, dim)

    def save(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH) -> None:
        if self._index is None or not self._chunks:
            logger.warning("save() - nothing to save.")
            return
        os.makedirs(os.path.dirname(str(index_path)) or ".", exist_ok=True)
        _faiss().write_index(self._index, str(index_path))
        with open(chunks_path, "wb") as fh:
            pickle.dump(self._chunks, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Index saved -> %s (%d chunks)", index_path, len(self._chunks))

    def load(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH, embedder: Optional[Embedder] = None) -> bool:
        """Loads a persisted index. Also loads the embedder so the query-time
        dimension is known. Returns True on success."""
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            logger.info("No persisted FAISS index found.")
            return False
        try:
            emb = embedder or get_embedder()
            emb.embed(["init"])  # force model load
            self._index = _faiss().read_index(str(index_path))
            with open(chunks_path, "rb") as fh:
                self._chunks = pickle.load(fh)
            logger.info("FAISS index loaded - %d vectors, %d chunks", self._index.ntotal, len(self._chunks))
            return True
        except ImportError as exc:
            # Expected when the optional RAG stack (faiss / sentence-transformers
            # / torch) isn't installed. Not an error worth a traceback — the
            # orchestrator falls back to its embedded knowledge base.
            logger.info("RAG dependencies unavailable (%s) - using knowledge-base fallback.", exc.name)
            self._index = None
            self._chunks = []
            return False
        except Exception:
            logger.exception("Error loading FAISS index")
            self._index = None
            self._chunks = []
            return False

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> Tuple[List[float], List[int]]:
        if self._index is None:
            return [], []
        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, actual_k)
        return scores[0].tolist(), indices[0].tolist()

    def clear(self, index_path=INDEX_PATH, chunks_path=CHUNKS_PATH) -> None:
        self._index = None
        self._chunks = []
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(chunks_path):
            os.remove(chunks_path)
        logger.info("FAISS index cleared.")

    @property
    def chunks(self) -> List[Chunk]:
        return self._chunks

    @property
    def is_ready(self) -> bool:
        return self._index is not None and bool(self._chunks)

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0


_indexer: Optional[FAISSIndexer] = None
_indexer_lock = threading.Lock()


def get_indexer() -> FAISSIndexer:
    global _indexer
    if _indexer is None:
        with _indexer_lock:
            if _indexer is None:
                _indexer = FAISSIndexer()
    return _indexer
