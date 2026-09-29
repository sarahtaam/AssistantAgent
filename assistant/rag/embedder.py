"""
assistant/rag/embedder.py — Multilingual embedding model (FR/AR/EN), lazily loaded.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - 50 languages including French and Arabic
  - 384 dimensions — good speed/quality tradeoff
  - ~120MB on disk, cached automatically by Hugging Face on first run

Single responsibility: turn text into normalized numpy vectors. Index
construction and search live in indexer.py and retriever.py.

Note: downloading the model requires internet access on first run. If
you're building this in a network-restricted environment (e.g. a CI
sandbox), the retriever's `is_ready` check means the rest of the agent
still works via the embedded knowledge-base fallback even if the index
was never built — see orchestrator.py.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64


def _resolve_device() -> str:
    """Picks CUDA if available. Imported lazily so torch is only required
    when the embedder is actually used, not at module import time."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    """Wraps SentenceTransformer with lazy loading, thread-safety, and
    L2-normalized output (compatible with FAISS IndexFlatIP = cosine similarity)."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:  # double-check after acquiring the lock
                # Imported here rather than at module level: sentence-transformers
                # pulls in torch (~2GB). Keeping it lazy means the API server
                # boots and every non-RAG endpoint works without it installed.
                from sentence_transformers import SentenceTransformer

                device = _resolve_device()
                logger.info("Loading embedding model: %s ...", self._model_name)
                self._model = SentenceTransformer(self._model_name, device=device)
                dim = self._model.get_sentence_embedding_dimension()
                logger.info("Model ready - dimension: %d", dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Vectorizes a list of texts. Returns a float32 array of shape
        (len(texts), dim), each vector L2-normalized (dot product of two
        normalized vectors = cosine similarity)."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        self._load()
        vecs = self._model.encode(
            texts, batch_size=BATCH_SIZE, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vecs.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
