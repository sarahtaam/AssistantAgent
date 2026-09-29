"""
scripts/build_index.py — Builds the FAISS knowledge-base index.

Run this once after installing dependencies:
    python scripts/build_index.py

Downloads the embedding model (~120MB) from Hugging Face on first run,
then vectorizes every document in assistant/rag/documents/ and persists
the FAISS index to assistant/rag/faiss.index.

If this fails (typically: no internet access to huggingface.co), that's
not fatal — the agent still runs. `retriever.is_ready` will simply stay
False, and the orchestrator falls back to its embedded knowledge base
instead of using RAG. See assistant/agent/orchestrator.py::_get_rag_context.
"""
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/build_index.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from assistant.rag.retriever import get_retriever  # noqa: E402


def main() -> int:
    retriever = get_retriever()
    try:
        n = retriever.build_from_docs()
    except Exception as exc:
        print(f"\nCould not build the RAG index: {exc}\n")
        print(
            "This is most likely a network issue downloading the embedding "
            "model from Hugging Face on first run. The agent will still work "
            "without it — it falls back to a small embedded knowledge base "
            "(see orchestrator.py::_get_rag_context) — but responses won't "
            "benefit from semantic search over assistant/rag/documents/ "
            "until this succeeds.\n"
        )
        return 1

    print(f"\nIndex built: {n} chunks indexed -> assistant/rag/faiss.index\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
