"""Retrieval part of the RAG pipeline.

Flow:  ML output -> build_query() -> embed with all-MiniLM-L6-v2
       -> FAISS IndexFlatIP search -> keep top 3 with cosine >= 0.35

Because the vectors are L2-normalised, the inner product that IndexFlatIP
returns is exactly the cosine similarity (1 = same meaning, 0 = unrelated).
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
THRESHOLD = 0.35                       # minimum cosine similarity to count as support
NO_SOURCE_MSG = "No supporting source found"


def load_embedder():
    """Load the sentence-transformers model (downloaded once, then cached)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(embedder, texts: list[str]) -> np.ndarray:
    """Embed texts and L2-normalise them so inner product = cosine similarity."""
    vecs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def build_query(prediction: dict) -> str:
    """Build the search query FROM the ML output (label, CIBIL, top factors).

    Example: "rejected loan application, CIBIL score 412, main factors
    CIBIL score, loan term and loan amount"
    """
    applicant = prediction["applicant"]
    names = [f["name"] for f in prediction["factors"]]
    factor_text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return (f"{prediction['label'].lower()} loan application, "
            f"CIBIL score {applicant['cibil_score']}, main factors {factor_text}")


class Retriever:
    """Loads the FAISS index + chunk metadata and searches them."""

    def __init__(self, embedder, index_dir: str | Path = INDEX_DIR):
        index_dir = Path(index_dir)
        self.embedder = embedder
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

    def search(self, query: str, k: int = TOP_K, threshold: float = THRESHOLD) -> list[dict]:
        """Return up to k chunks with cosine >= threshold (may be an empty list)."""
        scores, ids = self.index.search(embed(self.embedder, [query]), k)
        results = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1 or score < threshold:
                continue
            c = self.chunks[i]
            results.append({"text": c["text"], "source": c["source"], "title": c["title"],
                            "score": round(float(score), 3)})
        return results
