"""Build the RAG index from knowledge_base/*.txt.

Usage: python build_index.py

Each knowledge-base file starts with two header lines:
    SOURCE: <url>
    TITLE: <title>
followed by the readable text of a public RBI or TransUnion CIBIL page.

Steps:
1. Split each file's text into ~400-word chunks with a 50-word overlap
   (the overlap stops a sentence from being cut off between two chunks).
2. Embed every chunk with sentence-transformers all-MiniLM-L6-v2 (384 numbers).
3. L2-normalise the vectors and store them in a FAISS IndexFlatIP, so the
   inner-product search returns cosine similarity.
4. Save index/faiss.index and index/chunks.json (text + source URL + title).
"""
import json
from pathlib import Path

import faiss

from loanlens.rag import INDEX_DIR, embed, load_embedder

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
CHUNK_WORDS, OVERLAP = 400, 50


def read_kb_file(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    source = lines[0].removeprefix("SOURCE:").strip()
    title = lines[1].removeprefix("TITLE:").strip()
    return {"source": source, "title": title, "text": "\n".join(lines[2:]).strip()}


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP) -> list[str]:
    """Sliding window over words: 0-400, 350-750, 700-1100, ..."""
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def main():
    chunks = []
    for path in sorted(KB_DIR.glob("*.txt")):
        doc = read_kb_file(path)
        pieces = chunk_words(doc["text"])
        chunks += [{"text": t, "source": doc["source"], "title": doc["title"], "file": path.name}
                   for t in pieces]
        print(f"{path.name}: {len(pieces)} chunks")
    if not chunks:
        raise SystemExit("knowledge_base/ is empty - add the source .txt files first.")

    vectors = embed(load_embedder(), [c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])       # exact inner-product search
    index.add(vectors)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {index.ntotal} chunks of dimension {vectors.shape[1]} -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
