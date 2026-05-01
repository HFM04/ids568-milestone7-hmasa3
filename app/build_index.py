"""Build a FAISS index from the source-of-truth paper.

Usage:
    python app/build_index.py

Outputs:
    index/faiss.index
    index/chunks.json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    SOURCE_DOCX,
)


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    char_start: int
    char_end: int
    source_sha256: str


def read_docx(path: Path) -> str:
    """Extract non-empty paragraphs from a DOCX file."""
    if not path.exists():
        raise FileNotFoundError(f"Source document not found: {path}")

    document = Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        raise ValueError("The DOCX file did not contain extractable text.")
    return text


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash for source integrity tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_text(text: str, source_name: str, source_hash: str) -> List[Chunk]:
    """Split document text into overlapping character chunks."""
    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk,
                    source=source_name,
                    char_start=start,
                    char_end=end,
                    source_sha256=source_hash,
                )
            )
            chunk_id += 1
        if end == text_len:
            break
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings so inner product approximates cosine similarity."""
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    text = read_docx(SOURCE_DOCX)
    source_hash = file_sha256(SOURCE_DOCX)
    chunks = chunk_text(text, SOURCE_DOCX.name, source_hash)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode([c.text for c in chunks], convert_to_numpy=True, show_progress_bar=True)
    embeddings = normalize_embeddings(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)

    print(f"Built FAISS index with {len(chunks)} chunks")
    print(f"Index saved to: {FAISS_INDEX_PATH}")
    print(f"Chunks saved to: {CHUNKS_PATH}")
    print(f"Source SHA-256: {source_hash}")


if __name__ == "__main__":
    main()
