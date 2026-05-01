"""Closed-domain RAG pipeline for the warehouse inbound congestion paper.

The assistant answers only from retrieved source chunks. Unsupported questions
must return the configured refusal message.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    AUDIT_LOG_PATH,
    CHUNKS_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    LOG_DIR,
    MIN_RETRIEVAL_SCORE,
    REFUSAL_MESSAGE,
)
from app.ollama_client import OllamaClient


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float
    source: str


class ClosedDomainRAG:
    """FAISS + sentence-transformers retriever with Ollama generation."""

    def __init__(self, top_k: int = DEFAULT_TOP_K, min_score: float = MIN_RETRIEVAL_SCORE):
        if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError("Index files are missing. Run: python app/build_index.py")

        self.top_k = top_k
        self.min_score = min_score
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        with CHUNKS_PATH.open("r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.llm = OllamaClient()

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """Return top-k retrieved chunks with cosine similarity scores."""
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, self.top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = self.chunks[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=int(item["chunk_id"]),
                    text=item["text"],
                    score=float(score),
                    source=item["source"],
                )
            )
        return results

    def build_prompt(self, query: str, retrieved: List[RetrievedChunk]) -> str:
        """Create a strict source-grounded prompt."""
        context = "\n\n---\n\n".join(
            [f"[Chunk {c.chunk_id} | score={c.score:.3f}]\n{c.text}" for c in retrieved]
        )
        return f"""You are a closed-domain RAG assistant.
Answer the user's question using ONLY the provided source context.
Do not use outside knowledge.
If the answer is not explicitly supported by the context, respond exactly: {REFUSAL_MESSAGE}

SOURCE CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:"""

    @staticmethod
    def groundedness_score(answer: str, retrieved: List[RetrievedChunk]) -> float:
        """Simple lexical overlap groundedness heuristic for monitoring.

        This is intentionally lightweight for the course project. It is not a
        substitute for human evaluation, but it gives a useful operational signal.
        """
        if answer.strip() == REFUSAL_MESSAGE:
            return 1.0
        context_words = set(" ".join(c.text.lower() for c in retrieved).split())
        answer_words = [w for w in answer.lower().split() if len(w) > 3]
        if not answer_words:
            return 0.0
        overlap = sum(1 for w in answer_words if w in context_words)
        return overlap / len(answer_words)

    def answer(self, query: str) -> Dict[str, Any]:
        """Retrieve, generate, evaluate, and audit one answer."""
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        retrieved = self.retrieve(query)
        retrieval_latency = time.perf_counter() - retrieval_start

        top_score = retrieved[0].score if retrieved else 0.0
        if not retrieved or top_score < self.min_score:
            answer = REFUSAL_MESSAGE
            ollama_latency = 0.0
        else:
            prompt = self.build_prompt(query, retrieved)
            llm_start = time.perf_counter()
            answer = self.llm.generate(prompt)
            ollama_latency = time.perf_counter() - llm_start

        total_latency = time.perf_counter() - start
        groundedness = self.groundedness_score(answer, retrieved)
        result = {
            "query": query,
            "answer": answer,
            "refused": answer.strip() == REFUSAL_MESSAGE,
            "top_retrieval_score": top_score,
            "mean_retrieval_score": float(np.mean([c.score for c in retrieved])) if retrieved else 0.0,
            "groundedness_score": groundedness,
            "grounded": groundedness >= 0.70,
            "retrieved_chunks": [c.__dict__ for c in retrieved],
            "latency_seconds": total_latency,
            "retrieval_latency_seconds": retrieval_latency,
            "ollama_latency_seconds": ollama_latency,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.write_audit_event(result)
        return result

    @staticmethod
    def write_audit_event(event: Dict[str, Any]) -> None:
        """Append structured JSONL audit event."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        serializable = dict(event)
        serializable["retrieved_chunks"] = [
            {"chunk_id": c["chunk_id"], "score": c["score"], "source": c["source"]}
            for c in event.get("retrieved_chunks", [])
        ]
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rag = ClosedDomainRAG()
    while True:
        user_query = input("Question, or 'exit': ").strip()
        if user_query.lower() in {"exit", "quit"}:
            break
        output = rag.answer(user_query)
        print("\nAnswer:", output["answer"])
        print(f"Score: {output['top_retrieval_score']:.3f} | Groundedness: {output['groundedness_score']:.3f}\n")
