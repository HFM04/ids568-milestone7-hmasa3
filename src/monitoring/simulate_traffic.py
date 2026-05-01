"""Generate local traffic for Prometheus/Grafana dashboard testing.

Run after building the FAISS index and starting Ollama:
    python src/monitoring/simulate_traffic.py
"""

from __future__ import annotations

import random
import time

from app.rag_pipeline import ClosedDomainRAG
from src.monitoring.instrumentation import record_error, record_success, start_metrics_server

IN_SCOPE_QUESTIONS = [
    "What problem does the Warehouse Agent System address?",
    "What are the two agents in the proposed system?",
    "What KPIs are used to evaluate the system?",
    "How many Monte Carlo runs were used in the simulation?",
    "What reduction in carrier wait time was reported?",
    "How does Agent 1 detect congestion?",
    "How does Agent 2 prioritize tasks?",
    "What are the limitations mentioned in the paper?",
]

OUT_OF_SCOPE_QUESTIONS = [
    "What is the capital of France?",
    "Who won the latest Super Bowl?",
    "What should I invest in this year?",
    "Explain quantum computing from scratch.",
    "What is today's weather?",
]


def main() -> None:
    start_metrics_server(port=8001)
    rag = ClosedDomainRAG()
    questions = IN_SCOPE_QUESTIONS + OUT_OF_SCOPE_QUESTIONS

    print("Generating simulated RAG traffic. Press Ctrl+C to stop.")
    while True:
        query = random.choice(questions)
        try:
            result = rag.answer(query)
            record_success(result)
            print(
                f"query={query[:45]!r} refused={result['refused']} "
                f"score={result['top_retrieval_score']:.3f} "
                f"grounded={result['grounded']} latency={result['latency_seconds']:.2f}s"
            )
        except Exception as exc:  # noqa: BLE001 - monitoring should catch all runtime failures
            record_error()
            print(f"ERROR for query={query!r}: {exc}")
        time.sleep(random.uniform(0.5, 2.0))


if __name__ == "__main__":
    main()
