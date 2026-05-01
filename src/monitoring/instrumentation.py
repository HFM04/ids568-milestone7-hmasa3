"""Prometheus instrumentation for the closed-domain RAG service.

Six dashboard metrics are exposed:
1. request count
2. end-to-end latency
3. error count/rate
4. retrieval score
5. refusal count/rate
6. groundedness score/rate
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

RAG_REQUEST_COUNT = Counter(
    "rag_request_count_total",
    "Total number of RAG requests processed.",
)

RAG_LATENCY_SECONDS = Histogram(
    "rag_latency_seconds",
    "End-to-end RAG response latency in seconds.",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
)

RAG_ERROR_COUNT = Counter(
    "rag_error_count_total",
    "Total number of failed RAG requests.",
)

RAG_RETRIEVAL_SCORE = Gauge(
    "rag_retrieval_top_score",
    "Top FAISS similarity score for the most recent request.",
)

RAG_REFUSAL_COUNT = Counter(
    "rag_refusal_count_total",
    "Total number of source-scope refusals returned by the assistant.",
)

RAG_GROUNDEDNESS_SCORE = Gauge(
    "rag_groundedness_score",
    "Groundedness score for the most recent answer.",
)

RAG_GROUNDED_ANSWER_COUNT = Counter(
    "rag_grounded_answer_count_total",
    "Total number of answers classified as grounded.",
)

RAG_UNGROUNDED_ANSWER_COUNT = Counter(
    "rag_ungrounded_answer_count_total",
    "Total number of answers classified as ungrounded.",
)


def start_metrics_server(port: int = 8001) -> None:
    """Start Prometheus metrics endpoint at /metrics."""
    start_http_server(port, addr="0.0.0.0")
    print(f"Prometheus metrics server started on http://localhost:{port}/metrics")


def record_success(result: dict) -> None:
    """Record metrics from a successful RAG call."""
    RAG_REQUEST_COUNT.inc()
    RAG_LATENCY_SECONDS.observe(float(result.get("latency_seconds", 0.0)))
    RAG_RETRIEVAL_SCORE.set(float(result.get("top_retrieval_score", 0.0)))
    RAG_GROUNDEDNESS_SCORE.set(float(result.get("groundedness_score", 0.0)))

    if result.get("refused", False):
        RAG_REFUSAL_COUNT.inc()

    if result.get("grounded", False):
        RAG_GROUNDED_ANSWER_COUNT.inc()
    else:
        RAG_UNGROUNDED_ANSWER_COUNT.inc()


def record_error() -> None:
    """Record failed RAG request."""
    RAG_REQUEST_COUNT.inc()
    RAG_ERROR_COUNT.inc()
