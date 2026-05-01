from __future__ import annotations

import random
import time

from src.monitoring.instrumentation import record_success, start_metrics_server


def main() -> None:
    start_metrics_server(port=8001)
    print("Light metrics server running at http://localhost:8001/metrics")

    while True:
        result = {
            "latency_seconds": random.uniform(0.4, 1.5),
            "top_retrieval_score": random.uniform(0.65, 0.95),
            "groundedness_score": random.uniform(0.75, 0.98),
            "refused": random.choice([False, False, False, True]),
            "grounded": random.choice([True, True, True, False]),
        }

        record_success(result)
        print("Recorded one lightweight metrics event")
        time.sleep(5)


if __name__ == "__main__":
    main()
