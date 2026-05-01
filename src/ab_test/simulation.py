"""A/B test simulation for closed-domain RAG variants.

Variant A: baseline retrieval configuration.
Variant B: improved retrieval configuration.

The simulation creates synthetic evaluation outcomes for answer correctness,
groundedness, refusal accuracy, and latency, then performs statistical tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


@dataclass
class VariantResult:
    name: str
    n: int
    correctness_rate: float
    groundedness_rate: float
    refusal_accuracy_rate: float
    hallucination_rate: float
    latency_mean: float
    latency_p95: float


def required_sample_size(p1: float = 0.78, p2: float = 0.88, alpha: float = 0.05, power: float = 0.80) -> int:
    """Power calculation for a two-proportion test."""
    effect_size = proportion_effectsize(p1, p2)
    analysis = NormalIndPower()
    n = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
    return int(np.ceil(n))


def simulate_variant(name: str, n: int, seed: int, correctness_p: float, grounded_p: float,
                     refusal_p: float, hallucination_p: float, latency_mean: float) -> tuple[VariantResult, dict]:
    """Generate synthetic A/B outcomes for one RAG variant."""
    rng = np.random.default_rng(seed)
    correctness = rng.binomial(1, correctness_p, n)
    groundedness = rng.binomial(1, grounded_p, n)
    refusal_accuracy = rng.binomial(1, refusal_p, n)
    hallucination = rng.binomial(1, hallucination_p, n)
    latency = np.clip(rng.normal(latency_mean, 0.45, n), 0.2, None)

    result = VariantResult(
        name=name,
        n=n,
        correctness_rate=float(correctness.mean()),
        groundedness_rate=float(groundedness.mean()),
        refusal_accuracy_rate=float(refusal_accuracy.mean()),
        hallucination_rate=float(hallucination.mean()),
        latency_mean=float(latency.mean()),
        latency_p95=float(np.percentile(latency, 95)),
    )
    raw = {
        "correctness": correctness,
        "groundedness": groundedness,
        "refusal_accuracy": refusal_accuracy,
        "hallucination": hallucination,
        "latency": latency,
    }
    return result, raw


def two_proportion_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return z statistic and p-value for two independent proportions."""
    count = np.array([a.sum(), b.sum()])
    nobs = np.array([len(a), len(b)])
    z_stat, p_value = proportions_ztest(count, nobs)
    return float(z_stat), float(p_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Sample size per variant. Defaults to powered sample size.")
    parser.add_argument("--dry-run", action="store_true", help="Run quickly with 50 samples per variant.")
    args = parser.parse_args()

    n_powered = required_sample_size()
    n = 250 if args.dry_run else (args.n or n_powered)

    a, raw_a = simulate_variant(
        "A_baseline_topk3_chunk800",
        n=n,
        seed=42,
        correctness_p=0.78,
        grounded_p=0.82,
        refusal_p=0.80,
        hallucination_p=0.08,
        latency_mean=2.10,
    )
    b, raw_b = simulate_variant(
        "B_improved_topk5_threshold",
        n=n,
        seed=560,
        correctness_p=0.88,
        grounded_p=0.91,
        refusal_p=0.89,
        hallucination_p=0.035,
        latency_mean=2.55,
    )

    z_correct, p_correct = two_proportion_test(raw_a["correctness"], raw_b["correctness"])
    z_ground, p_ground = two_proportion_test(raw_a["groundedness"], raw_b["groundedness"])
    t_latency, p_latency = stats.ttest_ind(raw_a["latency"], raw_b["latency"], equal_var=False)

    print("=== A/B Test Simulation ===")
    print(f"Required powered sample size per variant: {n_powered}")
    print(f"Simulated sample size per variant: {n}")
    print("\nVariant A:", a)
    print("Variant B:", b)
    print("\nStatistical tests:")
    print(f"Correctness z={z_correct:.3f}, p={p_correct:.5f}")
    print(f"Groundedness z={z_ground:.3f}, p={p_ground:.5f}")
    print(f"Latency Welch t={t_latency:.3f}, p={p_latency:.5f}")

    guardrail_pass = b.latency_p95 <= 5.0 and b.hallucination_rate <= 0.05
    if p_correct < 0.05 and p_ground < 0.05 and guardrail_pass:
        decision = "Ship Variant B"
    elif not guardrail_pass:
        decision = "Run more data or tune Variant B because guardrails failed"
    else:
        decision = "Run more data"
    print(f"\nRecommendation: {decision}")


if __name__ == "__main__":
    main()
