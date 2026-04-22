"""Multi-seed aggregation with bootstrap confidence intervals.

Answers the paper's own §5 Limitations question: the single-seed scaling
trace cannot distinguish a real effect from seed variance. This module
aggregates metrics across seeds and reports mean ± bootstrap 95% CI.

Why bootstrap rather than the per-seed Wilson interval? Wilson gives a CI
for one binomial proportion at a time. What we actually want is a CI on the
*mean across seeds* — with three seeds we are ~resampling seed-level means,
which bootstrap handles cleanly without distributional assumptions.

Pure-python, no numpy dependency — the bootstrap is tiny (≤3×10³ resamples
per metric) and runs in milliseconds.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable

from metrics import ConditionMetrics, compute_metrics


@dataclass(frozen=True)
class AggregatedMetric:
    """Mean, standard deviation, and bootstrap 95% CI half-width across seeds."""

    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    n_seeds: int

    @property
    def ci95_halfwidth_pp(self) -> float:
        """Max of upper/lower half-width in percentage points."""
        return max(self.mean - self.ci95_low, self.ci95_high - self.mean) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"ci95_halfwidth_pp": self.ci95_halfwidth_pp}


@dataclass(frozen=True)
class AggregatedCondition:
    """All primary accuracy views aggregated across seeds for one condition."""

    n_seeds: int
    score_acc: AggregatedMetric
    text_acc: AggregatedMetric
    coherence_acc: AggregatedMetric
    text_score_agreement: AggregatedMetric

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_seeds": self.n_seeds,
            "score_acc": self.score_acc.to_dict(),
            "text_acc": self.text_acc.to_dict(),
            "coherence_acc": self.coherence_acc.to_dict(),
            "text_score_agreement": self.text_score_agreement.to_dict(),
        }


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of ``values``.

    Returns ``(low, high)``. With only 3 seeds the bootstrap is a
    rough-but-honest proxy for variability — it does not invent degrees of
    freedom the data doesn't have, but it does give a visually comparable
    error bar alongside the paper's Wilson intervals.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    k = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(sum(resample) / k)
    means.sort()
    alpha = (1 - confidence) / 2
    low = means[int(alpha * n_resamples)]
    high = means[int((1 - alpha) * n_resamples) - 1]
    return (low, high)


def _aggregate_values(values: list[float]) -> AggregatedMetric:
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = variance ** 0.5
    else:
        std = 0.0
    low, high = bootstrap_ci(values)
    return AggregatedMetric(
        mean=mean,
        std=std,
        ci95_low=low,
        ci95_high=high,
        n_seeds=len(values),
    )


def aggregate_across_seeds(
    seed_records: list[list[dict[str, Any]]],
    compute: Callable[[list[dict[str, Any]]], ConditionMetrics] = compute_metrics,
) -> AggregatedCondition:
    """Compute metrics for each seed's records and aggregate across seeds.

    ``seed_records[i]`` is the per-sample record list from seed ``i``. Every
    seed's records must be evaluated on the same eval split for the mean to
    be meaningful — this is not checked here; the caller owns it.
    """
    if not seed_records:
        raise ValueError("seed_records is empty")
    per_seed = [compute(r) for r in seed_records]
    return AggregatedCondition(
        n_seeds=len(per_seed),
        score_acc=_aggregate_values([m.score_acc for m in per_seed]),
        text_acc=_aggregate_values([m.text_acc for m in per_seed]),
        coherence_acc=_aggregate_values([m.coherence_acc for m in per_seed]),
        text_score_agreement=_aggregate_values(
            [m.text_score_agreement for m in per_seed if m.text_score_agreement == m.text_score_agreement]
            # filter NaNs (conditions where nothing parsed)
        ),
    )
