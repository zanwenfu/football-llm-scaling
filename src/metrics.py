"""Accuracy, MAE, prediction distribution, Wilson confidence intervals.

The JSON files in `results/raw/` store per-sample predictions produced by the
QLoRA, ICL, and CoT conditions. This module consumes those lists and computes
the metrics reported in Table 1 of the paper.

There are three views of accuracy, which `compute_all_metrics` computes
jointly so the parser-rescue gap is visible in a single pass:

1. ``score_acc`` — "score overrides text" legacy convention (primary number
   in the paper's headline table).
2. ``text_acc`` — label on the ``Prediction:`` line, ignoring the score.
3. ``coherence_acc`` — credit only when text == score == ground truth.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from config import WILSON_Z_95
from parsing import (
    Label,
    ParsedOutput,
    parse_output,
    resolve_score_overrides_text,
)

# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Returns ``(lower, upper)``. Used for every ±pp figure in the paper.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))


def wilson_halfwidth_pp(k: int, n: int, z: float = WILSON_Z_95) -> float:
    """Symmetric half-width of the Wilson interval in percentage points.

    The paper reports CIs as `p ± halfwidth_pp`; Wilson intervals are slightly
    asymmetric so this returns the larger side, matching the paper's rounding.
    """
    low, high = wilson_ci(k, n, z)
    p = k / n if n else 0.0
    return max(p - low, high - p) * 100.0


# ---------------------------------------------------------------------------
# Typed per-condition record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionMetrics:
    """Summary numbers for one condition × split (overall / named / anon)."""

    n: int
    parsed: int
    parse_rate: float

    # Three accuracy views
    score_acc: float  # parser-rescued (score overrides text when both exist)
    text_acc: float  # text label on Prediction: line
    coherence_acc: float  # text == score == GT

    # Wilson 95% two-sided intervals (half-widths in percentage points)
    score_acc_ci95_pp: float
    text_acc_ci95_pp: float
    coherence_acc_ci95_pp: float

    # Score regression
    score_exact_match: float
    goal_mae: float

    # Prediction distribution
    home_pred_rate: float
    away_pred_rate: float
    draw_pred_rate: float

    # Field-binding signal
    text_score_agreement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Result-record helpers (work on the raw/*.json schema)
# ---------------------------------------------------------------------------


def _gt_label(record: dict[str, Any]) -> Label | None:
    gt = record.get("gt") or {}
    return Label.from_str(gt.get("result", "")) if gt else None


def _gt_score(record: dict[str, Any]) -> tuple[int, int] | None:
    gt = record.get("gt") or {}
    h, a = gt.get("home_goals"), gt.get("away_goals")
    if h is None or a is None:
        return None
    return (int(h), int(a))


def _parsed(record: dict[str, Any]) -> ParsedOutput:
    return parse_output(record.get("raw_output", ""))


def _split_by_anon(records: Iterable[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    named, anon = [], []
    for r in records:
        (anon if r.get("is_anon") else named).append(r)
    return named, anon


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------


def compute_metrics(records: list[dict[str, Any]]) -> ConditionMetrics:
    """Compute the three accuracy views + ancillary metrics from raw records."""
    n = len(records)
    parsed_list = [_parsed(r) for r in records]
    gt_labels = [_gt_label(r) for r in records]
    gt_scores = [_gt_score(r) for r in records]

    parsed = sum(1 for p in parsed_list if p.has_prediction_line or p.has_score_line)

    score_correct = 0
    text_correct = 0
    coherence_correct = 0
    agreements = 0
    pred_counter: Counter[str] = Counter()
    score_matches = 0
    mae_total = 0.0
    mae_n = 0

    for p, gt_label, gt_score in zip(parsed_list, gt_labels, gt_scores, strict=True):
        effective = resolve_score_overrides_text(p)
        if effective is not None and effective == gt_label:
            score_correct += 1
        if p.text_label is not None and p.text_label == gt_label:
            text_correct += 1
        if p.is_coherent and p.text_label == gt_label:
            coherence_correct += 1
        if p.text_label is not None and p.score_label is not None:
            agreements += int(p.text_label == p.score_label)
        if effective is not None:
            pred_counter[effective.value] += 1
        if (
            p.home_goals is not None
            and p.away_goals is not None
            and gt_score is not None
        ):
            mae_total += abs(p.home_goals - gt_score[0]) + abs(p.away_goals - gt_score[1])
            mae_n += 2
            if (p.home_goals, p.away_goals) == gt_score:
                score_matches += 1

    agree_denom = sum(
        1 for p in parsed_list if p.text_label is not None and p.score_label is not None
    )

    return ConditionMetrics(
        n=n,
        parsed=parsed,
        parse_rate=parsed / n if n else 0.0,
        score_acc=score_correct / n if n else 0.0,
        text_acc=text_correct / n if n else 0.0,
        coherence_acc=coherence_correct / n if n else 0.0,
        score_acc_ci95_pp=wilson_halfwidth_pp(score_correct, n),
        text_acc_ci95_pp=wilson_halfwidth_pp(text_correct, n),
        coherence_acc_ci95_pp=wilson_halfwidth_pp(coherence_correct, n),
        score_exact_match=score_matches / n if n else 0.0,
        goal_mae=mae_total / mae_n if mae_n else float("nan"),
        home_pred_rate=pred_counter["home_win"] / n if n else 0.0,
        away_pred_rate=pred_counter["away_win"] / n if n else 0.0,
        draw_pred_rate=pred_counter["draw"] / n if n else 0.0,
        text_score_agreement=agreements / agree_denom if agree_denom else float("nan"),
    )


def compute_all_splits(records: list[dict[str, Any]]) -> dict[str, ConditionMetrics]:
    """Return metrics for the ``overall`` / ``named`` / ``anon`` splits."""
    named, anon = _split_by_anon(records)
    return {
        "overall": compute_metrics(records),
        "named": compute_metrics(named),
        "anon": compute_metrics(anon),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def always_home_accuracy(records: list[dict[str, Any]]) -> float:
    """Accuracy of the trivial always-home-win predictor on this eval set."""
    n = len(records)
    if n == 0:
        return 0.0
    return sum(1 for r in records if _gt_label(r) == Label.HOME_WIN) / n


def random_weighted_accuracy(
    records: list[dict[str, Any]],
    home_p: float,
    away_p: float,
    draw_p: float,
    seed: int = 42,
) -> float:
    """Accuracy of a class-weighted random predictor. Deterministic given seed."""
    import random

    rng = random.Random(seed)
    labels = [Label.HOME_WIN.value, Label.AWAY_WIN.value, Label.DRAW.value]
    weights = [home_p, away_p, draw_p]
    correct = 0
    for r in records:
        gt = _gt_label(r)
        if gt is None:
            continue
        pick = rng.choices(labels, weights=weights, k=1)[0]
        if pick == gt.value:
            correct += 1
    return correct / len(records) if records else 0.0


# ---------------------------------------------------------------------------
# Format compatibility with scaling_metrics.json
# ---------------------------------------------------------------------------


def metrics_to_legacy_schema(m: ConditionMetrics) -> dict[str, Any]:
    """Match the shape of the original ``scaling_metrics.json`` schema."""
    return {
        "total": m.n,
        "parsed": m.parsed,
        "parse_rate": m.parse_rate,
        "result_accuracy": m.score_acc,
        "score_exact_match": m.score_exact_match,
        "goal_mae": m.goal_mae,
        "pred_dist": {
            "home_win": round(m.home_pred_rate * m.n),
            "away_win": round(m.away_pred_rate * m.n),
            "draw": round(m.draw_pred_rate * m.n),
            "other": m.n
            - round(m.home_pred_rate * m.n)
            - round(m.away_pred_rate * m.n)
            - round(m.draw_pred_rate * m.n),
        },
        "home_pred_rate": m.home_pred_rate,
    }
