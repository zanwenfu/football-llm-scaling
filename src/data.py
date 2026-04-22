"""Dataset loading, named/anon splitting, stratified nested-prefix sampling.

The training data for the scaling ablation has two requirements that a naive
``dataset.shuffle().select(range(n))`` does not satisfy:

1. **Nested prefixes**: the 48-sample split must be a subset of the 96-sample
   split, which must be a subset of the 192-sample split. This is so that
   accuracy changes across budgets cannot be confounded with sample identity.
2. **Class stratification**: at every budget the home/away/draw balance
   matches the overall training-set balance. Without this, small budgets
   produce class-collapse adapters that look like artifacts of data mix.

:func:`stratified_nested_prefix_indices` builds such an index list
deterministically from a seed and verifies the subset property.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from config import SCALING_BUDGETS

# ---------------------------------------------------------------------------
# Prediction file I/O
# ---------------------------------------------------------------------------


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    """Load a `results/raw/*.json` prediction dump.

    The schema of each record is::

        {
            "sample_idx": int,
            "is_anon": bool,
            "gt": {"result": str, "home_goals": int, "away_goals": int, "parsed": bool},
            "pred": {"result": str, "home_goals": int, "away_goals": int, "parsed": bool},
            "raw_output": str,
            "output_len": Optional[int],  # CoT only
        }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected list of records in {path}, got {type(data).__name__}")
    return data


def save_predictions(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# ---------------------------------------------------------------------------
# Named / anonymized splitting
# ---------------------------------------------------------------------------


def split_named_anon(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition records into (named, anonymized) by the ``is_anon`` flag."""
    named, anon = [], []
    for r in records:
        (anon if r.get("is_anon") else named).append(r)
    return named, anon


# ---------------------------------------------------------------------------
# Stratified nested prefix sampling
# ---------------------------------------------------------------------------


def _gt_label(record: dict[str, Any]) -> str | None:
    return (record.get("gt") or record.get("metadata") or {}).get("result")


def stratified_nested_prefix_indices(
    train_records: list[dict[str, Any]],
    budgets: tuple[int, ...] = SCALING_BUDGETS,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Build nested stratified index lists — `idx[48] ⊂ idx[96] ⊂ idx[192]`.

    Returns ``{budget: [indices]}``. Each returned list is a subset of the
    next-larger list, and every list preserves the class balance of the full
    training set. Raises :class:`ValueError` if the nested-subset invariant
    cannot be satisfied (e.g. a class has too few samples).

    The algorithm: for each class, sort all indices in a shuffled order and
    then, at each budget, take the first ``round(budget × class_rate)``
    indices from that shuffled list. Because the lists are shared across
    budgets, the prefixes are nested by construction.
    """
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(train_records):
        label = _gt_label(rec)
        if label is None:
            continue
        by_label[label].append(idx)

    for idxs in by_label.values():
        rng.shuffle(idxs)

    total = sum(len(v) for v in by_label.values())
    if total == 0:
        raise ValueError("no records with a parseable ground-truth label")

    class_rates = {lab: len(idxs) / total for lab, idxs in by_label.items()}
    out: dict[int, list[int]] = {}

    for budget in sorted(budgets):
        if budget > total:
            raise ValueError(f"budget {budget} exceeds training total {total}")
        take: list[int] = []
        allocated = 0
        for lab in sorted(class_rates):  # deterministic class order
            k = round(budget * class_rates[lab])
            take.extend(by_label[lab][:k])
            allocated += k
        # Fix off-by-one from rounding by topping up / trimming the largest class
        while allocated < budget:
            lab = max(class_rates, key=class_rates.get)
            take.append(by_label[lab][allocated])
            allocated += 1
        while allocated > budget:
            take.pop()
            allocated -= 1
        take.sort()
        out[budget] = take

    _verify_nested(out)
    return out


def _verify_nested(indices_by_budget: dict[int, list[int]]) -> None:
    from itertools import pairwise

    for smaller, larger in pairwise(sorted(indices_by_budget)):
        if not set(indices_by_budget[smaller]).issubset(set(indices_by_budget[larger])):
            raise AssertionError(
                f"nested-subset invariant violated: n={smaller} not a subset of n={larger}"
            )


# ---------------------------------------------------------------------------
# HuggingFace loader (lazy; only needed for live training/eval)
# ---------------------------------------------------------------------------


def load_hf_dataset(
    dataset_id: str,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset as a list of dicts.

    Lazy-imports `datasets` so this module stays import-cheap.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    return list(ds)
