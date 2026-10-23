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

    The algorithm is the one that drew the adapters in the saved predictions
    (``notebooks/04_scaling_ablation.ipynb``): each class's indices are put in
    one seeded random order, each budget takes ``round(budget × class_rate)``
    from the front of every class, and any shortfall from rounding is topped
    up one index at a time from the largest class down. Because every budget
    reads the same per-class order, the prefixes are nested.
    """
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(train_records):
        label = _gt_label(rec)
        if label is None:
            continue
        by_label[label].append(idx)

    total = sum(len(v) for v in by_label.values())
    if total == 0:
        raise ValueError("no records with a parseable ground-truth label")

    # Classes in order of first appearance, as the notebook iterated them.
    rng = random.Random(seed)
    order = {lab: rng.sample(idxs, len(idxs)) for lab, idxs in by_label.items()}
    by_size = sorted(order, key=lambda lab: -len(order[lab]))
    out: dict[int, list[int]] = {}

    for budget in sorted(budgets):
        if budget > total:
            raise ValueError(f"budget {budget} exceeds training total {total}")
        picks: list[int] = []
        for idxs in order.values():
            picks.extend(idxs[: round(budget * len(idxs) / total)])
        picks = sorted(set(picks))
        chosen = set(picks)
        while len(picks) < budget:
            for lab in by_size:
                nxt = next((i for i in order[lab] if i not in chosen), None)
                if nxt is not None:
                    picks.append(nxt)
                    chosen.add(nxt)
                if len(picks) >= budget:
                    break
        out[budget] = sorted(picks[:budget])

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
# Training / eval records (only needed for live training/eval)
# ---------------------------------------------------------------------------


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Expose the fields the training and eval code reads.

    The files in ``data/`` keep the anonymized flag as ``metadata.anonymized``
    and the final score as a ``"home-away"`` string. Training and evaluation
    read ``is_anon`` and ``metadata.home_goals`` / ``metadata.away_goals``.
    """
    meta = dict(record.get("metadata") or {})
    if "is_anon" not in record:
        record["is_anon"] = bool(meta.get("anonymized", meta.get("is_anon", False)))
    score = meta.get("score")
    if isinstance(score, str) and "home_goals" not in meta:
        home, _, away = score.partition("-")
        if home.strip().isdigit() and away.strip().isdigit():
            meta["home_goals"], meta["away_goals"] = int(home), int(away)
    record["metadata"] = meta
    return record


def load_records(
    source: str | Path,
    split: str = "train",
) -> list[dict[str, Any]]:
    """Load training or eval records as a list of dicts.

    ``source`` is a local ``.jsonl`` file (the default, see ``config.TRAIN_DATA``
    and ``config.EVAL_DATA``) or a HuggingFace dataset id. Hub loading
    lazy-imports `datasets` so this module stays import-cheap.
    """
    path = Path(source)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        from datasets import load_dataset

        rows = list(load_dataset(str(source), split=split))
    return [_normalize_record(row) for row in rows]
