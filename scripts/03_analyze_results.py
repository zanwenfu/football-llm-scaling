"""Recompute metrics, consistency tables, and regime counts from raw predictions.

Usage::

    python scripts/03_analyze_results.py \\
        --raw-dir results/raw \\
        --tables-dir results/tables

Reads every ``*.json`` in ``raw-dir`` and emits:

* ``scaling_metrics.json`` — Table 1 of the paper (overall/named/anon per condition).
* ``consistency_table.json`` — text/score/coherence accuracies per condition × split.
* ``regime_counts.json`` — A/B/C/C'/D/U per condition.
* ``disagreement_directionality.json`` — Table 2 of the paper.
* ``aggregated_metrics.json`` — mean ± bootstrap 95% CI across seeds
  (only written when multi-seed dumps are present, e.g. ``..._s42.json``,
  ``..._s43.json``).
* ``final_tables.json`` — combined view that the paper's text references.
* ``consistency_table.csv`` — the same numbers as a flat CSV for spreadsheet work.

Multi-seed detection: any filename matching ``scaling_predictions_n<N>_s<S>.json``
(or the legacy ``scaling_eval_n<N>_s<S>.json``) is grouped by N, yielding a
list of record lists per budget. Single-seed dumps continue to work unchanged.

The script is pure-python and CPU-only: no model loading.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from aggregate import aggregate_across_seeds
from consistency import (
    disagreement_directionality,
    regime_counts,
)
from data import load_predictions
from metrics import compute_all_splits, compute_metrics, metrics_to_legacy_schema

# Filename stems for the non-scaling conditions (seed-insensitive).
SINGLE_NAME_MAP: dict[str, str] = {
    "icl_predictions": "ICL",
    "cot_predictions": "CoT",
    "icl_results": "ICL",
    "cot_results_v2": "CoT",
}

# Scaling filenames may optionally carry a `_s<seed>` suffix.
_SCALING_RE = re.compile(r"(?:scaling_predictions|scaling_eval)_n(\d+)(?:_s(\d+))?$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    # condition_name -> list of seed record-lists  (len == 1 if single-seed)
    seeded: dict[str, list[list[dict[str, Any]]]] = {}
    seed_ids: dict[str, list[str]] = {}

    for path in sorted(args.raw_dir.glob("*.json")):
        stem = path.stem
        if stem in SINGLE_NAME_MAP:
            name = SINGLE_NAME_MAP[stem]
            seeded.setdefault(name, []).append(load_predictions(path))
            seed_ids.setdefault(name, []).append("—")
            continue
        m = _SCALING_RE.match(stem)
        if not m:
            continue
        budget = int(m.group(1))
        seed = m.group(2)
        name = f"n{budget}"
        seeded.setdefault(name, []).append(load_predictions(path))
        seed_ids.setdefault(name, []).append(seed or "—")

    if not seeded:
        raise SystemExit(f"no recognized prediction dumps found in {args.raw_dir}")

    for name, lst in seeded.items():
        seed_label = ", ".join(seed_ids[name])
        print(f"loaded {name}: {len(lst)} seed(s) [{seed_label}]")

    # "Primary" per-condition records: seed 0 (or the only one) for the
    # tables that correspond 1:1 with the paper. Multi-seed aggregation goes
    # into its own aggregated_metrics.json.
    primary: dict[str, list[dict[str, Any]]] = {k: v[0] for k, v in seeded.items()}

    # --- Scaling / primary metrics -------------------------------------------
    scaling_metrics: dict[str, dict[str, Any]] = {}
    consistency_table: dict[str, dict[str, Any]] = {}
    for name, recs in primary.items():
        splits = compute_all_splits(recs)
        scaling_metrics[name] = {k: metrics_to_legacy_schema(v) for k, v in splits.items()}
        consistency_table[name] = {
            k: {
                "n": v.n,
                "score_acc": v.score_acc,
                "text_acc": v.text_acc,
                "coherence_acc": v.coherence_acc,
                "score_acc_ci95_pp": v.score_acc_ci95_pp,
                "text_acc_ci95_pp": v.text_acc_ci95_pp,
                "coherence_acc_ci95_pp": v.coherence_acc_ci95_pp,
                "text_score_agreement": v.text_score_agreement,
                "home_pred_rate": v.home_pred_rate,
                "away_pred_rate": v.away_pred_rate,
                "draw_pred_rate": v.draw_pred_rate,
            }
            for k, v in splits.items()
        }

    # --- Regimes + Table 2 ---------------------------------------------------
    regime_overall: dict[str, dict[str, Any]] = {}
    disagreement: dict[str, dict[str, Any]] = {}
    for name, recs in primary.items():
        regime_overall[name] = regime_counts(recs).to_dict()
        named = [r for r in recs if not r.get("is_anon")]
        disagreement[name] = disagreement_directionality(named).to_dict()

    # --- Multi-seed aggregation (only if any condition has >1 seed) ---------
    aggregated: dict[str, dict[str, Any]] = {}
    any_multi = any(len(v) > 1 for v in seeded.values())
    if any_multi:
        for name, seeds_recs in seeded.items():
            if len(seeds_recs) < 2:
                continue
            agg = aggregate_across_seeds(seeds_recs)
            aggregated[name] = agg.to_dict() | {
                "seeds": seed_ids[name],
                "per_seed_score_acc": [compute_metrics(r).score_acc for r in seeds_recs],
                "per_seed_coherence_acc": [compute_metrics(r).coherence_acc for r in seeds_recs],
            }

    # --- Write --------------------------------------------------------------
    _write_json(args.tables_dir / "scaling_metrics.json", scaling_metrics)
    _write_json(args.tables_dir / "consistency_table.json", consistency_table)
    _write_json(args.tables_dir / "regime_counts.json", regime_overall)
    _write_json(args.tables_dir / "disagreement_directionality.json", disagreement)
    combined: dict[str, Any] = {
        "scaling_metrics": scaling_metrics,
        "consistency_table": consistency_table,
        "regime_counts": regime_overall,
        "disagreement_directionality": disagreement,
    }
    if aggregated:
        _write_json(args.tables_dir / "aggregated_metrics.json", aggregated)
        combined["aggregated_metrics"] = aggregated
    _write_json(args.tables_dir / "final_tables.json", combined)
    _write_csv(args.tables_dir / "consistency_table.csv", consistency_table)

    print(f"\nWrote tables to {args.tables_dir}/")


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  {path.name}")


def _write_csv(path: Path, consistency_table: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for cond, splits in consistency_table.items():
        for split_name, stats in splits.items():
            rows.append({"condition": cond, "split": split_name, **stats})
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {path.name}")


if __name__ == "__main__":
    main()
