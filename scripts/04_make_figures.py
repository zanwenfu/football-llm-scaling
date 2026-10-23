"""Regenerate all paper figures from raw prediction dumps.

Usage::

    python scripts/04_make_figures.py \\
        --raw-dir results/raw \\
        --figures-dir results/figures

Figures produced:

* ``scaling_curve.png`` — accuracy vs. training budget (log-x) with ICL/CoT
  bands. Automatically switches to bootstrap error bars if multi-seed dumps
  (e.g. ``scaling_predictions_n96_s42.json`` + ``..._s43.json``) are present.
* ``distribution_curve.png`` — home/away/draw prediction rates per budget.
* ``confusion_matrices.png`` — 3×3 confusion matrix per condition.
* ``consistency_curve.png`` — Figure 1 of the paper (parser-rescue gap).
* ``consistency_by_split.png`` — text/score agreement on named vs. anon.
* ``regime_stack.png`` — stacked regime-taxonomy bar chart (new; not in paper).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from data import load_predictions
from plotting import (
    plot_confusion_matrices,
    plot_consistency_by_split,
    plot_consistency_curve,
    plot_distribution_curve,
    plot_regime_stack,
    plot_scaling_curve,
)

SINGLE_NAME_MAP: dict[str, str] = {
    "icl_predictions": "ICL",
    "cot_predictions": "CoT",
    "icl_results": "ICL",
    "cot_results_v2": "CoT",
}
_SCALING_RE = re.compile(r"(?:scaling_predictions|scaling_eval)_n(\d+)(?:_s(\d+))?$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    primary_conditions: dict[str, list[dict[str, Any]]] = {}
    seeded_scaling: dict[int, list[list[dict[str, Any]]]] = {}

    for path in sorted(args.raw_dir.glob("*.json")):
        stem = path.stem
        if stem in SINGLE_NAME_MAP:
            primary_conditions.setdefault(SINGLE_NAME_MAP[stem], load_predictions(path))
            continue
        m = _SCALING_RE.match(stem)
        if not m:
            continue
        budget = int(m.group(1))
        recs = load_predictions(path)
        seeded_scaling.setdefault(budget, []).append(recs)
        primary_conditions.setdefault(f"n{budget}", recs)

    def save(fig, name: str) -> None:
        out = args.figures_dir / name
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"  {out}")

    if seeded_scaling:
        # Error bars only when at least one budget has >1 seed.
        if any(len(v) > 1 for v in seeded_scaling.values()):
            scaling_input: dict[int, Any] = dict(seeded_scaling)
        else:
            scaling_input = {k: v[0] for k, v in seeded_scaling.items()}
        save(
            plot_scaling_curve(
                scaling_input,
                primary_conditions.get("ICL"),
                primary_conditions.get("CoT"),
            ),
            "scaling_curve.png",
        )
        save(
            plot_distribution_curve({k: v[0] for k, v in seeded_scaling.items()}),
            "distribution_curve.png",
        )

    ordered = [
        name for name in ("ICL", "CoT", "n48", "n96", "n192", "n384")
        if name in primary_conditions
    ]
    cond_ordered = {k: primary_conditions[k] for k in ordered}
    if cond_ordered:
        save(plot_confusion_matrices(cond_ordered), "confusion_matrices.png")
        save(plot_consistency_curve(cond_ordered), "consistency_curve.png")
        save(plot_consistency_by_split(cond_ordered), "consistency_by_split.png")
        save(plot_regime_stack(cond_ordered), "regime_stack.png")

    print(f"\nWrote figures to {args.figures_dir}/")


if __name__ == "__main__":
    main()
