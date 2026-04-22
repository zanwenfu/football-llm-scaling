"""Write a markdown file of representative parser-rescued and fragmented outputs.

Usage::

    python scripts/05_dump_contradictions.py \\
        --raw-dir results/raw \\
        --out results/examples/example_contradictions.md

Used to populate the qualitative appendix discussed in the paper ("On the 2022
Argentina vs. Saudi Arabia match..."). Selects up to ``--per-regime`` examples
from each of the interesting regimes (C, D) per condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from consistency import Regime, find_examples_by_regime
from data import load_predictions

NAME_MAP: dict[str, str] = {
    "scaling_predictions_n96": "n96",
    "scaling_predictions_n192": "n192",
    "scaling_predictions_n384": "n384",
    "scaling_eval_n96": "n96",
    "scaling_eval_n192": "n192",
    "scaling_eval_n384": "n384",
    "icl_results": "ICL",
    "cot_results_v2": "CoT",
    "icl_predictions": "ICL",
    "cot_predictions": "CoT",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/raw"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/examples/example_contradictions.md")
    )
    parser.add_argument("--per-regime", type=int, default=3)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sections: list[str] = [
        "# Example contradictions",
        "",
        "Representative model outputs from the three interesting regimes, sampled",
        "from the raw prediction dumps by `scripts/05_dump_contradictions.py`.",
        "",
    ]
    for stem, cond in NAME_MAP.items():
        path = args.raw_dir / f"{stem}.json"
        if not path.exists():
            continue
        recs = load_predictions(path)
        sections.append(f"## {cond}   ({path.name})")
        sections.append("")
        for regime in (Regime.PARSER_RESCUED, Regime.FRAGMENTED, Regime.SELF_CONSISTENT_MISTAKE):
            examples = find_examples_by_regime(recs, regime, limit=args.per_regime)
            if not examples:
                continue
            sections.append(f"### Regime {regime.value}: {regime.description()}")
            sections.append("")
            for ex in examples:
                sections.append(_format_example(ex))
                sections.append("")
        sections.append("")

    args.out.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {args.out}")


def _format_example(rec: dict[str, Any]) -> str:
    gt = rec.get("gt") or {}
    out = rec.get("raw_output", "").replace("\n", " ⏎ ")
    return (
        f"- **sample #{rec.get('sample_idx')}** "
        f"(anon={rec.get('is_anon')}, "
        f"GT={gt.get('result')} {gt.get('home_goals')}-{gt.get('away_goals')})\n"
        f"  ```\n  {out}\n  ```"
    )


if __name__ == "__main__":
    main()
