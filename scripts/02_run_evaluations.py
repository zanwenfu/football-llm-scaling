"""Run ICL, CoT, and QLoRA evaluations and write per-sample predictions.

Usage::

    python scripts/02_run_evaluations.py --condition icl
    python scripts/02_run_evaluations.py --condition cot
    python scripts/02_run_evaluations.py --condition qlora --adapter adapters/n192
    python scripts/02_run_evaluations.py --condition qlora --adapter zanwenfu/football-llm-qlora --out results/raw/scaling_predictions_n384.json

Output JSON files have the same schema as the files in ``results/raw/``, so
``scripts/03_analyze_results.py`` can consume them directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    EVAL_DATASET_HUB_ID,
    TRAIN_DATASET_HUB_ID,
)
from data import load_hf_dataset
from evaluation import evaluate_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition", required=True, choices=["qlora", "icl", "cot"]
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="HuggingFace id or local path of the QLoRA adapter "
             "(required when --condition qlora).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-dataset", default=EVAL_DATASET_HUB_ID)
    parser.add_argument("--train-dataset", default=TRAIN_DATASET_HUB_ID)
    args = parser.parse_args()

    eval_samples = load_hf_dataset(args.eval_dataset, split="eval")
    train_samples = None
    if args.condition == "icl":
        train_samples = load_hf_dataset(args.train_dataset, split="train")

    evaluate_condition(
        condition=args.condition,
        eval_samples=eval_samples,
        train_samples=train_samples,
        adapter_id_or_path=args.adapter,
        save_to=args.out,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
