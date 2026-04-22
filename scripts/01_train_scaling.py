"""Train QLoRA adapters at n ∈ {48, 96, 192} using nested stratified prefixes.

Single-seed (matches the paper)::

    python scripts/01_train_scaling.py --budgets 48 96 192 --out adapters/

Multi-seed sweep (answers the paper's §5 Limitations question: is the n=96
coherence collapse robust, or seed-level variance?)::

    python scripts/01_train_scaling.py \\
        --budgets 48 96 192 --seeds 42 43 44 --out adapters/

Adapters are written to ``{out}/n{budget}_s{seed}/`` (single-seed runs drop
the ``_s{seed}`` suffix and land in ``{out}/n{budget}/`` for
backward-compatibility with the paper's adapter paths). Each run skips
training if ``adapter_model.safetensors`` already exists at the target path,
so the sweep is restart-safe.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import DEFAULT_TRAIN, SCALING_BUDGETS, TRAIN_DATASET_HUB_ID
from data import (
    load_hf_dataset,
    stratified_nested_prefix_indices,
)
from training import train_qlora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[b for b in SCALING_BUDGETS if b <= 192],
        help="Training budgets to sweep (must all be <= named training size).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[DEFAULT_TRAIN.seed],
        help="Seeds to train. Each adapter is written to a seed-labelled path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("adapters"),
        help="Directory to write adapter checkpoints into.",
    )
    parser.add_argument(
        "--dataset",
        default=TRAIN_DATASET_HUB_ID,
        help="HuggingFace dataset id to train from.",
    )
    parser.add_argument(
        "--prefix-seed",
        type=int,
        default=DEFAULT_TRAIN.seed,
        help=(
            "Seed for the stratified nested-prefix sampler. Kept separate from "
            "training seeds so the same data subset is used across seeds at "
            "each budget, isolating optimization stochasticity."
        ),
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    all_train = load_hf_dataset(args.dataset, split="train")
    named = [
        r for r in all_train
        if not (r.get("is_anon") or (r.get("metadata") or {}).get("is_anon"))
    ]
    print(f"  total train rows: {len(all_train)};  named-only: {len(named)}")

    idx_by_budget = stratified_nested_prefix_indices(
        named, tuple(sorted(set(args.budgets))), seed=args.prefix_seed
    )
    single_seed = len(args.seeds) == 1

    for budget in sorted(args.budgets):
        subset = [named[i] for i in idx_by_budget[budget]]
        for seed in args.seeds:
            suffix = "" if single_seed else f"_s{seed}"
            out_dir = args.out / f"n{budget}{suffix}"
            print(
                f"\n=== training n={budget} seed={seed} ({len(subset)} examples) "
                f"-> {out_dir} ==="
            )
            train_cfg = replace(DEFAULT_TRAIN, seed=seed)
            train_qlora(subset, output_dir=out_dir, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
