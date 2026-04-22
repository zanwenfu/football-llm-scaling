#!/usr/bin/env bash
# Reproduce the paper end-to-end. Assumes raw prediction dumps already exist
# in results/raw/; if they don't, runs the evaluation scripts first (requires
# GPU and a HuggingFace login with access to meta-llama/Llama-3.1-8B-Instruct).
#
# Usage:
#   bash scripts/reproduce_paper.sh              # tables + figures from existing raw dumps
#   bash scripts/reproduce_paper.sh --train      # additionally train n={48,96,192} from scratch
#   bash scripts/reproduce_paper.sh --eval       # additionally regenerate all raw dumps via GPU

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN=${TRAIN:-0}
EVAL=${EVAL:-0}
for arg in "$@"; do
  case $arg in
    --train) TRAIN=1 ;;
    --eval)  EVAL=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$TRAIN" == "1" ]]; then
  echo "==> training QLoRA adapters at n=48,96,192"
  python scripts/01_train_scaling.py --budgets 48 96 192 --out adapters/
fi

if [[ "$EVAL" == "1" ]]; then
  mkdir -p results/raw
  echo "==> evaluating ICL"
  python scripts/02_run_evaluations.py --condition icl \
      --out results/raw/icl_predictions.json
  echo "==> evaluating CoT"
  python scripts/02_run_evaluations.py --condition cot \
      --out results/raw/cot_predictions.json
  for n in 48 96 192; do
    echo "==> evaluating QLoRA n=$n"
    python scripts/02_run_evaluations.py --condition qlora \
        --adapter adapters/n${n} \
        --out results/raw/scaling_predictions_n${n}.json
  done
  echo "==> evaluating QLoRA n=384 (reuses public adapter)"
  python scripts/02_run_evaluations.py --condition qlora \
      --adapter zanwenfu/football-llm-qlora \
      --out results/raw/scaling_predictions_n384.json
fi

echo "==> analysis + tables"
python scripts/03_analyze_results.py --raw-dir results/raw --tables-dir results/tables

echo "==> figures"
python scripts/04_make_figures.py --raw-dir results/raw --figures-dir results/figures

echo "==> example contradictions"
python scripts/05_dump_contradictions.py --raw-dir results/raw \
    --out results/examples/example_contradictions.md

echo
echo "Done. See results/tables/, results/figures/, results/examples/."
