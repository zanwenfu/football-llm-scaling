# Reproducibility

## Environment used to produce the paper

| component    | version |
| ------------ | ------- |
| Python       | 3.10    |
| torch        | 2.1+    |
| transformers | 4.44+   |
| peft         | 0.11+   |
| bitsandbytes | 0.43+   |
| trl          | 0.9+    |
| datasets     | 2.18+   |
| GPU          | 1× NVIDIA T4 (Colab) |

Canonical ranges live in [`pyproject.toml`](../pyproject.toml).
The adapter, dataset, and per-match statistics pipeline come from the base
repo ([zanwenfu/football-llm](https://github.com/zanwenfu/football-llm)).

## Seeds

All seeds default to 42 and live in `config.py`:

| where                              | knob                                                |
| ---------------------------------- | --------------------------------------------------- |
| SFTTrainer                         | `TrainingConfig.seed = 42`                          |
| `transformers.set_seed`            | propagated from `TrainingConfig.seed`               |
| `torch.manual_seed`                | set at the top of `generation.generate`             |
| Stratified nested-prefix sampling  | `stratified_nested_prefix_indices(..., seed=42)`    |
| ICL demo selection                 | `ICLConfig.seed = 42`                               |
| Random-weighted baseline           | `metrics.random_weighted_accuracy(..., seed=42)`    |

Decoding uses `do_sample=True` at `temperature=0.1` — runs are effectively
deterministic but not bit-for-bit reproducible across CUDA driver versions.

## Three levels of reproduction

### Level 1 — tables and figures from saved predictions (no GPU)

```bash
pip install -e .
make all    # ≈1 second; produces tables/ and figures/ in the paper
```

This is enough to confirm every number and every figure in the paper.

### Level 2 — re-run evaluations on existing adapters

```bash
pip install -e '.[train]'
huggingface-cli login   # needs meta-llama/Llama-3.1-8B-Instruct access
python scripts/02_run_evaluations.py --condition icl --out results/raw/icl_predictions.json
python scripts/02_run_evaluations.py --condition cot --out results/raw/cot_predictions.json
python scripts/02_run_evaluations.py --condition qlora \
    --adapter zanwenfu/football-llm-qlora \
    --out results/raw/scaling_predictions_n384.json
make all
```

### Level 3 — retrain from scratch

```bash
python scripts/01_train_scaling.py --budgets 48 96 192 --out adapters/
# then run the evaluations in Level 2 for each trained adapter
```

Roughly 20 minutes of T4 time for n=48, 35 for n=96, 70 for n=192. Each
budget writes to `adapters/n{n}/`; the script is idempotent and skips
training if `adapter_model.safetensors` already exists there.

## Verifying the scaling invariants

The scaling ablation depends on two properties that are easy to silently
break. Both are tested in `tests/test_data.py`:

1. **Nested subset**: `s_48 ⊂ s_96 ⊂ s_192`. Verified by the assertion
   inside `stratified_nested_prefix_indices` itself, and again by
   `tests/test_data.py::TestStratifiedNestedPrefix::test_nested_subset_invariant`.
2. **Class stratification**: at every budget the training-set class
   distribution matches the overall training distribution within ±2 samples.

If you change the prefix sampler, run `pytest tests/test_data.py` before
trusting any scaling trend.

## Known reproducibility caveats

* **Single seed per budget.** The paper's Limitations section flags this;
  multi-seed runs would be the first follow-up.
* **Bitsandbytes 4-bit quantization is not bit-exact across GPU models.**
  Expect ≲1% variance on score_acc if you switch from T4 to A100.
* **CoT token budget.** The first CoT pass used `max_new_tokens=300` and
  truncated before the concluding `Prediction:` line on ~40% of samples.
  The final reported numbers use 1024 (see `config.CoTGenerationConfig`).
  Don't regress this.
