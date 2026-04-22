# Results

Artifacts produced by the experiments. Everything here is either consumed by
or produced by the scripts in [`scripts/`](../scripts/).

```
results/
├── raw/            per-sample prediction dumps (inputs to all analysis)
├── tables/         aggregated metrics, consistency tables, regime counts
├── figures/        every figure in the paper, plus one extra
└── examples/       curated qualitative examples
```

## `raw/` — per-sample prediction dumps

Each JSON file is a 128-record list in the common schema:

```jsonc
{
  "sample_idx": 0,
  "is_anon": false,
  "gt":   { "result": "away_win", "home_goals": 0, "away_goals": 2, "parsed": true },
  "pred": { "result": "away_win", "home_goals": 0, "away_goals": 2, "parsed": true },
  "raw_output": "Prediction: home_win\nScore: 0-2\nReasoning: Qatar's squad...",
  "output_len": 149            // CoT only
}
```

| file                              | condition           | generator                                   |
| --------------------------------- | ------------------- | ------------------------------------------- |
| `icl_predictions.json`            | ICL (5-shot)        | `notebooks/03_icl_cot_baselines.ipynb`      |
| `cot_predictions.json`            | CoT (5-step, 1024 tok) | `notebooks/03b_cot_rerun.ipynb`          |
| `scaling_predictions_n{48,96,192}.json` | QLoRA n∈{48,96,192} | `notebooks/04_scaling_ablation.ipynb`   |
| `scaling_predictions_n384.json`   | QLoRA n=384 (reuses `zanwenfu/football-llm-qlora`) | `notebooks/04_scaling_ablation.ipynb` |

The `pred.result` field follows the legacy "score overrides text" convention
used by the original `eval_harness`. The parser-rescue analysis in the paper
recomputes per-channel labels from `raw_output` directly — don't assume
`pred.result` already reflects the coherent text label.

## `tables/` — aggregated metrics

| file                              | contents                                                        |
| --------------------------------- | --------------------------------------------------------------- |
| `scaling_metrics.json`            | Table 1 (overall / named / anon per condition)                  |
| `consistency_table.json`          | score_acc, text_acc, coherence_acc, text/score agreement, Wilson CIs |
| `consistency_table.csv`           | same, flat CSV                                                  |
| `regime_counts.json`              | A / B / C / C' / D / U counts per condition                     |
| `disagreement_directionality.json`| Table 2 of the paper (named split only)                         |
| `final_tables.json`               | all of the above in one file, matching the paper's text         |
| `baseline_metrics_cot.json`       | base (no adapter) CoT metrics for reference                     |
| `eval_results_base_repo.json`     | evaluation numbers inherited from the base football-llm repo    |

All numbers here are regenerable from `raw/` via:

```bash
python scripts/03_analyze_results.py --raw-dir results/raw --tables-dir results/tables
```

## `figures/` — paper figures

Reproducible from `raw/` via `scripts/04_make_figures.py`. Saved at 150 DPI.

| figure                      | shown in paper as |
| --------------------------- | ----------------- |
| `scaling_curve.png`         | §3.1 scaling      |
| `distribution_curve.png`    | §3.1 class collapse diagnostic |
| `confusion_matrices.png`    | §3.1 appendix     |
| `consistency_curve.png`     | **Figure 1** (the parser-rescue gap) |
| `consistency_by_split.png`  | §3.2 named vs. anon |
| `loss_curves.png`           | training loss for the n={48,96,192} adapters |
| `regime_stack.png`          | *new* — not in the paper; added to make regime composition visible per condition |

## `examples/`

`example_contradictions.md` contains a hand-curated set of representative
outputs. Regenerable via `scripts/05_dump_contradictions.py`.
