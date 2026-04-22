# Notebooks

Colab-runnable notebooks preserved for narrative value — they show the order in
which the research actually happened, including the corrections
(`03b_cot_rerun` after the 300-token truncation was discovered). For
production-quality reuse, prefer the functions in
[`src/`](../src/) and the
[`scripts/`](../scripts/) that wrap them.

| notebook                          | role                                                                                     | now lives in                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `01_train_qlora.ipynb`            | Original QLoRA training recipe (inherited from base `football-llm` repo).                | `src/training.py` + `scripts/01_train_scaling.py` |
| `02_eval_harness.ipynb`           | Parser, metric functions, always-home-win & random-weighted baselines.                   | `src/parsing.py`, `metrics.py`           |
| `03_icl_cot_baselines.ipynb`      | 5-shot stratified ICL and first CoT pass (300-token budget, found truncated).            | `src/prompts.py`, `evaluation.py`        |
| `03b_cot_rerun.ipynb`             | Hardened parser + CoT rerun at 1024 tokens. Fixes the `303-167` squad-goal false positive. | same as above; parser lives in `parsing.py`                 |
| `04_scaling_ablation.ipynb`       | Trains n={48,96,192}, evaluates all four budgets, produces the scaling + distribution + confusion plots. | `scripts/01_train_scaling.py`, `scripts/02_run_evaluations.py`, `scripts/04_make_figures.py` |
| `05_consistency_analysis.ipynb`   | **The paper's novel contribution**: the regime taxonomy and parser-rescue gap.           | `src/consistency.py` + `scripts/03_analyze_results.py` |

The notebooks are kept verbatim as a research log. If you want to *extend* the
work, write a new script against the package API rather than forking a notebook.
