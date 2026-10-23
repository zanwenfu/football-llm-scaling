# Figures

Every figure in the paper is reproducible from `results/raw/*.json` via
`scripts/04_make_figures.py` and the functions in
[`src/plotting.py`](../src/plotting.py).

| file                                                  | what it shows                                                                             | paper reference |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------- | --------------- |
| [`consistency_curve.png`](../results/figures/consistency_curve.png)           | Score-acc vs. text-acc vs. text/score agreement across all conditions. **The parser-rescue gap.** | **Figure 1**    |
| [`scaling_curve.png`](../results/figures/scaling_curve.png)                  | QLoRA result accuracy vs. training budget, log-x. Horizontal lines for ICL, CoT, always-home. | §3.1            |
| [`distribution_curve.png`](../results/figures/distribution_curve.png)        | Home / away / draw prediction rates per QLoRA budget. Makes the n=48 class-collapse visible. | §3.1            |
| [`confusion_matrices.png`](../results/figures/confusion_matrices.png)        | 3×3 confusion matrix per condition. Column collapse = class collapse.                     | §3.1 appendix   |
| [`consistency_by_split.png`](../results/figures/consistency_by_split.png)    | Text/score agreement on named vs. anonymized eval halves.                                 | §3.2            |
| [`loss_curves.png`](../results/figures/loss_curves.png)                      | Training loss for the n = 48, 96, 192 adapters.                                           | §3.1            |
| [`regime_stack.png`](../results/figures/regime_stack.png)                    | Stacked bar: share of regime A / B / C / C' / D / U per condition. *New — not in paper.*  | extension       |

## Regenerating

```bash
python scripts/04_make_figures.py --raw-dir results/raw --figures-dir results/figures
```

## Adding a figure

All figures go through the same shape: receive a `dict[str, list[dict]]`
mapping condition name to per-sample records, compute metrics via
`compute_metrics` / `regime_counts`, and return a `matplotlib.Figure`. Add
the plotter to `src/plotting.py` and a `save(...)` line
to `scripts/04_make_figures.py`.

Match the established aesthetic: `figsize=(7–8, 4.5)`, 150 DPI, Wilson CIs
reported in the tables rather than on the figure, always-home-win drawn as a
dotted horizontal reference, and a `tight_layout()` at the end.
