"""Paper figures, reproduced from per-sample records in ``results/raw/``.

Each function takes a dict ``condition_name -> records`` and returns a
matplotlib Figure. The plotting code is deliberately kept thin — we compute
metrics via :mod:`metrics` and :mod:`consistency` so the numbers on every
figure match the tables.

Figures reproduced (see :file:`results/figures/` for the paper versions):

* ``plot_scaling_curve`` → ``scaling_curve.png``
* ``plot_distribution_curve`` → ``distribution_curve.png``
* ``plot_confusion_matrices`` → ``confusion_matrices.png``
* ``plot_consistency_curve`` → ``consistency_curve.png`` (Figure 1 of the paper)
* ``plot_consistency_by_split`` → ``consistency_by_split.png``
"""

from __future__ import annotations

from typing import Any, Mapping

from config import BASELINE_HOME_WIN_RATE
from consistency import regime_counts
from metrics import compute_metrics
from parsing import Label, parse_output


def _is_list_of_lists(v: Any) -> bool:
    """True if `v` is a list of record-lists (multi-seed) rather than one list."""
    return isinstance(v, list) and len(v) > 0 and isinstance(v[0], list)

# ---------------------------------------------------------------------------
# Scaling curve
# ---------------------------------------------------------------------------


def plot_scaling_curve(
    scaling_records: Mapping[int, Any],
    icl_records: list[dict[str, Any]] | None = None,
    cot_records: list[dict[str, Any]] | None = None,
):
    """Accuracy vs. training budget (log-x), with ICL/CoT reference lines.

    ``scaling_records[n]`` may be either a single per-sample list (one seed)
    or a list-of-lists (one per seed). In the multi-seed case we plot the
    mean across seeds with a bootstrap 95% error bar.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    budgets = sorted(scaling_records)
    means: list[float] = []
    low_err: list[float] = []
    high_err: list[float] = []
    multi_seed = False
    for n in budgets:
        v = scaling_records[n]
        if _is_list_of_lists(v):
            multi_seed = True
            from aggregate import aggregate_across_seeds  # noqa: PLC0415
            agg = aggregate_across_seeds(v).score_acc
            means.append(agg.mean)
            low_err.append(agg.mean - agg.ci95_low)
            high_err.append(agg.ci95_high - agg.mean)
        else:
            means.append(compute_metrics(v).score_acc)
            low_err.append(0.0)
            high_err.append(0.0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if multi_seed:
        ax.errorbar(
            budgets, means,
            yerr=[low_err, high_err],
            marker="o", linewidth=2, capsize=4,
            label="QLoRA score_acc (mean ± bootstrap 95% CI)",
        )
    else:
        ax.plot(budgets, means, marker="o", linewidth=2, label="QLoRA score_acc")
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(n) for n in budgets])
    ax.set_xlabel("training budget n")
    ax.set_ylabel("result accuracy")
    ax.axhline(
        BASELINE_HOME_WIN_RATE,
        color="gray",
        linestyle=":",
        label=f"always-home-win ({BASELINE_HOME_WIN_RATE:.1%})",
    )
    if icl_records is not None:
        ax.axhline(
            compute_metrics(icl_records).score_acc,
            color="tab:green",
            linestyle="--",
            label="ICL (5-shot)",
        )
    if cot_records is not None:
        ax.axhline(
            compute_metrics(cot_records).score_acc,
            color="tab:orange",
            linestyle="--",
            label="CoT (5-step)",
        )
    ax.set_ylim(0.30, 0.75)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("QLoRA scaling on 2022 World Cup eval (n=128)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Prediction-distribution curve
# ---------------------------------------------------------------------------


def plot_distribution_curve(
    scaling_records: Mapping[int, list[dict[str, Any]]],
):
    """Stacked-bar view of home / away / draw prediction rates across budgets."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    budgets = sorted(scaling_records)
    home = np.array([compute_metrics(scaling_records[n]).home_pred_rate for n in budgets])
    away = np.array([compute_metrics(scaling_records[n]).away_pred_rate for n in budgets])
    draw = np.array([compute_metrics(scaling_records[n]).draw_pred_rate for n in budgets])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(budgets))
    ax.bar(x, home, label="home_win", color="tab:blue")
    ax.bar(x, away, bottom=home, label="away_win", color="tab:orange")
    ax.bar(x, draw, bottom=home + away, label="draw", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in budgets])
    ax.axhline(BASELINE_HOME_WIN_RATE, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("prediction rate")
    ax.set_title("Predicted-class distribution by training budget")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confusion matrices (one per condition)
# ---------------------------------------------------------------------------


def plot_confusion_matrices(conditions: Mapping[str, list[dict[str, Any]]]):
    """3×3 confusion matrix per condition. Column collapse = class collapse."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    labels = [Label.HOME_WIN, Label.AWAY_WIN, Label.DRAW]
    label_names = ["home_win", "away_win", "draw"]
    names = list(conditions)
    n = len(names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes)

    for i, name in enumerate(names):
        cm = np.zeros((3, 3), dtype=int)
        for rec in conditions[name]:
            gt = Label.from_str((rec.get("gt") or {}).get("result", ""))
            p = parse_output(rec.get("raw_output", ""))
            pred = p.score_label or p.text_label
            if gt is None or pred is None:
                continue
            cm[labels.index(gt), labels.index(pred)] += 1
        ax = axes[i // cols, i % cols]
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(label_names, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title(name, fontsize=10)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Consistency curve (Figure 1 of the paper)
# ---------------------------------------------------------------------------


def plot_consistency_curve(conditions: Mapping[str, list[dict[str, Any]]]):
    """Score-acc vs. text-acc, with text/score agreement on a secondary axis."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    names = list(conditions)
    score_acc = [compute_metrics(conditions[n]).score_acc for n in names]
    text_acc = [compute_metrics(conditions[n]).text_acc for n in names]
    agreement = [compute_metrics(conditions[n]).text_score_agreement for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = list(range(len(names)))
    ax.plot(x, score_acc, marker="o", label="score_acc (parser-rescued)", color="tab:blue")
    ax.plot(x, text_acc, marker="s", linestyle="--",
            label="text_acc (what Prediction: says)", color="tab:orange")
    ax.axhline(BASELINE_HOME_WIN_RATE, color="gray", linestyle=":",
               label=f"always-home ({BASELINE_HOME_WIN_RATE:.1%})")
    ax.axhline(1 / 3, color="lightgray", linestyle=":", label="random (33.3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, fontsize=9)
    ax.set_ylim(0.20, 0.75)
    ax.set_ylabel("accuracy")
    ax.legend(loc="upper left", fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(x, agreement, marker="^", linestyle=":", color="tab:green",
             label="text/score agreement")
    ax2.set_ylim(0.40, 1.0)
    ax2.set_ylabel("text/score agreement", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.legend(loc="lower right", fontsize=9)

    ax.set_title("The parser-rescue gap: score_acc masks internal inconsistency")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Named vs. anonymized agreement
# ---------------------------------------------------------------------------


def plot_consistency_by_split(conditions: Mapping[str, list[dict[str, Any]]]):
    """Text/score agreement split by named vs. anonymized eval halves."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    names = list(conditions)
    named_agree, anon_agree = [], []
    for cname in names:
        named = [r for r in conditions[cname] if not r.get("is_anon")]
        anon = [r for r in conditions[cname] if r.get("is_anon")]
        named_agree.append(compute_metrics(named).text_score_agreement)
        anon_agree.append(compute_metrics(anon).text_score_agreement)

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, named_agree, width, label="named (in-dist.)", color="tab:blue")
    ax.bar(x + width / 2, anon_agree, width, label="anonymized", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("text/score agreement")
    ax.set_title("Field-binding fidelity by split")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regime taxonomy stacked bar
# ---------------------------------------------------------------------------


def plot_regime_stack(conditions: Mapping[str, list[dict[str, Any]]]):
    """Stacked-bar view of regime counts per condition (A / B / C / C_inv / D / U)."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    names = list(conditions)
    labels = ["A: coherent ✓", "B: self-consistent ✗", "C: parser-rescued",
              "C': parser-penalized", "D: fragmented", "U: unparseable"]
    colors = ["tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:red", "lightgray"]
    rows = np.zeros((6, len(names)), dtype=float)
    for i, cname in enumerate(names):
        rc = regime_counts(conditions[cname])
        n = rc.total or 1
        rows[:, i] = [rc.A / n, rc.B / n, rc.C / n, rc.C_inv / n, rc.D / n, rc.U / n]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for i in range(6):
        ax.bar(x, rows[i], bottom=bottom, label=labels[i], color=colors[i])
        bottom += rows[i]
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("share of eval samples")
    ax.set_ylim(0, 1)
    ax.set_title("Regime taxonomy per condition")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    return fig
