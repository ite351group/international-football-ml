"""Plot measured probability diagnostics from probability_eval.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    folder = args.input.resolve()
    metrics = json.loads((folder / "probability_metrics.json").read_text(encoding="utf-8"))
    bins = pd.read_csv(folder / "draw_calibration_bins.csv")
    names = ["logistic_regression", "random_forest", "extra_trees"]
    labels = ["Logistic Regression", "Random Forest", "Extra Trees"]
    colors = ["#275d9d", "#dc9147", "#8a4ca2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("#fbfbfd")

    ax = axes[0]
    x = np.arange(len(names))
    w = .26
    vals = [
        [metrics[n]["actual_draw_rate"] * 100 for n in names],
        [metrics[n]["mean_predicted_draw_probability"] * 100 for n in names],
        [metrics[n]["predicted_draw_rate"] * 100 for n in names],
    ]
    bar_colors = ["#a6a8ad", "#4386bd", "#ce7549"]
    for offset, values, label, color in zip([-w, 0, w], vals,
                                            ["Actual draws", "Mean P(draw)", "Predicted draws"], bar_colors):
        bars = ax.bar(x + offset, values, w, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x, ["LR", "RF", "ET"])
    ax.set_ylim(0, 36)
    ax.set_ylabel("Share of matches / mean probability (%)")
    ax.set_title("Draw probability is not the same as a draw decision", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot([0, .6], [0, .6], color="#777777", ls="--", lw=1, label="Perfect calibration")
    for name, label, color in zip(names, labels, colors):
        current = bins[(bins.model == name) & (bins.n >= 30)]
        ax.plot(current.mean_predicted_draw_probability,
                current.observed_draw_rate, marker="o", ms=5, lw=1.8,
                color=color, label=label)
    ax.set_xlim(0, .55)
    ax.set_ylim(0, .55)
    ax.set_xlabel("Mean predicted P(draw) within bin")
    ax.set_ylabel("Observed draw frequency within bin")
    ax.set_title("Draw calibration by 0.1-wide probability bins", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(.02, .02, "Bins with fewer than 30 matches hidden", transform=ax.transAxes,
            fontsize=8, color="#555555")
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle("ITE351 | Published holdout, 6,609 matches (2019-11-19 onward)",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, .03, 1, .94])
    fig.savefig(folder / "probability_and_draws.png", dpi=170, bbox_inches="tight")
    fig.savefig(folder / "probability_and_draws.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6))
    fig.patch.set_facecolor("#fbfbfd")
    for ax, (key, title) in zip(axes, [
        ("log_loss", "Log loss ↓"),
        ("multiclass_brier_sum", "Multiclass Brier ↓"),
        ("rps_normalized", "Normalized RPS ↓"),
    ]):
        y = [metrics[n][key] for n in names]
        bars = ax.bar(labels, y, color=colors, width=.62)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in y], padding=3, fontsize=9)
        ax.set_title(title, weight="bold", fontsize=11)
        ax.set_ylim(0, max(y) * 1.22)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Probability metrics on the same 6,609 matches", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, .92])
    fig.savefig(folder / "probability_metrics.png", dpi=170, bbox_inches="tight")
    fig.savefig(folder / "probability_metrics.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
