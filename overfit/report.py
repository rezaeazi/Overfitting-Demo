"""Summary table and plot of walk-forward results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS = ["train_acc", "test_acc", "baseline_acc", "train_loss", "test_loss"]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics by variant, with cross-seed sample standard deviations."""
    if "seed" in results and results["seed"].nunique() > 1:
        by_seed = results.groupby(["seed", "variant"], sort=False)[METRICS].mean()
        by_seed["gap"] = by_seed["train_acc"] - by_seed["test_acc"]
        means = by_seed.groupby("variant", sort=False).mean()
        stds = by_seed.groupby("variant", sort=False).std(ddof=1)
        summary = means.join(stds.add_suffix("_std"))
        return summary

    summary = results.groupby("variant", sort=False)[METRICS].mean()
    summary["gap"] = summary["train_acc"] - summary["test_acc"]
    return summary


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_options = {}
    if {"train_acc_std", "test_acc_std"}.issubset(summary.columns):
        plot_options["yerr"] = summary[["train_acc_std", "test_acc_std"]].set_axis(
            ["train_acc", "test_acc"], axis="columns"
        )
        plot_options["capsize"] = 3
    ax = summary[["train_acc", "test_acc"]].plot.bar(
        figsize=(8, 4.5), rot=0, **plot_options
    )
    ax.axhline(summary["baseline_acc"].mean(), color="gray", linestyle="--", label="majority baseline")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Train vs out-of-sample accuracy (walk-forward mean)")
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=150)
    plt.close(ax.figure)
