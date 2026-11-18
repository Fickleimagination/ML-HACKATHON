"""Task-3 step 5: diagnostic plots for the Optuna study and the two CSVs
produced by eval_student_test.py / sampling_rate.py.

    python -m pathsnap.experiments.plots

Writes to results/figures/:
    optuna_history.png        objective value per trial
    optuna_importance.png     sigma/beta/uturn_penalty importance
    top1_per_trip.png         nearest vs tuned HMM Top-1, one bar pair per trip
    sampling_rate_curve.png   mean Top-1 vs sampling step, nearest vs HMM
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import optuna.visualization.matplotlib as opt_mpl
import pandas as pd

from pathsnap.hmm.tune_decoder import DB_PATH, STUDY_NAME

FIG_DIR = Path("results/figures")
DECODER_CSV = Path("results/decoder_student_test.csv")
SAMPLING_CSV = Path("results/sampling_rate.csv")


def plot_optuna(study):
    ax = opt_mpl.plot_optimization_history(study)
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "optuna_history.png", dpi=150)
    plt.close(ax.figure)

    ax = opt_mpl.plot_param_importances(study)
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "optuna_importance.png", dpi=150)
    plt.close(ax.figure)


def plot_top1_per_trip():
    df = pd.read_csv(DECODER_CSV)
    df = df[df["trip"] != "MEAN"]
    pivot = df.pivot(index="trip", columns="config", values="top1").sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(pivot))
    width = 0.27
    for i, config in enumerate(["nearest", "default", "tuned"]):
        if config in pivot.columns:
            ax.bar([xi + (i - 1) * width for xi in x], pivot[config], width, label=config)
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Nearest vs default vs tuned HMM -- student_test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top1_per_trip.png", dpi=150)
    plt.close(fig)


def plot_sampling_rate():
    """Solid lines: Top-1 scored on the common subset (multiples of 30),
    the fair, rate-to-rate comparable number. Dashed/faint: the old
    own-subset numbers, kept for reference only -- they score each rate on
    a different set of points, so they are NOT comparable across rates."""
    df = pd.read_csv(SAMPLING_CSV)
    means = df[df["trip"] == "MEAN"].sort_values("step")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(means["step"], means["nearest_top1_common"], "o-", color="tab:blue",
            label="nearest (common subset, fair)")
    ax.plot(means["step"], means["hmm_top1_common"], "o-", color="tab:orange",
            label="tuned HMM (common subset, fair)")
    ax.plot(means["step"], means["nearest_top1_own_subset"], "o--", color="tab:blue",
            alpha=0.4, label="nearest (own subset, not comparable)")
    ax.plot(means["step"], means["hmm_top1_own_subset"], "o--", color="tab:orange",
            alpha=0.4, label="tuned HMM (own subset, not comparable)")
    ax.set_xlabel("keep every Nth point (decoded on the thinned track)")
    ax.set_ylabel("mean Top-1 accuracy (student_test)")
    ax.set_title("Accuracy vs GPS sampling rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sampling_rate_curve.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}")
    plot_optuna(study)
    plot_top1_per_trip()
    plot_sampling_rate()
    print(f"wrote plots to {FIG_DIR}")
    print("(uturn_plateau.png is written by pathsnap.hmm.uturn_plateau)")


if __name__ == "__main__":
    main()
