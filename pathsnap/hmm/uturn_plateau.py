"""Task 3c-2: U-turn-penalty plateau check -- no new Optuna study.

Holds sigma and beta at the current Optuna best and sweeps uturn_penalty
across a wide range (including values far past the tuned range's edge, up
to 1000) using the training cache, to see whether mean Top-1 actually
flattens out past ~28 -- justifying the Optuna search's upper bound in the
report instead of re-tuning with an even wider range.

    python -m pathsnap.hmm.uturn_plateau

Writes results/uturn_plateau.csv and results/figures/uturn_plateau.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.evaluation.metrics import route_precision_recall, top1
from pathsnap.hmm.decode_cache import score_trip
from pathsnap.hmm.tune_decoder import _load_training_set
from pathsnap.hmm.viterbi import build_route
from pathsnap.predict import load_config

SIGMA = 1.6788762605299112   # current Optuna best, held fixed
BETA = 18.2956835903205      # current Optuna best, held fixed
UTURN_VALUES = [0, 3, 10, 20, 28.2, 50, 100, 1000]

OUT_CSV = Path("results/uturn_plateau.csv")
OUT_PNG = Path("results/figures/uturn_plateau.png")


def main():
    cfg = load_config()
    print("loading training cache + labels + networks ...")
    training_set = _load_training_set(cfg)
    print(f"loaded {len(training_set)} training trips")

    rows = []
    for uturn in UTURN_VALUES:
        top1_scores, precisions, total_disc = [], [], 0
        for trip in training_set:
            idx, arc = score_trip(trip["cache"], SIGMA, BETA, uturn, cfg["no_path_logp"])
            point_arcs = np.full(trip["n_points"], -1, dtype=np.int64)
            point_arcs[idx] = arc
            top1_scores.append(top1(point_arcs, trip["labels"]))

            route, disconnections = build_route(trip["net"], point_arcs, cfg["connector_limit"])
            total_disc += disconnections
            precision, _ = route_precision_recall(trip["net"], route, trip["ref_route"])
            if np.isfinite(precision):
                precisions.append(precision)
            trip["net"].clear_cache()

        row = dict(uturn_penalty=uturn,
                   mean_top1=round(float(np.mean(top1_scores)), 4),
                   mean_route_precision=round(float(np.mean(precisions)), 4),
                   total_disconnections=total_disc)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(df["uturn_penalty"], df["mean_top1"], "o-", color="tab:blue", label="mean Top-1 (train)")
    ax1.set_xlabel(f"uturn_penalty (sigma={SIGMA:.2f}, beta={BETA:.2f} fixed)")
    ax1.set_ylabel("mean Top-1 (train)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xscale("symlog")
    ax1.axvline(28.2, color="gray", linestyle=":", linewidth=1, label="Optuna best (28.2)")

    ax2 = ax1.twinx()
    ax2.plot(df["uturn_penalty"], df["mean_route_precision"], "s--", color="tab:orange",
             label="mean route precision")
    ax2.set_ylabel("mean route precision", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax1.set_title("U-turn penalty plateau (sigma/beta fixed at tuned best)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
