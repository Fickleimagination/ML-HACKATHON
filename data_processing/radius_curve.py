"""Candidate-radius study: miss rate vs number of candidates per point.

A point's reference arc is among its candidates exactly when the point lies within the
radius R of that arc, so miss_rate(R) = share of points with label_dist_m > R.
Candidates per point are counted with one query at the largest radius.

The radius is chosen on the TRAIN partition only; student_test is shown for information.
Trips 54, 33, 2 and 96 contain stretches where the GPS leaves the road network or disagrees
with the reference route (data_quality_log.md); results are given with and without them.

Usage:
    python -m data_processing.radius_curve --raw "data/raw/Student folder" --processed data/processed --out results/radius
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import list_trips, load_trip
from .preprocess import preprocess_trip

RADII = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150]
PROBLEM_TRIPS = {"00000054", "00000033", "00000002", "00000096"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="results/radius")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for part in ["train", "student_test"]:
        for d in list_trips(a.raw, part):
            trip = load_trip(d, with_route=False)
            prep = preprocess_trip(trip.track, trip.nodes, trip.arcs, radius=max(RADII))
            pts = pd.read_parquet(os.path.join(a.processed, "points", f"{trip.trip_id}.parquet"))
            ld = pts["label_dist_m"].to_numpy()
            dist = prep.cand["dist_m"].to_numpy()
            n = len(pts)
            for R in RADII:
                rows.append({"trip_id": trip.trip_id, "partition": part, "radius_m": R, "n_obs": n,
                             "n_missed": int(np.sum(~(ld <= R))), "n_candidates": int(np.sum(dist <= R))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "radius_by_trip.csv"), index=False)

    def agg(sub):
        g = sub.groupby("radius_m")[["n_obs", "n_missed", "n_candidates"]].sum()
        return pd.DataFrame({"miss_rate": g.n_missed / g.n_obs, "cands_per_obs": g.n_candidates / g.n_obs})

    groups = {
        "train, all 80 trips": df[df.partition == "train"],
        "train, excl. trips 54/33/2/96": df[(df.partition == "train") & ~df.trip_id.isin(PROBLEM_TRIPS)],
        "student_test (info only)": df[df.partition == "student_test"],
    }
    summary = pd.concat({k: agg(v) for k, v in groups.items()}, names=["group"]).reset_index()
    summary.to_csv(os.path.join(a.out, "radius_summary.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    styles = {"train, all 80 trips": ("#2a6fdb", "-"), "train, excl. trips 54/33/2/96": ("#2a6fdb", "--"),
              "student_test (info only)": ("#e8833a", "-")}
    for k, s in summary.groupby("group"):
        c, ls = styles[k]
        ax[0].plot(s.radius_m, s.miss_rate * 100, marker="o", color=c, ls=ls, label=k)
        ax[1].plot(s.cands_per_obs, s.miss_rate * 100, marker="o", color=c, ls=ls, label=k)
        for _, r in s[s.radius_m.isin([20, 30, 50, 100])].iterrows():
            ax[1].annotate(f"{int(r.radius_m)} m", (r.cands_per_obs, r.miss_rate * 100), fontsize=7,
                           textcoords="offset points", xytext=(4, 4))
    for x in ax:
        x.axhline(0, color="#999", lw=0.5); x.set_ylabel("candidate miss rate (%)"); x.grid(alpha=0.3)
    ax[0].axvline(50, color="k", ls=":", lw=1); ax[0].set_xlabel("candidate radius R (m)")
    ax[0].set_title("Miss rate vs radius"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("candidates per GPS point (mean)"); ax[1].set_title("Trade-off: misses vs candidate load")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "radius_curve.png"), dpi=150); plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
