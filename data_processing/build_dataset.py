"""Build per-trip observation and candidate tables (Person A -> B/C hand-off).

Usage (from the repo root):
    python -m data_processing.build_dataset --raw data --out data/processed

Writes, for each trip in train/ and student_test/:
    <out>/points/<trip_id>.parquet      one row per GPS observation
    <out>/candidates/<trip_id>.parquet  one row per (observation, candidate arc)
and <out>/summary_by_trip.csv, <out>/config.json.

Evaluator-test trips are never processed by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .candidates import DEFAULT_RADIUS_M
from .cleaning import GAP_SECONDS
from .io_utils import list_trips, load_trip
from .labels import labels_v0
from .preprocess import SETTINGS, preprocess_trip, reference_labels

PARTITIONS = ["train", "student_test"]


def process_trip(trip_dir: str, partition: str, radius: float, gap_s: float):
    trip = load_trip(trip_dir)
    prep = preprocess_trip(trip.track, trip.nodes, trip.arcs, radius=radius, gap_s=gap_s)
    obs, cand = prep.obs, prep.cand

    # v1 reference labels (order-preserving alignment) -- the labels everyone uses
    ref = reference_labels(prep, trip.route)
    for c in ["label_arc", "label_route_pos", "label_dist_m", "label_low_conf"]:
        obs[c] = ref[c].to_numpy()
    # v0 kept only for comparison in the report
    valid = ~obs["flag_invalid"].to_numpy()
    v0 = np.full(len(obs), -1, dtype=np.int64)
    v0[valid], _ = labels_v0(prep.pts_xy[valid], obs["heading_deg"].to_numpy(np.float64)[valid],
                             trip.route, prep.index.A, prep.index.B, prep.index.bearing)
    obs["label_arc_v0"] = v0.astype(np.int32)
    label = obs["label_arc"].to_numpy()
    ldist = obs["label_dist_m"].to_numpy()

    cand["label"] = (cand["cand_arc"].to_numpy() == label[cand["obs_idx"].to_numpy()]).astype(np.int8)
    n_cand = np.bincount(cand["obs_idx"], minlength=len(obs))
    has_pos = np.bincount(cand["obs_idx"], weights=cand["label"], minlength=len(obs)) > 0
    obs["n_candidates"] = n_cand.astype(np.int16)
    obs["label_in_candidates"] = has_pos

    obs.insert(0, "trip_id", trip.trip_id)
    cand.insert(0, "trip_id", trip.trip_id)
    obs.insert(1, "partition", partition)
    cand.insert(1, "partition", partition)

    summary = {
        "trip_id": trip.trip_id, "partition": partition,
        "n_obs": len(obs), "n_nodes": len(trip.nodes), "n_arcs": len(trip.arcs),
        "route_len": len(trip.route), "route_unique": len(np.unique(trip.route)),
        "n_segments": int(obs["segment_id"].max() + 1),
        "max_gap_s": float(np.nanmax(obs["dt_prev"])) if len(obs) > 1 else 0.0,
        "frac_1s_steps": float(np.mean(obs["dt_prev"].iloc[1:] == 1)),
        "n_invalid": int(obs["flag_invalid"].sum()), "n_dup_time": int(obs["flag_dup_time"].sum()),
        "n_time_backwards": int(obs["flag_time_backwards"].sum()), "n_jump": int(obs["flag_jump"].sum()),
        "cands_per_obs_mean": float(n_cand.mean()), "obs_no_candidates": int((n_cand == 0).sum()),
        "candidate_miss_rate": float(1 - has_pos.mean()),
        "label_dist_median_m": float(np.nanmedian(ldist)), "label_dist_p90_m": float(np.nanpercentile(ldist, 90)),
        "v1_vs_v0_label_change_frac": float(np.mean(obs["label_arc"] != obs["label_arc_v0"])),
        "label_low_conf_frac": float(obs["label_low_conf"].mean()),
    }
    return obs, cand, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder holding train/ and student_test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M)
    ap.add_argument("--gap", type=float, default=GAP_SECONDS)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "points"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "candidates"), exist_ok=True)
    rows = []
    t0 = time.time()
    for part in PARTITIONS:
        for d in list_trips(args.raw, part):
            obs, cand, s = process_trip(d, part, args.radius, args.gap)
            obs.to_parquet(os.path.join(args.out, "points", f"{s['trip_id']}.parquet"), index=False)
            cand.to_parquet(os.path.join(args.out, "candidates", f"{s['trip_id']}.parquet"), index=False)
            rows.append(s)
            print(f"{part:12s} {s['trip_id']}  obs={s['n_obs']:6d}  cands/obs={s['cands_per_obs_mean']:5.1f}"
                  f"  miss={s['candidate_miss_rate']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(args.out, "summary_by_trip.csv"), index=False)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({"label_version": "v1_order_preserving_alignment", **SETTINGS,
                   "candidate_radius_m": args.radius, "segment_gap_s": args.gap}, f, indent=2)


if __name__ == "__main__":
    main()
