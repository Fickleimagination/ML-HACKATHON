"""Build per-trip observation and candidate tables (Person A -> B/C hand-off).

Usage (from the repo root):
    python -m data_processing.build_dataset --raw "data/raw/Student folder" --out data/processed

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

from .candidates import DEFAULT_RADIUS_M, ArcIndex
from .cleaning import GAP_SECONDS, observation_table
from .io_utils import list_trips, load_trip
from .labels import LOW_CONF_DIST_M, TIE_TOL_M, labels_v0
from .projection import TripProjection

PARTITIONS = ["train", "student_test"]


def process_trip(trip_dir: str, partition: str, radius: float, gap_s: float):
    trip = load_trip(trip_dir)
    proj = TripProjection.for_track(trip.track)
    pts_xy = proj.to_xy(trip.track[:, :2])
    node_xy = proj.to_xy(trip.nodes)

    obs = observation_table(trip.track, pts_xy, gap_s=gap_s)
    index = ArcIndex(node_xy, trip.arcs)
    cand = index.query(pts_xy, radius=radius)

    label, ldist = labels_v0(pts_xy, obs["heading_deg"].to_numpy(np.float64),
                             trip.route, index.A, index.B, index.bearing)
    obs["label_arc"] = label.astype(np.int32)
    obs["label_dist_m"] = ldist.astype(np.float32)
    obs["label_low_conf"] = ldist > LOW_CONF_DIST_M

    cand["label"] = (cand["cand_arc"].to_numpy() == label[cand["obs_idx"].to_numpy()]).astype(np.int8)
    n_cand = np.bincount(cand["obs_idx"], minlength=len(obs))
    has_pos = np.bincount(cand["obs_idx"], weights=cand["label"], minlength=len(obs)) > 0
    obs["n_candidates"] = n_cand.astype(np.int16)
    obs["label_in_candidates"] = has_pos

    obs.insert(0, "trip_id", trip.trip_id)
    cand.insert(0, "trip_id", trip.trip_id)
    obs.insert(1, "partition", partition)

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
        "label_dist_median_m": float(np.median(ldist)), "label_dist_p90_m": float(np.percentile(ldist, 90)),
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
        json.dump({"label_version": "v0_nearest_route_arc_heading_tiebreak", "radius_m": args.radius,
                   "gap_seconds": args.gap, "tie_tol_m": TIE_TOL_M, "low_conf_dist_m": LOW_CONF_DIST_M,
                   "projection": "per-trip AEQD centred on mean GPS position"}, f, indent=2)


if __name__ == "__main__":
    main()
