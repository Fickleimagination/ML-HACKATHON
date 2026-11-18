"""Task-3 step 4 (fixed in 3c-1): how much does a lower GPS sampling rate
hurt accuracy?

Keeps every 1st/5th/10th/30th point of each student_test trip (same
candidates, re-run the decoder on the sparser sequence so segmentation/gaps
react to the larger time steps exactly as they would on real low-rate GPS),
and compares nearest-candidate vs tuned-HMM Top-1.

Fairness fix: scoring Top-1 on each rate's OWN thinned subset isn't
comparable across rates -- a sparser subset can look "easier" or "harder"
just because it's a different set of points, not because of the sampling
rate itself (e.g. step=30 was scoring on 1/30 of the points, not the same
points step=1 is scored on). Since every step here divides 30, the step=30
index set (multiples of 30) is a subset of every other step's index set, so
it's used as the COMMON subset: every rate's Top-1 is also reported scored
ONLY on those shared obs_idx, which makes the four rates directly
comparable. The old per-rate ("own subset") numbers are kept alongside,
clearly labelled, for reference.

    python -m pathsnap.experiments.sampling_rate

Writes results/sampling_rate.csv (per trip + mean, per step).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pathsnap.evaluation.metrics import load_point_labels, top1
from pathsnap.experiments.eval_student_test import load_tuned_params, load_trip_geometry
from pathsnap.hmm.viterbi import decode_trip
from pathsnap.predict import load_config

STUDENT_TEST_DIR = Path("data/student_test")
OUT_CSV = Path("results/sampling_rate.csv")
STEPS = [1, 5, 10, 30]
COMMON_STEP = max(STEPS)
assert all(COMMON_STEP % s == 0 for s in STEPS), "COMMON_STEP must be a multiple of every step"


def main():
    default_cfg = load_config()
    tuned_params, _ = load_tuned_params()
    cfg = {**default_cfg, **tuned_params}

    rows = []
    for trip_dir in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        name = trip_dir.name
        folder = str(trip_dir)
        track, nodes, arcs, xy, net, cands, _ = load_trip_geometry(
            folder, name, cfg["radius"], cfg["gap_s"])
        times = track[:, 2]
        labels = load_point_labels(name)

        common_idx = np.arange(0, len(xy), COMMON_STEP)
        common_labels = labels.iloc[common_idx].reset_index(drop=True)

        for step in STEPS:
            idx = np.arange(0, len(xy), step)
            sub_cands = [cands[i] for i in idx]
            own_labels = labels.iloc[idx].reset_index(drop=True)

            nearest = np.array([min(c, key=lambda x: x[1])[0] if c else -1 for c in sub_cands])
            point_arcs, _, _ = decode_trip(net, xy[idx], times[idx], sub_cands, cfg=cfg)
            net.clear_cache()

            # idx is a multiple of `step`; common_idx is a multiple of COMMON_STEP, which is
            # itself a multiple of step, so every common_idx value is in idx at position
            # common_idx // step.
            common_pos = common_idx // step
            nearest_common_top1 = top1(nearest[common_pos], common_labels)
            hmm_common_top1 = top1(point_arcs[common_pos], common_labels)

            rows.append(dict(
                trip=name, step=step, points=len(idx), common_points=len(common_idx),
                nearest_top1_own_subset=round(top1(nearest, own_labels), 4),
                hmm_top1_own_subset=round(top1(point_arcs, own_labels), 4),
                nearest_top1_common=round(nearest_common_top1, 4),
                hmm_top1_common=round(hmm_common_top1, 4),
            ))

    df = pd.DataFrame(rows)
    mean_cols = ["points", "common_points", "nearest_top1_own_subset", "hmm_top1_own_subset",
                 "nearest_top1_common", "hmm_top1_common"]
    means = df.groupby("step")[mean_cols].mean().round(4).reset_index()
    means.insert(0, "trip", "MEAN")
    means = means[df.columns]
    out = pd.concat([df, means], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
