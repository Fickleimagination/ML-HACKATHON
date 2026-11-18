"""Task-3 step 4: how much does a lower GPS sampling rate hurt accuracy?

Keeps every 1st/5th/10th/30th point of each student_test trip (same
candidates, re-run the decoder on the sparser sequence so segmentation/gaps
react to the larger time steps exactly as they would on real low-rate GPS),
and compares nearest-candidate vs tuned-HMM Top-1.

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
        label_arc = labels["label_arc"].to_numpy()

        for step in STEPS:
            idx = np.arange(0, len(xy), step)
            sub_cands = [cands[i] for i in idx]
            sub_labels = labels.iloc[idx].reset_index(drop=True)

            nearest = np.array([min(c, key=lambda x: x[1])[0] if c else -1 for c in sub_cands])
            nearest_top1 = top1(nearest, sub_labels)

            point_arcs, _, _ = decode_trip(net, xy[idx], times[idx], sub_cands, cfg=cfg)
            hmm_top1 = top1(point_arcs, sub_labels)
            net.clear_cache()

            rows.append(dict(trip=name, step=step, points=len(idx),
                              nearest_top1=round(nearest_top1, 4),
                              hmm_top1=round(hmm_top1, 4)))

    df = pd.DataFrame(rows)
    means = df.groupby("step")[["points", "nearest_top1", "hmm_top1"]].mean().round(4).reset_index()
    means.insert(0, "trip", "MEAN")
    means = means[df.columns]
    out = pd.concat([df, means], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
