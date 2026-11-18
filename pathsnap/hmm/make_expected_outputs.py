"""Run the joint predictor directly on student-test trips and save its outputs,
so D can check the API returns exactly the same thing.

    python -m pathsnap.hmm.make_expected_outputs data/student_test

Writes results/expected_outputs/<trip>.json and prints a summary showing which
trips exercise gaps (segments > 1) or disconnected joins (disconnections > 0).
Reads only .track/.nodes/.arcs, never .route.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from pathsnap.predict import predict_trip

OUT = Path("results/expected_outputs")


def main(folder):
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'trip':<10}{'points':>8}{'segments':>10}{'disconn':>9}{'no_cand':>9}{'secs':>7}")
    for trip in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        name = trip.name
        track = np.loadtxt(trip / f"{name}.track", ndmin=2)
        nodes = np.loadtxt(trip / f"{name}.nodes", ndmin=2)
        arcs = np.loadtxt(trip / f"{name}.arcs", dtype=np.int64, ndmin=2)
        t0 = time.time()
        out = predict_trip(track, nodes, arcs)
        secs = time.time() - t0
        (OUT / f"{name}.json").write_text(json.dumps(out))
        d = out["diagnostics"]
        print(f"{name:<10}{d['n_points']:>8}{d['segments']:>10}"
              f"{d['disconnections']:>9}{d['no_candidate_points']:>9}{secs:>7.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
