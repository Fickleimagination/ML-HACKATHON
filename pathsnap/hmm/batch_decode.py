"""Run the legacy v0 candidate-generator experiment over a trip folder.

For current decoder metrics, use `python -m pathsnap.experiments.eval_student_test`.
This legacy runner remains available to reproduce the original route-membership CSV.

    python -m pathsnap.hmm.batch_decode data/student_test
    python -m pathsnap.hmm.batch_decode data/train

Writes results/decode_<partition>.csv with one row per trip: trip, points,
segments, disconnections, no_candidate_points, runtime_s, continuity,
nearest_share, hmm_share.

nearest_share/hmm_share are a dev-only accuracy check against the trip's
.route file, computed by `_route_share` below. That is the ONLY place in
this module that reads .route. pathsnap/predict.py (the API path) never
imports this module and never reads .route.

Processes one trip at a time and drops the RoadNetwork/candidates before
moving on, per the 16 GB RAM constraint.
"""
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np

from pathsnap.hmm.candidates_v0 import build_index, candidates
from pathsnap.hmm.routing import RoadNetwork, load_trip, make_projector
from pathsnap.hmm.viterbi import continuity, decode_trip
from pathsnap.predict import load_config

OUT_DIR = Path("results")
FIELDS = ["trip", "points", "segments", "disconnections", "no_candidate_points",
          "runtime_s", "continuity", "nearest_share", "hmm_share"]


def _route_share(folder, name, cands, point_arcs):
    """Dev-only metric: share of points whose assigned arc is on the
    reference route. Reads the trip's .route file -- never called from
    prediction code. Returns (nearest_share, hmm_share), either None if
    there is no .route file (e.g. a real evaluator-test trip)."""
    try:
        ref = set(np.loadtxt(f"{folder}/{name}.route", dtype=np.int64, ndmin=1).tolist())
    except OSError:
        return None, None
    nearest = [min(c, key=lambda x: x[1])[0] if c else -1 for c in cands]

    def on_ref(arr):
        kept = [a in ref for a in arr if a >= 0]
        return float(np.mean(kept)) if kept else None

    return on_ref(nearest), on_ref(point_arcs.tolist())


def decode_one(folder, name, cfg):
    """Decode a single trip. `folder` is the trip's own directory
    (e.g. data/student_test/00000056), `name` is the file stem."""
    track, nodes, arcs = load_trip(folder, name)
    proj = make_projector(track[:, 0], track[:, 1])
    xy = np.column_stack(proj.transform(track[:, 0], track[:, 1]))
    nxy = np.column_stack(proj.transform(nodes[:, 0], nodes[:, 1]))
    net = RoadNetwork(nxy, arcs)

    t0 = time.time()
    cands = candidates(net, build_index(net), xy, cfg["radius"])
    no_cand = sum(not c for c in cands)
    point_arcs, route, stats = decode_trip(net, xy, track[:, 2], cands, cfg=cfg)
    runtime = time.time() - t0

    nearest_share, hmm_share = _route_share(folder, name, cands, point_arcs)
    row = {
        "trip": name,
        "points": len(xy),
        "segments": stats["segments"],
        "disconnections": stats["disconnections"],
        "no_candidate_points": no_cand,
        "runtime_s": round(runtime, 3),
        "continuity": round(continuity(net, route), 4),
        "nearest_share": None if nearest_share is None else round(nearest_share, 4),
        "hmm_share": None if hmm_share is None else round(hmm_share, 4),
    }
    del net, cands, point_arcs, route
    gc.collect()
    return row


def main(folder, cfg=None):
    folder = Path(folder)
    partition = folder.name
    cfg = cfg or load_config()
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for trip_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        name = trip_dir.name
        row = decode_one(str(trip_dir), name, cfg)
        rows.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()))

    out_path = OUT_DIR / f"decode_{partition}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
