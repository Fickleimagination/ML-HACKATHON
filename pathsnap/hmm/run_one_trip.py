"""Checkpoint script: decode one trip with distance-only emissions.

    python -m pathsnap.hmm.run_one_trip data/student_test/56 56
"""
import sys
import time

import numpy as np

from pathsnap.hmm.candidates_v0 import build_index, candidates
from pathsnap.hmm.routing import RoadNetwork, load_trip, make_projector
from pathsnap.hmm.viterbi import continuity, decode_trip


def main(folder, name, radius=50.0):
    track, nodes, arcs = load_trip(folder, name)
    proj = make_projector(track[:, 0], track[:, 1])
    xy = np.column_stack(proj.transform(track[:, 0], track[:, 1]))
    nxy = np.column_stack(proj.transform(nodes[:, 0], nodes[:, 1]))
    net = RoadNetwork(nxy, arcs)

    t0 = time.time()
    cands = candidates(net, build_index(net), xy, radius)
    miss_pts = sum(not c for c in cands)
    point_arcs, route, stats = decode_trip(net, xy, track[:, 2], cands)
    dt = time.time() - t0

    # nearest-candidate baseline (what the HMM must beat)
    nearest = np.array([min(c, key=lambda x: x[1])[0] if c else -1 for c in cands])

    print(f"points={len(xy)} arcs={len(arcs)} no-candidate points={miss_pts} "
          f"time={dt:.1f}s")
    print(f"route arcs={len(route)} continuity HMM={continuity(net, route):.3f} "
          f"segments={stats['segments']} disconnections={stats['disconnections']}")

    try:  # reference only for a quick dev sanity check, never in the API path
        ref = set(np.loadtxt(f"{folder}/{name}.route", dtype=np.int64, ndmin=1).tolist())
        on_ref = lambda arr: np.mean([a in ref for a in arr if a >= 0])
        print(f"share of points on a reference arc: nearest={on_ref(nearest):.3f} "
              f"HMM={on_ref(point_arcs):.3f}")
    except OSError:
        pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
