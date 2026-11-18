"""One-off check: does score_trip(cache, ...) reproduce decode_trip's
point_arcs exactly, for several parameter settings? Not part of the
pipeline -- run manually, then delete or keep as a regression check."""
import numpy as np

from pathsnap.hmm.candidates_v0 import build_index, candidates
from pathsnap.hmm.decode_cache import load_cache, precompute_trip, score_trip
from pathsnap.hmm.routing import RoadNetwork, load_trip, make_projector
from pathsnap.hmm.viterbi import decode_trip
from pathsnap.predict import load_config

PARAM_SETS = [
    dict(sigma=10.0, beta=5.0, uturn_penalty=3.0),
    dict(sigma=5.0, beta=20.0, uturn_penalty=0.0),
    dict(sigma=25.0, beta=2.0, uturn_penalty=10.0),
]


def check(folder, name):
    cfg = load_config()
    track, nodes, arcs = load_trip(folder, name)
    proj = make_projector(track[:, 0], track[:, 1])
    xy = np.column_stack(proj.transform(track[:, 0], track[:, 1]))
    nxy = np.column_stack(proj.transform(nodes[:, 0], nodes[:, 1]))
    net = RoadNetwork(nxy, arcs)
    cands = candidates(net, build_index(net), xy, cfg["radius"])

    precompute_trip(folder, name, cfg)
    cache = load_cache(name)

    for params in PARAM_SETS:
        full_cfg = {**cfg, **params}
        point_arcs, _, _ = decode_trip(net, xy, track[:, 2], cands, cfg=full_cfg)
        net.clear_cache()
        idx, arc = score_trip(cache, no_path_logp=cfg["no_path_logp"], **params)
        expected = point_arcs[idx]
        ok = np.array_equal(expected, arc)
        print(f"{name} {params}: {'MATCH' if ok else 'MISMATCH'} "
              f"({len(idx)} cached points vs {(point_arcs >= 0).sum()} decoded)")
        if not ok:
            bad = np.flatnonzero(expected != arc)[:5]
            print("  first mismatches at cache-positions", bad.tolist(),
                  "expected", expected[bad].tolist(), "got", arc[bad].tolist())


if __name__ == "__main__":
    check("data/student_test/00000056", "00000056")
    check("data/student_test/00000065", "00000065")
