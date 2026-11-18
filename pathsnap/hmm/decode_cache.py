"""Per-trip cache that separates the expensive, parameter-independent part of
decoding (candidate geometry + road distances, which need Dijkstra over the
RoadNetwork) from the cheap, parameter-dependent part (the Viterbi forward
pass, which only needs sigma/beta/uturn_penalty to turn those numbers into
log-probabilities). Built for task 3: Optuna can call `score_trip` thousands
of times on the cached arrays with pure NumPy, no Dijkstra, no shapely.

    precompute_trip(folder, name, cfg)   -> writes results/cache/<name>.npz
    load_cache(name)                     -> dict of arrays
    score_trip(cache, sigma, beta, uturn_penalty, no_path_logp) -> point_arcs

gap_s is baked into the cache at precompute time (it controls segmentation,
which changes which Dijkstra calls are needed) -- re-run precompute_trip if
gap_s changes. sigma/beta/uturn_penalty/no_path_logp are NOT baked in; they
are the knobs score_trip takes.

score_trip reproduces exactly the arithmetic of viterbi._decode_segment, just
reading `d` (perp distance) and `rd` (road distance) from the cache instead
of recomputing them, so it must be kept in sync with that function by hand.
"""
import sys
import time
from pathlib import Path

import numpy as np

from pathsnap.hmm.candidates_v0 import build_index, candidates
from pathsnap.hmm.routing import RoadNetwork, load_trip, make_projector
from pathsnap.hmm.viterbi import split_segments
from pathsnap.predict import load_config

CACHE_DIR = Path("results/cache")


def precompute_trip(folder, name, cfg=None):
    """Compute and save everything decode_trip needs except sigma, beta,
    uturn_penalty, no_path_logp, alpha: candidate (arc, dist, frac) per
    point, the straight-line distance between consecutive points in a
    segment, and the full road-distance + is-reverse matrix between their
    candidate sets. One Dijkstra-heavy pass per trip; after this, scoring
    any parameter set is pure NumPy."""
    cfg = {**load_config(), **(cfg or {})}
    track, nodes, arcs = load_trip(folder, name)
    proj = make_projector(track[:, 0], track[:, 1])
    xy = np.column_stack(proj.transform(track[:, 0], track[:, 1]))
    nxy = np.column_stack(proj.transform(nodes[:, 0], nodes[:, 1]))
    net = RoadNetwork(nxy, arcs)
    cands = candidates(net, build_index(net), xy, cfg["radius"])
    segs = split_segments(track[:, 2], cands, cfg["gap_s"])

    seg_lens = np.array([len(s) for s in segs], dtype=np.int64)
    seg_idx = np.concatenate(segs).astype(np.int64) if segs else np.empty(0, np.int64)
    n_cands = np.array([len(cands[t]) for t in seg_idx], dtype=np.int64)
    cand_arc = np.array([c[0] for t in seg_idx for c in cands[t]], dtype=np.int64)
    cand_dist = np.array([c[1] for t in seg_idx for c in cands[t]], dtype=np.float64)
    cand_frac = np.array([c[2] for t in seg_idx for c in cands[t]], dtype=np.float64)

    straight, rd_shapes, rd_chunks, ut_chunks = [], [], [], []
    for seg in segs:
        for prev_t, t in zip(seg[:-1], seg[1:]):
            d = float(np.linalg.norm(xy[t] - xy[prev_t]))
            limit = 2 * d + 200.0
            P, C = cands[prev_t], cands[t]
            rd = np.full((len(P), len(C)), np.inf)
            ut = np.zeros((len(P), len(C)), dtype=bool)
            for i, (a, _, fa) in enumerate(P):
                for j, (b, _, fb) in enumerate(C):
                    rd[i, j] = net.route_distance(a, fa, b, fb, limit)
                    ut[i, j] = net.is_reverse(a, b)
            straight.append(d)
            rd_shapes.append((len(P), len(C)))
            rd_chunks.append(rd.ravel())
            ut_chunks.append(ut.ravel())
    net.clear_cache()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_DIR / f"{name}.npz",
        seg_lens=seg_lens, seg_idx=seg_idx, n_cands=n_cands,
        cand_arc=cand_arc, cand_dist=cand_dist, cand_frac=cand_frac,
        straight=np.array(straight, dtype=np.float64),
        rd_shapes=np.array(rd_shapes, dtype=np.int64).reshape(-1, 2) if rd_shapes else np.empty((0, 2), np.int64),
        rd_flat=np.concatenate(rd_chunks) if rd_chunks else np.empty(0, np.float64),
        uturn_flat=np.concatenate(ut_chunks) if ut_chunks else np.empty(0, dtype=bool),
        gap_s=np.array(cfg["gap_s"]), radius=np.array(cfg["radius"]),
    )


def load_cache(name):
    with np.load(CACHE_DIR / f"{name}.npz") as d:
        return {k: d[k] for k in d.files}


def score_trip(cache, sigma, beta, uturn_penalty, no_path_logp=-50.0):
    """Replay Viterbi from the cache. Returns (point_idx, arc_id) arrays
    aligned the same way decode_trip's point_arcs is for points that have
    candidates (points with none are simply absent, as in `seg_idx`)."""
    seg_lens, seg_idx = cache["seg_lens"], cache["seg_idx"]
    n_cands, cand_arc, cand_dist = cache["n_cands"], cache["cand_arc"], cache["cand_dist"]
    straight, rd_shapes = cache["straight"], cache["rd_shapes"]
    rd_flat, uturn_flat = cache["rd_flat"], cache["uturn_flat"]
    c_off = np.concatenate([[0], np.cumsum(n_cands)])

    out_idx, out_arc = [], []
    pos, pair, rd_off = 0, 0, 0
    for seg_len in seg_lens:
        seg_t = seg_idx[pos:pos + seg_len]
        local_off = c_off[pos:pos + seg_len + 1] - c_off[pos]
        base = c_off[pos]

        def cand(k):
            s, e = base + local_off[k], base + local_off[k + 1]
            return cand_dist[s:e], cand_arc[s:e]

        d0, arc0 = cand(0)
        score = -0.5 * (d0 / sigma) ** 2
        back = []
        for k in range(1, seg_len):
            i0, j0 = rd_shapes[pair]
            rd = rd_flat[rd_off:rd_off + i0 * j0].reshape(i0, j0)
            ut = uturn_flat[rd_off:rd_off + i0 * j0].reshape(i0, j0)
            trans = np.where(np.isfinite(rd),
                              -np.abs(rd - straight[pair]) / beta - ut * uturn_penalty,
                              no_path_logp)
            total = score[:, None] + trans
            back.append(total.argmax(0))
            dk, arck = cand(k)
            score = total.max(0) - 0.5 * (dk / sigma) ** 2
            pair += 1
            rd_off += i0 * j0

        path = [int(score.argmax())]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        path.reverse()

        for k, p in enumerate(path):
            _, arck = cand(k)
            out_idx.append(int(seg_t[k]))
            out_arc.append(int(arck[p]))
        pos += seg_len

    return np.array(out_idx, dtype=np.int64), np.array(out_arc, dtype=np.int64)


if __name__ == "__main__":
    # one-off precompute for a trip, e.g.
    #   python -m pathsnap.hmm.decode_cache data/student_test 00000056
    folder, name = sys.argv[1], sys.argv[2]
    t0 = time.time()
    precompute_trip(folder, name)
    print(f"cached {name} in {time.time() - t0:.1f}s -> {CACHE_DIR / (name + '.npz')}")
