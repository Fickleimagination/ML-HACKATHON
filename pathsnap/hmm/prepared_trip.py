"""Bridge between A's shared preprocessing (data_processing.preprocess) and
the HMM decoder.

preprocess_trip never reads .route (confirmed: it only touches track, nodes,
arcs), so it is safe to call at inference time -- unlike reference_labels,
which is a separate, route-reading function used only for scoring.

This replaces pathsnap/hmm/candidates_v0.py as the candidate source for
prediction and for the decoder's Optuna cache: candidates_v0 was always a
throwaway stand-in (its own docstring says so) with its own ad hoc radius
query and no deterministic tie-break order. Going through preprocess_trip
instead means arc ids, perpendicular distances and projection fractions are
byte-for-byte what A's v1 labels and B's features were computed against --
not a second, independently-computed approximation of the same thing.
"""
import numpy as np

from data_processing.preprocess import preprocess_trip
from pathsnap.hmm.routing import RoadNetwork


def candidates_from_df(cand_df, n_points):
    """A's per-candidate DataFrame (prep.cand) -> a per-point list of
    (arc_id, dist_m, frac) tuples, the format pathsnap.hmm.viterbi expects.
    candidates.py already sorts rows by (obs_idx, dist_mm, cand_arc), so the
    per-point candidate order is deterministic across machines."""
    out = [[] for _ in range(n_points)]
    obs = cand_df["obs_idx"].to_numpy()
    arc = cand_df["cand_arc"].to_numpy()
    dist = cand_df["dist_m"].to_numpy()
    frac = cand_df["proj_t"].to_numpy()
    for i in range(len(obs)):
        out[int(obs[i])].append((int(arc[i]), float(dist[i]), float(frac[i])))
    return out


def prepare_trip(track, nodes, arcs, radius=50.0, gap_s=30.0):
    """Returns (net, xy, times, cands, tree, prep):
        net   RoadNetwork built on A's own projected node coordinates
        xy    (N, 2) projected point coordinates, NaN rows zeroed (invalid
              points never appear in `cands` so they're never read)
        times the raw .track timestamps, unchanged
        cands per-point candidate list, see candidates_from_df
        tree  A's shapely STRtree over the arcs, for predict.py's
              global-nearest-arc fallback when nothing decoded at all
        prep  the raw data_processing.preprocess.Prepared object (prep.cand,
              prep.obs), for callers that need A's/B's own DataFrame schema
              directly (e.g. feeding B's feature function) instead of
              re-running preprocess_trip a second time
    gap_s only affects A's internal obs/heading table, not the candidates
    (candidate generation depends only on radius); it's threaded through so
    callers can keep that table consistent too if they start using it.
    """
    track = np.asarray(track, float)
    nodes = np.asarray(nodes, float)
    arcs = np.asarray(arcs, np.int64)
    prep = preprocess_trip(track, nodes, arcs, radius=radius, gap_s=gap_s)
    net = RoadNetwork(prep.node_xy, arcs)
    xy = np.nan_to_num(prep.pts_xy)
    cands = candidates_from_df(prep.cand, len(track))
    return net, xy, track[:, 2], cands, prep.index.tree, prep
