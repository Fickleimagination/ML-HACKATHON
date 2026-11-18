"""Single entry point for the API:  pathsnap.predict:predict_trip

    predict_trip(track, nodes, arcs) -> dict

Inputs are the raw NumPy arrays in source row order:
    track (N, 3) lon, lat, t      nodes (M, 2) lon, lat      arcs (K, 2) from, to
All returned IDs are row indices of the original .arcs file.

This module never reads .route files and never imports evaluation code.
"""
import json
from pathlib import Path

import numpy as np
from shapely import points

from pathsnap.hmm.prepared_trip import prepare_trip
from pathsnap.hmm.viterbi import decode_trip

CONFIG_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "hmm_config.json"


def load_config(path=CONFIG_PATH):
    raw = json.loads(Path(path).read_text())
    return {
        "radius": raw["radius_m"],
        "sigma": raw["emission"]["sigma_m"],
        "alpha": raw["emission"]["alpha_classifier_weight"],
        "beta": raw["transition"]["beta_m"],
        "uturn_penalty": raw["transition"]["uturn_penalty"],
        "no_path_logp": raw["transition"]["no_path_logp"],
        "gap_s": raw["gaps"]["gap_s"],
        "connector_limit": raw["fallbacks"]["connector_search_limit_m"],
    }


def _classifier_probs(track, nodes, arcs, xy, net, cands):
    """Hook for B's model. Return a list (one array per point, aligned with
    `cands`) of P(candidate is correct), or None for distance-only decoding.
    TODO: call B's feature function + artifacts/model.joblib here."""
    return None


def _fill_missing(point_arcs, tree, xy):
    """Give every observation an arc: copy from the nearest-in-time decoded
    point; if nothing was decoded at all, use the globally nearest arc."""
    have = np.flatnonzero(point_arcs >= 0)
    miss = np.flatnonzero(point_arcs < 0)
    if len(miss) == 0:
        return point_arcs
    if len(have):
        pos = np.searchsorted(have, miss).clip(1, len(have)) - 1
        left, right = have[pos], have[np.minimum(pos + 1, len(have) - 1)]
        pick = np.where(np.abs(miss - left) <= np.abs(right - miss), left, right)
        point_arcs[miss] = point_arcs[pick]
    else:
        point_arcs[miss] = tree.nearest(points(xy[miss]))
    return point_arcs


def predict_trip(track, nodes, arcs, config=None):
    cfg = config or load_config()
    track, nodes = np.asarray(track, float), np.asarray(nodes, float)
    arcs = np.asarray(arcs, np.int64)

    net, xy, times, cands, tree = prepare_trip(track, nodes, arcs,
                                                radius=cfg["radius"], gap_s=cfg["gap_s"])
    probs = _classifier_probs(track, nodes, arcs, xy, net, cands)
    point_arcs, route, stats = decode_trip(net, xy, times, cands, probs, cfg)
    no_cand = int(sum(not c for c in cands))
    point_arcs = _fill_missing(point_arcs, tree, xy)

    return {
        "arcs": point_arcs.tolist(),                       # one per observation
        "route": route,                                    # ordered, with connectors
        "confidence": None,                                # added with classifier
        "candidate_arcs": [[c[0] for c in cs] for cs in cands],
        "diagnostics": {
            "n_points": len(xy),
            "no_candidate_points": no_cand,
            "segments": stats["segments"],
            "disconnections": stats["disconnections"],
            "config_version": "see artifacts/hmm_config.json",
        },
    }
