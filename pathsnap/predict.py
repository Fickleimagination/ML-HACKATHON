"""Single entry point for the API:  pathsnap.predict:predict_trip

    predict_trip(track, nodes, arcs) -> dict

Inputs are the raw NumPy arrays in source row order:
    track (N, 3) lon, lat, t      nodes (M, 2) lon, lat      arcs (K, 2) from, to
All returned IDs are row indices of the original .arcs file.

This module never reads .route files and never imports evaluation code.
"""
import json
from pathlib import Path

import joblib
import numpy as np
from shapely import points

from features.build_features import trip_features
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
        "model_path": raw.get("model_path"),
    }


def _classifier_probs(prep, arcs, cfg, cands):
    """B's tuned LightGBM binary classifier (features.build_features +
    models.zoo), scored on A's live preprocessing output -- no reference
    labels, no .route, so this still works at inference and for evaluator
    trips. Returns a list (one array per point, aligned with `cands`) of
    P(candidate is correct), or None if no model is configured (falls back
    to distance-only decoding).

    features.build_features.trip_features() is written for the labeled
    training pipeline: it requires a "label" column on the candidates table
    and "label_low_conf"/"label_in_candidates" columns on the points table,
    used only to build its own "label"/"usable" OUTPUT columns (never read
    by the FEATURES computation itself). At inference none of that exists,
    so dummy placeholder columns are added purely to satisfy the function's
    input schema; the resulting "label"/"usable" output columns are never
    used below -- only the FEATURES columns are.
    """
    model_path = cfg.get("model_path")
    if not model_path or not Path(model_path).exists():
        return None

    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["features"]

    c = prep.cand.copy()
    c["label"] = np.int8(0)
    p = prep.obs.copy()
    p["partition"] = "inference"
    p["label_low_conf"] = False
    p["label_in_candidates"] = True

    df = trip_features(c, p, arcs, "inference")
    proba = model.score(df[feature_cols].to_numpy(np.float32))

    by_point = {}
    for o, a, pr in zip(df["obs_idx"].to_numpy(), df["cand_arc"].to_numpy(), proba):
        by_point.setdefault(int(o), {})[int(a)] = float(pr)

    probs = []
    for t, cs in enumerate(cands):
        if not cs:
            probs.append(np.array([]))
            continue
        have = by_point.get(t, {})
        arc_ids = [a for a, _, _ in cs]
        assert set(have) == set(arc_ids), (
            f"classifier/candidate arc mismatch at point {t}: "
            f"classifier arcs={sorted(have)} decoder candidate arcs={sorted(arc_ids)}")
        probs.append(np.array([have[a] for a in arc_ids]))
    return probs


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


def _confidence(point_arcs, cands, probs):
    """Classifier P(chosen arc is correct) per observation, or None where
    there's no classifier or the chosen arc isn't one of that point's own
    candidates (e.g. copied in by _fill_missing's nearest-in-time fallback)."""
    if probs is None:
        return [None] * len(point_arcs)
    conf = []
    for t, a in enumerate(point_arcs.tolist()):
        arc_ids = [c[0] for c in cands[t]] if cands[t] else []
        conf.append(float(probs[t][arc_ids.index(a)]) if a in arc_ids else None)
    return conf


def predict_trip(track, nodes, arcs, config=None):
    cfg = config or load_config()
    track, nodes = np.asarray(track, float), np.asarray(nodes, float)
    arcs = np.asarray(arcs, np.int64)

    net, xy, times, cands, tree, prep = prepare_trip(track, nodes, arcs,
                                                      radius=cfg["radius"], gap_s=cfg["gap_s"])
    probs = _classifier_probs(prep, arcs, cfg, cands)
    point_arcs, route, stats = decode_trip(net, xy, times, cands, probs, cfg)
    no_cand = int(sum(not c for c in cands))
    point_arcs = _fill_missing(point_arcs, tree, xy)
    confidence = _confidence(point_arcs, cands, probs)

    return {
        "arcs": point_arcs.tolist(),                       # one per observation
        "route": route,                                    # ordered, with connectors
        "confidence": confidence,                           # P(chosen arc correct), None without a classifier
        "candidate_arcs": [[c[0] for c in cs] for cs in cands],
        "diagnostics": {
            "n_points": len(xy),
            "no_candidate_points": no_cand,
            "segments": stats["segments"],
            "disconnections": stats["disconnections"],
            "config_version": "see artifacts/hmm_config.json",
        },
    }
