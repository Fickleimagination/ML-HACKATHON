import importlib
import os
from collections.abc import Mapping

import numpy as np
import shapely

from data_processing.preprocess import preprocess_trip


def nearest_arc(track, nodes, arcs):
    prep = preprocess_trip(track, nodes, arcs)
    valid_rows = np.flatnonzero(~prep.obs["flag_invalid"].to_numpy())
    pairs = prep.index.tree.query_nearest(shapely.points(prep.pts_xy[valid_rows]))
    # Resolve equal distances by arc ID for reproducible baseline runs.
    nearest = np.full(len(valid_rows), len(arcs), dtype=np.int64)
    np.minimum.at(nearest, pairs[0], pairs[1])
    chosen = np.full(len(track), -1, dtype=np.int64)
    chosen[valid_rows] = nearest
    route = nearest[np.r_[True, nearest[1:] != nearest[:-1]]]
    candidates = [[] for _ in track]
    for obs, arc in prep.cand[["obs_idx", "cand_arc"]].itertuples(index=False, name=None):
        candidates[obs].append(arc)
    # Include the fallback actually used even when no arc is inside the search radius.
    for obs, arc in zip(valid_rows, nearest, strict=True):
        if arc not in candidates[obs]:
            candidates[obs].append(int(arc))
    return {
        "arcs": chosen,
        "route": route,
        "candidate_arcs": candidates,
    }


HMM_PREDICTOR = "pathsnap.predict:predict_trip"


def load_predictor():
    target = os.getenv("PATHSNAP_PREDICTOR", HMM_PREDICTOR)
    if target == "nearest":
        return nearest_arc, "Nearest-arc baseline"
    module, separator, name = target.partition(":")
    if not separator or not module or not name:
        raise ValueError("PATHSNAP_PREDICTOR must be nearest or module:function")
    predictor = getattr(importlib.import_module(module), name)
    if not callable(predictor):
        raise ValueError("Configured predictor is not callable")
    name = "PathSnap HMM" if target == HMM_PREDICTOR else target
    return predictor, name


def arc_ids(values, arc_count, *, allow_unmatched=False):
    ids = np.asarray(values)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError("Arc IDs must be a one-dimensional integer sequence")
    minimum = -1 if allow_unmatched else 0
    if ids.size and (ids.min() < minimum or ids.max() >= arc_count):
        raise ValueError("Arc ID is outside this trip's network")
    return ids.astype(np.int64, copy=False)


def validate_prediction(result, obs_count, arc_count):
    if not isinstance(result, Mapping):
        raise ValueError("Predictor must return a mapping")
    predicted = arc_ids(result["arcs"], arc_count, allow_unmatched=True)
    route = arc_ids(result["route"], arc_count)
    if len(predicted) != obs_count or (not len(route) and (predicted >= 0).any()):
        raise ValueError("Return one ID per observation and a route when any observation matches")
    confidence = result.get("confidence")
    if confidence is not None:
        confidence = np.array(confidence, dtype=object)
        if confidence.shape != predicted.shape:
            raise ValueError("Confidence requires one entry per observation")
        # Fallback arcs can be valid predictions without a classifier score.
        scored = np.array([value is not None for value in confidence], dtype=bool)
        probabilities = confidence[scored].astype(float)
        if not (np.isfinite(probabilities).all()
                and ((0 <= probabilities) & (probabilities <= 1)).all()):
            raise ValueError("Confidence entries must be finite probabilities or null")
        confidence[scored] = probabilities
    candidates = result.get("candidate_arcs")
    if candidates is not None:
        if len(candidates) != obs_count:
            raise ValueError("Candidate lists must cover every observation")
        candidates = [arc_ids(row, arc_count) for row in candidates]
    return predicted, route, confidence, candidates
