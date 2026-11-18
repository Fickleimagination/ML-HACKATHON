"""Evaluation metrics for the HMM decoder.

NEVER imported by prediction code (pathsnap/predict.py or pathsnap/hmm/*,
other than the one-directional re-use of viterbi.continuity below) -- only
by dev/tuning/eval scripts. Accuracy numbers are computed here, not inside
the decoder.

Point labels come from A's precomputed v1 reference labels in
data/processed/points/<trip>.parquet (one row per .track row, in obs_idx
order); this module reads them, it never recomputes them. Reference routes
are read directly from <raw_folder>/<trip_id>.route, the same dev-only
pattern already used in pathsnap/hmm/batch_decode.py and run_one_trip.py.
"""
import numpy as np
import pandas as pd

from pathsnap.hmm.viterbi import continuity

PROCESSED_DIR = "data/processed"

__all__ = ["load_point_labels", "load_reference_route", "top1",
           "route_precision_recall", "continuity"]


def load_point_labels(trip_id, processed_dir=PROCESSED_DIR):
    """A's v1 per-observation reference labels, row order = obs_idx."""
    df = pd.read_parquet(f"{processed_dir}/points/{trip_id}.parquet")
    return df.sort_values("obs_idx").reset_index(drop=True)


def load_reference_route(folder, trip_id):
    """Ordered reference arc ids from <folder>/<trip_id>.route. Dev-only."""
    return np.loadtxt(f"{folder}/{trip_id}.route", dtype=np.int64, ndmin=1)


def top1(pred_arcs, labels):
    """Observation-level Top-1 accuracy.

    Points with no usable label (flag_invalid -> label_arc == -1) are
    excluded from the denominator. Points where the HMM had no candidate
    (pred_arcs == -1) or whose candidate set never included the true arc
    both land as misses automatically, since pred can't equal label_arc --
    matching data_processing/README.md's "keep the rest in evaluation,
    counted as misses" rule without needing label_in_candidates here.
    """
    label_arc = labels["label_arc"].to_numpy()
    usable = label_arc >= 0
    if not usable.any():
        return float("nan")
    pred_arcs = np.asarray(pred_arcs)[usable]
    return float(np.mean(pred_arcs == label_arc[usable]))


def route_precision_recall(net, pred_route, ref_route):
    """Length-weighted precision/recall over the SET of arcs used (a loop
    that revisits an arc counts its length once, not per visit):
        precision = length shared with reference / length of predicted route
        recall    = length shared with reference / length of reference route
    """
    pred_arcs = sorted(set(int(a) for a in pred_route))
    ref_arcs = sorted(set(int(a) for a in ref_route))
    shared = sorted(set(pred_arcs) & set(ref_arcs))

    pred_len = float(net.length[pred_arcs].sum()) if pred_arcs else 0.0
    ref_len = float(net.length[ref_arcs].sum()) if ref_arcs else 0.0
    shared_len = float(net.length[shared].sum()) if shared else 0.0

    precision = shared_len / pred_len if pred_len > 0 else float("nan")
    recall = shared_len / ref_len if ref_len > 0 else float("nan")
    return precision, recall
