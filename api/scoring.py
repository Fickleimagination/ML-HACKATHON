import numpy as np

from api.predictor import arc_ids
from data_processing.preprocess import preprocess_trip, reference_labels


def load_reference(stem, arc_count):
    return arc_ids(np.loadtxt(stem.with_suffix(".route"), dtype=np.int64, ndmin=1), arc_count)


def continuity(ids, arcs):
    before, after = ids[:-1], ids[1:]
    comparable = (before >= 0) & (after >= 0)
    if not comparable.any():
        return None
    before, after = before[comparable], after[comparable]
    return float(np.mean((before == after) | (arcs[before, 1] == arcs[after, 0])))


def candidate_scores(labels, predicted, candidates):
    if candidates is None or not len(labels):
        return {"macro_f1": None, "candidate_miss_rate": None}
    tp = fp = fn = tn = misses = 0
    for label, selected, row in zip(labels, predicted, candidates, strict=True):
        row = np.unique(row)
        has_label = label in row
        has_selected = selected in row
        correct = has_label and label == selected
        tp += int(correct)
        fp += int(has_selected and not correct)
        fn += int(has_label and not correct)
        tn += len(row) - int(has_label) - int(has_selected) + int(correct)
        misses += int(not has_label)
    if not tp + fp + fn + tn:
        macro = None
    else:
        positive = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        negative = 2 * tn / (2 * tn + fp + fn) if tn + fp + fn else 0.0
        macro = (positive + negative) / 2
    return {"macro_f1": macro, "candidate_miss_rate": misses / len(labels)}


def evaluate(stem, track, nodes, arcs, predicted, route, candidates):
    metrics = {
        "top1_accuracy": None,
        "top1_high_confidence": None,
        "macro_f1": None,
        "candidate_miss_rate": None,
        "continuity": continuity(predicted, arcs),
        "route_continuity": continuity(route, arcs),
        "route_precision": None,
        "route_recall": None,
    }
    if not stem.with_suffix(".route").is_file():
        return metrics

    reference = load_reference(stem, len(arcs))
    prep = preprocess_trip(track, nodes, arcs)
    aligned = reference_labels(prep, reference)
    labels = aligned["label_arc"].to_numpy()
    labelled = labels >= 0
    high_confidence = labelled & ~aligned["label_low_conf"].to_numpy()
    matches = labels == predicted
    lengths = prep.index.length
    labelled_candidates = (
        [candidates[i] for i in np.flatnonzero(labelled)] if candidates is not None else None
    )
    unique_predicted, unique_reference = np.unique(route), np.unique(reference)
    matched = lengths[np.intersect1d(unique_predicted, unique_reference)].sum()
    predicted_length = lengths[unique_predicted].sum()
    reference_length = lengths[unique_reference].sum()
    metrics.update(
        top1_accuracy=float(matches[labelled].mean()) if labelled.any() else None,
        top1_high_confidence=float(matches[high_confidence].mean()) if high_confidence.any() else None,
        route_precision=float(matched / predicted_length) if predicted_length else None,
        route_recall=float(matched / reference_length) if reference_length else None,
        **candidate_scores(labels[labelled], predicted[labelled], labelled_candidates),
    )
    return metrics
