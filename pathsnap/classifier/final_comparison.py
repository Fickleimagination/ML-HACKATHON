"""Task 4-4 (time-boxed): nearest / classifier-only / distance-only HMM /
classifier+HMM on student_test.

Route-level metrics (continuity, route precision/recall, disconnections)
are computed the same way for all four configs by running build_route
(network-aware connector search) over each config's per-point arc choice,
so "disconnections" is comparable across rows -- a naive dedup-only route
(as used for the "nearest" baseline in earlier tasks) never attempts a
connector, so it wouldn't have a meaningful disconnection count here.

    python -m pathsnap.classifier.final_comparison

Writes results/final_comparison.csv and results/figures/final_comparison.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pathsnap.evaluation.metrics import (continuity, load_point_labels,
                                          load_reference_route,
                                          route_precision_recall, top1)
from pathsnap.hmm.prepared_trip import prepare_trip
from pathsnap.hmm.routing import load_trip
from pathsnap.hmm.viterbi import build_route, decode_trip
from pathsnap.predict import _classifier_probs, load_config

STUDENT_TEST_DIR = Path("data/student_test")
FEATURES_DIR = Path("data/features")
OUT_CSV = Path("results/final_comparison.csv")
OUT_PNG = Path("results/figures/final_comparison.png")


def macro_f1_pick(name, chosen_arc):
    """B's macro_f1_pick, defined for any method: 1 on the candidate row
    that was picked for its point, compared to the true label -- no
    continuous score needed, so it works for nearest/HMM picks too."""
    df = pd.read_parquet(FEATURES_DIR / f"{name}.parquet")
    y = df["label"].to_numpy()
    obs_idx = df["obs_idx"].to_numpy()
    cand_arc = df["cand_arc"].to_numpy()
    chosen = np.array([chosen_arc.get(int(o), -1) for o in obs_idx])
    pick = (cand_arc == chosen).astype(int)
    return f1_score(y, pick, average="macro")


def row_for(name, label, net, point_arcs, labels, ref_route):
    route, disconnections = build_route(net, point_arcs, cfg["connector_limit"])
    chosen = {t: int(a) for t, a in enumerate(point_arcs) if a >= 0}
    prec, rec = route_precision_recall(net, route, ref_route)
    return dict(trip=name, config=label,
                top1=round(top1(point_arcs, labels), 4),
                macro_f1=round(macro_f1_pick(name, chosen), 4),
                continuity=round(continuity(net, route), 4),
                route_precision=round(prec, 4), route_recall=round(rec, 4),
                disconnections=disconnections)


def main():
    global cfg
    cfg = load_config()
    cfg_clf = {**cfg, "model_path": "artifacts/models/lgbm_tuned.joblib"}
    alpha_df = pd.read_csv("results/alpha_grid.csv")
    best_alpha = float(alpha_df.loc[alpha_df["objective"].idxmax(), "alpha"])
    cfg_hmm_clf = {**cfg_clf, "alpha": best_alpha}
    print(f"using alpha={best_alpha} from results/alpha_grid.csv")

    rows = []
    for trip_dir in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        name = trip_dir.name
        folder = str(trip_dir)
        track, nodes, arcs = load_trip(folder, name)
        net, xy, times, cands, tree, prep = prepare_trip(track, nodes, arcs,
                                                          radius=cfg["radius"], gap_s=cfg["gap_s"])
        labels = load_point_labels(name)
        ref_route = load_reference_route(folder, name)

        probs = _classifier_probs(prep, arcs, cfg_clf, cands)

        nearest = np.array([min(c, key=lambda x: x[1])[0] if c else -1 for c in cands])
        rows.append(row_for(name, "nearest", net, nearest, labels, ref_route))
        net.clear_cache()

        clf_only = np.array([cs[int(np.argmax(pr))][0] if cs else -1
                             for cs, pr in zip(cands, probs)])
        rows.append(row_for(name, "classifier_only", net, clf_only, labels, ref_route))
        net.clear_cache()

        pa_dist, _, _ = decode_trip(net, xy, times, cands, probs=None, cfg={**cfg, "alpha": 0.0})
        rows.append(row_for(name, "distance_hmm", net, pa_dist, labels, ref_route))
        net.clear_cache()

        pa_clf, _, _ = decode_trip(net, xy, times, cands, probs=probs, cfg=cfg_hmm_clf)
        rows.append(row_for(name, "classifier_hmm", net, pa_clf, labels, ref_route))
        net.clear_cache()

    df = pd.DataFrame(rows)
    means = (df.groupby("config")[["top1", "macro_f1", "continuity", "route_precision",
                                    "route_recall", "disconnections"]]
            .mean().round(4).reset_index())
    means.insert(0, "trip", "MEAN")
    means = means[df.columns]
    out = pd.concat([df, means], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"wrote {OUT_CSV}")

    order = ["nearest", "classifier_only", "distance_hmm", "classifier_hmm"]
    m = means.set_index("config").loc[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    ax.bar(x - 0.2, m["top1"], 0.4, label="Top-1")
    ax.bar(x + 0.2, m["macro_f1"], 0.4, label="macro F1 (pick)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel("score")
    ax.set_title(f"Final comparison -- student_test (alpha={best_alpha})")
    ax.legend()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
