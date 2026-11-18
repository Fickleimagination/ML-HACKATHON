"""Train + grouped-CV + student-test evaluation (Person B).

Phase 1:  python -m models.train_models --models lr gnb rf
Later:    python -m models.train_models --models knn lgbm lgbm_rank mlp

Reads data/features/*.parquet (partition column = train / student_test).
Training + CV use only train-partition points with usable == True, every 3rd point.
Student-test is scored on ALL its points (unusable ones count as misses).
Evaluator-test trips are never present.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from models import zoo
from models.common import (FEATURES, SEED, list_trips, load_split, metrics,
                           route_continuity)


def nm(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def sd(v):
    v = [x for x in v if x == x]
    return float(np.std(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts/models"))
    ap.add_argument("--models", nargs="+", default=zoo.ALL)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--exclude", nargs="*", default=[], help="train trips to drop (e.g. shifted trip 33)")
    a = ap.parse_args()

    out = Path(a.out) / "models"
    out.mkdir(parents=True, exist_ok=True)
    a.artifacts.mkdir(parents=True, exist_ok=True)
    sp = list_trips(a.features)
    print({k: len(v) for k, v in sp.items()})
    train_trips = [t for t in sp["train"] if t not in set(a.exclude)]

    tr = load_split(a.features, train_trips, step=a.step, usable_only=True)
    te = load_split(a.features, sp["student_test"], step=1)   # all points for test
    X, y, g = tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy(np.int8), tr["g"].to_numpy()
    trip = pd.factorize(tr["trip_id"])[0]
    Xt, yt, gt = te[FEATURES].to_numpy(np.float32), te["label"].to_numpy(np.int8), te["g"].to_numpy()
    print(f"train rows {len(X)}, test rows {len(Xt)}")

    for name in a.models:
        print(f"== {name}")
        folds = []
        t0 = time.perf_counter()
        for a_i, b_i in GroupKFold(n_splits=5).split(X, y, trip):
            m = zoo.build(name).fit(X[a_i], y[a_i], g=g[a_i], trip=trip[a_i])
            folds.append(metrics(y[b_i], g[b_i], m.score(X[b_i]), m.proba))
        cv_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = zoo.build(name).fit(X, y, g=g, trip=trip)
        train_time = time.perf_counter() - t0

        # train-set Top-1 on 3000 random GPS points (overfitting evidence)
        rng = np.random.RandomState(SEED)
        sel = np.isin(g, rng.choice(np.unique(g), 3000, replace=False))
        tr_m = metrics(y[sel], g[sel], m.score(X[sel]), m.proba)

        t0 = time.perf_counter()
        s = m.score(Xt)
        inf = time.perf_counter() - t0
        te_m = metrics(yt, gt, s, m.proba)
        cont = route_continuity(te, s)

        res = {
            "model": name, "family": m.family, "config": m.config, "seed": SEED,
            "step": a.step, "n_train_rows": len(X), "n_test_rows": len(Xt),
            "train_time_s": train_time, "cv_time_s": cv_time,
            "infer_ms_per_1k_rows": 1e6 * inf / len(Xt),
            "train_top1": tr_m["top1"],
            "route_continuity": cont,
            "cv_fold_top1": [f["top1"] for f in folds],
        }
        for k in ("macro_f1", "macro_f1_pick", "top1", "log_loss"):
            res[f"cv_{k}"] = nm([f[k] for f in folds])
            res[f"cv_{k}_std"] = sd([f[k] for f in folds])
        for k, v in te_m.items():
            res[f"test_{k}"] = v
        json.dump(res, open(out / f"{name}_metrics.json", "w"), indent=1)
        if hasattr(m, "history"):
            json.dump(m.history, open(out / f"{name}_history.json", "w"), indent=1)
        joblib.dump({"model": m, "features": FEATURES, "name": name}, a.artifacts / f"{name}.joblib")
        print(f"cv_top1 {res['cv_top1']:.4f} | test_top1 {res['test_top1']:.4f} "
              f"| continuity {cont:.4f} | train_top1 {res['train_top1']:.4f}")


if __name__ == "__main__":
    main()
