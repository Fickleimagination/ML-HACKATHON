"""Task 4-3: refit the Optuna-tuned LightGBM on all 80 training trips, and
compare it against B's existing untuned artifacts/models/lgbm.joblib on
student_test.

Never overwrites lgbm.joblib -- writes a new file, artifacts/models/
lgbm_tuned.joblib, in B's own joblib format ({"model", "features", "name"})
so pathsnap/predict.py and B's own tooling can both load it unchanged.

The untuned row's CV/test metrics are B's own already-computed numbers from
results/models/lgbm_metrics.json (the actual evaluation of lgbm.joblib,
not a fresh refit of the same hyperparameters) -- this script additionally
re-scores the existing lgbm.joblib artifact on the freshly rebuilt
data/features test rows as a drift check (logged, not re-fit).

    python -m pathsnap.classifier.refit_compare

Requires results/optuna_classifier.db (python -m pathsnap.classifier.tune_classifier).

Writes:
    artifacts/models/lgbm_tuned.joblib
    results/classifier_tuned_vs_untuned.csv
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from features.build_features import FEATURES
from models import zoo
from models.common import list_trips, load_split, metrics, route_continuity
from pathsnap.classifier.tune_classifier import DB_PATH, N_SPLITS, STEP, STUDY_NAME

FEATURES_DIR = Path("data/features")
UNTUNED_PATH = Path("artifacts/models/lgbm.joblib")
TUNED_PATH = Path("artifacts/models/lgbm_tuned.joblib")
UNTUNED_METRICS_JSON = Path("results/models/lgbm_metrics.json")
OUT_CSV = Path("results/classifier_tuned_vs_untuned.csv")


def cv_macro_f1_top1_logloss(params, X, y, g, trip):
    folds = []
    for a_i, b_i in GroupKFold(n_splits=N_SPLITS).split(X, y, trip):
        m = zoo.build("lgbm", **params).fit(X[a_i], y[a_i], g=g[a_i], trip=trip[a_i])
        folds.append(metrics(y[b_i], g[b_i], m.score(X[b_i]), m.proba))
    return (float(np.mean([f["macro_f1"] for f in folds])),
            float(np.mean([f["top1"] for f in folds])),
            float(np.mean([f["log_loss"] for f in folds])))


def main():
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}")
    best_params = study.best_params
    print("tuned params:", best_params)

    sp = list_trips(FEATURES_DIR)
    tr = load_split(FEATURES_DIR, sp["train"], step=STEP, usable_only=True)
    te = load_split(FEATURES_DIR, sp["student_test"], step=1)
    X, y, g = tr[FEATURES].to_numpy(np.float32), tr["label"].to_numpy(np.int8), tr["g"].to_numpy()
    trip = pd.factorize(tr["trip_id"])[0]
    Xt, yt, gt = te[FEATURES].to_numpy(np.float32), te["label"].to_numpy(np.int8), te["g"].to_numpy()
    print(f"train rows (all 80 trips) {len(X)}, test rows {len(Xt)}")

    # tuned: full CV (needed, can't be read off disk) + refit on all 80 training trips
    cv_f1, cv_top1, cv_ll = cv_macro_f1_top1_logloss(best_params, X, y, g, trip)
    t0 = time.perf_counter()
    m_tuned = zoo.build("lgbm", **best_params).fit(X, y, g=g, trip=trip)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_tuned = m_tuned.score(Xt)
    inf_tuned = time.perf_counter() - t0
    te_tuned = metrics(yt, gt, s_tuned, m_tuned.proba)
    cont_tuned = route_continuity(te, s_tuned)
    Path("artifacts/models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m_tuned, "features": FEATURES, "name": "lgbm_tuned"}, TUNED_PATH)
    print(f"wrote {TUNED_PATH} (refit on all 80 training trips, {fit_time:.1f}s)")

    # untuned: B's own already-computed numbers for lgbm.joblib, plus a drift
    # check re-scoring the SAME saved artifact on the freshly rebuilt test rows
    b_metrics = json.loads(UNTUNED_METRICS_JSON.read_text())
    bundle = joblib.load(UNTUNED_PATH)
    m_untuned = bundle["model"]
    t0 = time.perf_counter()
    s_untuned = m_untuned.score(Xt)
    inf_untuned = time.perf_counter() - t0
    te_untuned_now = metrics(yt, gt, s_untuned, m_untuned.proba)
    cont_untuned_now = route_continuity(te, s_untuned)
    drift = abs(te_untuned_now["top1"] - b_metrics["test_top1"])
    print(f"untuned lgbm.joblib test_top1 drift vs stored metrics.json: {drift:.5f} "
          f"({'OK' if drift < 1e-6 else 'rebuilt data/features differs from what B trained on!'})")

    rows = [
        dict(model="lgbm_untuned", source="artifacts/models/lgbm.joblib (B's existing fit)",
             cv_macro_f1=round(b_metrics["cv_macro_f1"], 4), cv_top1=round(b_metrics["cv_top1"], 4),
             cv_log_loss=round(b_metrics["cv_log_loss"], 4),
             test_top1=round(b_metrics["test_top1"], 4), test_macro_f1=round(b_metrics["test_macro_f1"], 4),
             test_log_loss=round(b_metrics["test_log_loss"], 4),
             route_continuity=round(b_metrics["route_continuity"], 4),
             inference_ms_per_1k_rows=round(b_metrics["infer_ms_per_1k_rows"], 3)),
        dict(model="lgbm_untuned_rescored", source="same artifact, re-scored on rebuilt data/features (drift check)",
             cv_macro_f1=float("nan"), cv_top1=float("nan"), cv_log_loss=float("nan"),
             test_top1=round(te_untuned_now["top1"], 4), test_macro_f1=round(te_untuned_now["macro_f1"], 4),
             test_log_loss=round(te_untuned_now["log_loss"], 4),
             route_continuity=round(cont_untuned_now, 4),
             inference_ms_per_1k_rows=round(1e6 * inf_untuned / len(Xt), 3)),
        dict(model="lgbm_tuned", source="artifacts/models/lgbm_tuned.joblib (this script, refit on all 80 trips)",
             cv_macro_f1=round(cv_f1, 4), cv_top1=round(cv_top1, 4), cv_log_loss=round(cv_ll, 4),
             test_top1=round(te_tuned["top1"], 4), test_macro_f1=round(te_tuned["macro_f1"], 4),
             test_log_loss=round(te_tuned["log_loss"], 4),
             route_continuity=round(cont_tuned, 4),
             inference_ms_per_1k_rows=round(1e6 * inf_tuned / len(Xt), 3)),
    ]
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
