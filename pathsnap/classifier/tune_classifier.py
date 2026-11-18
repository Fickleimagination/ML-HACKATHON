"""Optuna TPE tuning of B's LightGBM binary classifier.

Reuses B's own code end to end -- this file only imports features/ and
models/, never edits or copies them:
    features.build_features.FEATURES    the ordered feature list
    models.common.list_trips/load_split train-partition rows, step=3,
                                         usable_only=True (B's default CV rows)
    models.common.metrics               macro F1 / Top-1 / log loss
    models.zoo.build("lgbm", **params)  B's LGBMClassifier wrapper; a trial's
                                         params simply override B's defaults,
                                         exactly like train_models.py's --models
                                         CLI overrides would

    python -m pathsnap.classifier.tune_classifier

Requires data/features/*.parquet (python -m features.build_features ...).

Writes:
    results/optuna_classifier.db
    results/optuna_classifier_trials.csv
"""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

from features.build_features import FEATURES
from models import zoo
from models.common import SEED, list_trips, load_split, metrics

FEATURES_DIR = Path("data/features")
STEP = 3                     # B's default row sampling for training/CV
N_SPLITS = 5                  # B's GroupKFold split count
N_TRIALS = 40
DB_PATH = Path("results/optuna_classifier.db")
TRIALS_CSV = Path("results/optuna_classifier_trials.csv")
STUDY_NAME = "lgbm_classifier"


def load_cv_data():
    sp = list_trips(FEATURES_DIR)
    tr = load_split(FEATURES_DIR, sp["train"], step=STEP, usable_only=True)
    X = tr[FEATURES].to_numpy(np.float32)
    y = tr["label"].to_numpy(np.int8)
    g = tr["g"].to_numpy()
    trip = pd.factorize(tr["trip_id"])[0]
    return X, y, g, trip


def make_objective(X, y, g, trip):
    def objective(trial):
        params = dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=1,   # required for bagging_fraction to take effect; matches B's subsample_freq=1
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        )
        folds = []
        for a_i, b_i in GroupKFold(n_splits=N_SPLITS).split(X, y, trip):
            m = zoo.build("lgbm", **params).fit(X[a_i], y[a_i], g=g[a_i], trip=trip[a_i])
            folds.append(metrics(y[b_i], g[b_i], m.score(X[b_i]), m.proba))

        cv_macro_f1 = float(np.mean([f["macro_f1"] for f in folds]))
        trial.set_user_attr("cv_top1", float(np.mean([f["top1"] for f in folds])))
        trial.set_user_attr("cv_log_loss", float(np.mean([f["log_loss"] for f in folds])))
        return cv_macro_f1
    return objective


def main():
    print("loading CV rows (B's load_split: train partition, step=3, usable_only=True) ...")
    X, y, g, trip = load_cv_data()
    print(f"rows={len(X)}  trips={len(np.unique(trip))}")

    Path("results").mkdir(exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}",
                                direction="maximize", sampler=sampler, load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)
    if done:
        print(f"resuming study: {done} trials already complete, running {remaining} more "
              f"(kept below the tool's 10-minute cap by running in chunks if needed)")
    if remaining:
        study.optimize(make_objective(X, y, g, trip), n_trials=remaining, n_jobs=1)

    print("best value (cv_macro_f1):", study.best_value)
    print("best params:", study.best_params)
    print("best trial user attrs:", study.best_trial.user_attrs)

    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"].reset_index(drop=True)
    df.to_csv(TRIALS_CSV, index=False)
    print(f"wrote {TRIALS_CSV}")
    print(f"wrote {DB_PATH}")
    return study


if __name__ == "__main__":
    main()
