"""Optuna TPE tuning of the distance-only decoder: sigma, beta, uturn_penalty.

Uses the task-2 cache (pathsnap.hmm.decode_cache) so each trial is pure
NumPy -- no Dijkstra calls -- scored against A's v1 reference labels
(pathsnap.evaluation.metrics). gap_s, alpha, radius, no_path_logp stay
fixed at the artifacts/hmm_config.json defaults; only the three geometry
weights are tuned, on TRAINING trips only.

    python -m pathsnap.hmm.tune_decoder

Requires results/cache/<trip>.npz for every training trip
(python -m pathsnap.hmm.decode_cache data/train) to already exist.

Writes:
    results/optuna_decoder.db           sqlite study (sampler=TPE, seed=42)
    results/optuna_decoder_trials.csv   one row per trial
"""
from pathlib import Path

import numpy as np
import optuna

from pathsnap.hmm.decode_cache import CACHE_DIR, load_cache, score_trip
from pathsnap.evaluation.metrics import load_point_labels, top1
from pathsnap.predict import load_config

TRAIN_DIR = Path("data/train")
N_TRIALS = 60
SEED = 42
DB_PATH = Path("results/optuna_decoder.db")
TRIALS_CSV = Path("results/optuna_decoder_trials.csv")
STUDY_NAME = "decoder_v1_distance_only"


def _load_training_set():
    """Preload every training trip's cache + label array once (pure NumPy
    after this, no file I/O inside the Optuna loop)."""
    trips = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    data = []
    for name in trips:
        cache_path = CACHE_DIR / f"{name}.npz"
        if not cache_path.exists():
            continue
        cache = load_cache(name)
        labels = load_point_labels(name)
        n_points = len(labels)
        data.append((name, cache, labels, n_points))
    return data


def make_objective(training_set, no_path_logp):
    def objective(trial):
        sigma = trial.suggest_float("sigma", 2.0, 50.0, log=True)
        beta = trial.suggest_float("beta", 0.5, 50.0, log=True)
        uturn_penalty = trial.suggest_float("uturn_penalty", 0.0, 10.0)

        scores = []
        for name, cache, labels, n_points in training_set:
            idx, arc = score_trip(cache, sigma, beta, uturn_penalty, no_path_logp)
            point_arcs = np.full(n_points, -1, dtype=np.int64)
            point_arcs[idx] = arc
            scores.append(top1(point_arcs, labels))
        return float(np.mean(scores))
    return objective


def main():
    cfg = load_config()
    print("loading training cache + labels ...")
    training_set = _load_training_set()
    print(f"loaded {len(training_set)} training trips with a cache")
    missing = len(sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())) - len(training_set)
    if missing:
        print(f"WARNING: {missing} training trips have no cache and were skipped")

    Path("results").mkdir(exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{DB_PATH}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=False,
    )
    objective = make_objective(training_set, cfg["no_path_logp"])
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    print("best value:", study.best_value)
    print("best params:", study.best_params)

    df = study.trials_dataframe()
    df.to_csv(TRIALS_CSV, index=False)
    print(f"wrote {TRIALS_CSV}")
    print(f"wrote {DB_PATH}")
    return study


if __name__ == "__main__":
    main()
