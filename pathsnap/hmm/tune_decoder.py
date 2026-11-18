"""Optuna TPE tuning of the distance-only decoder: sigma, beta, uturn_penalty.

Uses the decoder cache (pathsnap.hmm.decode_cache, built from A's aligned
candidates + 30s gap_s -- see pathsnap.hmm.prepared_trip) so each trial's
Viterbi pass is pure NumPy -- no Dijkstra -- scored against A's v1
reference labels (pathsnap.evaluation.metrics). gap_s, alpha, radius,
no_path_logp stay fixed at the artifacts/hmm_config.json defaults; only the
three geometry weights are tuned, on TRAINING trips only.

Route precision and disconnections DO need a RoadNetwork (connector search
is a bounded Dijkstra), but only run once per trip per trial at the handful
of points where the decoded arc sequence actually breaks -- cheap compared
to the candidate-pair Dijkstra matrix that's cached away -- so RoadNetwork
objects for all training trips are preloaded once and reused.

    python -m pathsnap.hmm.tune_decoder

Requires results/cache/<trip>.npz for every training trip
(python -m pathsnap.hmm.decode_cache data/train) to already exist.

Writes:
    results/optuna_decoder.db           sqlite study (sampler=TPE, seed=42)
    results/optuna_decoder_trials.csv   one row per trial, plus per-trial
                                         mean route precision and total
                                         disconnections as user attributes
"""
from pathlib import Path

import numpy as np
import optuna

from pathsnap.evaluation.metrics import load_point_labels, load_reference_route, route_precision_recall, top1
from pathsnap.hmm.decode_cache import CACHE_DIR, load_cache, score_trip
from pathsnap.hmm.prepared_trip import prepare_trip
from pathsnap.hmm.routing import load_trip
from pathsnap.hmm.viterbi import build_route
from pathsnap.predict import load_config

TRAIN_DIR = Path("data/train")
N_TRIALS = 60
SEED = 42
DB_PATH = Path("results/optuna_decoder.db")
TRIALS_CSV = Path("results/optuna_decoder_trials.csv")
STUDY_NAME = "decoder_v1_1_aligned"

SIGMA_RANGE = (0.5, 50.0)
BETA_RANGE = (0.5, 100.0)
UTURN_RANGE = (0.0, 30.0)


def _load_training_set(cfg):
    """Preload every training trip's cache, labels, RoadNetwork and
    reference route once (no file I/O inside the Optuna loop other than
    the occasional connector-search Dijkstra)."""
    trips = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    data = []
    for name in trips:
        cache_path = CACHE_DIR / f"{name}.npz"
        if not cache_path.exists():
            continue
        folder = str(TRAIN_DIR / name)
        track, nodes, arcs = load_trip(folder, name)
        net, _, _, _, _ = prepare_trip(track, nodes, arcs, radius=cfg["radius"], gap_s=cfg["gap_s"])
        cache = load_cache(name)
        labels = load_point_labels(name)
        ref_route = load_reference_route(folder, name)
        data.append(dict(name=name, net=net, cache=cache, labels=labels,
                          n_points=len(labels), ref_route=ref_route))
    return data


def make_objective(training_set, no_path_logp, connector_limit):
    def objective(trial):
        sigma = trial.suggest_float("sigma", *SIGMA_RANGE, log=True)
        beta = trial.suggest_float("beta", *BETA_RANGE, log=True)
        uturn_penalty = trial.suggest_float("uturn_penalty", *UTURN_RANGE)

        top1_scores, precisions, total_disconnections = [], [], 0
        for trip in training_set:
            idx, arc = score_trip(trip["cache"], sigma, beta, uturn_penalty, no_path_logp)
            point_arcs = np.full(trip["n_points"], -1, dtype=np.int64)
            point_arcs[idx] = arc
            top1_scores.append(top1(point_arcs, trip["labels"]))

            route, disconnections = build_route(trip["net"], point_arcs, connector_limit)
            total_disconnections += disconnections
            precision, _ = route_precision_recall(trip["net"], route, trip["ref_route"])
            if np.isfinite(precision):
                precisions.append(precision)
            trip["net"].clear_cache()

        trial.set_user_attr("mean_route_precision", float(np.mean(precisions)) if precisions else float("nan"))
        trial.set_user_attr("total_disconnections", int(total_disconnections))
        return float(np.mean(top1_scores))
    return objective


def main():
    cfg = load_config()
    print("loading training cache + labels + networks ...")
    training_set = _load_training_set(cfg)
    n_trip_dirs = len(sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir()))
    print(f"loaded {len(training_set)} training trips with a cache")
    missing = n_trip_dirs - len(training_set)
    if missing:
        print(f"WARNING: {missing} training trips have no cache and were skipped")

    Path("results").mkdir(exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{DB_PATH}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, N_TRIALS - done)
    if done:
        print(f"resuming study: {done} trials already complete, running {remaining} more "
              f"(each run below the tool's 10-minute cap, so a long study is done in chunks)")
    objective = make_objective(training_set, cfg["no_path_logp"], cfg["connector_limit"])
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1)

    print("best value:", study.best_value)
    print("best params:", study.best_params)
    print("best trial user attrs:", study.best_trial.user_attrs)

    df = study.trials_dataframe()
    df.to_csv(TRIALS_CSV, index=False)
    print(f"wrote {TRIALS_CSV}")
    print(f"wrote {DB_PATH}")
    return study


if __name__ == "__main__":
    main()
