"""Task 4-3 (time-boxed): choose alpha with NO new decoder Optuna study.
sigma/beta/uturn_penalty stay at whatever is already in hmm_config.json.

2-fold out-of-fold classifier probabilities for the 80 training trips:
split trips into two groups, fit the tuned LightGBM on one group's rows
and score the OTHER group's full (unsampled) candidate tables, and vice
versa -- so every training trip's probabilities come from a model that
never saw that trip, covering every point (not just B's 1-in-3 CV rows).

    python -m pathsnap.classifier.alpha_grid

Writes results/alpha_grid.csv.
"""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from features.build_features import FEATURES
from models import zoo
from models.common import list_trips, load_split
from pathsnap.classifier.tune_classifier import DB_PATH, STEP, STUDY_NAME
from pathsnap.evaluation.metrics import load_point_labels, top1
from pathsnap.hmm.decode_cache import CACHE_DIR, load_cache, score_trip
from pathsnap.hmm.prepared_trip import prepare_trip
from pathsnap.hmm.routing import load_trip
from pathsnap.hmm.viterbi import build_route
from pathsnap.predict import load_config

FEATURES_DIR = Path("data/features")
TRAIN_DIR = Path("data/train")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
OUT_CSV = Path("results/alpha_grid.csv")


def split_trips(trips, seed=42):
    t = sorted(trips)
    rng = np.random.RandomState(seed)
    rng.shuffle(t)
    half = len(t) // 2
    return t[:half], t[half:]


def fit_on(trips, params):
    tr = load_split(FEATURES_DIR, trips, step=STEP, usable_only=True)
    X = tr[FEATURES].to_numpy(np.float32)
    y = tr["label"].to_numpy(np.int8)
    g = tr["g"].to_numpy()
    trip = pd.factorize(tr["trip_id"])[0]
    return zoo.build("lgbm", **params).fit(X, y, g=g, trip=trip)


def score_full_trip(model, trip_id):
    df = pd.read_parquet(FEATURES_DIR / f"{trip_id}.parquet")
    X = df[FEATURES].to_numpy(np.float32)
    return df["obs_idx"].to_numpy(np.int64), df["cand_arc"].to_numpy(np.int64), model.score(X)


def aligned_cand_prob(geo_cache, obs_idx, cand_arc, prob):
    point_idx = np.repeat(geo_cache["seg_idx"], geo_cache["n_cands"])
    lookup = {(int(o), int(a)): float(p) for o, a, p in zip(obs_idx, cand_arc, prob)}
    return np.array([lookup[(int(o), int(a))] for o, a in zip(point_idx, geo_cache["cand_arc"])])


def main():
    cfg = load_config()
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}")
    best_params = study.best_params
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"classifier study: {n_complete} completed trials, using best: {best_params}")
    print(f"decoder params held fixed: sigma={cfg['sigma']:.4f} beta={cfg['beta']:.4f} "
          f"uturn_penalty={cfg['uturn_penalty']:.4f}")

    sp = list_trips(FEATURES_DIR)
    g0, g1 = split_trips(sp["train"])
    print(f"OOF split: group0={len(g0)} trips, group1={len(g1)} trips")

    model_on_g0 = fit_on(g0, best_params)   # scores group1 (out-of-fold)
    model_on_g1 = fit_on(g1, best_params)   # scores group0 (out-of-fold)

    trip_data = []
    for trips, model in [(g1, model_on_g0), (g0, model_on_g1)]:
        for name in trips:
            if not (CACHE_DIR / f"{name}.npz").exists():
                continue
            folder = str(TRAIN_DIR / name)
            track, nodes, arcs = load_trip(folder, name)
            net, _, _, _, _, _ = prepare_trip(track, nodes, arcs, radius=cfg["radius"], gap_s=cfg["gap_s"])
            geo = load_cache(name)
            obs_idx, cand_arc, prob = score_full_trip(model, name)
            cand_prob = aligned_cand_prob(geo, obs_idx, cand_arc, prob)
            labels = load_point_labels(name)
            ref_route = np.loadtxt(f"{folder}/{name}.route", dtype=np.int64, ndmin=1)
            trip_data.append(dict(name=name, net=net, cache=geo, cand_prob=cand_prob,
                                  labels=labels, n_points=len(labels), ref_route=ref_route))
    print(f"OOF-scored {len(trip_data)} training trips")

    rows = []
    for alpha in ALPHAS:
        top1_scores, total_disc = [], 0
        for trip in trip_data:
            idx, arc = score_trip(trip["cache"], cfg["sigma"], cfg["beta"], cfg["uturn_penalty"],
                                  cfg["no_path_logp"], cand_prob=trip["cand_prob"], alpha=alpha)
            point_arcs = np.full(trip["n_points"], -1, dtype=np.int64)
            point_arcs[idx] = arc
            top1_scores.append(top1(point_arcs, trip["labels"]))
            route, disc = build_route(trip["net"], point_arcs, cfg["connector_limit"])
            total_disc += disc
            trip["net"].clear_cache()
        mean_top1 = float(np.mean(top1_scores))
        mean_disc = total_disc / len(trip_data)
        objective = mean_top1 - 0.002 * mean_disc
        rows.append(dict(alpha=alpha, mean_top1=round(mean_top1, 4),
                         mean_disconnections=round(mean_disc, 4), objective=round(objective, 4)))
        print(rows[-1])

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    best_alpha = float(df.loc[df["objective"].idxmax(), "alpha"])
    print(f"wrote {OUT_CSV}; best alpha = {best_alpha}")
    return best_alpha


if __name__ == "__main__":
    main()
