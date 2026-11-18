"""The seven models (Person B). build(name, **overrides) -> wrapper.

Every wrapper has: fit(X, y, g=, trip=), score(X), .proba, .family, .config
Ranker scores are NOT probabilities: Person C must softmax them per GPS point
before using them as HMM emissions.
"""
import copy
import json

import numpy as np
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ALL = ["lr", "gnb", "rf", "knn", "lgbm", "lgbm_rank", "mlp"]


class Sk:
    proba = True

    def __init__(self, est, family, params):
        self.est, self.family = est, family
        self.config = json.dumps(params, default=str)

    def fit(self, X, y, **kw):
        self.est.fit(X, y)
        return self

    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class Knn(Sk):
    N_FIT, BATCH = 50_000, 50_000

    def fit(self, X, y, **kw):
        if len(X) > self.N_FIT:
            i = np.random.RandomState(SEED).choice(len(X), self.N_FIT, replace=False)
            X, y = X[i], y[i]
        self.est.fit(X, y)
        return self

    def score(self, X):
        return np.concatenate([self.est.predict_proba(X[i:i + self.BATCH])[:, 1]
                               for i in range(0, len(X), self.BATCH)])


class Ranker:
    proba = False
    family = "Learning to rank"

    def __init__(self, params):
        self.est = LGBMRanker(**params)
        self.config = json.dumps(params, default=str)

    def fit(self, X, y, g=None, **kw):
        edges = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
        self.est.fit(X, y, group=np.diff(edges))   # one query group per GPS point
        return self

    def score(self, X):
        return self.est.predict(X)


class MlpHist:
    """sklearn MLP trained epoch by epoch so we get REAL train/val loss curves.
    Validation = 15% of the training TRIPS (group-aware). Train loss from sklearn
    includes the L2 term; val loss is plain log loss."""
    proba = True
    family = "Neural network"

    def __init__(self, hidden=(64, 32), epochs=40, patience=5, lr=1e-3,
                 alpha=1e-4, batch=1024):
        self.p = dict(hidden=hidden, epochs=epochs, patience=patience,
                      lr=lr, alpha=alpha, batch=batch)
        self.config = json.dumps(self.p)
        self.history = []

    def fit(self, X, y, trip=None, **kw):
        p = self.p
        trips = np.unique(trip)
        rng = np.random.RandomState(SEED)
        val_t = rng.choice(trips, max(1, int(0.15 * len(trips))), replace=False)
        vm = np.isin(trip, val_t)
        self.sc = StandardScaler().fit(X[~vm])
        Xa, Xb = self.sc.transform(X[~vm]), self.sc.transform(X[vm])
        ya, yb = y[~vm], y[vm]
        self.net = MLPClassifier(hidden_layer_sizes=p["hidden"], batch_size=p["batch"],
                                 learning_rate_init=p["lr"], alpha=p["alpha"],
                                 random_state=SEED)
        best, bad, state = np.inf, 0, None
        self.history = []
        for ep in range(p["epochs"]):
            self.net.partial_fit(Xa, ya, classes=[0, 1])   # one epoch
            vl = log_loss(yb, np.clip(self.net.predict_proba(Xb)[:, 1], 1e-6, 1 - 1e-6),
                          labels=[0, 1])
            self.history.append({"epoch": ep + 1,
                                 "train_loss": float(self.net.loss_curve_[-1]),
                                 "val_loss": float(vl)})
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = (copy.deepcopy(self.net.coefs_), copy.deepcopy(self.net.intercepts_))
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        self.net.coefs_, self.net.intercepts_ = state   # restore best epoch
        return self

    def score(self, X):
        return self.net.predict_proba(self.sc.transform(X))[:, 1]


def build(name, **o):
    if name == "lr":
        p = dict(C=1.0, max_iter=300) | o
        return Sk(make_pipeline(StandardScaler(), LogisticRegression(**p)),
                  "Linear", p)
    if name == "gnb":
        return Sk(GaussianNB(**o), "Probabilistic", o)
    if name == "rf":
        p = dict(n_estimators=150, max_depth=20, min_samples_leaf=20,
                 max_samples=0.3, n_jobs=-1, random_state=SEED) | o
        return Sk(RandomForestClassifier(**p), "Ensemble (bagging)", p)
    if name == "knn":
        p = dict(n_neighbors=25, weights="distance", n_jobs=-1) | o
        return Knn(make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
                   "Instance-based", p)
    lgb = dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
               min_child_samples=50, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1) | o
    if name == "lgbm":
        return Sk(LGBMClassifier(**lgb), "Boosting", lgb)
    if name == "lgbm_rank":
        return Ranker(lgb | dict(objective="lambdarank"))
    if name == "mlp":
        return MlpHist(**o)
    raise ValueError(name)
