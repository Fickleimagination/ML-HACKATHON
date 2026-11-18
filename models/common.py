"""Shared loading + metrics (Person B)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

from features.build_features import FEATURES

SEED = 42


def list_trips(feat_dir):
    """{'train': [...], 'student_test': [...]} read from the partition column."""
    parts = {}
    for f in sorted(Path(feat_dir).glob("*.parquet")):
        p = pd.read_parquet(f, columns=["partition"])["partition"].iloc[0]
        parts.setdefault(p, []).append(f.stem)
    return parts


def load_split(feat_dir, trips, step=1, usable_only=False):
    """step=3 keeps every 3rd GPS point; usable_only drops untrusted-label points."""
    parts = []
    for tid in trips:
        df = pd.read_parquet(Path(feat_dir) / f"{tid}.parquet")
        if step > 1:
            df = df[df["obs_idx"] % step == 0]
        if usable_only:
            df = df[df["usable"]]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["trip_id", "obs_idx"], kind="stable").reset_index(drop=True)
    codes = pd.factorize(df["trip_id"])[0].astype(np.int64)
    df["g"] = codes * 10_000_000 + df["obs_idx"].astype(np.int64)  # one id per GPS point
    return df


def top1_stats(y, g, s):
    """Observation-level Top-1: best-scored candidate of each point == label.
    Points whose label arc is not a candidate count as misses."""
    d = pd.DataFrame({"g": g, "s": s, "y": y})
    best_idx = d.groupby("g")["s"].idxmax().to_numpy()   # positions (RangeIndex)
    best = d.loc[best_idx]
    has_pos = d.groupby("g")["y"].max()
    n, npos = len(best), int(has_pos.sum())
    hits = int((best["y"] == 1).sum())
    return {"top1": hits / n,
            "top1_given_candidate": hits / max(npos, 1),
            "miss_rate": 1 - npos / n}, best_idx


def metrics(y, g, s, proba):
    t1, best_idx = top1_stats(y, g, s)
    pick = np.zeros(len(y), dtype=int)
    pick[best_idx] = 1
    out = dict(t1)
    out["macro_f1_pick"] = f1_score(y, pick, average="macro")   # defined for ALL models
    if proba:   # thresholded at 0.5 on the candidate probability
        yh = (s >= 0.5).astype(int)
        out["macro_f1"] = f1_score(y, yh, average="macro")
        out["macro_precision"] = precision_score(y, yh, average="macro", zero_division=0)
        out["macro_recall"] = recall_score(y, yh, average="macro", zero_division=0)
        out["log_loss"] = log_loss(y, np.clip(s, 1e-6, 1 - 1e-6), labels=[0, 1])
        out["confusion"] = confusion_matrix(y, yh, labels=[0, 1]).tolist()
    else:       # ranker: raw scores are not probabilities
        for k in ("macro_f1", "macro_precision", "macro_recall", "log_loss"):
            out[k] = float("nan")
        out["confusion"] = confusion_matrix(y, pick, labels=[0, 1]).tolist()
    return out


def route_continuity(df, s):
    """Share of consecutive (obs_idx+1) predicted arcs that are the same or connected
    (head node of arc i == tail node of arc i+1). Uses from/to nodes stored in df."""
    d = pd.DataFrame({"g": df["g"].values, "s": s, "trip": df["trip_id"].values,
                      "obs": df["obs_idx"].values, "arc": df["cand_arc"].values,
                      "fr": df["from_node"].values, "to": df["to_node"].values})
    best = d.loc[d.groupby("g")["s"].idxmax()].sort_values("g")
    trip = best["trip"].to_numpy()
    obs = best["obs"].to_numpy()
    arc, fr, to = best["arc"].to_numpy(), best["fr"].to_numpy(), best["to"].to_numpy()
    m = (trip[:-1] == trip[1:]) & (np.diff(obs) == 1)
    good = (arc[:-1] == arc[1:]) | (to[:-1] == fr[1:])
    return float(good[m].sum() / max(m.sum(), 1))
