"""Feature extraction (Person B), built on Person A's processed files.

Inputs : <processed>/candidates/<trip>.parquet, <processed>/points/<trip>.parquet
         raw <trip>.arcs  (only for node degree + reverse-arc existence)
Output : <out>/<trip>.parquet  (one row per point x candidate arc)

Run from repo root:
  python -m features.build_features --processed data/processed --raw data --out data/features
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "dist", "dist_rank", "dist_gap", "frac_along", "arc_len", "head_cos",
    "gps_moving", "deg_from", "deg_to", "has_reverse", "n_cands", "persist",
    "speed", "dt_prev", "dt_next",
]
WIN = 5  # persistence window: +-5 neighbouring points


def index_arcs(raw_root):
    """Index the raw arc files for training and student-test trips."""
    root = Path(raw_root)
    return {
        path.stem: path
        for partition in ("train", "student_test")
        for path in (root / partition).glob("*/*.arcs")
    }


def read_arcs(path):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=np.int64).to_numpy()


def trip_features(c, p, arcs, trip_id):
    pm = p[["obs_idx", "partition", "t", "heading_deg", "speed_ms", "dt_prev",
            "dt_next", "label_low_conf", "label_in_candidates"]]
    # newer candidate files may already carry some point-level columns: keep the points copy
    c = c.drop(columns=[x for x in pm.columns if x != "obs_idx" and x in c.columns])
    d = c.merge(pm, on="obs_idx", how="left", validate="many_to_one")
    d = d.sort_values(["obs_idx", "dist_m", "cand_arc"], kind="stable").reset_index(drop=True)

    o = d["obs_idx"].to_numpy(np.int64)
    k = d["cand_arc"].to_numpy(np.int64)
    fr = d["from_node"].to_numpy(np.int64)
    to = d["to_node"].to_numpy(np.int64)
    if not ((arcs[k, 0] == fr).all() and (arcs[k, 1] == to).all()):
        raise ValueError(f"{trip_id}: .arcs rows do not match candidate from/to nodes")

    # relative distance features
    g = d.groupby("obs_idx")["dist_m"]
    dist_rank = g.rank(method="first").to_numpy()
    dist_gap = (d["dist_m"] - g.transform("min")).to_numpy()
    n_cands = g.transform("size").to_numpy()

    # signed heading agreement: +1 same direction, -1 opposite (separates twin arcs)
    heading = d["heading_deg"]
    moving = heading.notna().to_numpy()
    diff = np.deg2rad((d["arc_bearing_deg"] - heading).to_numpy(np.float64))
    head_cos = np.where(moving, np.cos(np.nan_to_num(diff)), 0.0)

    # node degree + reverse-arc existence from the FULL network
    nn = int(max(arcs.max(), fr.max(), to.max())) + 1
    deg = np.bincount(arcs[:, 0], minlength=nn) + np.bincount(arcs[:, 1], minlength=nn)
    ekeys = np.sort(arcs[:, 0] * nn + arcs[:, 1])
    q = to * nn + fr
    pos = np.searchsorted(ekeys, q)
    pos[pos == len(ekeys)] = len(ekeys) - 1
    has_rev = ekeys[pos] == q

    # window persistence: same arc is also a candidate at +-WIN neighbouring points
    na = int(k.max()) + 1
    keys = o * na + k
    skeys = np.sort(keys)
    persist = np.zeros(len(keys), dtype=np.int16)
    for off in range(-WIN, WIN + 1):
        if off == 0:
            continue
        qq = keys + off * na
        ps = np.searchsorted(skeys, qq)
        ps[ps == len(skeys)] = len(skeys) - 1
        persist += (skeys[ps] == qq)

    dtp = d["dt_prev"].fillna(d["dt_next"]).fillna(1.0)
    dtn = d["dt_next"].fillna(d["dt_prev"]).fillna(1.0)

    out = pd.DataFrame({
        "trip_id": trip_id,
        "partition": d["partition"].astype(str),
        "obs_idx": o.astype(np.int32),
        "cand_arc": k.astype(np.int32),
        "from_node": fr.astype(np.int32),
        "to_node": to.astype(np.int32),
        "t": d["t"].to_numpy(),
        "dist": d["dist_m"], "dist_rank": dist_rank, "dist_gap": dist_gap,
        "frac_along": d["proj_t"], "arc_len": d["arc_len_m"],
        "head_cos": head_cos, "gps_moving": moving.astype(float),
        "deg_from": deg[fr], "deg_to": deg[to], "has_reverse": has_rev.astype(float),
        "n_cands": n_cands, "persist": persist,
        "speed": d["speed_ms"].fillna(0.0), "dt_prev": dtp, "dt_next": dtn,
    })
    out[FEATURES] = out[FEATURES].astype(np.float32)
    out["label"] = d["label"].to_numpy(np.int8)
    # A's rule: train only where the label arc is a candidate and label is trusted
    out["usable"] = (d["label_in_candidates"] & ~d["label_low_conf"]).to_numpy(bool)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--raw", default="data")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="*", help="build just these trip ids")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    arcs_idx = index_arcs(a.raw)
    proc = Path(a.processed)
    for f in sorted((proc / "candidates").glob("*.parquet")):
        tid = f.stem
        if a.only and tid not in a.only:
            continue
        dst = Path(a.out) / f"{tid}.parquet"
        if a.skip_existing and dst.exists():
            continue
        df = trip_features(pd.read_parquet(f), pd.read_parquet(proc / "points" / f.name),
                           read_arcs(arcs_idx[tid]), tid)   # one trip at a time
        df.to_parquet(dst, index=False)
        print(tid, df["partition"].iloc[0], len(df), "rows")


if __name__ == "__main__":
    main()
