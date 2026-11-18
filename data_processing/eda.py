"""Exploratory data analysis for PathSnap (report section 2).

Usage (from the repo root, after build_dataset has run):
    python -m data_processing.eda --raw data --processed data/processed --out results/eda

Uses the train and student_test partitions only. Every table and figure states its partition.
Outputs:
    tables/*.csv          inventory, missing/invalid values, gaps, GPS noise, network connectivity
    figures/*.png         static plots for the PDF
    maps/*.html           interactive Folium maps (GPS, road network, reference route)
    data_quality_log.md   issues found, assumptions and decisions, with numbers
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io_utils import list_trips, load_trip

PARTS = ["train", "student_test"]
PART_COLOURS = {"train": "#2a6fdb", "student_test": "#e8833a"}
MAP_TRIPS = {  # trip -> why it is shown
    "00000048": "clean trip (train)",
    "00000033": "GPS circles in a ~120 x 160 m area off the reference route for 55 min (train)",
    "00000056": "route revisits arcs / loops (student_test)",
    "00000090": "very large time gap, 153,064 s (train)",
}


# --------------------------------------------------------------------------- tables
def raw_tables(raw_root: str):
    inv, miss, net, gaps = [], [], [], []
    all_dt = {p: [] for p in PARTS}
    for part in PARTS:
        for d in list_trips(raw_root, part):
            tr = load_trip(d)
            tid = tr.trip_id
            sizes = {ext: os.path.getsize(os.path.join(d, f"{tid}.{ext}")) for ext in ["track", "nodes", "arcs", "route"]}
            t = tr.track[:, 2]
            dt = np.diff(t)
            all_dt[part].append(dt)
            E, V = len(tr.arcs), len(tr.nodes)

            inv.append({"trip_id": tid, "partition": part, "n_obs": len(tr.track), "n_nodes": V, "n_arcs": E,
                        "route_len": len(tr.route), "duration_s": float(t[-1] - t[0]),
                        **{f"bytes_{k}": v for k, v in sizes.items()}})

            valid_route = (tr.route >= 0) & (tr.route < E)
            r = tr.route[valid_route]
            cont = tr.arcs[r[:-1], 1] == tr.arcs[r[1:], 0] if len(r) > 1 else np.array([True])
            miss.append({"trip_id": tid, "partition": part,
                         "track_nan": int(np.isnan(tr.track).sum()), "nodes_nan": int(np.isnan(tr.nodes).sum()),
                         "track_bad_coord": int(((np.abs(tr.track[:, 0]) > 180) | (np.abs(tr.track[:, 1]) > 90)).sum()),
                         "dup_timestamps": int((dt == 0).sum()), "time_backwards": int((dt < 0).sum()),
                         "arcs_bad_node_id": int(((tr.arcs < 0) | (tr.arcs >= V)).sum()),
                         "route_bad_arc_id": int((~valid_route).sum()),
                         "route_breaks": int((~cont).sum())})

            gaps.append({"trip_id": tid, "partition": part, "frac_1s_steps": float(np.mean(dt == 1)),
                         "median_dt_s": float(np.median(dt)), "max_gap_s": float(dt.max()),
                         "n_gaps_over_30s": int((dt > 30).sum()), "n_gaps_over_300s": int((dt > 300).sum())})

            # connectivity on the directed road graph
            g = coo_matrix((np.ones(E), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(V, V)).tocsr()
            n_weak, weak_lab = connected_components(g, directed=True, connection="weak")
            n_strong, strong_lab = connected_components(g, directed=True, connection="strong")
            deg = np.bincount(tr.arcs[:, 0], minlength=V) + np.bincount(tr.arcs[:, 1], minlength=V)
            pairs = set(map(tuple, tr.arcs.tolist()))
            two_way = np.mean([(b, a) in pairs for a, b in tr.arcs.tolist()])
            net.append({"trip_id": tid, "partition": part, "n_nodes": V, "n_arcs": E,
                        "weak_components": n_weak,
                        "largest_weak_frac": float(np.bincount(weak_lab).max() / V),
                        "strong_components": n_strong,
                        "largest_strong_frac": float(np.bincount(strong_lab).max() / V),
                        "isolated_nodes": int((deg == 0).sum()),
                        "self_loops": int((tr.arcs[:, 0] == tr.arcs[:, 1]).sum()),
                        "duplicate_arcs": int(E - len(pairs)),
                        "two_way_frac": float(two_way),
                        "mean_node_degree": float(deg.mean()),
                        "route_in_largest_strong": float(np.mean(strong_lab[tr.arcs[r, 0]] == np.bincount(strong_lab).argmax()))})
    return (pd.DataFrame(inv), pd.DataFrame(miss), pd.DataFrame(gaps), pd.DataFrame(net),
            {p: np.concatenate(v) for p, v in all_dt.items()})


def load_points(processed: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(processed, "points", "*.parquet")))],
                     ignore_index=True)


# --------------------------------------------------------------------------- figures
def fig_dataset(inv: pd.DataFrame, P: pd.DataFrame, path: str):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part in PARTS:
        s = inv[inv.partition == part]
        ax[0].hist(s.n_obs, bins=np.logspace(2, 4.5, 25), alpha=0.7, label=part, color=PART_COLOURS[part])
        ax[1].hist(s.n_arcs, bins=np.logspace(2.5, 5.6, 25), alpha=0.7, label=part, color=PART_COLOURS[part])
    for a, t in zip(ax[:2], ["GPS observations per trip", "Road-network arcs per trip"]):
        a.set_xscale("log"); a.set_title(t); a.set_ylabel("trips"); a.legend()
    cen = P.groupby(["trip_id", "partition"])[["lon", "lat"]].mean().reset_index()
    for part in PARTS:
        s = cen[cen.partition == part]
        ax[2].scatter(s.lon, s.lat, s=18, label=part, color=PART_COLOURS[part])
    ax[2].set_xlim(-180, 180); ax[2].set_ylim(-60, 75); ax[2].set_title("Trip locations (mean GPS position)")
    ax[2].set_xlabel("longitude"); ax[2].set_ylabel("latitude"); ax[2].grid(alpha=0.3); ax[2].legend()
    fig.suptitle("Dataset overview (train: 80 trips, student_test: 10 trips)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_sampling(all_dt: dict, gaps: pd.DataFrame, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.array([0, 0.5, 1.5, 2.5, 5.5, 10.5, 30.5, 60.5, 300.5, 3600.5, 2e5])
    labels = ["0", "1", "2", "3-5", "6-10", "11-30", "31-60", "61-300", "301-3600", ">3600"]
    w = 0.4
    for i, part in enumerate(PARTS):
        c, _ = np.histogram(all_dt[part], bins=bins)
        ax[0].bar(np.arange(len(c)) + (i - 0.5) * w, c / c.sum(), width=w, label=part, color=PART_COLOURS[part])
    ax[0].set_yscale("log"); ax[0].set_xticks(range(len(labels))); ax[0].set_xticklabels(labels, rotation=30)
    ax[0].set_xlabel("time step between observations (s)"); ax[0].set_ylabel("share of steps (log)")
    ax[0].set_title("Sampling intervals"); ax[0].legend()
    g = gaps.sort_values("frac_1s_steps")
    ax[1].barh(g.trip_id.head(12), g.frac_1s_steps.head(12),
               color=[PART_COLOURS[p] for p in g.partition.head(12)])
    ax[1].set_xlabel("share of steps exactly 1 s"); ax[1].set_title("Least regular trips (12 lowest)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_noise(P: pd.DataFrame, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for part in PARTS:
        d = P.loc[P.partition == part, "label_dist_m"].to_numpy()
        ax[0].hist(np.clip(d, 0, 60), bins=60, density=True, alpha=0.6, label=part, color=PART_COLOURS[part])
    ax[0].axvline(50, color="k", ls="--", lw=1); ax[0].text(50.5, ax[0].get_ylim()[1] * 0.8, "50 m radius", fontsize=8)
    ax[0].text(59, ax[0].get_ylim()[1] * 0.95, ">=60 m\n(mostly trips\n54 and 33)", fontsize=7, ha="right", va="top")
    ax[0].set_xlabel("distance from GPS point to its reference arc (m, clipped at 60)")
    ax[0].set_ylabel("density"); ax[0].set_title("GPS noise relative to the reference route"); ax[0].legend()
    per = P.groupby(["trip_id", "partition"]).label_dist_m.median().reset_index().sort_values("label_dist_m")
    ax[1].bar(range(len(per)), per.label_dist_m, color=[PART_COLOURS[p] for p in per.partition])
    ax[1].set_yscale("log"); ax[1].set_xlabel("trips, sorted"); ax[1].set_ylabel("median distance (m, log)")
    ax[1].set_title("Median GPS-to-route distance per trip")
    for _, r in per.tail(2).iterrows():
        ax[1].annotate(r.trip_id, (list(per.trip_id).index(r.trip_id), r.label_dist_m), fontsize=7, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_network(net: pd.DataFrame, path: str):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    c = [PART_COLOURS[p] for p in net.partition]
    ax[0].scatter(net.n_nodes, net.n_arcs, c=c, s=18); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].set_title("Network size per trip")
    ax[1].scatter(net.n_arcs, net.largest_strong_frac, c=c, s=18); ax[1].set_xscale("log")
    ax[1].set_xlabel("arcs"); ax[1].set_ylabel("share of nodes in largest strongly connected component")
    ax[1].set_title("Directed connectivity")
    ax[2].hist(net.two_way_frac, bins=20, color="#666"); ax[2].set_xlabel("share of arcs with a reverse arc")
    ax[2].set_ylabel("trips"); ax[2].set_title("Two-way roads")
    for p in PARTS:
        ax[0].scatter([], [], c=PART_COLOURS[p], label=p)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _route_polylines(trip, xy_nodes):
    """Split the reference route into continuous polylines (node coordinate lists)."""
    lines, cur = [], []
    for k, a in enumerate(trip.route):
        f, t = trip.arcs[a]
        if cur and cur[-1] != f:
            lines.append(cur); cur = []
        if not cur:
            cur = [f]
        cur.append(t)
    if cur:
        lines.append(cur)
    return [xy_nodes[l] for l in lines]


def _clip_arcs(trip, buffer_deg=0.003):
    lo = trip.track[:, :2].min(0) - buffer_deg
    hi = trip.track[:, :2].max(0) + buffer_deg
    A, B = trip.nodes[trip.arcs[:, 0]], trip.nodes[trip.arcs[:, 1]]
    keep = ((A >= lo) & (A <= hi)).all(1) | ((B >= lo) & (B <= hi)).all(1)
    return A[keep], B[keep]


def fig_trajectories(raw_root: str, trips: dict, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    for ax, (tid, why) in zip(axes.ravel(), trips.items()):
        d = glob.glob(os.path.join(raw_root, "*", tid))[0]
        tr = load_trip(d)
        A, B = _clip_arcs(tr)
        ax.plot(np.c_[A[:, 0], B[:, 0], np.full(len(A), np.nan)].ravel(),
                np.c_[A[:, 1], B[:, 1], np.full(len(A), np.nan)].ravel(), color="#c8c8c8", lw=0.5, label="road network")
        for i, ln in enumerate(_route_polylines(tr, tr.nodes)):
            ax.plot(ln[:, 0], ln[:, 1], color="#1a9850", lw=2.2, label="reference route" if i == 0 else None)
        ax.scatter(tr.track[:, 0], tr.track[:, 1], s=1.5, color="#2a6fdb", label="GPS observations", zorder=3)
        lat0 = np.radians(tr.track[:, 1].mean())
        ax.set_aspect(1 / np.cos(lat0))
        ax.set_title(f"Trip {tid}: {why}", fontsize=10); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- folium
def folium_map(raw_root: str, processed: str, tid: str, why: str, path: str):
    import folium
    from branca.element import MacroElement, Template

    d = glob.glob(os.path.join(raw_root, "*", tid))[0]
    tr = load_trip(d)
    pts = pd.read_parquet(os.path.join(processed, "points", f"{tid}.parquet"))
    # OpenStreetMap's own tile servers refuse requests from local HTML files (HTTP 403, no Referer),
    # so the maps use Esri basemaps (no API key needed for this use).
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
        name="Basemap: satellite (Esri)", max_zoom=19).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
        name="Basemap: light grey (Esri)", max_zoom=16).add_to(m)

    A, B = _clip_arcs(tr)
    net = folium.FeatureGroup(name="Road network (clipped to trip)", show=True)
    folium.PolyLine([[(a[1], a[0]), (b[1], b[0])] for a, b in zip(A, B)], color="#9a9a9a", weight=1,
                    opacity=0.6).add_to(net)
    net.add_to(m)

    route = folium.FeatureGroup(name="Reference route", show=True)
    for ln in _route_polylines(tr, tr.nodes):
        folium.PolyLine([(y, x) for x, y in ln], color="#1a9850", weight=5, opacity=0.8).add_to(route)
    route.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations (every 3rd)", show=True)
    for _, r in pts.iloc[::3].iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=2, color="#2a6fdb", fill=True, fill_opacity=0.9, weight=0,
                            tooltip=f"obs {r.obs_idx} | t={r.t:.0f}s | dist to ref {r.label_dist_m:.1f} m").add_to(gps)
    gps.add_to(m)

    low = pts[pts.label_low_conf]
    if len(low):
        lc = folium.FeatureGroup(name=f"Low-confidence labels (>50 m), {len(low)} pts", show=True)
        for _, r in low.iloc[::3].iterrows():
            folium.CircleMarker((r.lat, r.lon), radius=3, color="#d73027", fill=True, fill_opacity=0.9,
                                weight=0).add_to(lc)
        lc.add_to(m)

    breaks = pts[(pts.segment_id.diff() > 0)]
    if len(breaks):
        gb = folium.FeatureGroup(name=f"Segment starts after gaps >30 s ({len(breaks)})", show=True)
        for _, r in breaks.iterrows():
            folium.Marker((r.lat, r.lon), icon=folium.Icon(color="orange", icon="pause"),
                          tooltip=f"obs {r.obs_idx}: gap {r.dt_prev:.0f} s").add_to(gb)
        gb.add_to(m)

    folium.Marker((pts.lat.iloc[0], pts.lon.iloc[0]), tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((pts.lat.iloc[-1], pts.lon.iloc[-1]), tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[pts.lat.min(), pts.lon.min()], [pts.lat.max(), pts.lon.max()]])

    legend = Template(f"""{{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 10px 12px;
                border: 1px solid #bbb; border-radius: 6px; font: 12px sans-serif; line-height: 1.6;">
      <b>Trip {tid}</b> &mdash; {why}<br>
      <span style="color:#2a6fdb">&#9679;</span> GPS observation<br>
      <span style="color:#1a9850">&#9644;</span> Reference route<br>
      <span style="color:#9a9a9a">&#9644;</span> Road network<br>
      <span style="color:#d73027">&#9679;</span> Low-confidence label (&gt;50 m from route)<br>
      <span style="color:#f0932b">&#9873;</span> New segment after time gap
    </div>{{% endmacro %}}""")
    el = MacroElement(); el._template = legend
    m.get_root().add_child(el)
    m.save(path)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="results/eda")
    a = ap.parse_args()
    for sub in ["tables", "figures", "maps"]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    T = lambda n: os.path.join(a.out, "tables", n)
    F = lambda n: os.path.join(a.out, "figures", n)

    inv, miss, gaps, net, all_dt = raw_tables(a.raw)
    P = load_points(a.processed)
    inv.to_csv(T("inventory_by_trip.csv"), index=False)
    miss.to_csv(T("missing_invalid_by_trip.csv"), index=False)
    gaps.to_csv(T("sampling_gaps_by_trip.csv"), index=False)
    net.to_csv(T("network_by_trip.csv"), index=False)

    # partition-level summaries
    inv.groupby("partition").agg(trips=("trip_id", "count"), obs_total=("n_obs", "sum"), obs_median=("n_obs", "median"),
                                 obs_min=("n_obs", "min"), obs_max=("n_obs", "max"), arcs_median=("n_arcs", "median"),
                                 arcs_max=("n_arcs", "max"), route_len_median=("route_len", "median"),
                                 duration_median_s=("duration_s", "median")).to_csv(T("inventory_by_partition.csv"))
    miss.drop(columns="trip_id").groupby("partition").sum().to_csv(T("missing_invalid_by_partition.csv"))
    flags = P.groupby("partition")[["flag_invalid", "flag_dup_time", "flag_time_backwards", "flag_jump",
                                    "label_low_conf"]].mean()
    flags["heading_undefined"] = P.groupby("partition").heading_deg.apply(lambda s: s.isna().mean())
    flags.to_csv(T("observation_flags_rate_by_partition.csv"))
    noise = P.groupby("partition").label_dist_m.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    noise.to_csv(T("gps_noise_by_partition.csv"))
    gaps.sort_values("max_gap_s", ascending=False).head(10).to_csv(T("largest_gaps_top10.csv"), index=False)

    fig_dataset(inv, P, F("01_dataset_overview.png"))
    fig_sampling(all_dt, gaps, F("02_sampling_intervals.png"))
    fig_noise(P, F("03_gps_noise.png"))
    fig_network(net, F("04_road_network.png"))
    fig_trajectories(a.raw, MAP_TRIPS, F("05_trajectories.png"))
    for tid, why in MAP_TRIPS.items():
        folium_map(a.raw, a.processed, tid, why, os.path.join(a.out, "maps", f"trip_{tid}.html"))
    print("EDA written to", a.out)


if __name__ == "__main__":
    main()
