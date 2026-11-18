"""Label-quality analysis for report section 2.

Compares v1 (order-preserving alignment) with v0 (nearest route arc, heading tie-break) and
measures how often a reference label points against the direction of travel.

Usage:
    python -m data_processing.label_quality --raw data --processed data/processed --out results/labels

Outputs:
    label_quality_by_trip.csv
    label_quality.png                 per-trip change rate and wrong-direction rate
    trip_00000056_v0_vs_v1.html       Folium map of where v0 and v1 disagree on a loop trip
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

from .io_utils import load_trip

MOVING_MS = 3.0        # heading is only trusted when the vehicle moves faster than this
AGAINST_DEG = 90.0     # label arc points more than this away from the heading


def _angle(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def trip_stats(raw: str, processed: str, tid: str, part: str) -> dict:
    trip = load_trip(os.path.join(raw, part, tid))
    p = pd.read_parquet(os.path.join(processed, "points", f"{tid}.parquet"))
    c = pd.read_parquet(os.path.join(processed, "candidates", f"{tid}.parquet"),
                        columns=["cand_arc", "arc_bearing_deg"]).drop_duplicates("cand_arc")
    brg = pd.Series(c.arc_bearing_deg.to_numpy(), index=c.cand_arc.to_numpy())
    use = ((p.speed_ms > MOVING_MS) & ~p.label_low_conf & p.heading_deg.notna()).to_numpy()

    def against(col):
        b = brg.reindex(p[col]).to_numpy()
        ok = use & ~np.isnan(b)
        return float(np.mean(_angle(b[ok], p.heading_deg.to_numpy()[ok]) > AGAINST_DEG)) if ok.any() else np.nan

    return {"trip_id": tid, "partition": part, "n_obs": len(p),
            "v1_vs_v0_change": float(np.mean(p.label_arc != p.label_arc_v0)),
            "against_travel_v1": against("label_arc"), "against_travel_v0": against("label_arc_v0"),
            "low_conf": float(p.label_low_conf.mean()),
            "route_repeat_share": float(1 - len(np.unique(trip.route)) / len(trip.route))}


def figure(df: pd.DataFrame, path: str):
    d = df.sort_values("against_travel_v1", ascending=False).reset_index(drop=True)
    col = ["#e8833a" if p == "student_test" else "#2a6fdb" for p in d.partition]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(d))
    ax[0].bar(x, d.against_travel_v1 * 100, color=col)
    ax[0].plot(x, d.against_travel_v0 * 100, "k.", ms=4, label="v0 labels (nearest route arc + heading)")
    ax[0].axhline(d.against_travel_v1.median() * 100, color="#999", ls="--", lw=1)
    ax[0].text(len(d) - 1, d.against_travel_v1.median() * 100 + 0.8, f"median {d.against_travel_v1.median():.1%}",
               ha="right", fontsize=8, color="#555")
    ax[0].set_ylabel("% of moving points")
    ax[0].set_title("Reference label points against the direction of travel (> 90° from heading, speed > 3 m/s)")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, d.v1_vs_v0_change * 100, color=col)
    ax[1].set_ylabel("% of points")
    ax[1].set_title("Labels changed by the order-preserving alignment (v1 vs v0)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([t[-2:] for t in d.trip_id], fontsize=6, rotation=90)
    ax[1].set_xlabel("trip (last two digits), sorted by top panel; orange = student_test, blue = train")
    for k in range(3):
        ax[0].annotate(d.trip_id[k][-2:], (k, d.against_travel_v1[k] * 100), fontsize=7, ha="center",
                       textcoords="offset points", xytext=(0, 3))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def loop_map(raw: str, processed: str, tid: str, part: str, path: str):
    import folium
    from branca.element import MacroElement, Template

    trip = load_trip(os.path.join(raw, part, tid))
    p = pd.read_parquet(os.path.join(processed, "points", f"{tid}.parquet"))
    nodes = trip.nodes
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
                     name="Basemap: satellite (Esri)", max_zoom=19).add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri", name="Basemap: light grey (Esri)", max_zoom=16).add_to(m)

    def arc_line(a):
        f, t = trip.arcs[a]
        return [(nodes[f, 1], nodes[f, 0]), (nodes[t, 1], nodes[t, 0])]

    diff = p[p.label_arc != p.label_arc_v0]
    same = p[p.label_arc == p.label_arc_v0]
    g_same = folium.FeatureGroup(name=f"Points where v0 = v1 ({len(same)})")
    for _, r in same.iloc[::4].iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=2, color="#2a6fdb", weight=0, fill=True, fill_opacity=0.7).add_to(g_same)
    g_v0 = folium.FeatureGroup(name="v0 label arcs at disagreements")
    g_v1 = folium.FeatureGroup(name="v1 label arcs at disagreements")
    for a in diff.label_arc_v0.unique():
        folium.PolyLine(arc_line(a), color="#d73027", weight=6, opacity=0.8).add_to(g_v0)
    for a in diff.label_arc.unique():
        folium.PolyLine(arc_line(a), color="#1a9850", weight=4, opacity=0.9).add_to(g_v1)
    g_diff = folium.FeatureGroup(name=f"Points where v0 ≠ v1 ({len(diff)})")
    for _, r in diff.iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=3, color="#f0932b", weight=0, fill=True, fill_opacity=0.95,
                            tooltip=f"obs {r.obs_idx}: v0 arc {r.label_arc_v0}, v1 arc {r.label_arc} "
                                    f"(route pos {r.label_route_pos})").add_to(g_diff)
    for g in (g_same, g_v0, g_v1, g_diff):
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[p.lat.min(), p.lon.min()], [p.lat.max(), p.lon.max()]])
    legend = Template(f"""{{% macro html(this, kwargs) %}}
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; padding: 10px 12px;
                border: 1px solid #bbb; border-radius: 6px; font: 12px sans-serif; line-height: 1.6;">
      <b>Trip {tid}</b> &mdash; where v0 and v1 labels disagree ({len(diff)} of {len(p)} points)<br>
      <span style="color:#2a6fdb">&#9679;</span> GPS point, labels agree<br>
      <span style="color:#f0932b">&#9679;</span> GPS point, labels disagree (hover for arcs)<br>
      <span style="color:#d73027">&#9644;</span> v0 label arc (nearest route arc)<br>
      <span style="color:#1a9850">&#9644;</span> v1 label arc (order-preserving alignment)
    </div>{{% endmacro %}}""")
    el = MacroElement(); el._template = legend
    m.get_root().add_child(el)
    m.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="results/labels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for part in ["train", "student_test"]:
        for d in sorted(glob.glob(os.path.join(a.raw, part, "*"))):
            rows.append(trip_stats(a.raw, a.processed, os.path.basename(d), part))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "label_quality_by_trip.csv"), index=False)
    figure(df, os.path.join(a.out, "label_quality.png"))
    loop_map(a.raw, a.processed, "00000056", "student_test", os.path.join(a.out, "trip_00000056_v0_vs_v1.html"))
    for part, s in df.groupby("partition"):
        w = s.n_obs / s.n_obs.sum()
        print(f"{part}: v1 vs v0 change {np.sum(s.v1_vs_v0_change * w):.3f} | against travel v1 median "
              f"{s.against_travel_v1.median():.3f}, v0 median {s.against_travel_v0.median():.3f}")


if __name__ == "__main__":
    main()
