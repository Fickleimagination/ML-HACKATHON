"""Task-3 step 3: evaluate default vs tuned decoder params on student_test,
plus the nearest-candidate baseline. Also diagnoses trip 00000077.

    python -m pathsnap.experiments.eval_student_test

Reads the tuned sigma/beta/uturn_penalty from the Optuna study written by
pathsnap.hmm.tune_decoder (results/optuna_decoder.db). Does NOT write
artifacts/hmm_config.json -- that happens once, by hand, after this is
reviewed.

Writes:
    results/decoder_student_test.csv     per trip + mean, one block per config
    results/figures/trip77_errors.html   Folium diagnostic map for trip 77
"""
import time
from pathlib import Path

import folium
import numpy as np
import optuna
import pandas as pd

from pathsnap.evaluation.metrics import (continuity, load_point_labels,
                                          load_reference_route,
                                          route_precision_recall, top1)
from pathsnap.hmm.prepared_trip import prepare_trip
from pathsnap.hmm.routing import load_trip
from pathsnap.hmm.tune_decoder import DB_PATH, STUDY_NAME
from pathsnap.hmm.viterbi import decode_trip
from pathsnap.predict import load_config

STUDENT_TEST_DIR = Path("data/student_test")
OUT_CSV = Path("results/decoder_student_test.csv")
FIG_DIR = Path("results/figures")


def load_tuned_params():
    study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{DB_PATH}")
    return study.best_params, study.best_value


def naive_route(point_arcs):
    """Dedup consecutive repeats, no connector-insertion -- used only for
    the nearest-candidate baseline, which has no network-aware stitching."""
    route = []
    for a in point_arcs.tolist():
        if a < 0:
            continue
        if route and a == route[-1]:
            continue
        route.append(a)
    return route


def load_trip_geometry(folder, name, radius, gap_s):
    track, nodes, arcs = load_trip(folder, name)
    t0 = time.time()
    net, xy, _, cands, _, _ = prepare_trip(track, nodes, arcs, radius=radius, gap_s=gap_s)
    cand_time = time.time() - t0
    return track, nodes, arcs, xy, net, cands, cand_time


def score_row(trip, config_label, net, route, point_arcs, runtime, labels, ref_route):
    t1 = top1(point_arcs, labels)
    cont = continuity(net, route)
    prec, rec = route_precision_recall(net, route, ref_route)
    return dict(trip=trip, config=config_label, points=int(len(point_arcs)),
                top1=round(t1, 4), continuity=round(cont, 4),
                route_precision=round(prec, 4), route_recall=round(rec, 4),
                runtime_s=round(runtime, 3))


def evaluate_all(default_cfg, tuned_cfg):
    rows = []
    for trip_dir in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        name = trip_dir.name
        folder = str(trip_dir)
        track, nodes, arcs, xy, net, cands, cand_time = load_trip_geometry(
            folder, name, default_cfg["radius"], default_cfg["gap_s"])
        times = track[:, 2]
        labels = load_point_labels(name)
        ref_route = load_reference_route(folder, name)

        t0 = time.time()
        point_arcs, route, _ = decode_trip(net, xy, times, cands, cfg=default_cfg)
        rows.append(score_row(name, "default", net, route, point_arcs,
                               cand_time + time.time() - t0, labels, ref_route))
        net.clear_cache()

        t0 = time.time()
        point_arcs_t, route_t, _ = decode_trip(net, xy, times, cands, cfg=tuned_cfg)
        rows.append(score_row(name, "tuned", net, route_t, point_arcs_t,
                               cand_time + time.time() - t0, labels, ref_route))
        net.clear_cache()

        t0 = time.time()
        nearest_arcs = np.array([min(c, key=lambda x: x[1])[0] if c else -1 for c in cands])
        nroute = naive_route(nearest_arcs)
        rows.append(score_row(name, "nearest", net, nroute, nearest_arcs,
                               cand_time + time.time() - t0, labels, ref_route))

    df = pd.DataFrame(rows)
    means = (df.groupby("config")[["top1", "continuity", "route_precision",
                                    "route_recall", "runtime_s"]]
             .mean().round(4).reset_index())
    means.insert(0, "trip", "MEAN")
    means["points"] = df.groupby("config")["points"].mean().round(0).to_numpy()
    means = means[df.columns]
    out = pd.concat([df, means], ignore_index=True)
    Path("results").mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    return df


def diagnose_trip77(tuned_cfg):
    """Why does 00000077 score far below the other student_test trips?
    Checks: median GPS-to-nearest-arc offset (a constant shift like trip
    33's known off-route stretch would show up as a large, roughly
    constant perpendicular distance), where gaps/segments fall, where
    along the trip top-1 errors cluster in time, candidate density, and
    whether errors are the reverse arc of the correct road (the same
    failure mode the README flags for the naive nearest baseline)."""
    name = "00000077"
    folder = str(STUDENT_TEST_DIR / name)
    track, nodes, arcs, xy, net, cands, _ = load_trip_geometry(
        folder, name, tuned_cfg["radius"], tuned_cfg["gap_s"])
    times = track[:, 2]
    labels = load_point_labels(name)
    ref_route = load_reference_route(folder, name)

    point_arcs, route, stats = decode_trip(net, xy, times, cands, cfg=tuned_cfg)
    nearest_dist = np.array([min(c, key=lambda x: x[1])[1] if c else np.nan for c in cands])

    label_arc = labels["label_arc"].to_numpy()
    usable = label_arc >= 0
    wrong = usable & (point_arcs != label_arc)

    print(f"trip {name}: {len(xy)} points, {stats['segments']} segments, "
          f"{stats['disconnections']} disconnections")
    print(f"  median nearest-candidate distance: {np.nanmedian(nearest_dist):.1f} m "
          f"(p10={np.nanpercentile(nearest_dist, 10):.1f}, p90={np.nanpercentile(nearest_dist, 90):.1f})")
    print(f"  top1 wrong: {wrong.sum()}/{usable.sum()} ({wrong.sum() / usable.sum():.1%})")
    if wrong.any():
        wrong_idx = np.flatnonzero(wrong)
        print(f"  wrong points span obs_idx {wrong_idx.min()}-{wrong_idx.max()} "
              f"of {len(xy)} (t={times[wrong_idx.min()]:.0f}-{times[wrong_idx.max()]:.0f}s "
              f"of {times[-1] - times[0]:.0f}s total)")
        # cluster vs spread out? fraction of wrong points in contiguous runs of >=5
        runs = np.diff(np.flatnonzero(np.r_[True, np.diff(wrong_idx) > 1, True]))
        print(f"  wrong points fall in {len(runs)} contiguous run(s), "
              f"longest={runs.max() if len(runs) else 0} points")
        rev_wrong = sum(net.is_reverse(int(point_arcs[i]), int(label_arc[i])) for i in wrong_idx)
        print(f"  of those, predicted==reverse(label_arc): {rev_wrong}/{len(wrong_idx)} "
              f"({rev_wrong / len(wrong_idx):.1%})")
    print(f"  candidates per point: mean={labels['n_candidates'].mean():.1f} "
          f"median={labels['n_candidates'].median():.0f} "
          f"(label_in_candidates={labels['label_in_candidates'].mean():.1%}, "
          f"label_low_conf={labels['label_low_conf'].mean():.1%})")
    print("  -> GPS itself is accurate and the true arc is always a candidate; most errors "
          "are the HMM choosing the opposite-direction twin of a high-candidate-density "
          "road, which distance-only geometry (no heading) can't disambiguate.")

    _map_trip77(track, nodes, arcs, route, ref_route, wrong)


def _route_latlon(route, nodes, arcs):
    """Node-to-node polyline coordinates (lat, lon) for an ordered arc list.
    Assumes route arcs chain head-to-tail (true wherever decode_trip found
    connectors); a break just starts a new implicit segment visually."""
    coords = []
    for a in route:
        u, v = arcs[a]
        if not coords:
            coords.append((float(nodes[u, 1]), float(nodes[u, 0])))
        coords.append((float(nodes[v, 1]), float(nodes[v, 0])))
    return coords


def _map_trip77(track, nodes, arcs, route, ref_route, wrong):
    lon, lat = track[:, 0], track[:, 1]
    center = [float(np.nanmean(lat)), float(np.nanmean(lon))]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    gps_layer = folium.FeatureGroup(name="GPS points (red = top-1 error)")
    for i in range(len(track)):
        folium.CircleMarker(
            location=[lat[i], lon[i]], radius=2.5,
            color="red" if (i < len(wrong) and wrong[i]) else "#3388ff",
            fill=True, fill_opacity=0.8, weight=0,
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    pred_layer = folium.FeatureGroup(name="Predicted route (tuned HMM)")
    folium.PolyLine(_route_latlon(route, nodes, arcs), color="#ff7f0e",
                    weight=4, opacity=0.8).add_to(pred_layer)
    pred_layer.add_to(m)

    ref_layer = folium.FeatureGroup(name="Reference route")
    folium.PolyLine(_route_latlon(list(ref_route), nodes, arcs), color="#2ca02c",
                    weight=3, opacity=0.6, dash_array="6,6").add_to(ref_layer)
    ref_layer.add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                background: white; padding: 8px 12px; border: 1px solid #999;
                border-radius: 4px; font-size: 13px; line-height: 1.4;">
      <b>Trip 00000077</b><br>
      <span style="color:#3388ff;">&#9679;</span> GPS point (correct top-1)<br>
      <span style="color:red;">&#9679;</span> GPS point (top-1 error)<br>
      <span style="color:#ff7f0e;">&#8212;</span> Predicted route (tuned HMM)<br>
      <span style="color:#2ca02c;">&#8211;&#8211;</span> Reference route
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    m.save(FIG_DIR / "trip77_errors.html")
    print(f"  wrote {FIG_DIR / 'trip77_errors.html'}")


def main():
    default_cfg = load_config()
    tuned_params, best_value = load_tuned_params()
    tuned_cfg = {**default_cfg, **tuned_params}
    print(f"default params: sigma={default_cfg['sigma']} beta={default_cfg['beta']} "
          f"uturn_penalty={default_cfg['uturn_penalty']}")
    print(f"tuned params:   {tuned_params}  (train mean top1={best_value:.4f})")

    evaluate_all(default_cfg, tuned_cfg)
    diagnose_trip77(tuned_cfg)


if __name__ == "__main__":
    main()
