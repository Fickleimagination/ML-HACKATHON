import argparse
from pathlib import Path

import folium
import numpy as np

from api.predictor import load_predictor, validate_prediction
from api.trips import discover_trips, load_trip
from data_processing.preprocess import valid_mask
from data_processing.projection import TripProjection

# Bundle the Leaflet runtime so the demo map renders on an internet-isolated LAN.
LEAFLET = Path(__file__).with_name("vendor") / "leaflet"
LEAFLET_ASSETS = (
    "<style>" + (LEAFLET / "leaflet.css").read_text(encoding="utf-8") + "</style>"
    "<script>" + (LEAFLET / "leaflet.js").read_text(encoding="utf-8") + "</script>"
)


class RouteMap(folium.Map):
    default_js = []
    default_css = []


COLORS = {
    "Road network": "#64706f",
    "GPS observations": "#2455b5",
    "Predicted route": "#cc4828",
    "Reference route": "#157850",
}


def route_segments(nodes, arcs, ids):
    """Merge consecutive edges only when the directed endpoints touch."""
    segments = []
    previous_end = None
    for start, end in arcs[ids]:
        if start != previous_end:
            segments.append([nodes[start, ::-1].tolist()])
        segments[-1].append(nodes[end, ::-1].tolist())
        previous_end = end
    return segments


def build_map(track, nodes, arcs, route, reference=None):
    valid = np.flatnonzero(valid_mask(track))
    gps = track[valid, 1::-1]
    view = RouteMap(tiles=None, control_scale=True, prefer_canvas=True)
    view.get_root().header.add_child(folium.Element(LEAFLET_ASSETS), name="leaflet_assets")
    view.get_root().header.add_child(folium.Element(
        "<style>.leaflet-container { background: #edf0e7; }</style>"
    ))
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap basemap", overlay=True, show=False).add_to(view)
    projection = TripProjection.for_track(track[valid])
    track_xy = projection.to_xy(track[valid, :2])
    nodes_xy = projection.to_xy(nodes)
    bbox_min = track_xy.min(axis=0) - 200
    bbox_max = track_xy.max(axis=0) + 200
    endpoints = nodes_xy[arcs]
    visible = (
        (endpoints.max(axis=1) >= bbox_min).all(axis=1)
        & (endpoints.min(axis=1) <= bbox_max).all(axis=1)
    )
    layers = {"Road network": nodes[arcs[visible]][:, :, ::-1].tolist()}
    if reference is not None:
        layers["Reference route"] = route_segments(nodes, arcs, reference)
    layers["Predicted route"] = route_segments(nodes, arcs, route)

    for name, segments in layers.items():
        group = folium.FeatureGroup(name=name)
        if segments:
            folium.PolyLine(
                segments, color=COLORS[name],
                weight={"Road network": 1, "Predicted route": 4, "Reference route": 7}[name],
                opacity={"Road network": 0.4, "Predicted route": 0.95, "Reference route": 0.65}[name],
                tooltip=name,
            ).add_to(group)
        group.add_to(view)

    observations = folium.FeatureGroup(name="GPS observations")
    indices = sorted(set(range(0, len(gps), 5)) | {len(gps) - 1})
    for i in indices:
        folium.CircleMarker(
            gps[i].tolist(), radius=2.5, stroke=False, fill=True,
            fill_color=COLORS["GPS observations"], fill_opacity=0.9,
            tooltip=f"GPS observation {valid[i] + 1}",
        ).add_to(observations)
    observations.add_to(view)

    extent = np.concatenate((gps, nodes[arcs[route].ravel()][:, ::-1]))
    view.fit_bounds([extent.min(axis=0).tolist(), extent.max(axis=0).tolist()],
                    padding=(18, 18), max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(view)
    legend = []
    for name, color in COLORS.items():
        if name == "Reference route" and reference is None:
            continue
        legend.append(f'<div><span style="color:{color}">●</span> {name}</div>')
    view.get_root().html.add_child(folium.Element(
        '<div style="position:absolute;bottom:58px;left:12px;z-index:999;'
        'padding:7px 10px;background:white;color:#202827;border:1px solid #c8cfcd;'
        'font:13px/1.55 system-ui;border-radius:4px">'
        + "".join(legend) + "</div>"
    ))
    return view


def main():
    from api.scoring import load_reference

    parser = argparse.ArgumentParser(description="Export a local student-test report map")
    parser.add_argument("trip_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/student_test"))
    parser.add_argument("--reference", action="store_true", help="Include the local reference layer")
    args = parser.parse_args()
    trips = discover_trips(args.data)
    if args.trip_id not in trips:
        parser.error("trip ID was not found")
    stem = trips[args.trip_id]
    track, nodes, arcs = load_trip(stem)
    predictor, _ = load_predictor()
    _, route, _, _ = validate_prediction(predictor(track, nodes, arcs), len(track), len(arcs))
    reference = load_reference(stem, len(arcs)) if args.reference else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_map(track, nodes, arcs, route, reference).save(str(args.output))


if __name__ == "__main__":
    main()
