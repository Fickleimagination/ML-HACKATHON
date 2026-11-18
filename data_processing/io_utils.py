"""Loading raw PathSnap trip files.

Each trip folder <trip_id>/ contains four header-less, whitespace-separated files:
    <trip_id>.track  -> lon, lat, t (seconds)       one row per GPS observation
    <trip_id>.nodes  -> lon, lat                    row index = node_id (0-based)
    <trip_id>.arcs   -> from_node_id, to_node_id    row index = arc_id (0-based)
    <trip_id>.route  -> arc_id                      ordered reference route (no timestamps)

The .route file is missing for evaluator trips at prediction time, so it is optional here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) from_node, to_node
    route: np.ndarray | None     # (M,) arc ids, or None if not supplied


def _load(path: str, dtype=float, ndmin=2) -> np.ndarray:
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_trip(trip_dir: str, with_route: bool = True) -> Trip:
    """Load one trip folder. Set with_route=False in the prediction pipeline."""
    tid = os.path.basename(os.path.normpath(trip_dir))
    track = _load(os.path.join(trip_dir, f"{tid}.track"))
    nodes = _load(os.path.join(trip_dir, f"{tid}.nodes"))
    arcs = _load(os.path.join(trip_dir, f"{tid}.arcs"), dtype=np.int64)
    route = None
    rpath = os.path.join(trip_dir, f"{tid}.route")
    if with_route and os.path.exists(rpath):
        route = _load(rpath, dtype=np.int64, ndmin=1)
    return Trip(tid, track, nodes, arcs, route)


def list_trips(raw_root: str, partition: str) -> list[str]:
    """Return sorted trip folders for a partition ('train' or 'student_test')."""
    return sorted(d for d in glob(os.path.join(raw_root, partition, "*")) if os.path.isdir(d))
