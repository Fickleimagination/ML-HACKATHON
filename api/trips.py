from pathlib import Path

import numpy as np

from data_processing.io_utils import load_trip as load_raw_trip
from data_processing.preprocess import valid_mask


def discover_trips(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"Trip directory does not exist: {root}")
    trips = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not folder.name.isascii() or not folder.name.isdigit():
            continue
        stem = folder / folder.name
        if not all(stem.with_suffix(ext).is_file() for ext in (".track", ".nodes", ".arcs")):
            raise ValueError(f"Incomplete trip: {folder}")
        trips[folder.name] = stem
    if not trips:
        raise ValueError(f"No trips found under {root}")
    return trips


def load_trip(stem: Path):
    trip = load_raw_trip(str(stem.parent), with_route=False)
    track, nodes, arcs = trip.track, trip.nodes, trip.arcs
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Expected track, nodes and arcs to have 3, 2 and 2 columns")
    if not len(track) or not len(nodes) or not len(arcs):
        raise ValueError("Trip files must contain observations, nodes and arcs")
    if not np.isfinite(nodes).all() or (abs(nodes) > [180, 90]).any():
        raise ValueError("Invalid road-network coordinates")
    if not valid_mask(track).any():
        raise ValueError("Trip has no valid GPS observations")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError("Arc endpoint refers to a missing node")
    return track, nodes, arcs
