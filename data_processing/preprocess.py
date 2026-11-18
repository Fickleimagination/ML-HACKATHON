"""Single preprocessing entry point, shared by training, the API and evaluation.

    from data_processing.preprocess import preprocess_trip, reference_labels

    prep = preprocess_trip(track, nodes, arcs)        # inference: never reads the route
    obs, cand = prep.obs, prep.cand                    # per-observation and per-candidate tables

    ref = reference_labels(prep, route)                # scoring only (server-side)

Conventions (identical everywhere):
  * Projection: AEQD centred on the mean lon/lat of the valid GPS points (projection.py).
  * Nothing is dropped or reordered: obs row k == .track row k (obs_idx).
  * Invalid points (NaN / out-of-range / 0,0) are flagged; they get no candidates and label -1.
  * Segments: a new segment starts after a time step > 30 s, time going backwards, or an
    invalid point (cleaning.py).
  * No GPS offset correction is applied.
  * Candidates: all directed arcs within 50 m (candidates.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alignment import GAP_SECONDS as ALIGN_GAP_S, SKIP_PENALTY, align_route
from .candidates import DEFAULT_RADIUS_M, ArcIndex
from .cleaning import GAP_SECONDS, observation_table
from .projection import TripProjection

LOW_CONF_DIST_M = 50.0


@dataclass
class Prepared:
    proj: TripProjection
    pts_xy: np.ndarray
    node_xy: np.ndarray
    index: ArcIndex
    obs: pd.DataFrame
    cand: pd.DataFrame


def valid_mask(track: np.ndarray) -> np.ndarray:
    lon, lat = track[:, 0], track[:, 1]
    return (np.isfinite(track).all(axis=1) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
            & ~((lon == 0) & (lat == 0)))


def preprocess_trip(track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray,
                    radius: float = DEFAULT_RADIUS_M, gap_s: float = GAP_SECONDS) -> Prepared:
    valid = valid_mask(track)
    proj = TripProjection(track[valid, 0].mean(), track[valid, 1].mean())
    pts_xy = proj.to_xy(np.where(valid[:, None], track[:, :2], 0.0))
    pts_xy[~valid] = np.nan
    node_xy = proj.to_xy(nodes)
    obs = observation_table(track, pts_xy, gap_s=gap_s)
    index = ArcIndex(node_xy, arcs)
    cand = index.query(np.nan_to_num(pts_xy[valid]), radius=radius)
    cand["obs_idx"] = np.flatnonzero(valid)[cand["obs_idx"].to_numpy()].astype(np.int32)
    return Prepared(proj, pts_xy, node_xy, index, obs, cand)


def reference_labels(prep: Prepared, route: np.ndarray) -> pd.DataFrame:
    """v1 order-preserving reference labels, one row per original .track row."""
    valid = ~prep.obs["flag_invalid"].to_numpy()
    lab, pos, dist = align_route(prep.pts_xy, prep.obs["t"].to_numpy(), valid, route,
                                 prep.index.A, prep.index.B)
    return pd.DataFrame({"obs_idx": prep.obs["obs_idx"].to_numpy(),
                         "label_arc": lab.astype(np.int32), "label_route_pos": pos.astype(np.int32),
                         "label_dist_m": dist.astype(np.float32),
                         "label_low_conf": ~(dist <= LOW_CONF_DIST_M)})


SETTINGS = {"projection": "AEQD centred on mean lon/lat of valid GPS points",
            "candidate_radius_m": DEFAULT_RADIUS_M, "segment_gap_s": GAP_SECONDS,
            "align_skip_penalty_per_m": SKIP_PENALTY, "align_gap_s": ALIGN_GAP_S,
            "low_conf_dist_m": LOW_CONF_DIST_M, "offset_correction": None}


def reference_labels_for_dir(trip_dir: str) -> pd.DataFrame:
    """Convenience for server-side scoring: load a trip folder and return v1 labels by .track row."""
    from .io_utils import load_trip
    trip = load_trip(trip_dir)
    return reference_labels(preprocess_trip(trip.track, trip.nodes, trip.arcs), trip.route)
