"""Candidate generation: every directed arc within radius R of each GPS point.

Both directions of a two-way road are separate arcs and are kept as separate candidates;
heading features (built in features/) decide between them.
Candidate generation never reads the route, so it runs identically for evaluator trips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from shapely import STRtree

from .cleaning import bearing_deg
from .geometry import project_pairs

DEFAULT_RADIUS_M = 50.0


class ArcIndex:
    """Spatial index over one trip's arcs, in that trip's projected metres."""

    def __init__(self, node_xy: np.ndarray, arcs: np.ndarray):
        self.node_xy = node_xy
        self.arcs = arcs
        self.A = node_xy[arcs[:, 0]]
        self.B = node_xy[arcs[:, 1]]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = bearing_deg(d[:, 0], d[:, 1])
        self.tree = STRtree(shapely.linestrings(np.stack([self.A, self.B], axis=1)))
        # node degree (in + out arcs) over the whole network, not just nearby candidates
        n_nodes = len(node_xy)
        self.degree = (np.bincount(arcs[:, 0], minlength=n_nodes)
                       + np.bincount(arcs[:, 1], minlength=n_nodes))
        # does the opposite-direction arc (to -> from) exist? -1 if not
        key = {(int(a), int(b)): i for i, (a, b) in enumerate(arcs)}
        self.reverse_arc = np.array([key.get((int(b), int(a)), -1) for a, b in arcs], dtype=np.int64)

    def query(self, pts_xy: np.ndarray, radius: float = DEFAULT_RADIUS_M) -> pd.DataFrame:
        pi, ai = self.tree.query(shapely.points(pts_xy), predicate="dwithin", distance=radius)
        d, t = project_pairs(pts_xy[pi], self.A[ai], self.B[ai])
        return pd.DataFrame({
            "obs_idx": pi.astype(np.int32),
            "cand_arc": ai.astype(np.int32),
            "dist_m": d.astype(np.float32),
            "proj_t": t.astype(np.float32),
            "arc_len_m": self.length[ai].astype(np.float32),
            "arc_bearing_deg": self.bearing[ai].astype(np.float32),
            "from_node": self.arcs[ai, 0].astype(np.int32),
            "to_node": self.arcs[ai, 1].astype(np.int32),
            "from_degree": self.degree[self.arcs[ai, 0]].astype(np.int16),
            "to_degree": self.degree[self.arcs[ai, 1]].astype(np.int16),
            "reverse_arc": self.reverse_arc[ai].astype(np.int32),
        }).sort_values(["obs_idx", "dist_m"], kind="stable").reset_index(drop=True)
