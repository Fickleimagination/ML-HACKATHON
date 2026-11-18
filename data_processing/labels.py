"""Per-observation reference labels derived from the .route file.

The route lists the arcs driven, in order, but gives no time for each arc, so labels
for individual GPS points have to be constructed. Two versions:

v0 (this file, quick): each point takes the closest arc among the route's arcs. When
    several route arcs are equally close (within TIE_TOL_M, typically the two
    directions of the same road), the one whose bearing best matches the GPS heading
    wins. Known weakness: ignores route order, so loops and out-and-back trips can
    be mislabelled. v1 (order-preserving alignment) fixes this.

Labels are reference information, not verified truth: label_dist_m is kept so that
points far from every route arc can be flagged as low-confidence.
"""
from __future__ import annotations

import numpy as np

from .geometry import dist_matrix

TIE_TOL_M = 1.0
LOW_CONF_DIST_M = 50.0
CHUNK = 2000


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def labels_v0(pts_xy, heading_deg, route, arc_A, arc_B, arc_bearing):
    """Return (label_arc, label_dist_m) for every point."""
    route_arcs = np.unique(route)
    A, B, brg = arc_A[route_arcs], arc_B[route_arcs], arc_bearing[route_arcs]
    n = len(pts_xy)
    label = np.empty(n, dtype=np.int64)
    ldist = np.empty(n, dtype=np.float64)
    for s in range(0, n, CHUNK):
        D = dist_matrix(pts_xy[s:s + CHUNK], A, B)                  # (c, M)
        dmin = D.min(axis=1, keepdims=True)
        tied = D <= dmin + TIE_TOL_M
        h = heading_deg[s:s + CHUNK]
        ang = _angle_diff(h[:, None], brg[None, :])                 # NaN where heading unknown
        ang = np.where(np.isnan(ang), 0.0, ang)
        # among tied arcs pick best heading match, then the closer one
        score = np.where(tied, ang + D * 1e-3, np.inf)
        j = score.argmin(axis=1)
        label[s:s + CHUNK] = route_arcs[j]
        ldist[s:s + CHUNK] = D[np.arange(len(j)), j]
    return label, ldist
