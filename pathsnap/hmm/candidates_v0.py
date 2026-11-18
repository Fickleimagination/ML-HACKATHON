"""Throwaway candidate generator so the HMM can run before A's parquet files exist.

Swap for A's candidate tables once they land (same fields: arc, dist, frac).
"""
import numpy as np
from shapely import STRtree, linestrings, points


def build_index(net):
    lines = linestrings(np.stack([net.xy[net.u], net.xy[net.v]], axis=1))
    return STRtree(lines)


def candidates(net, tree, pts_xy, radius=50.0):
    """For each point -> list of (arc_id, perp_distance_m, fraction_along_arc)."""
    pts = points(pts_xy)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius)
    out = [[] for _ in range(len(pts_xy))]
    p0, p1 = net.xy[net.u[ai]], net.xy[net.v[ai]]
    seg = p1 - p0
    L2 = np.maximum((seg ** 2).sum(1), 1e-9)
    f = np.clip(((pts_xy[pi] - p0) * seg).sum(1) / L2, 0, 1)
    d = np.linalg.norm(p0 + f[:, None] * seg - pts_xy[pi], axis=1)
    for i, a, dd, ff in zip(pi, ai, d, f):
        out[i].append((int(a), float(dd), float(ff)))
    return out
