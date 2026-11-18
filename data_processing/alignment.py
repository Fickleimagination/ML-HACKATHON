"""v1 reference labels: order-preserving alignment of GPS observations to the route.

Problem: the .route file lists the arcs driven, in order, with no timestamps. We assign
every GPS observation i a position p_i in the route list (0..M-1) such that positions
never go backwards in time (p_0 <= p_1 <= ... <= p_{N-1}), minimising

    sum_i dist(point_i, route arc p_i)                                   (metres)
  + sum_i lambda_i * (length of route arcs skipped between p_{i-1} and p_i)

The label for observation i is route[p_i].

Why positions and not arc ids: a route can use the same arc more than once (loops,
out-and-back trips). Positions keep each visit separate; the order constraint forces
early observations onto early visits.

Rules (these must be identical in the API's scoring code):
  * Distances: point-to-segment distance in the trip's AEQD projection (projection.py),
    centred on the mean of the valid GPS points, rounded to whole millimetres. All DP
    arithmetic is integer (mm), so results are identical across machines and library builds.
  * Skip penalty: SKIP_PENALTY per metre of route arcs jumped over (integer mm:
    penalty = floor(1000*SKIP_PENALTY * cumulative_route_mm / 1000) differences). Waived (lambda_i = 0)
    when the time step into observation i exceeds GAP_SECONDS, because after a long gap
    the vehicle may legitimately have covered many arcs. Arcs skipped before the first
    observation and after the last are penalised the same way.
  * Repeated arcs: handled by working on route positions (above).
  * Invalid observations (NaN / out-of-range coordinates): excluded from the DP and given
    label -1 / position -1. They do not break the order constraint.
  * Ties: when staying on the current position and jumping ahead cost the same, stay;
    among equal jump origins take the earliest; at the end take the lowest final position.
    np.argmin / strict '<' comparisons implement this deterministically.
  * Nothing is dropped or reordered: output arrays are indexed by the original .track row.
"""
from __future__ import annotations

import numpy as np

from .geometry import dist_matrix

SKIP_PENALTY = 0.1      # cost (m) per metre of skipped route length
GAP_SECONDS = 30.0      # no skip penalty after a time step longer than this
CHUNK = 1024


def align_route(pts_xy: np.ndarray, t: np.ndarray, valid: np.ndarray, route: np.ndarray,
                arc_A: np.ndarray, arc_B: np.ndarray,
                skip_penalty: float = SKIP_PENALTY, gap_s: float = GAP_SECONDS):
    """Return (label_arc, route_pos, dist_m), each of length N (original .track rows).

    pts_xy: (N, 2) projected points; t: (N,) seconds; valid: (N,) bool;
    route: (M,) arc ids; arc_A/arc_B: (E, 2) projected arc end points for all arcs.
    """
    n = len(pts_xy)
    label = np.full(n, -1, dtype=np.int64)
    pos = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.nan)
    idx = np.flatnonzero(valid)
    if len(idx) == 0 or len(route) == 0:
        return label, pos, dist

    A, B = arc_A[route], arc_B[route]
    M = len(route)
    jidx = np.arange(M)
    # All costs are integers in millimetres so every machine makes identical decisions:
    # floating-point noise (~1e-7 m between numpy/pyproj builds) cannot flip a near-tie.
    seg_mm = np.rint(np.hypot(*(B - A).T) * 1000.0).astype(np.int64)
    S_mm = np.concatenate([[0], np.cumsum(seg_mm)])            # S_mm[k] = length of route[0:k]
    lam_milli = int(round(skip_penalty * 1000))
    Sp = (lam_milli * S_mm) // 1000                             # skip penalty (mm) up to position k
    zeros = np.zeros_like(Sp)
    BIG = np.iinfo(np.int64).max // 4

    tv = t[idx]
    no_pen = np.zeros(len(idx), dtype=bool)
    no_pen[1:] = np.diff(tv) > gap_s                            # waive penalty after long gaps

    back = np.empty((len(idx), M), dtype=np.int32)              # best previous position
    Cbuf = None
    D = None
    for k in range(len(idx)):
        if k % CHUNK == 0:
            Cbuf = np.rint(dist_matrix(pts_xy[idx[k:k + CHUNK]], A, B) * 1000.0).astype(np.int64)
        c = Cbuf[k % CHUNK]
        if k == 0:
            D = c + Sp[:M]                                      # arcs skipped before the start
            back[0] = jidx
            continue
        P = zeros if no_pen[k] else Sp
        # jump from j' < j costs D[j'] + P[j] - P[j'+1]
        v = D - P[1:M + 1]
        run = np.minimum.accumulate(v)
        new_min = np.empty(M, dtype=bool)
        new_min[0] = True
        new_min[1:] = v[1:] < run[:-1]                           # strict: earliest origin on ties
        arg = np.maximum.accumulate(np.where(new_min, jidx, 0))
        jump = np.full(M, BIG, dtype=np.int64)
        jump[1:] = P[1:M] + run[:-1]                             # origins strictly before j
        jump_from = np.zeros(M, dtype=np.int32)
        jump_from[1:] = arg[:-1]
        stay_better = D <= jump                                  # tie -> stay
        back[k] = np.where(stay_better, jidx, jump_from)
        D = c + np.where(stay_better, D, jump)

    final = D + (Sp[M] - Sp[1:M + 1])                           # arcs skipped after the end
    j = int(np.argmin(final))                                    # tie -> lowest position
    path = np.empty(len(idx), dtype=np.int64)
    for k in range(len(idx) - 1, -1, -1):
        path[k] = j
        j = back[k, j]

    pos[idx] = path
    label[idx] = route[path]
    P = pts_xy[idx]
    dist[idx] = dist_matrix_rowwise(P, A[path], B[path])
    return label, pos, dist


def dist_matrix_rowwise(P, A, B):
    from .geometry import project_pairs
    return project_pairs(P, A, B)[0]
