"""Point-to-segment geometry in projected metres."""
from __future__ import annotations

import numpy as np


def project_pairs(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Row-wise projection of points P onto segments A->B (all arrays (K, 2)).

    Returns (distance_m, t) where t in [0, 1] is the fraction along the segment of the
    closest point (0 = from-node, 1 = to-node).
    """
    AB = B - A
    L2 = (AB ** 2).sum(axis=1)
    safe = np.where(L2 > 0, L2, 1.0)
    t = np.where(L2 > 0, ((P - A) * AB).sum(axis=1) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    Q = A + t[:, None] * AB
    d = np.hypot(P[:, 0] - Q[:, 0], P[:, 1] - Q[:, 1])
    return d, t


def dist_matrix(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distances from every point in P (N, 2) to every segment A->B (M, 2): (N, M)."""
    AB = B - A
    L2 = (AB ** 2).sum(axis=1)
    safe = np.where(L2 > 0, L2, 1.0)
    PA = P[:, None, :] - A[None, :, :]
    t = np.clip(np.where(L2 > 0, (PA * AB[None]).sum(axis=2) / safe, 0.0), 0.0, 1.0)
    Q = A[None] + t[..., None] * AB[None]
    return np.hypot(P[:, None, 0] - Q[..., 0], P[:, None, 1] - Q[..., 1])
