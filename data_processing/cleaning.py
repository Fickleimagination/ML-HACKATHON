"""GPS observation quality flags, segmentation and heading.

Rows are never dropped: evaluation is per observation, so obs_idx (the row number in
.track) must stay stable. Problems are recorded as boolean flags instead, and the
sequence is split into segments at large time gaps so the HMM can restart there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GAP_SECONDS = 30.0        # new segment when the time step exceeds this
JUMP_SPEED_MS = 70.0      # ~250 km/h: faster implied speed is treated as a GPS jump
HEADING_HALF_WINDOW = 3   # heading from points i-3 .. i+3
MIN_HEADING_MOVE_M = 2.0  # below this displacement the heading is undefined (stationary)


def bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass bearing in degrees: 0 = north (+y), 90 = east (+x)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def observation_table(track: np.ndarray, xy: np.ndarray, gap_s: float = GAP_SECONDS) -> pd.DataFrame:
    n = len(track)
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]

    invalid = ~np.isfinite(track).all(axis=1) | (np.abs(lon) > 180) | (np.abs(lat) > 90) | ((lon == 0) & (lat == 0))

    dt_prev = np.full(n, np.nan)
    dt_prev[1:] = np.diff(t)
    dt_next = np.full(n, np.nan)
    dt_next[:-1] = np.diff(t)

    step = np.full(n, np.nan)
    step[1:] = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt_prev > 0, step / dt_prev, np.nan)

    dup_time = dt_prev == 0
    time_backwards = dt_prev < 0
    jump = speed > JUMP_SPEED_MS

    # segment breaks: big gap, time going backwards, or an invalid point
    brk = (dt_prev > gap_s) | time_backwards | invalid
    brk[0] = True
    segment_id = np.cumsum(brk) - 1

    # heading from a centred window, restricted to the same segment
    k = HEADING_HALF_WINDOW
    lo = np.clip(np.arange(n) - k, 0, n - 1)
    hi = np.clip(np.arange(n) + k, 0, n - 1)
    # pull window ends back inside the point's own segment
    while True:
        bad_lo = segment_id[lo] != segment_id
        bad_hi = segment_id[hi] != segment_id
        if not (bad_lo.any() or bad_hi.any()):
            break
        lo[bad_lo] += 1
        hi[bad_hi] -= 1
    dx = xy[hi, 0] - xy[lo, 0]
    dy = xy[hi, 1] - xy[lo, 1]
    move = np.hypot(dx, dy)
    heading = np.where(move >= MIN_HEADING_MOVE_M, bearing_deg(dx, dy), np.nan)

    return pd.DataFrame({
        "obs_idx": np.arange(n, dtype=np.int32),
        "t": t,
        "lon": lon, "lat": lat,
        "x": xy[:, 0].astype(np.float32), "y": xy[:, 1].astype(np.float32),
        "dt_prev": dt_prev.astype(np.float32), "dt_next": dt_next.astype(np.float32),
        "speed_ms": speed.astype(np.float32),
        "heading_deg": heading.astype(np.float32),
        "segment_id": segment_id.astype(np.int32),
        "flag_invalid": invalid, "flag_dup_time": dup_time,
        "flag_time_backwards": time_backwards, "flag_jump": jump,
    })
