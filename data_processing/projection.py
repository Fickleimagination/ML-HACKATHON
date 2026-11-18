"""Per-trip metric projection.

Trips are spread across continents, so a single projected CRS cannot serve them all.
Each trip gets an azimuthal equidistant (AEQD) projection centred on the mean GPS
position. Distances from the centre are exact and distortion within a few tens of km
is far below GPS noise (metres), so ordinary Euclidean geometry in metres is valid
inside one trip.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


class TripProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        proj = f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs"
        self._fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
        self._inv = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, track: np.ndarray) -> "TripProjection":
        """Centre on the GPS observations (available at prediction time too)."""
        return cls(np.nanmean(track[:, 0]), np.nanmean(track[:, 1]))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])
