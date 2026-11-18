"""Road network in metric coordinates + bounded, cached shortest paths.

Uses scipy.sparse.csgraph (NOT networkx) so 300k-arc trips stay cheap.
"""
import numpy as np
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def load_trip(folder, name):
    """Read the four raw files of one trip. `name` is the file stem."""
    track = np.loadtxt(f"{folder}/{name}.track", ndmin=2)        # lon, lat, t
    nodes = np.loadtxt(f"{folder}/{name}.nodes", ndmin=2)        # lon, lat
    arcs = np.loadtxt(f"{folder}/{name}.arcs", dtype=np.int64, ndmin=2)
    return track, nodes, arcs


def make_projector(lon, lat):
    """Azimuthal-equidistant projection centred on the trip -> metres."""
    lon0, lat0 = float(np.mean(lon)), float(np.mean(lat))
    crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


class RoadNetwork:
    def __init__(self, nodes_xy, arcs):
        self.xy = np.asarray(nodes_xy, dtype=np.float64)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        self.u, self.v = self.arcs[:, 0], self.arcs[:, 1]
        self.length = np.linalg.norm(self.xy[self.v] - self.xy[self.u], axis=1)
        n = len(self.xy)
        # keep the shortest arc if there are parallel duplicates
        w = np.maximum(self.length, 1e-3)
        order = np.argsort(-w)
        self.G = csr_matrix((w[order], (self.u[order], self.v[order])), shape=(n, n))
        self.arc_of = {}                                   # (u, v) -> arc id
        for a in np.argsort(self.length)[::-1]:
            self.arc_of[(int(self.u[a]), int(self.v[a]))] = int(a)
        self._cache = {}

    def is_reverse(self, a, b):
        return self.u[a] == self.v[b] and self.v[a] == self.u[b]

    def node_dists(self, src, limit):
        """Distances from node `src` to every node within `limit` metres (cached)."""
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2]
        d, pred = dijkstra(self.G, directed=True, indices=src,
                           limit=limit, return_predecessors=True)
        reach = np.flatnonzero(np.isfinite(d))
        dist = dict(zip(reach.tolist(), d[reach].tolist()))
        preds = dict(zip(reach.tolist(), pred[reach].tolist()))
        self._cache[src] = (limit, dist, preds)
        return dist, preds

    def route_distance(self, a, fa, b, fb, limit):
        """Driving distance from arc a at fraction fa to arc b at fraction fb."""
        if a == b and fb >= fa:
            return (fb - fa) * self.length[a]
        dist, _ = self.node_dists(int(self.v[a]), limit)
        mid = dist.get(int(self.u[b]))
        if mid is None:
            return np.inf
        return (1 - fa) * self.length[a] + mid + fb * self.length[b]

    def connecting_arcs(self, a, b, limit=5000.0):
        """Arc ids strictly between arc a and arc b on a shortest path ([] if none)."""
        src, dst = int(self.v[a]), int(self.u[b])
        if src == dst:
            return []
        _, preds = self.node_dists(src, limit)
        if dst not in preds:
            return []
        path, n = [], dst
        while n != src:
            p = preds[n]
            if p < 0:
                return []
            path.append(self.arc_of[(p, n)])
            n = p
        return path[::-1]

    def clear_cache(self):
        self._cache.clear()
