"""HMM map matching (Newson & Krumm 2009 style) with Viterbi in log space.

Hidden state at point t = one candidate arc (with its projection fraction).
Emission  : alpha * log p_classifier + (1 - alpha) * log N(dist; sigma)
            (alpha = 0 -> distance only, used until B's model exists)
Transition: -|route_dist - straight_dist| / beta, minus a U-turn penalty
            if the next arc is the reverse twin; NO_PATH_LOGP if unreachable.
Gaps      : a time gap > gap_s or a point with no candidates ends the segment;
            segments are decoded separately and joined with shortest paths.
"""
import numpy as np

DEFAULTS = dict(sigma=10.0, beta=5.0, alpha=0.0, uturn_penalty=3.0,
                gap_s=60.0, no_path_logp=-50.0)


def _emission(cands, probs, cfg):
    d = np.array([c[1] for c in cands])
    e = -0.5 * (d / cfg["sigma"]) ** 2
    if probs is not None and cfg["alpha"] > 0:
        e = cfg["alpha"] * np.log(np.clip(probs, 1e-9, 1)) + (1 - cfg["alpha"]) * e
    return e


def _decode_segment(net, xy, cands, probs, idx, cfg):
    """Viterbi over the points in `idx` (all have >=1 candidate)."""
    score = _emission(cands[idx[0]], None if probs is None else probs[idx[0]], cfg)
    back = []
    for prev_t, t in zip(idx[:-1], idx[1:]):
        straight = float(np.linalg.norm(xy[t] - xy[prev_t]))
        limit = 2 * straight + 200.0
        P, C = cands[prev_t], cands[t]
        trans = np.full((len(P), len(C)), cfg["no_path_logp"])
        for i, (a, _, fa) in enumerate(P):
            for j, (b, _, fb) in enumerate(C):
                rd = net.route_distance(a, fa, b, fb, limit)
                if np.isfinite(rd):
                    trans[i, j] = -abs(rd - straight) / cfg["beta"]
                    if net.is_reverse(a, b):
                        trans[i, j] -= cfg["uturn_penalty"]
        total = score[:, None] + trans
        back.append(total.argmax(0))
        score = total.max(0) + _emission(C, None if probs is None else probs[t], cfg)
    # backtrack
    path = [int(score.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    path.reverse()
    return [cands[t][k][0] for t, k in zip(idx, path)]


def split_segments(times, cands, gap_s):
    segs, cur = [], []
    for t in range(len(cands)):
        broken = not cands[t] or (cur and times[t] - times[cur[-1]] > gap_s)
        if broken and cur:
            segs.append(cur)
            cur = []
        if cands[t]:
            cur.append(t)
    if cur:
        segs.append(cur)
    return segs


def build_route(net, point_arcs, connector_limit=5000.0):
    """Dedup consecutive repeats and insert shortest-path connectors between
    route arcs that aren't already head-to-tail. Returns (route, disconnections).
    Split out of decode_trip so callers that already have a point_arcs array
    (e.g. the Optuna cache scorer) can build the route without re-decoding."""
    route, disconnections = [], 0
    for a in point_arcs[point_arcs >= 0].tolist():
        if route and a == route[-1]:
            continue
        if route and net.v[route[-1]] != net.u[a]:
            link = net.connecting_arcs(route[-1], a, connector_limit)
            if not link:
                disconnections += 1
            route += link
        route.append(a)
    return route, disconnections


def decode_trip(net, xy, times, cands, probs=None, cfg=None):
    """Returns per-point arc ids (-1 = no candidate), a route with connector
    arcs inserted, and stats {segments, disconnections}."""
    cfg = {**DEFAULTS, **(cfg or {})}
    point_arcs = np.full(len(xy), -1, dtype=np.int64)
    segs = split_segments(times, cands, cfg["gap_s"])
    for seg in segs:
        point_arcs[seg] = _decode_segment(net, xy, cands, probs, seg, cfg)

    route, disconnections = build_route(net, point_arcs, cfg.get("connector_limit", 5000.0))
    net.clear_cache()
    return point_arcs, route, {"segments": len(segs), "disconnections": disconnections}


def continuity(net, route):
    """Share of consecutive route arcs that are actually connected."""
    if len(route) < 2:
        return 1.0
    ok = sum(net.v[a] == net.u[b] for a, b in zip(route[:-1], route[1:]))
    return ok / (len(route) - 1)
