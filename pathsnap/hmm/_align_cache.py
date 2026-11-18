"""One-off migration: compare each existing results/cache/<trip>.npz (built
from the old candidates_v0 + gap_s=60) against what A's aligned candidates
+ gap_s=30 would produce, and only re-run the expensive precompute_trip
(Dijkstra-heavy) for trips where candidates or segmentation actually
changed. Not part of the pipeline -- run manually once, then keep for the
record of what changed.

    python -m pathsnap.hmm._align_cache [data/train] [data/student_test]
"""
import sys
from pathlib import Path

import numpy as np

from pathsnap.hmm.decode_cache import CACHE_DIR, load_cache, precompute_trip
from pathsnap.hmm.prepared_trip import prepare_trip
from pathsnap.hmm.routing import load_trip
from pathsnap.hmm.viterbi import split_segments
from pathsnap.predict import load_config

NEW_GAP_S = 30.0


def _segs_from_cache(cache):
    segs = []
    pos = 0
    for n in cache["seg_lens"]:
        segs.append(cache["seg_idx"][pos:pos + n])
        pos += n
    return segs


def _cand_sets(seg_idx, n_cands, cand_arc):
    offsets = np.concatenate([[0], np.cumsum(n_cands)])
    return {int(seg_idx[k]): frozenset(cand_arc[offsets[k]:offsets[k + 1]].tolist())
            for k in range(len(seg_idx))}


def differs(folder, name, cfg):
    old = load_cache(name)
    old_segs = _segs_from_cache(old)
    old_sets = _cand_sets(old["seg_idx"], old["n_cands"], old["cand_arc"])

    track, nodes, arcs = load_trip(folder, name)
    _, _, times, cands, _, _ = prepare_trip(track, nodes, arcs, radius=cfg["radius"], gap_s=NEW_GAP_S)
    new_segs = split_segments(times, cands, NEW_GAP_S)
    new_sets = {t: frozenset(c[0] for c in cands[t]) for seg in new_segs for t in seg}

    if len(old_segs) != len(new_segs) or any(
            not np.array_equal(a, b) for a, b in zip(old_segs, new_segs)):
        return True, "segmentation"
    if old_sets != new_sets:
        return True, "candidates"
    return False, None


def main(folders):
    cfg = load_config()
    changed, unchanged = [], []
    for name in sorted(p.stem for p in CACHE_DIR.glob("*.npz")):
        folder = next((str(Path(f) / name) for f in folders if (Path(f) / name).is_dir()), None)
        if folder is None:
            print(f"{name}: SKIP (not found in {folders})")
            continue
        is_diff, reason = differs(folder, name, cfg)
        if is_diff:
            print(f"{name}: CHANGED ({reason}) -- rebuilding")
            precompute_trip(folder, name, {**cfg, "gap_s": NEW_GAP_S})
            changed.append(name)
        else:
            unchanged.append(name)
    print(f"\n{len(changed)} changed, {len(unchanged)} unchanged "
          f"(of {len(changed) + len(unchanged)} cached trips)")
    if changed:
        print("changed:", changed)


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/train", "data/student_test"])
