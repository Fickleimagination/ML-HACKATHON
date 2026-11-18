# data_processing (Person A)

Turns raw trip files into per-observation reference labels and per-candidate geometry, and holds the one preprocessing entry point that training, the API and scoring all share.

## Run

```bash
pip install numpy pandas pyproj shapely pyarrow scipy matplotlib folium
# raw data at data/raw/Student folder/{train,student_test}/<trip_id>/
python -m data_processing.build_dataset --raw "data/raw/Student folder" --out data/processed   # ~40 s
python -m data_processing.eda --raw "data/raw/Student folder" --processed data/processed --out results/eda
```

Evaluator-test trips are never touched by these scripts.

## Entry points for B, C and D

```python
from data_processing.io_utils import load_trip
from data_processing.preprocess import preprocess_trip, reference_labels, reference_labels_for_dir, SETTINGS

trip = load_trip("path/to/<trip_id>", with_route=False)       # API: no route at inference
prep = preprocess_trip(trip.track, trip.nodes, trip.arcs)      # same code as training
prep.obs    # one row per .track row (obs_idx), flags, heading, segment_id, x/y
prep.cand   # candidate arcs within 50 m with geometry
prep.proj   # projection: prep.proj.to_lonlat(xy) for map output

ref = reference_labels_for_dir("path/to/<trip_id>")           # server-side scoring only
```

| Module | Contents |
| --- | --- |
| `io_utils.py` | Load the four trip files |
| `projection.py` | Per-trip AEQD projection (metres) |
| `cleaning.py` | Quality flags, segments, heading |
| `geometry.py` | Point-to-arc distance and projection |
| `candidates.py` | Spatial index; arcs within radius |
| `alignment.py` | v1 order-preserving alignment (reference labels) |
| `labels.py` | v0 labels (kept only for comparison) |
| `preprocess.py` | Shared entry point + settings |
| `build_dataset.py` | Writes the parquet files |
| `eda.py` | EDA tables, figures, Folium maps |

## Settings (`preprocess.SETTINGS`, also in `data/processed/config.json`)

| Setting | Value |
| --- | --- |
| Projection | AEQD centred on the mean lon/lat of valid GPS points |
| Candidate radius | 50 m, all directed arcs |
| Segment break | time step > 30 s, time going backwards, or an invalid point |
| Alignment skip penalty | 0.1 per metre of skipped route, waived after a step > 30 s |
| Low-confidence label | distance to labelled arc > 50 m |
| GPS offset correction | none |

## Reference labels (v1, order-preserving alignment)

Each GPS observation gets a position in the route list. Positions never go backwards in time. The alignment minimises total point-to-arc distance plus a small penalty for skipping route arcs. The label is the arc at that position.

- **Repeated arcs:** the alignment works on route positions, not arc ids, so each visit is separate; order forces early points onto early visits.
- **Large gaps:** the skip penalty is waived after a time step > 30 s, so the vehicle may jump many arcs.
- **Invalid points:** excluded from the alignment; label -1, position -1.
- **Ties:** stay on the current position over jumping; the earliest origin among equal jumps; the lowest final position.
- **Reproducibility:** distances are rounded to whole millimetres and all alignment arithmetic is integer, so every machine gives identical labels (floating-point differences between library builds cannot flip a near-tie).
- **Rows:** nothing is dropped or reordered; output row k = `.track` row k.
- **Known limitation:** where the GPS drives a stretch the reference route doesn't include in that direction, the order constraint can assign the reverse arc. The reference route is treated as reference information, not verified truth.

## Outputs (`data/processed/`)

### `points/<trip_id>.parquet`: one row per GPS observation (row order = `.track`)

| Column | Meaning |
| --- | --- |
| `trip_id`, `partition` | Trip and partition |
| `obs_idx` | Row number in `.track` |
| `t`, `lon`, `lat` | Raw values |
| `x`, `y` | Metres in the trip's projection |
| `dt_prev`, `dt_next` | Seconds to previous / next observation |
| `speed_ms` | Step distance / `dt_prev` |
| `heading_deg` | Bearing (0 = north, clockwise) from points i-3 to i+3 in the same segment; NaN if moved < 2 m |
| `segment_id` | Segment number |
| `flag_invalid`, `flag_dup_time`, `flag_time_backwards`, `flag_jump` | Quality flags (jump = implied speed > 70 m/s) |
| `label_arc` | **v1 reference arc** |
| `label_route_pos` | Position of that arc in the `.route` list |
| `label_dist_m` | Distance from point to `label_arc` |
| `label_low_conf` | `label_dist_m` > 50 m |
| `label_arc_v0` | Old v0 label, for comparison only |
| `n_candidates` | Candidate arcs within 50 m |
| `label_in_candidates` | Whether `label_arc` is among the candidates |

### `candidates/<trip_id>.parquet`: one row per (observation, arc within 50 m), sorted by `obs_idx`, then `dist_m`

| Column | Meaning |
| --- | --- |
| `trip_id`, `partition`, `obs_idx`, `cand_arc` | Keys; `cand_arc` = row number in `.arcs` |
| `dist_m` | Perpendicular distance from point to arc |
| `proj_t` | Position of the closest point along the arc (0 = from-node, 1 = to-node) |
| `arc_len_m`, `arc_bearing_deg` | Arc geometry |
| `from_node`, `to_node` | Node ids |
| `from_degree`, `to_degree` | Arcs touching each end node (in + out), whole network |
| `reverse_arc` | Arc id of the same road in the opposite direction, or -1 if one-way |
| `label` | 1 if `cand_arc == label_arc`, else 0 |

Also `summary_by_trip.csv` and `config.json`.

## Numbers to know (v1, 50 m radius)

| | Train | Student-test |
| --- | --- | --- |
| Observations | 208,417 | 19,160 |
| Candidate rows (per observation) | 1,866,677 (9.0) | 209,038 (10.9) |
| Positive rate | 9.9% | 9.2% |
| Candidate miss rate = low-confidence labels | 11.2% | 0.14% |
| Labels changed from v0 | 7.6% | 9.4% |

Most v0-to-v1 changes are points within 1 m of the junction between two consecutive route arcs. On loop trips such as 56, v1 also fixes points that v0 had put on the wrong visit of a repeated road.

The train miss rate comes almost entirely from four trips:

- **Trip 54:** after about point 2,700, the GPS leaves the supplied road network.
- **Trip 33:** for points 2,085–5,126 (55 minutes), the vehicle moves around inside a ~120 × 160 m area away from the reference route, probably a car park or yard; the reference route is ~120 m away.
- **Trips 2 and 96:** the trips end off the network.

Baselines on usable points (label in candidates and not low-confidence):

- Nearest arc: 54.7% Top-1. 73% of its errors are the reverse arc of the same road.
- Nearest arc + heading-direction rule: 85.4% Top-1.

## Rules for B and C

- Train only on points with `label_in_candidates & ~label_low_conf`; keep the rest in evaluation, counted as misses.
- Cross-validation must group by `trip_id`.
- Subsample every 3rd `obs_idx` for model training, the same for every model.
- Use `preprocess_trip` at inference so features match training exactly.
