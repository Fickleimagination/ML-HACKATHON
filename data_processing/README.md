# data_processing (Person A)

Turns raw trip files into per-observation labels and per-candidate geometry.

## Run

```bash
pip install numpy pandas pyproj shapely pyarrow
# raw data at data/raw/Student folder/{train,student_test}/<trip_id>/
python -m data_processing.build_dataset --raw "data/raw/Student folder" --out data/processed
```

Takes about 20 s for all 90 trips. Evaluator trips are never touched.

## Outputs (`data/processed/`)

### `points/<trip_id>.parquet`: one row per GPS observation (row order = `.track`)

| Column | Meaning |
| --- | --- |
| `trip_id`, `partition` | Trip and partition (`train` / `student_test`) |
| `obs_idx` | Row number in `.track` (never renumbered; nothing is dropped) |
| `t`, `lon`, `lat` | Raw values |
| `x`, `y` | Metres in the trip's own AEQD projection (centre = mean GPS position) |
| `dt_prev`, `dt_next` | Seconds to previous / next observation |
| `speed_ms` | Step distance / `dt_prev` |
| `heading_deg` | Bearing (0 = north, clockwise) from points i-3 to i+3 in the same segment; NaN if moved < 2 m |
| `segment_id` | New segment after a gap > 30 s, time going backwards, or an invalid point |
| `flag_invalid`, `flag_dup_time`, `flag_time_backwards`, `flag_jump` | Quality flags (jump = implied speed > 70 m/s) |
| `label_arc` | v0 reference arc for this point |
| `label_dist_m` | Distance from the point to `label_arc` |
| `label_low_conf` | `label_dist_m` > 50 m: label not trustworthy |
| `n_candidates` | Candidate arcs within 50 m |
| `label_in_candidates` | Whether `label_arc` is among the candidates |

### `candidates/<trip_id>.parquet`: one row per (observation, arc within 50 m), sorted by `obs_idx`, then `dist_m`

| Column | Meaning |
| --- | --- |
| `trip_id`, `obs_idx`, `cand_arc` | Keys; `cand_arc` = row number in `.arcs` |
| `dist_m` | Perpendicular distance from point to arc |
| `proj_t` | Position of the closest point along the arc (0 = from-node, 1 = to-node) |
| `arc_len_m`, `arc_bearing_deg` | Arc geometry |
| `from_node`, `to_node` | Node ids (for degree, reverse-twin and graph features) |
| `label` | 1 if `cand_arc == label_arc`, else 0 |

Also `summary_by_trip.csv` (per-trip inventory and quality numbers) and `config.json`.

## Label version v0

Each point takes the closest arc among the route's arcs. Ties within 1 m (usually the two directions of one road) go to the arc whose bearing best matches the GPS heading. It ignores route order, so loops can be mislabelled; v1 (order-preserving alignment) replaces it later with the same columns.

## Numbers to know (v0, 50 m radius)

| | Train | Student-test |
| --- | --- | --- |
| Observations | 208,417 | 19,160 |
| Candidate rows | 1,866,677 | 209,038 |
| Positive rate | 10.0% | 9.2% |
| Candidate miss rate | 10.8% | 0.1% |
| Low-confidence labels | 10.8% | 0.1% |

The train miss rate comes almost entirely from four trips:

- **Trip 54:** after about point 2,700, the GPS leaves the supplied road network; no arc within 50 m.
- **Trip 33:** points 2,085–5,126 follow real roads, but the reference route is about 120 m away; the reference is suspect there.
- **Trips 2 and 96:** the trips end off the network.

Baselines on usable points (label in candidates and not low-confidence), all trips:

- Nearest arc: 53.7% Top-1. 69% of its errors are the reverse arc of the same road.
- Nearest arc + heading-direction rule: 83.7% Top-1.

## Rules for B and C

- Train only on rows where the point has `label_in_candidates & ~label_low_conf`; keep those points in evaluation, counted as misses.
- Cross-validation must group by `trip_id`.
- Subsample every 3rd `obs_idx` for model training, the same for every model.
