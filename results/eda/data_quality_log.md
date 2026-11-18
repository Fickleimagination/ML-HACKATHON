# Data-quality log (Person A)

Partitions examined: train (80 trips) and student_test (10 trips). Evaluator-test was not opened.
All numbers come from `data_processing/eda.py` and `data_processing/build_dataset.py`; tables are in `results/eda/tables/`.

## Dataset structure

| | Train | Student-test |
| --- | --- | --- |
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Observations per trip (median, min–max) | 1,529 (361–19,313) | 1,420 (262–3,958) |
| Arcs per trip (median, max) | 19,527 (307,392) | 13,145 (56,027) |
| Route length in arcs (median) | 280 | 250 |

Trips are spread across Europe, Russia, North and South America, East and Southeast Asia, and Africa.

## Issues found and decisions

| # | Issue | Evidence | Decision |
| --- | --- | --- | --- |
| 1 | Coordinates are lon/lat in degrees, worldwide | Trip centres span -100° to 137° longitude | Project each trip with its own azimuthal equidistant projection (metres), centred on the mean GPS position |
| 2 | No missing values or impossible coordinates | 0 NaN in track and nodes; 0 out-of-range coordinates; 0 bad node or arc ids | No imputation needed; checks kept in code |
| 3 | Timestamps are clean but irregular | 0 duplicate and 0 backward timestamps. Most steps are 1 s, but trips 22, 4, 90, 39 have only 53–58% 1 s steps | Keep `dt_prev`/`dt_next` as features |
| 4 | Very large time gaps | Max gaps: 153,064 s (trip 90), 52,810 s (78), 2,658 s (69), 2,161 s (21) | Split each trip into segments at gaps > 30 s (`segment_id`); HMM restarts at each segment |
| 5 | GPS jumps | Implied speed > 70 m/s on 0.07% of train and 0.005% of student-test observations | Flag (`flag_jump`), keep the row |
| 6 | `.route` has no timestamps | Ordered arc list only | Build per-observation labels ourselves with an order-preserving alignment (v1). v0 (nearest route arc, heading tie-break) kept for comparison; v1 changes 7.6% (train) and 9.4% (student-test) of labels, mostly at arc junctions and on loops |
| 7 | GPS noise around the route | Distance to reference arc: median 5.7 m (train), 5.3 m (student-test); p90 13.5 m on student-test | Candidate radius 50 m (to be confirmed by the miss-rate curve) |
| 8 | GPS leaves the road network | Trip 54 after ~obs 2,700, and the ends of trips 2 and 96: no arc within 50 m | Labels > 50 m from the route flagged `label_low_conf`; excluded from training, counted as misses in evaluation |
| 9 | GPS stays off the reference route | Trip 33, obs 2,085–5,126 (55 min): the vehicle moves around inside a ~120 × 160 m area (probably a car park or yard), ~120 m from the reference route | Same flag; reference route treated as reference information, not verified truth |
| 10 | Most roads are two-way | 92.3% of arcs have a reverse arc | Directed arcs kept as separate candidates; nearest-arc gets 54.7% Top-1 and 73% of its errors are the reverse arc; nearest + heading rule gets 85.4%; heading is essential |
| 11 | Route discontinuities | 13 breaks in train (trips 2, 19, 33, 41, 42, 69, 75, 80, 86, 89), 1 in student-test (76) | HMM evaluation must not assume the reference route is fully connected |
| 12 | Directed network is not fully connected | Largest strongly connected component holds 95% of nodes on average; trip 5 only 64%, with 47% of its route outside it | Shortest-path routing needs a fallback when no directed path exists |
| 13 | Duplicate arcs and self-loops | Duplicate arcs in 60 trips (median 2.5, max 322); self-loops in 9 trips; only 11 of 34,491 route arcs have an exact duplicate | Negligible for labels; noted |
| 14 | Undefined heading | 1.0% (train), 0.6% (student-test) of points moved < 2 m in their ±3-point window | Heading = NaN; models must handle missing heading |
| 15 | Route skips a stretch the GPS drives in that direction | Seen on out-and-back trip 61 | The order-preserving alignment can then assign the reverse arc; accepted as a limitation of the reference route |
| 16 | Reference labels that point against the direction of travel | For moving points (> 3 m/s, heading defined, not low-confidence), the labelled arc points > 90° from the GPS heading for a median of 3.1% of points per trip (v0 labels: 2.0%). Outliers: trips 33 (35%), 10 (25%), 77 (21%, student-test), 82 (20%), 61 (19%). Part is heading noise at sharp turns; the rest is the reference route not containing the travelled direction. Source: `results/labels/label_quality_by_trip.csv` (`data_processing/label_quality.py`) | Labels kept; trip 77's low score is mainly a reference-route issue, so the decoder is not tuned to it; reported as a limitation |

## Label summary (v1 order-preserving alignment, 50 m radius)

| | Train | Student-test |
| --- | --- | --- |
| Candidate rows | 1,866,677 | 209,038 |
| Candidates per observation (mean) | 9.0 | 10.9 |
| Positive rate | 9.9% | 9.2% |
| Candidate miss rate | 11.2% | 0.14% |
| Low-confidence labels | 11.2% | 0.14% |

The train miss rate comes almost entirely from trips 54, 33, 2 and 96 (rows 8 and 9 above).
