# PathSnap: Distributed Map-Matching and Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University RR campus.

PathSnap matches noisy GPS observations to a road network. For every GPS point it finds nearby candidate road arcs, scores them with a trained classifier, and reconstructs a connected route with an HMM decoded by Viterbi. A FastAPI backend on one laptop serves predictions to an evaluator UI on a second laptop.

| Member | Role | Code |
| --- | --- | --- |
| Bhavyaa Garg | A: data processing, reference labels, EDA | `data_processing/` |
| Person B | Features and models | `features/`, `models/` |
| Arundhati K | C: Bayesian optimisation, HMM and Viterbi | `pathsnap/` |
| Bhavesh V | D: API, UI and deployment | `api/`, `ui/`, `visualisation/` |

## Repository layout

```text
data_processing/     A: loaders, per-trip projection, cleaning, candidates, v1 labels, EDA
  preprocess.py      shared entry point used by training, the API and scoring
features/            B: feature extraction from processed trips
models/              B: training, comparison and plots
pathsnap/
  predict.py         single prediction entry point: predict_trip(track, nodes, arcs)
  hmm/               C: road graph, Dijkstra, HMM/Viterbi, decoder cache and tuning
  evaluation/        C: Top-1, continuity, route precision/recall (never imported by prediction code)
  experiments/       C: student-test evaluation, sampling-rate test, plots
api/                 D: FastAPI backend and server-side scoring
ui/                  D: static evaluator interface
visualisation/       D: Folium maps for the UI and report
scripts/check_api.py D: HTTP readiness and prediction parity checks
artifacts/           HMM config and fitted models under models/
results/             metrics, figures, Folium maps, Optuna studies and demo evidence
data/                raw and processed data (not in git)
```

## Setup

Use Python 3.11 or newer. The demo was verified on Python 3.13.

```bash
git clone https://github.com/Fickleimagination/ML-HACKATHON.git
cd ML-HACKATHON
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The data is not committed. Unzip the organisers' `Student folder.zip` and place the partitions so the layout is:

```text
data/
  train/<trip_id>/<trip_id>.track .nodes .arcs .route        80 trips
  student_test/<trip_id>/...                                 10 trips
  evaluator_test/<trip_id>/...                               10 reserved trips, API laptop only
  processed/                                                 created by step 1 below
  features/                                                  created by step 2 below
```

Each trip has four header-less, whitespace-separated files: `.track` (lon, lat, t), `.nodes` (lon, lat; row = node id), `.arcs` (from node, to node; row = arc id) and `.route` (ordered arc ids). Evaluator-test trips are only ever placed on the API laptop and are never used for training, tuning or model selection.

## Reproducing the results

Run from the repository root, in this order.

### 1. Data processing and EDA (A)

```bash
python -m data_processing.build_dataset --raw data --out data/processed        # labels + candidates, ~40 s
python -m data_processing.eda           --raw data --processed data/processed --out results/eda
python -m data_processing.radius_curve  --raw data --processed data/processed --out results/radius
python -m data_processing.label_quality --raw data --processed data/processed --out results/labels
```

Outputs: `data/processed/points/` and `data/processed/candidates/` (one parquet file per trip), `results/eda/` (figures, tables, Folium maps, data-quality log), `results/radius/`, `results/labels/`. Column definitions and settings: `data_processing/README.md`.

### 2. Features and models (B)

```bash
python -m features.build_features --processed data/processed --raw data --out data/features
python -m models.train_models --features data/features --out results --artifacts artifacts/models
python -m models.make_plots results
```

The trainer defaults to all seven models. Use `--models lr gnb rf`, for example, to run a subset. It uses five-fold CV grouped by trip, samples every third training observation, and keeps training points with a trusted reference label present in the candidates. See the [model comparison](results/model_comparison.csv), [per-model metrics](results/models/) and [saved fitted artifacts](artifacts/models/). The comparison reports agreement with the aligned reference labels. KNN fits a 50,000-row training sample. The artifacts retain each estimator and its ordered feature list.

### 3. HMM decoder: tuning and evaluation (C)

```bash
python -m pathsnap.hmm.decode_cache data/train               # cache road distances per training trip (results/cache/, not in git)
python -m pathsnap.hmm.tune_decoder                          # Optuna TPE, 60 trials, seed 42, training trips only
python -m pathsnap.hmm.uturn_plateau                         # U-turn penalty plateau check
python -m pathsnap.experiments.eval_student_test             # nearest vs default vs tuned on student-test
python -m pathsnap.experiments.sampling_rate                 # every 1st/5th/10th/30th point, common point subset
python -m pathsnap.experiments.plots                         # figures in results/figures/
python -m pathsnap.hmm.make_expected_outputs data/student_test   # reference outputs for the API check
```

Decoder settings live in `artifacts/hmm_config.json`. The original `pathsnap.hmm.batch_decode` command remains available to reproduce the legacy v0 candidate experiment. For current decoder metrics, use the student-test evaluation command above.

### 4. Classifier tuning and integration (B and C)

```bash
python -m pathsnap.classifier.tune_classifier     # Optuna TPE on B's LightGBM, same CV rows/folds as models.train_models
python -m pathsnap.classifier.refit_compare       # refit best params on all 80 training trips -> lgbm_tuned.joblib
python -m pathsnap.classifier.alpha_grid          # choose alpha (0/.25/.5/.75/1) by OOF training Top-1 - 0.002x disconnections
python -m pathsnap.classifier.final_comparison    # nearest / classifier-only / distance HMM / classifier+HMM on student-test
```

B's `lgbm.joblib` is untouched; the tuned classifier is `artifacts/models/lgbm_tuned.joblib`, loaded by `pathsnap.predict._classifier_probs` (no labels, no `.route`, same candidate order as the decoder, asserted). The study was time-boxed to 20 completed trials (`results/optuna_classifier.db`, `results/optuna_classifier_trials.csv`) rather than the originally planned 40, under a hard wall-clock budget; its `cv_macro_f1` ties B's untuned default (0.9274 vs 0.9273) but `test_top1` is slightly lower (0.8718 vs 0.8818) at ~18x faster inference -- see `results/classifier_tuned_vs_untuned.csv`. The decoder's sigma/beta/uturn_penalty were **not** retuned jointly with alpha (no new decoder Optuna study, same time budget); alpha=0.5 was chosen from a 5-point grid using 2-fold out-of-fold classifier probabilities on all 80 training trips against the existing decoder cache (`results/alpha_grid.csv`). Final comparison: `results/final_comparison.csv`, `results/figures/final_comparison.png` -- classifier+HMM keeps almost all of classifier-only's Top-1 (0.882 vs 0.884) while fixing route coherence (mean disconnections 0.6 vs 14.6 per trip, route precision 0.82 vs 0.65), and beats distance-only HMM on Top-1 (0.882 vs 0.867) at similar continuity.

New dependency: `lightgbm==4.7.0` (already in `requirements.txt` for B's models). No new packages for this step.

## Deployment: two laptops

Laptop 1 hosts the FastAPI backend, decoder configuration and the trip dataset. Laptop 2 only needs the static `ui/` directory and a browser. The active predictor is `pathsnap.predict:predict_trip`, the classifier+HMM decoder (shared candidates, 30-second gap rule, B's tuned LightGBM blended in at alpha=0.5) described in step 4 above.

From the repository root on laptop 1, with the environment active:

```sh
# Student-test rehearsal. The API defaults to data/student_test.
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Final-demo launch. Replace the running rehearsal server first.
PATHSNAP_DATA=data/evaluator_test python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

For the final-demo launch in PowerShell, set `$env:PATHSNAP_DATA = "data/evaluator_test"` before running the same Uvicorn command. Set `PATHSNAP_PREDICTOR=nearest` for the nearest-arc baseline or set it to a compatible `module:function` to exercise another predictor. The callable takes raw `track`, `nodes` and `arcs` arrays and returns observation arc IDs and an ordered route. It may include per-observation confidence (a probability or null when the selected arc has no classifier score), candidate lists in predictor order, and decoder diagnostics.

On laptop 2, run this from the repository root with the UI directory present:

```sh
python -m http.server 8080 --bind 127.0.0.1 --directory ui
```

1. Join the same Wi-Fi or hotspot on both laptops. On macOS, find laptop 1’s Wi-Fi address with `ipconfig getifaddr en0`. On Windows, use `ipconfig`.
2. On laptop 2, visit `http://LAPTOP_1_IP:8000/trips` to verify reachability. Open [the evaluator](http://127.0.0.1:8080), enter `http://LAPTOP_1_IP:8000` under **API connection**, and select **Connect**. For a one-laptop rehearsal, use `http://127.0.0.1:8000`.
3. Select a trip and click **Run prediction**. The Folium map’s GPS, road network and predicted route layers render offline. Enable the optional OpenStreetMap layer only if you want an online basemap.
4. For the reserved partition, run `python -m scripts.check_api --readiness-only` on laptop 1 to verify all 10 trip responses without printing or exporting scores. Keep the API terminal visible during the demo for request and inference logs.

Set `PATHSNAP_CORS_ORIGINS=http://127.0.0.1:8080` on laptop 1 to limit browser requests to that UI origin. The default allows cross-origin requests from the demo UI.

## API and verification

- `GET /trips` returns the trip IDs and active predictor.
- `POST /predict` takes `{"trip_id": "00000052"}` and returns predicted arcs, the route, candidate arc IDs in predictor order, decoder diagnostics, aggregate metrics, inference time and a Folium map. Use `"include_map": false` for a smaller response.
- FastAPI's interactive API documentation is available at `/docs`.

```sh
curl http://127.0.0.1:8000/trips
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"trip_id":"00000052","include_map":false}'
python -m scripts.check_api --api http://127.0.0.1:8000 \
  --direct-data data/student_test --csv results/deployment/student_test_classifier_hmm.csv
```

These examples use the API’s default student-test partition. The smoke check calls the same HTTP endpoints as the UI for all 10 trips and verifies response structure, scoring and map layers. The `--direct-data` option compares the response exactly with a direct run on the same machine. The `--expected-dir results/expected_outputs` option performs a strict comparison with C’s saved fixtures. On the development Mac, the API matches local direct inference on all 10 student-test trips. For the final classifier and HMM, all ten saved fixtures match point-level arcs, candidate ordering and diagnostics. Nine fixture routes match every arc ID. Trip `00000009` differs at three parallel connector arc IDs with the same directed endpoints. Confidence differences are at floating-point rounding scale. On the API laptop, capture the terminal showing `POST /predict` and HTTP 200 for the report.

## Scoring and report maps

After inference, the API calls the team's shared v1 reference alignment in `data_processing.preprocess`. It minimizes point-to-segment distance plus a penalty of 0.1 per metre of route skipped, waived across GPS gaps over 30 seconds. Top-1 uses all labelled observations. The high-confidence Top-1 also excludes observations over 50 metres from their aligned arc. Unlabelled observations are excluded from reference-based point scores. Candidate macro F1 averages the positive and negative binary F1 across candidates for labelled observations. Continuity counts adjacent matched predictions that stay on the same arc or follow directed endpoints. Route precision and recall measure overlap of unique directed arcs weighted by their lengths. A metric with no usable denominator is shown as unavailable.

For an offline report comparison, run this command locally on a machine with the student-test reference files. Run it with the team's predictor configured to generate a final-model map.

```sh
python -m visualisation.maps 00000052 results/student_test_00000052.html --reference
```

The report command can include a reference layer. The evaluator API keeps reference geometry and per-observation labels on the server. See [the report handoff](results/report_sections_6_7.md) for the combined predictor evidence and final capture checklist.

## Rules we follow

- Evaluator-test data never enters training, tuning, feature selection or model selection.
- Reference labels and routes stay on the API laptop and are never sent to the UI or used by the prediction code.
- No hardcoded predictions; every number in the report comes from code and outputs in this repository.
- Reported accuracy is agreement with reference labels (v1, order-preserving alignment), not verified ground truth.
