# PathSnap: Distributed Map-Matching and Route Reconstruction

UE24CS352A Machine Learning hackathon, PES University RR campus.

PathSnap matches noisy GPS observations to a road network. For every GPS point it finds nearby candidate road arcs, scores them with a trained classifier, and reconstructs a connected route with an HMM decoded by Viterbi. A FastAPI backend on one laptop serves predictions to an evaluator UI on a second laptop.

| Member | Role | Code |
| --- | --- | --- |
| Bhavyaa Garg | A: data processing, reference labels, EDA | `data_processing/` |
| [name] | B: features and models | `features/`, `models/` *(to be pushed)* |
| Arundhati K | C: Bayesian optimisation, HMM and Viterbi | `pathsnap/` |
| [name] | D: API, UI and deployment | `api/`, `ui/` *(to be pushed)* |

## Repository layout

```text
data_processing/     A: loaders, per-trip projection, cleaning, candidates, v1 labels, EDA
  preprocess.py      shared entry point used by training, the API and scoring
features/            B: feature extraction                         (to be pushed)
models/              B: training, comparison, plots                (to be pushed)
pathsnap/
  predict.py         single prediction entry point: predict_trip(track, nodes, arcs)
  hmm/               C: road graph, Dijkstra, HMM/Viterbi, decoder cache and tuning
  evaluation/        C: Top-1, continuity, route precision/recall (never imported by prediction code)
  experiments/       C: student-test evaluation, sampling-rate test, plots
api/                 D: FastAPI backend                            (to be pushed)
ui/                  D: evaluator interface                        (to be pushed)
artifacts/           hmm_config.json and saved model files
results/             tables, figures, Folium maps, Optuna studies, logs
data/                raw and processed data (not in git)
```

## Setup

Python 3.10 or newer.

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
  processed/                                                 created by step 1 below
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
# [B: add the exact commands, e.g.]
# python -m features.build_features ...
# python -m models.train_models ...
# python -m models.make_plots ...
```

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

Decoder settings live in `artifacts/hmm_config.json`.

### 4. Classifier tuning and integration (B and C)

```bash
# [B/C: Optuna study for the chosen classifier, and how its probabilities enter the HMM (alpha)]
```

## Deployment: two laptops

| Laptop | Runs | Holds |
| --- | --- | --- |
| Laptop 1 | FastAPI backend | Trained model, HMM config, road networks, evaluator trips and their reference routes |
| Laptop 2 | Evaluator UI | Nothing but the UI; never receives reference routes or labels |

```bash
# Laptop 1 (API)  [D: confirm]
# uvicorn api.main:app --host 0.0.0.0 --port 8000

# Laptop 2 (UI)   [D: confirm]
# set the API address to http://<laptop-1-IP>:8000, then start the UI
```

Both laptops must be on the same network; a phone hotspot is the most reliable. On macOS, find laptop 1's IP with `ipconfig getifaddr en0` and allow incoming connections when prompted. The API logs every request in its terminal.

Flow: the UI lists the evaluation trips → the evaluator picks one → the UI sends the trip ID → the API runs `preprocess_trip`, the classifier and the HMM → the API scores the prediction server-side with the aligner → the UI shows the predicted route and metrics.

## Rules we follow

- Evaluator-test data never enters training, tuning, feature selection or model selection.
- Reference labels and routes stay on the API laptop and are never sent to the UI or used by the prediction code.
- No hardcoded predictions; every number in the report comes from code and outputs in this repository.
- Reported accuracy is agreement with reference labels (v1, order-preserving alignment), not verified ground truth.
