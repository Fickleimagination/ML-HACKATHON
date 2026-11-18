import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.predictor import load_predictor, validate_prediction
from api.scoring import evaluate
from api.trips import discover_trips, load_trip
from visualisation.maps import build_map

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    root = Path(os.getenv("PATHSNAP_DATA", "data/student_test")).resolve()
    app.state.trips = discover_trips(root)
    app.state.predictor, app.state.predictor_name = load_predictor()
    app.state.inference_lock = Lock()
    logger.info("Loaded %d trips from %s | predictor=%s",
                len(app.state.trips), root, app.state.predictor_name)
    yield


app = FastAPI(title="PathSnap", lifespan=lifespan)
origins = [
    origin.strip() for origin in os.getenv(
        "PATHSNAP_CORS_ORIGINS", "http://127.0.0.1:8080,http://localhost:8080"
    ).split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_private_network=bool(origins) and "*" not in origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/ui/", include_in_schema=False)
@app.get("/ui/{asset}", include_in_schema=False)
def serve_evaluator(asset: str = "index.html"):
    if asset not in {"index.html", "app.js", "styles.css"}:
        raise HTTPException(404, "Evaluator asset is not available")
    return FileResponse(Path(__file__).resolve().parents[1] / "ui" / asset)


class PredictRequest(BaseModel):
    trip_id: str = Field(pattern=r"^[0-9]{1,32}$")
    include_map: bool = True


@app.get("/trips")
def list_trips(request: Request):
    return {
        "trips": list(request.app.state.trips),
        "predictor": request.app.state.predictor_name,
    }


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    state = request.app.state
    stem = state.trips.get(body.trip_id)
    if stem is None:
        raise HTTPException(404, "Trip ID is not available")
    started = perf_counter()
    try:
        track, nodes, network = load_trip(stem)
        # Score against the original trip arrays even if the plug-in preprocesses in place.
        with state.inference_lock:
            inference_started = perf_counter()
            result = state.predictor(track.copy(), nodes.copy(), network.copy())
            predicted, route, confidence, candidates = validate_prediction(
                result, len(track), len(network)
            )
            inference_ms = (perf_counter() - inference_started) * 1000
        metrics = evaluate(stem, track, nodes, network, predicted, route, candidates)
        map_html = (
            build_map(track, nodes, network, route).get_root().render()
            if body.include_map else None
        )
    except Exception:
        logger.exception("Prediction failed | trip=%s", body.trip_id)
        raise HTTPException(500, "Trip processing failed. See the API terminal.") from None

    logger.info("Prediction complete | trip=%s observations=%d inference_ms=%.1f "
                "elapsed_ms=%.1f top1=%s", body.trip_id, len(track), inference_ms,
                (perf_counter() - started) * 1000, metrics["top1_accuracy"])
    return {
        "trip_id": body.trip_id,
        "predictor": state.predictor_name,
        "observations": len(track),
        "arcs": predicted.tolist(),
        "confidence": confidence.tolist() if confidence is not None else None,
        "route": route.tolist(),
        "candidate_arcs": [row.tolist() for row in candidates] if candidates is not None else None,
        "diagnostics": result.get("diagnostics"),
        "metrics": metrics,
        "inference_ms": round(inference_ms, 2),
        "map_html": map_html,
    }
