import os
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.lightgbm
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = (
    ["rolling_1h_mean", "rolling_24h_std", "volume_zscore", "pos", "neg", "neu"]
    + [f"gnn_dim_{i}" for i in range(64)]
)

REQUEST_COUNT = Counter("fin_predict_requests_total", "Total prediction requests", ["endpoint"])
REQUEST_LATENCY = Histogram("fin_predict_latency_seconds", "Prediction latency", ["endpoint"],
                            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
ERROR_COUNT = Counter("fin_predict_errors_total", "Total prediction errors")

_state: dict = {}
_recent_X: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    from pathlib import Path
    mlflow.set_tracking_uri(Path("./mlruns").resolve().as_uri())
    model_uri = "models:/fin-platform-lgbm/Production"
    log.info("Loading model from %s", model_uri)
    _state["model"] = mlflow.lightgbm.load_model(model_uri)
    _state["explainer"] = shap.TreeExplainer(_state["model"])
    log.info("Model loaded. Ready.")
    yield
    _state.clear()


app = FastAPI(title="Financial Intelligence Platform API", lifespan=lifespan)


class PredictRequest(BaseModel):
    rolling_1h_mean: float
    rolling_24h_std: float
    volume_zscore: float = 0.0
    pos: float = Field(ge=0.0, le=1.0)
    neg: float = Field(ge=0.0, le=1.0)
    neu: float = Field(ge=0.0, le=1.0)
    gnn_dims: list[float] = Field(default_factory=lambda: [0.0] * 64)


def build_feature_vector(req: PredictRequest) -> np.ndarray:
    gnn = req.gnn_dims
    if len(gnn) < 64:
        gnn = gnn + [0.0] * (64 - len(gnn))
    return np.array(
        [req.rolling_1h_mean, req.rolling_24h_std, req.volume_zscore,
         req.pos, req.neg, req.neu] + gnn[:64],
        dtype=np.float32,
    ).reshape(1, -1)


@app.post("/predict")
def predict(req: PredictRequest):
    REQUEST_COUNT.labels(endpoint="predict").inc()
    start = time.time()
    try:
        x = build_feature_vector(req)
        prob = float(_state["model"].predict(x)[0])
        direction = "up" if prob >= 0.5 else "down"
        _recent_X.append(x[0].tolist())
        if len(_recent_X) > 200:
            _recent_X.pop(0)
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.time() - start)
        return {"direction": direction, "confidence": round(prob, 4)}
    except Exception as exc:
        ERROR_COUNT.inc()
        log.error("Predict error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/explain")
def explain(req: PredictRequest):
    REQUEST_COUNT.labels(endpoint="explain").inc()
    start = time.time()
    try:
        x = build_feature_vector(req)
        shap_vals = _state["explainer"].shap_values(x)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        REQUEST_LATENCY.labels(endpoint="explain").observe(time.time() - start)
        return {"shap_values": shap_vals[0].tolist()}
    except Exception as exc:
        ERROR_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/shap-summary")
def shap_summary():
    if not _recent_X:
        raise HTTPException(status_code=404, detail="No predictions yet — call /predict first")
    X = np.array(_recent_X, dtype=np.float32)
    shap_vals = _state["explainer"].shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X, feature_names=FEATURE_COLS, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return {"image": f"data:image/png;base64,{encoded}"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
