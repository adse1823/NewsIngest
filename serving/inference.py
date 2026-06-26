"""SageMaker serving entry point."""
import os
import json
import numpy as np
import mlflow.lightgbm


def model_fn(model_dir: str):
    model_uri = os.path.join(model_dir, "model")
    return mlflow.lightgbm.load_model(model_uri)


def input_fn(request_body: str, content_type: str = "application/json"):
    data = json.loads(request_body)
    gnn = data.get("gnn_dims", [0.0] * 64)
    if len(gnn) < 64:
        gnn = gnn + [0.0] * (64 - len(gnn))
    features = [
        data["rolling_1h_mean"],
        data["rolling_24h_std"],
        data.get("volume_zscore", 0.0),
        data["pos"],
        data["neg"],
        data["neu"],
    ] + gnn[:64]
    return np.array(features, dtype=np.float32).reshape(1, -1)


def predict_fn(input_data: np.ndarray, model):
    prob = float(model.predict_proba(input_data)[0, 1])
    return {"direction": "up" if prob >= 0.5 else "down", "confidence": round(prob, 4)}


def output_fn(prediction: dict, accept: str = "application/json") -> str:
    return json.dumps(prediction)
