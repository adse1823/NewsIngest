import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── load serving.main by file path, bypassing package resolution ──────────────

def _load_serving_main():
    if "serving.main" in sys.modules:
        return sys.modules["serving.main"]
    path = Path(__file__).parent.parent.parent / "serving" / "main.py"
    spec = importlib.util.spec_from_file_location("serving.main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serving.main"] = mod
    sys.modules.setdefault("serving", mod)
    spec.loader.exec_module(mod)
    return mod


# ── mock objects ──────────────────────────────────────────────────────────────

class _MockModel:
    def __init__(self, prob: float = 0.7):
        self._prob = prob

    def predict(self, X):
        return np.array([self._prob] * len(X))


class _MockExplainer:
    def shap_values(self, X):
        n = X.shape[0] if hasattr(X, "shape") else 1
        return np.zeros((n, 70))


# ── shared fixture ────────────────────────────────────────────────────────────

def _make_client(prob: float = 0.7):
    """
    Load serving.main via importlib, patch out all external calls in the
    lifespan, and return a context-managed TestClient.
    """
    sm = _load_serving_main()
    from fastapi.testclient import TestClient

    mock_version = MagicMock(version="9", run_id="abc123")

    mlflow_patch    = patch.object(sm, "mlflow")
    artifact_patch  = patch.object(sm, "_find_artifact", return_value="/fake/path")
    explainer_patch = patch("shap.TreeExplainer", return_value=_MockExplainer())

    mock_mlflow = mlflow_patch.start()
    artifact_patch.start()
    explainer_patch.start()

    mock_mlflow.tracking.MlflowClient.return_value \
        .get_model_version_by_alias.return_value = mock_version
    mock_mlflow.lightgbm.load_model.return_value = _MockModel(prob=prob)

    sm._recent_X.clear()

    client = TestClient(sm.app, raise_server_exceptions=True)
    client.__enter__()

    yield client

    client.__exit__(None, None, None)
    mlflow_patch.stop()
    artifact_patch.stop()
    explainer_patch.stop()


@pytest.fixture
def client():
    yield from _make_client(prob=0.7)


VALID_BODY = {
    "rolling_1h_mean": 1.5,
    "rolling_24h_std": 0.03,
    "volume_zscore": 0.8,
    "pos": 0.6,
    "neg": 0.1,
    "neu": 0.3,
}


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_correct_shape(client):
    r = client.post("/predict", json=VALID_BODY)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"direction", "confidence"}
    assert body["direction"] in ("up", "down")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_high_prob_is_up(client):
    r = client.post("/predict", json=VALID_BODY)
    assert r.json()["direction"] == "up"
    assert r.json()["confidence"] == pytest.approx(0.7, abs=0.0001)


def test_predict_low_prob_is_down():
    for c in _make_client(prob=0.3):
        r = c.post("/predict", json=VALID_BODY)
        assert r.json()["direction"] == "down"
        assert r.json()["confidence"] == pytest.approx(0.3, abs=0.0001)


def test_predict_missing_required_field(client):
    body = {k: v for k, v in VALID_BODY.items() if k != "rolling_1h_mean"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_pos_out_of_range(client):
    r = client.post("/predict", json={**VALID_BODY, "pos": 1.5})
    assert r.status_code == 422


def test_predict_gnn_dims_auto_padded(client):
    r = client.post("/predict", json={**VALID_BODY, "gnn_dims": [0.1, 0.2]})
    assert r.status_code == 200


def test_predict_populates_recent_x(client):
    client.post("/predict", json=VALID_BODY)
    r = client.get("/shap-summary")
    assert r.status_code == 200
    assert "image" in r.json()


def test_shap_summary_404_before_any_prediction(client):
    r = client.get("/shap-summary")
    assert r.status_code == 404


def test_explain_returns_shap_values(client):
    r = client.post("/explain", json=VALID_BODY)
    assert r.status_code == 200
    body = r.json()
    assert "shap_values" in body
    assert isinstance(body["shap_values"], list)
    assert len(body["shap_values"]) == 70


def test_predict_boundary_prob_is_up():
    for c in _make_client(prob=0.5):
        r = c.post("/predict", json=VALID_BODY)
        assert r.json()["direction"] == "up"
        assert r.json()["confidence"] == pytest.approx(0.5, abs=0.0001)


def test_predict_neg_out_of_range(client):
    r = client.post("/predict", json={**VALID_BODY, "neg": -0.1})
    assert r.status_code == 422


def test_predict_neu_out_of_range(client):
    r = client.post("/predict", json={**VALID_BODY, "neu": 1.1})
    assert r.status_code == 422


def test_metrics_returns_prometheus_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"fin_predict_requests_total" in r.content
