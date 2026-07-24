# Layer 9 — Serving Layer: FastAPI + Docker + SageMaker

## What This Layer Does

Wraps the production LightGBM model in a REST API so any application, dashboard, or downstream system can request predictions without knowing anything about how the model was trained.

---

## API Architecture

```
                          ┌─────────────────────────────────────────┐
                          │            FastAPI App                  │
                          │          (serving/main.py)              │
                          │                                         │
  Client request ────────►│  ┌─────────────────────────────────┐   │
  POST /predict           │  │  Pydantic validation             │   │
  {features...}           │  │  (rejects bad input with 422)    │   │
                          │  └──────────────┬──────────────────┘   │
                          │                 │                       │
                          │                 ▼                       │
                          │  ┌─────────────────────────────────┐   │
                          │  │  LightGBM champion model         │   │
                          │  │  (loaded from MLflow at startup) │   │
                          │  └──────────────┬──────────────────┘   │
                          │                 │                       │
                          │                 ▼                       │
                          │  ┌─────────────────────────────────┐   │
                          │  │  SHAP TreeExplainer              │   │
                          │  │  (initialized once at startup)   │   │
                          │  └──────────────┬──────────────────┘   │
                          │                 │                       │
                          │                 ▼                       │
  Response ◄─────────────│  {direction, confidence}                │
  {direction: "up",       │                                         │
   confidence: 0.673}     └─────────────────────────────────────────┘
```

---

## Endpoints

### POST /predict

```
Request:
  {
    "rolling_1h_mean": 2.0,
    "rolling_24h_std": 0.45,
    "volume_zscore": 1.82,
    "pos": 0.82,
    "neg": 0.05,
    "neu": 0.13,
    "gnn_dims": [0.12, -0.05, 0.33, ...]   ← 64 floats
  }

Processing:
  concat features → (69,) vector → LightGBM.predict_proba → [P(down), P(up)]
  direction = "up" if P(up) > 0.5 else "down"
  confidence = max(P(up), P(down))

Response:
  {
    "direction": "up",
    "confidence": 0.673
  }
```

### POST /explain

```
Same request body as /predict.

Processing:
  SHAP TreeExplainer.shap_values(feature_vector)
  → one SHAP value per feature (69 values)

  SHAP value interpretation:
  Positive → this feature pushed prediction toward "up"
  Negative → this feature pushed prediction toward "down"

Response:
  {
    "shap_values": [0.14, -0.03, 0.22, -0.08, 0.01, ...]
  }

  Values in order:
  [rolling_1h_mean, rolling_24h_std, volume_zscore, pos, neg, neu,
   gnn_dim_0, ..., gnn_dim_63]
```

### GET /shap-summary

```
Returns a SHAP summary plot (beeswarm) over the last 200 predictions.
Shows which features have the most impact on model output across many requests.

Response:
  {
    "image": "data:image/png;base64,iVBORw0KGgo..."
  }

Useful for: monitoring, presentations, model auditing.
```

### GET /health

```
Response: {"status": "ok"}
Used by: Docker health checks, load balancers, uptime monitoring
```

### GET /metrics

```
Prometheus-format metrics scraped every 15 seconds:

  prediction_total{direction="up"}    42
  prediction_total{direction="down"}  31
  request_latency_seconds{quantile="0.5"}   0.021
  request_latency_seconds{quantile="0.95"}  0.048
  request_errors_total                       0
```

---

## Why FastAPI and Not Flask

```
Feature              Flask              FastAPI
─────────────────────────────────────────────────
Concurrency          Sync (blocking)    Async (non-blocking)
Auto Swagger docs    No (needs Flask-RESTful + flasgger)  YES (built-in at /docs)
Request validation   Manual             Pydantic (automatic)
Bad input behavior   500 / manual check 422 Unprocessable Entity (automatic)
Type hints           Optional           First-class (powers validation + docs)
Performance          WSGI               ASGI (Uvicorn/Gunicorn)
```

FastAPI's Pydantic validation means a request missing `pos` or with `gnn_dims` as a string returns a clear 422 error — the model code never sees the bad input.

---

## Model Loading at Startup

```python
# serving/main.py (startup)

@app.on_event("startup")
async def load_model():
    client = MlflowClient()
    version = client.get_model_version_by_alias("FinPlatform", "champion")
    model = mlflow.lightgbm.load_model(f"models:/FinPlatform/{version.version}")

    # Initialize SHAP explainer ONCE — not per request
    # TreeExplainer on 300 trees takes ~200ms to init
    explainer = shap.TreeExplainer(model)
```

Initialization happens once. Every `/predict` and `/explain` call reuses the same objects.

---

## Docker Containerization

```
┌─────────────────────────────────────────────────┐
│  serving/Dockerfile                             │
│                                                 │
│  FROM python:3.11-slim                          │
│  COPY requirements.txt .                        │
│  RUN pip install -r requirements.txt            │
│  COPY serving/ .                                │
│  EXPOSE 8000                                    │
│  CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]  │
└─────────────────────────────────────────────────┘

Benefits of containerization:
  - Reproducible environment: same behavior on any machine
  - Dependency isolation: no conflicts with other Python packages
  - Portable: runs locally, on EC2, on SageMaker, on Kubernetes
  - Easy scaling: run multiple replicas behind a load balancer
```

---

## AWS SageMaker Path

For the managed deployment sprint, `serving/inference.py` wraps the same model in the SageMaker serving contract:

```
SageMaker contract:

  model_fn(model_dir)     ← load model from model_dir
       │
  predict_fn(data, model) ← run inference
       │
  output_fn(prediction, accept) ← serialize response

  SageMaker calls these functions.
  The model logic inside is identical to the FastAPI version.

FastAPI (local/Docker):
  You manage: server, routing, scaling

SageMaker managed endpoint:
  AWS manages: server, routing, auto-scaling, health checks, TLS
  You provide: serving/inference.py + model artifact
```

---

## Request Flow End-to-End

```
User / dashboard / predict_loop.py

        │
        │  POST http://localhost:8000/predict
        │  {rolling_1h_mean: 2.0, pos: 0.82, gnn_dims: [...]}
        │
        ▼
FastAPI (port 8000)
        │
        ├── Pydantic validates input schema
        │   └── missing field? → 422 Unprocessable Entity
        │
        ├── Concatenate features → (69,) numpy array
        │
        ├── lgbm_model.predict_proba([[features]])
        │   → [[0.327, 0.673]]
        │
        ├── direction = "up", confidence = 0.673
        │
        ├── log to Prometheus counter + latency histogram
        │
        └── return {"direction": "up", "confidence": 0.673}
        │
        ▼
        Client receives response (~21ms p50)
```

---

## Files in This Layer

| File | Role |
|------|------|
| [serving/main.py](../../serving/main.py) | FastAPI app: all endpoints, model loading, SHAP |
| [serving/Dockerfile](../../serving/Dockerfile) | Container definition for the API |
| [serving/inference.py](../../serving/inference.py) | SageMaker serving entry point |
| [scripts/predict_loop.py](../../scripts/predict_loop.py) | Background loop: calls /predict for all tickers → feeds Grafana |
