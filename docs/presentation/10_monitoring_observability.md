# Layer 10 — Monitoring & Observability

## What This Layer Does

Keeps the system healthy after deployment. Detects when data or model behavior drifts from baseline, surfaces operational metrics in real time, and triggers automatic retraining when drift exceeds the threshold.

---

## Three Layers of Observability

```
┌──────────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY STACK                              │
│                                                                      │
│   Layer A: DATA & MODEL DRIFT                                        │
│   Evidently AI — daily drift reports — auto-retrain trigger          │
│                                                                      │
│   Layer B: REQUEST METRICS                                           │
│   Prometheus — scrapes /metrics every 15s                            │
│   Grafana — dashboards for predictions, latency, drift, versions     │
│                                                                      │
│   Layer C: PIPELINE ORCHESTRATION & ALERTING                         │
│   Airflow — schedules daily DAG — retries failed steps               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Layer A: Evidently AI — Drift Detection

### What Drift Is

```
TRAINING TIME (reference window: first 30 days of production data)

  Feature: rolling_1h_mean
  Distribution:  mean=1.2,  std=0.4
                 ████████
                █████████████
               ███████████████████
              ██████████████████████
         ─────────────────────────────►
         0.2  0.5  1.0  1.5  2.0  2.5

CURRENT WINDOW (new data, 3 months later)

  Feature: rolling_1h_mean
  Distribution has SHIFTED (news volume increased dramatically)
  mean=3.1, std=0.9
                                  ████████
                                 █████████████
                                ███████████████████
         ─────────────────────────────────────────►
         0.2  0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0

  The model was trained on data centered at 1.2.
  It's now seeing data centered at 3.1.
  Its learned thresholds are no longer calibrated.
  → Performance degrades silently.
```

### How Evidently Detects It

```
monitoring/drift_report.py (runs via Airflow daily):

  reference_data = features from first 30 days of production
  current_data   = features from today

  Evidently runs statistical tests per feature:
    - Kolmogorov-Smirnov test (continuous features)
    - Chi-squared test (categorical features)
    Each test produces: is_drifted (bool) + p-value

  Share of drifted columns = drifted_features / total_features

  If share > 0.30 (30% threshold):
    → trigger retraining
    → Airflow runs modeling/train.py
    → new model version registered in MLflow
    → new version promoted to champion
    → FastAPI reloads champion on next request
```

### Drift Decision Flow

```
                         Evidently drift report
                                │
                    ┌───────────▼────────────┐
                    │  share_drifted > 0.30? │
                    └───────────┬────────────┘
                          │           │
                         YES          NO
                          │           │
                          ▼           ▼
                    Trigger       Log "no drift"
                    retrain       in Airflow
                          │
                          ▼
                   modeling/train.py
                          │
                          ▼
                   New model version
                   in MLflow registry
                          │
                          ▼
                   Promote to champion alias
                          │
                          ▼
                   FastAPI serves new model
                   (next request picks it up)
```

---

## Layer B: Prometheus + Grafana

### Metric Collection

```
FastAPI app (serving/main.py)
       │
       │  every prediction:
       │    prediction_counter.labels(direction=direction).inc()
       │    request_latency.observe(elapsed_seconds)
       │
       ▼
  GET /metrics  (Prometheus format)
       │
       ▼
Prometheus scrapes every 15 seconds:
  prediction_total{direction="up"}     = 842
  prediction_total{direction="down"}   = 611
  request_latency_seconds{q="0.5"}     = 0.021
  request_latency_seconds{q="0.95"}    = 0.048
  request_errors_total                 = 0
       │
       ▼
Grafana queries Prometheus and renders dashboards
```

### Grafana Dashboard Panels

```
┌────────────────────────────────────────────────────────────────────┐
│                    GRAFANA DASHBOARD                               │
│                                                                    │
│  ┌──────────────────────────┐  ┌───────────────────────────────┐  │
│  │  Predictions per minute  │  │  Request latency              │  │
│  │                          │  │                               │  │
│  │  ▁▂▃▄▅▆▅▄▃▄▅▆▇▆▅▄▃▂▁   │  │  p50:  ──────────────── 21ms │  │
│  │  up ████ down ████       │  │  p95:  ──────────────── 48ms │  │
│  │                          │  │                               │  │
│  └──────────────────────────┘  └───────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────┐  ┌───────────────────────────────┐  │
│  │  Drift score over time   │  │  Model version history        │  │
│  │                          │  │                               │  │
│  │  0.35─── ─── ─── ─── ── │  │  v3  ████████████████████     │  │
│  │  0.30 ·threshold·········│  │  v2  ████████                 │  │
│  │  0.15────────────────    │  │  v1  ███                      │  │
│  │  retrain ↑ triggered     │  │                               │  │
│  └──────────────────────────┘  └───────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
scrape_configs:
  - job_name: 'fastapi'
    scrape_interval: 15s
    static_configs:
      - targets: ['api:8000']
```

Prometheus connects to the FastAPI container by Docker network hostname.

---

## Layer C: Airflow — Pipeline Orchestration

### DAG Structure

```
fin_pipeline DAG (daily at 2 AM):

  [export_features]
         │
         ▼
    [run_nlp]
         │
         ▼
   [train_gnn]
         │
         ▼
  [train_model]
         │
         ▼
   [run_drift]
```

Each box is an Airflow Task. If a task fails, Airflow:
- Retries up to N times with configurable delay
- Marks the task red in the UI
- Sends an alert (email, Slack, etc.)
- Stops downstream tasks from running

Airflow's job is to make the pipeline reproducible and observable — you can see in the UI exactly which step failed, when, and what the logs say.

### LocalExecutor

```
Airflow modes:

  SequentialExecutor (default):
    Runs tasks one at a time in the same process.
    OK for demos. Not parallel.

  LocalExecutor (used here):
    Spawns subprocesses for each task.
    Can run parallel tasks concurrently.
    No extra infrastructure.
    Good for: single-machine production.

  CeleryExecutor / KubernetesExecutor:
    Distributed task workers.
    For: multi-machine production at scale.
```

---

## Why This Matters: The Model Decay Problem

```
Without monitoring:

  Model trained: July 14 → ROC-AUC 0.67
  
  August: market enters new volatility regime
  Model: still predicts using July patterns
  Real performance: degrades to ~0.55
  
  Nobody notices until a downstream system breaks.
  Root cause investigation takes days.

With this monitoring stack:

  Day 1 of new regime:
    Evidently detects 38% of features have drifted
    → triggers retrain automatically
    → new model trained on updated data
    → champion alias updated
    → performance restored to ~0.65
    
  Grafana shows the event: drift spike → model version change
  Full audit trail in MLflow.
```

---

## Service URLs

| Service | URL | What you see |
|---------|-----|-------------|
| Grafana | `http://localhost:3000` | Predictions, latency, drift, versions |
| Prometheus | `http://localhost:9090` | Raw metrics, query interface |
| Airflow | `http://localhost:8888` | DAG runs, task logs, schedule |
| MLflow | `http://localhost:5000` | Experiment runs, model registry |
| Streamlit | `http://localhost:8501` | Sentiment, prices, predictions, SHAP plots |

---

## Files in This Layer

| File | Role |
|------|------|
| [monitoring/drift_report.py](../../monitoring/drift_report.py) | Evidently drift detection + auto-retrain trigger |
| [monitoring/dashboard.py](../../monitoring/dashboard.py) | Streamlit dashboard |
| [monitoring/prometheus.yml](../../monitoring/prometheus.yml) | Prometheus scrape config |
| [monitoring/grafana/dashboard.json](../../monitoring/grafana/dashboard.json) | Importable Grafana dashboard |
| [dags/fin_pipeline.py](../../dags/fin_pipeline.py) | Airflow DAG wiring all pipeline steps |
