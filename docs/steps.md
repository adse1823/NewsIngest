# How to Run the Pipeline

## First Time Setup

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# If Activate.ps1 is blocked, run this once first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your NewsAPI key to .env
# Get a free key at https://newsapi.org/register
# Open .env and set: NEWS_API_KEY=your_key_here
```

---

## Running the Full Pipeline (Recommended)

Use the pipeline runner. It runs all steps in order with a progress bar.

```powershell
.\venv\Scripts\Activate.ps1
python run_pipeline.py
```

**Skip flags** — reuse expensive saved state instead of re-running:

```powershell
# Most common re-run: new data already fetched, skip FinBERT (slow)
python run_pipeline.py --skip-backfill --skip-sentiment --skip-embeddings --skip-graph --skip-gnn

# Retrain model only (fastest)
python run_pipeline.py --skip-backfill --skip-features --skip-sentiment --skip-embeddings --skip-graph --skip-gnn

# All available flags
--skip-backfill     Skip news + price backfill (NewsAPI already fetched today)
--skip-features     Skip DuckDB feature export
--skip-sentiment    Skip FinBERT sentiment scoring (~3-5 min) — reuses DuckDB scores
--skip-embeddings   Skip FinBERT embeddings (~1-2 min) — reuses headline_embeddings.npy
--skip-graph        Skip graph build — reuses graph.pt
--skip-gnn          Skip GNN training — reuses gnn_embeddings.parquet
--skip-model        Skip LightGBM training
```

**API request budget:** NewsAPI free tier = 100 requests/day.
The backfill uses 40 requests (4 weeks × 10 tickers). The live producer uses 10 per poll at 15-min intervals.
Do not run the full backfill more than once per day or you will exhaust the daily quota.

---

## Running the Demo

Open 3 terminals, activate venv in each.

### Terminal 1 — Dashboard

```powershell
.\venv\Scripts\Activate.ps1
streamlit run monitoring/dashboard.py
```

Opens at <http://localhost:8501>

### Terminal 2 — Prediction API

```powershell
.\venv\Scripts\Activate.ps1
uvicorn serving.main:app --port 8000
```

Interactive docs at <http://localhost:8000/docs>

### Terminal 3 — MLflow experiment tracker

```powershell
.\venv\Scripts\Activate.ps1
mlflow ui --port 5000 --backend-store-uri ./mlruns
```

Opens at <http://localhost:5000>

---

## Manual Step-by-Step (if not using run_pipeline.py)

### Step 1 — Backfill news (run once)
```powershell
python ingestion/backfill_news.py
```
Fetches last 28 days in weekly windows. Uses 40 API requests.

### Step 2 — Backfill prices (run once)
```powershell
python ingestion/backfill_prices.py
```
Downloads 1 month of 5-minute bars from yfinance. No API key needed.

### Step 3 — Start live news ingestion (keep running)
```powershell
python ingestion/news_producer.py
```
Polls NewsAPI every 15 minutes. Inserts only new articles (deduped by URL).

### Step 4 — Start live price ingestion (keep running)
```powershell
python ingestion/price_producer.py
```
Polls yfinance every 30 seconds. Writes latest OHLCV tick to SQLite.

### Step 5 — Build features
```powershell
python feature_store/export.py
```
DuckDB reads SQLite, computes 5-min windows, rolling features, between-window pct_change.

### Step 6 — Run NLP sentiment
```powershell
python nlp/sentiment.py
```
FinBERT scores every headline window. ~3-5 min. Weights cached after first run.

### Step 7 — Generate embeddings
```powershell
python nlp/embeddings.py
```
Mean-pools FinBERT hidden states per ticker. Saves to `./data/headline_embeddings.npy`.

### Step 8 — Build company graph
```powershell
python graph/build_graph.py
```
Co-occurrence graph from headline mentions. Saves to `./data/graph.pt`.

### Step 9 — Train GNN
```powershell
python graph/train_gnn.py
```
GraphSAGE 100 epochs. Saves 64-dim embeddings to `./data/gnn_embeddings.parquet`.

### Step 10 — Train forecasting model
```powershell
python modeling/train.py
```
LightGBM with TimeSeriesSplit CV. Registers model in `./mlruns/`, promotes to Production.

### Step 11 — Start the API
```powershell
uvicorn serving.main:app --port 8000
```
Loads Production model. Serves predictions at http://localhost:8000.

### Step 12 — Start the dashboard
```powershell
streamlit run monitoring/dashboard.py
```
Opens at http://localhost:8501.

---

## API Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}` |
| `/predict` | POST | Returns price direction and confidence |
| `/explain` | POST | Returns SHAP values for the prediction |
| `/shap-summary` | GET | Returns SHAP summary chart as base64 PNG |
| `/metrics` | GET | Prometheus metrics (request count, latency, errors) |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Example prediction request (PowerShell)
```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"rolling_1h_mean\": 8.5, \"rolling_24h_std\": 1.2, \"volume_zscore\": 2.1, \"pos\": 0.72, \"neg\": 0.08, \"neu\": 0.20, \"gnn_dims\": []}'
```

### Demo input values

| Scenario | rolling_1h_mean | rolling_24h_std | volume_zscore | pos | neg | neu |
|---|---|---|---|---|---|---|
| Bullish | 8.5 | 1.2 | 2.1 | 0.72 | 0.08 | 0.20 |
| Bearish | 12.0 | 2.8 | 3.5 | 0.05 | 0.81 | 0.14 |
| Neutral | 1.2 | 0.3 | -0.4 | 0.31 | 0.18 | 0.51 |

---

## Data Files Reference

| File | Created by | Used by |
|---|---|---|
| `./data/raw.db` | producers + backfill scripts | feature_store/export.py |
| `./data/feature_store.duckdb` | export.py, sentiment.py | modeling/train.py, dashboard.py |
| `./data/features_export.parquet` | export.py | modeling/train.py |
| `./data/headline_embeddings.npy` | embeddings.py | graph/build_graph.py |
| `./data/embedding_meta.json` | embeddings.py | graph/build_graph.py |
| `./data/graph.pt` | build_graph.py | graph/train_gnn.py |
| `./data/gnn_embeddings.parquet` | train_gnn.py | modeling/train.py |
| `./data/drift_report.html` | drift_report.py | view in browser |
| `./mlruns/` | modeling/train.py | serving/main.py, mlflow ui |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No such file: embedding_meta.json` | Skipped `nlp/embeddings.py` | Run embeddings.py before build_graph.py |
| `Only one class present in y_true` | Too few data rows or all same price direction | Run backfill scripts first |
| `JSONDecodeError` from yfinance | Market is closed | Already fixed — uses `period="5d"` |
| `ModuleNotFoundError: prometheus_client` | Missing package | `pip install prometheus-client==0.20.0` |
| `pyarrow` conflict with mlflow | Version mismatch | Already fixed — pinned to pyarrow==15.0.2 |
| MLflow connecting to localhost:5000 | Stale env var | Already fixed — tracking URI hardcoded to `./mlruns` |
| `predict_proba` AttributeError in API | Model saved as raw Booster | Already fixed — serving uses `model.predict()` |
| Backfill inserts 0 new rows | All articles already in DB | Normal if run twice same day — articles are deduplicated by URL |
