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
# Open .env and replace: NEWS_API_KEY=your_newsapi_key_here
```

---

## Every Session

Each step below must be run in order. Steps 1 and 2 run continuously in the background.
Everything else is run once per session.

### Step 1 — Start news ingestion (Terminal 1, keep running)
```powershell
.\venv\Scripts\Activate.ps1
python ingestion/news_producer.py
```
Polls NewsAPI every 30 seconds. Writes headlines to `./data/raw.db`.

### Step 2 — Start price ingestion (Terminal 2, keep running)
```powershell
.\venv\Scripts\Activate.ps1
python ingestion/price_producer.py
```
Polls yfinance every 30 seconds. Writes OHLCV ticks to `./data/raw.db`.

> Let both producers run for at least 5 minutes before continuing.

### Quick check — confirm data is arriving
```powershell
python -c "import sqlite3; con = sqlite3.connect('./data/raw.db'); print('News:', con.execute('SELECT COUNT(*) FROM news_raw').fetchone()[0]); print('Prices:', con.execute('SELECT COUNT(*) FROM price_ticks').fetchone()[0])"
```

---

### Step 3 — Build features (Terminal 3)
```powershell
python feature_store/export.py
```
Reads SQLite via DuckDB. Runs 3 SQL models (raw_news → features_sentiment → entity_table).
Exports final feature table to `./data/features_export.parquet`.

### Step 4 — Run NLP sentiment
```powershell
python nlp/sentiment.py
```
Loads FinBERT (downloads ~400 MB on first run, then cached).
Runs batch inference over all headlines. Writes pos/neg/neu scores to DuckDB.

### Step 5 — Generate embeddings
```powershell
python nlp/embeddings.py
```
Mean-pools FinBERT hidden states into one 768-dim vector per ticker.
Saves to `./data/headline_embeddings.npy` and `./data/embedding_meta.json`.

### Step 6 — Build company graph
```powershell
python graph/build_graph.py
```
Scans all headlines for company name mentions. Connects tickers that appear in the same
headline. Saves graph to `./data/graph.pt`.

### Step 7 — Train GNN
```powershell
python graph/train_gnn.py
```
Trains a 2-layer GraphSAGE model for 100 epochs.
Saves 64-dim embeddings per ticker to `./data/gnn_embeddings.parquet`.

### Step 8 — Start MLflow (Terminal 4, keep running)
```powershell
.\venv\Scripts\Activate.ps1
mlflow ui --port 5000
```
Opens experiment tracking UI at http://localhost:5000

### Step 9 — Train the forecasting model
```powershell
python modeling/train.py
```
Combines tabular features + GNN embeddings. Trains LightGBM with TimeSeriesSplit.
Registers and promotes best model to Production in MLflow.

### Step 10 — Start the API (Terminal 5, keep running)
```powershell
.\venv\Scripts\Activate.ps1
uvicorn serving.main:app --port 8000
```
Loads Production model from MLflow. Serves predictions at http://localhost:8000.

### Step 11 — Start the dashboard (Terminal 6, keep running)
```powershell
.\venv\Scripts\Activate.ps1
streamlit run monitoring/dashboard.py
```
Opens live dashboard at http://localhost:8501.

---

## API Endpoints

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "ok"}` |
| `/predict` | POST | Returns price direction and confidence |
| `/explain` | POST | Returns SHAP values for the prediction |
| `/shap-summary` | GET | Returns SHAP summary chart as base64 PNG |
| `/metrics` | GET | Prometheus metrics (request count, latency, errors) |
| `/docs` | GET | Interactive API documentation (auto-generated) |

### Example prediction request
```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"rolling_1h_mean": 5.0, "rolling_24h_std": 1.2, "volume_zscore": 0.3, "pos": 0.6, "neg": 0.1, "neu": 0.3, "gnn_dims": []}'
```

Expected response:
```json
{"direction": "up", "confidence": 0.67}
```

---

## Data Files Reference

| File | Created by | Used by |
|---|---|---|
| `./data/raw.db` | producers | feature_store/export.py |
| `./data/feature_store.duckdb` | export.py, sentiment.py | modeling/train.py, dashboard.py |
| `./data/features_export.parquet` | export.py | modeling/train.py |
| `./data/headline_embeddings.npy` | embeddings.py | graph/build_graph.py |
| `./data/embedding_meta.json` | embeddings.py | graph/build_graph.py |
| `./data/graph.pt` | build_graph.py | graph/train_gnn.py |
| `./data/gnn_embeddings.parquet` | train_gnn.py | modeling/train.py |
| `./data/drift_report.html` | drift_report.py | (view in browser) |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No such file: embedding_meta.json` | Skipped `nlp/embeddings.py` | Run embeddings.py before build_graph.py |
| `Only one class present in y_true` | Too few data rows for CV | Collect more data, or reduce N_SPLITS |
| `JSONDecodeError` from yfinance | Market is closed | Already fixed — uses `period="5d"` |
| `ModuleNotFoundError: prometheus_client` | Missing package | `pip install prometheus-client==0.20.0` |
| `pyarrow` conflict with mlflow | Version mismatch | Already fixed — pinned to pyarrow==15.0.2 |
