#!/usr/bin/env bash
set -euo pipefail

PYTHON="./venv/Scripts/python.exe"

echo "=================================================="
echo "  Financial Intelligence Platform — startup"
echo "=================================================="

# ── 1. Redpanda ────────────────────────────────────────
echo ""
echo "[1/8] Starting Redpanda..."
docker compose up -d redpanda

echo "      Waiting for Redpanda Kafka port (29092)..."
until $PYTHON -c "import socket; socket.create_connection(('localhost', 29092), timeout=2).close()" 2>/dev/null; do
  sleep 2
done
echo "      Kafka port ready."

echo "      Waiting for Schema Registry port (8081)..."
until $PYTHON -c "import socket; socket.create_connection(('localhost', 8081), timeout=2).close()" 2>/dev/null; do
  sleep 2
done
echo "      Redpanda ready."

# ── 2. Schema registration ─────────────────────────────
echo ""
echo "[2/8] Registering Avro schemas..."
$PYTHON scripts/register_schemas.py

# ── 3. Backfill + synthetic seed ───────────────────────
echo ""
echo "[3/8] Checking database..."
ROW_COUNT=$($PYTHON -c "
import sqlite3, os
db = './data/raw.db'
if not os.path.exists(db):
    print(0)
else:
    c = sqlite3.connect(db)
    print(c.execute('SELECT COUNT(*) FROM news_raw').fetchone()[0])
    c.close()
" 2>/dev/null || echo 0)

if [ "$ROW_COUNT" -gt 100 ]; then
  echo "      Skipping backfill — $ROW_COUNT rows already in news_raw."
else
  echo "      Running backfill (this takes ~2 min)..."
  $PYTHON ingestion/backfill_news.py
  $PYTHON ingestion/backfill_prices.py
fi

echo "      Seeding synthetic news for non-live tickers..."
$PYTHON ingestion/synthetic_news.py

# ── 4. ML pipeline ─────────────────────────────────────
echo ""
echo "[4/8] Running ML pipeline..."
if [ -f "./data/features_export.parquet" ] && [ -f "./data/gnn_embeddings.parquet" ]; then
  echo "      Features and embeddings already exist — skipping slow steps."
  $PYTHON run_pipeline.py --skip-backfill --skip-sentiment --skip-embeddings --skip-graph --skip-gnn
else
  echo "      First run — building features, running FinBERT, training GNN (~8 min)..."
  $PYTHON run_pipeline.py --skip-backfill
fi
echo "      Pipeline complete."

# ── 5. Smoke test ──────────────────────────────────────
echo ""
echo "[5/8] Running smoke test..."
$PYTHON scripts/smoke_test.py --skip-train
echo "      Smoke test passed."

# ── 6. Replay SQLite data into Kafka topics ────────────
echo ""
echo "[6/10] Seeding Kafka topics from SQLite history..."
$PYTHON ingestion/kafka_replay.py
echo "       Topics seeded — Spark will find existing records at offset earliest."

# ── 7. Start Spark streaming cluster ──────────────────
echo ""
echo "[7/10] Starting Spark streaming cluster..."
echo "       First run: downloads Kafka + Avro JARs (~100 MB, ~2 min). Cached after that."
docker compose --profile spark up -d
echo "       Spark master UI: http://localhost:8080"
echo "       Streaming job consuming news-raw + price-ticks → ./data/windowed/"

# ── 8. Start Docker services ───────────────────────────
echo ""
echo "[8/10] Starting monitoring + Airflow..."
docker compose --profile monitoring --profile airflow up -d

# ── 9. Start API + MLflow in background ────────────────
echo ""
echo "[9/10] Starting prediction API (PID saved below)..."
$PYTHON -m uvicorn serving.main:app --port 8000 &
API_PID=$!

echo "       Waiting for API on port 8000..."
until $PYTHON -c "import socket; socket.create_connection(('localhost', 8000), timeout=2).close()" 2>/dev/null; do
  sleep 2
done
echo "       API ready (PID $API_PID)."

# ── 10. Streamlit + MLflow UI + live producers ─────────
echo ""
echo "[10/10] Starting MLflow UI, Streamlit, and live data producers..."
$PYTHON -m mlflow ui --port 5000 --backend-store-uri ./mlruns &
MLFLOW_PID=$!

$PYTHON -m streamlit run monitoring/dashboard.py --server.port 8501 &
STREAMLIT_PID=$!

$PYTHON ingestion/news_producer.py &
NEWS_PID=$!

$PYTHON ingestion/price_producer.py &
PRICE_PID=$!

$PYTHON scripts/predict_loop.py &
LOOP_PID=$!

echo ""
echo "=================================================="
echo "  All services started."
echo ""
echo "  Streamlit dashboard: http://localhost:8501"
echo "  Prediction API:      http://localhost:8000/docs"
echo "  Grafana:             http://localhost:3000  (admin / admin)"
echo "  Prometheus:          http://localhost:9090"
echo "  Airflow:             http://localhost:8888  (admin / admin)"
echo "  MLflow:              http://localhost:5000"
echo "  Spark master:        http://localhost:8080"
echo ""
echo "  PIDs — API: $API_PID  MLflow: $MLFLOW_PID  Streamlit: $STREAMLIT_PID"
echo "        News: $NEWS_PID  Prices: $PRICE_PID  Predict loop: $LOOP_PID"
echo "  To stop: kill $API_PID $MLFLOW_PID $STREAMLIT_PID $NEWS_PID $PRICE_PID $LOOP_PID"
echo "=================================================="

wait
