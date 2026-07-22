#!/usr/bin/env bash
# Resume script — use this when data and models already exist.
# Starts all services without re-running backfill, pipeline, or smoke tests.
set -euo pipefail

PYTHON="./venv/Scripts/python.exe"

echo "=================================================="
echo "  Financial Intelligence Platform — resume"
echo "=================================================="

# ── 1. Docker services ─────────────────────────────────
echo ""
echo "[1/3] Starting Docker services..."
docker compose up -d redpanda 2>/dev/null || docker start redpanda 2>/dev/null || true
docker compose --profile monitoring --profile airflow up -d 2>/dev/null || true

echo "      Waiting for Redpanda..."
until $PYTHON -c "import socket; socket.create_connection(('localhost', 29092), timeout=2).close()" 2>/dev/null; do
  sleep 2
done
echo "      Redpanda ready."

# ── 2. Prediction API ──────────────────────────────────
echo ""
echo "[2/3] Starting prediction API..."
$PYTHON -m uvicorn serving.main:app --port 8000 &
API_PID=$!

echo "      Waiting for API on port 8000..."
until $PYTHON -c "import socket; socket.create_connection(('localhost', 8000), timeout=2).close()" 2>/dev/null; do
  sleep 2
done
echo "      API ready (PID $API_PID)."

# ── 3. MLflow UI + Streamlit ───────────────────────────
echo ""
echo "[3/3] Starting MLflow UI and Streamlit dashboard..."
$PYTHON -m mlflow ui --port 5000 --backend-store-uri ./mlruns &
MLFLOW_PID=$!

$PYTHON -m streamlit run monitoring/dashboard.py --server.port 8501 &
STREAMLIT_PID=$!

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
echo ""
echo "  PIDs — API: $API_PID  MLflow: $MLFLOW_PID  Streamlit: $STREAMLIT_PID"
echo "  To stop: kill $API_PID $MLFLOW_PID $STREAMLIT_PID"
echo "=================================================="

wait
