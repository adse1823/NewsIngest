#!/usr/bin/env bash
set -euo pipefail

PYTHON="./venv/Scripts/python.exe"

echo "=================================================="
echo "  Financial Intelligence Platform — startup"
echo "=================================================="

# ── 1. Redpanda ────────────────────────────────────────
echo ""
echo "[1/5] Starting Redpanda..."
docker compose up -d redpanda

echo "      Waiting for Redpanda Kafka port (29092)..."
until $PYTHON -c "import socket; socket.create_connection(('localhost', 29092), timeout=2).close()" 2>/dev/null; do
  sleep 2
done
echo "      Redpanda ready."

# ── 2. Schema registration ─────────────────────────────
echo ""
echo "[2/5] Registering Avro schemas..."
$PYTHON scripts/register_schemas.py

# ── 3. Backfill (skip if data already exists) ──────────
echo ""
echo "[3/5] Checking database..."
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

# ── 4. Smoke test ──────────────────────────────────────
echo ""
echo "[4/5] Running smoke test..."
$PYTHON scripts/smoke_test.py --skip-train
echo "      Smoke test passed."

# ── 5. Start services ──────────────────────────────────
echo ""
echo "[5/5] Starting monitoring stack + Spark..."
docker compose --profile monitoring --profile spark up -d

echo ""
echo "=================================================="
echo "  All services started."
echo ""
echo "  Streamlit dashboard:"
echo "    $PYTHON -m streamlit run monitoring/dashboard.py"
echo ""
echo "  Prediction API:"
echo "    $PYTHON -m uvicorn serving.main:app --port 8000"
echo ""
echo "  Grafana:    http://localhost:3000  (admin / admin)"
echo "  Prometheus: http://localhost:9090"
echo "  Spark UI:   http://localhost:8080"
echo "=================================================="
