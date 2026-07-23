"""
Background prediction loop — calls /predict for every ticker on each cycle.

Keeps Prometheus metrics (request count, latency, errors) continuously populated
so Grafana panels show live data rather than empty charts.

Reads latest feature row per ticker from DuckDB, merges GNN embeddings,
then fires one HTTP POST per ticker to the local FastAPI.

Usage:
    python scripts/predict_loop.py                  # default: every 300s
    python scripts/predict_loop.py --interval 60    # every 60s
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from tickers import TICKERS

API_URL          = os.getenv("API_URL", "http://localhost:8000")
DUCKDB_PATH      = os.getenv("FEATURE_STORE_PATH", "./data/feature_store.duckdb")
GNN_PATH         = "./data/gnn_embeddings.parquet"
DEFAULT_INTERVAL = 300  # seconds


def wait_for_api(timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f"{API_URL}/health", timeout=2).raise_for_status()
            log.info("API ready at %s", API_URL)
            return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"API at {API_URL} did not become ready within {timeout}s")


def load_latest_features() -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        return con.execute("""
            SELECT ticker,
                   rolling_1h_mean, rolling_24h_std, volume_zscore,
                   pos, neg, neu
            FROM (
                SELECT f.ticker,
                       f.rolling_1h_mean, f.rolling_24h_std, f.volume_zscore,
                       s.pos, s.neg, s.neu,
                       ROW_NUMBER() OVER (PARTITION BY f.ticker ORDER BY f.window_start DESC) AS rn
                FROM features_sentiment f
                LEFT JOIN sentiment_scores s USING (window_start, ticker)
            )
            WHERE rn = 1
        """).df()
    finally:
        con.close()


def load_gnn_embeddings() -> dict[str, list[float]]:
    if not Path(GNN_PATH).exists():
        return {}
    df = pd.read_parquet(GNN_PATH)
    dim_cols = sorted([c for c in df.columns if c.startswith("dim_")],
                      key=lambda c: int(c.split("_")[1]))
    return {
        row["ticker"]: row[dim_cols].tolist()
        for _, row in df.iterrows()
    }


def run_cycle(gnn_embeddings: dict[str, list[float]]) -> dict:
    features = load_latest_features()
    feat_by_ticker = {row["ticker"]: row for _, row in features.iterrows()}

    results = {"up": 0, "down": 0, "skipped": 0, "errors": 0}

    for ticker in TICKERS:
        row = feat_by_ticker.get(ticker)
        if row is None:
            results["skipped"] += 1
            continue

        gnn_dims = gnn_embeddings.get(ticker, [0.0] * 64)
        payload = {
            "rolling_1h_mean": float(row["rolling_1h_mean"] or 0),
            "rolling_24h_std":  float(row["rolling_24h_std"]  or 0),
            "volume_zscore":    float(row["volume_zscore"]    or 0),
            "pos": float(row["pos"] or 0.33),
            "neg": float(row["neg"] or 0.33),
            "neu": float(row["neu"] or 0.34),
            "gnn_dims": gnn_dims,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            direction = resp.json()["direction"]
            results[direction] += 1
        except Exception as exc:
            log.debug("Ticker %s failed: %s", ticker, exc)
            results["errors"] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Seconds between cycles (default: 300)")
    args = parser.parse_args()

    wait_for_api()

    gnn_embeddings = load_gnn_embeddings()
    log.info("GNN embeddings loaded for %d tickers", len(gnn_embeddings))

    log.info("Prediction loop started — %d tickers every %ds", len(TICKERS), args.interval)

    while True:
        t0 = time.time()
        try:
            results = run_cycle(gnn_embeddings)
            elapsed = time.time() - t0
            log.info(
                "Cycle done in %.1fs — up=%d down=%d skipped=%d errors=%d",
                elapsed, results["up"], results["down"], results["skipped"], results["errors"],
            )
        except Exception as exc:
            log.error("Cycle failed: %s", exc)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
