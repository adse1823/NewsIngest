import io
import json
import os
import sqlite3
import struct
import time
import logging
from datetime import datetime, timezone

import fastavro
import pandas as pd
import requests
import yfinance as yf
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS       = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
POLL_INTERVAL = 30
DB_PATH       = "./data/raw.db"
BROKER        = os.getenv("REDPANDA_BROKERS", "localhost:29092")
REGISTRY_URL  = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
TOPIC         = "price-ticks"
SCHEMA_FILE   = os.path.join(os.path.dirname(__file__), "..", "schemas", "price_tick_v1.avsc")
SUBJECT       = "price-ticks-value"


def _load_avro_schema():
    with open(SCHEMA_FILE) as f:
        return fastavro.parse_schema(json.load(f))


def _get_schema_id(schema_json: str) -> int:
    url = f"{REGISTRY_URL}/subjects/{SUBJECT}/versions"
    resp = requests.post(url, json={"schemaType": "AVRO", "schema": schema_json}, timeout=5)
    resp.raise_for_status()
    return resp.json()["id"]


def _serialize(schema, schema_id: int, record: dict) -> bytes:
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, schema, record)
    return b"\x00" + struct.pack(">I", schema_id) + buf.getvalue()


def init_db(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS price_ticks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            ts          INTEGER NOT NULL,
            inserted_at INTEGER NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker ON price_ticks (ticker)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_price_ts    ON price_ticks (ts)")
    con.commit()


def fetch_tick(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="5d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            log.warning("No data returned for %s (market may be closed)", ticker)
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        row = df.iloc[-1]
        return {
            "ticker": ticker,
            "open":   float(row["Open"]),
            "high":   float(row["High"]),
            "low":    float(row["Low"]),
            "close":  float(row["Close"]),
            "volume": int(row["Volume"]),
            "ts":     int(df.index[-1].timestamp() * 1000),
        }
    except Exception as exc:
        log.warning("yfinance error for %s: %s", ticker, exc)
        return None


def insert_tick(con: sqlite3.Connection, producer, schema, schema_id, tick: dict):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    con.execute(
        """INSERT OR IGNORE INTO price_ticks (ticker, open, high, low, close, volume, ts, inserted_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (tick["ticker"], tick["open"], tick["high"], tick["low"],
         tick["close"], tick["volume"], tick["ts"], now_ms),
    )
    con.commit()
    if producer is None:
        return
    try:
        producer.send(TOPIC, key=tick["ticker"].encode(), value=_serialize(schema, schema_id, tick))
    except Exception as exc:
        log.warning("Kafka publish failed for %s: %s", tick["ticker"], exc)


def main():
    os.makedirs("./data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    schema = _load_avro_schema()
    schema_json = open(SCHEMA_FILE).read()

    producer = None
    schema_id = None
    try:
        schema_id = _get_schema_id(schema_json)
        producer = KafkaProducer(bootstrap_servers=BROKER)
        log.info("Avro producer ready (schema_id=%d)", schema_id)
    except Exception as e:
        log.warning("Kafka/Schema Registry unavailable: %s — running SQLite-only mode", e)

    log.info("Price producer started (dual-write: SQLite + Redpanda). Polling every %ds.", POLL_INTERVAL)

    while True:
        total = 0
        for ticker in TICKERS:
            tick = fetch_tick(ticker)
            if tick:
                insert_tick(con, producer, schema, schema_id, tick)
                total += 1
                log.debug("Inserted tick for %s: close=%.2f", ticker, tick["close"])

        if producer is not None:
            producer.flush()
        log.info("Batch complete — %d ticks written to %s + Redpanda", total, DB_PATH)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
