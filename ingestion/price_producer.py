import os
import json
import time
import sqlite3
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
POLL_INTERVAL = 30
DB_PATH = "./data/raw.db"
BROKER = os.getenv("REDPANDA_BROKERS", "localhost:29092")
TOPIC = "price-ticks"


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
        # period="5d" ensures we get the last available candle even outside market hours
        df = yf.download(ticker, period="5d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            log.warning("No data returned for %s (market may be closed)", ticker)
            return None
        # Flatten multi-level columns yfinance sometimes returns
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


def insert_tick(con: sqlite3.Connection, producer, tick: dict):
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
        producer.send(TOPIC, key=tick["ticker"].encode(), value=tick)
    except Exception as exc:
        log.warning("Kafka publish failed for %s: %s", tick["ticker"], exc)


def main():
    os.makedirs("./data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    try:
        producer = KafkaProducer(
            bootstrap_servers=BROKER,
            value_serializer=lambda v: json.dumps(v).encode(),
        )
    except Exception as e:
        log.warning("Kafka unavailable: %s — running SQLite-only mode", e)
        producer = None

    log.info("Price producer started (dual-write: SQLite + Redpanda). Polling every %ds.", POLL_INTERVAL)

    while True:
        total = 0
        for ticker in TICKERS:
            tick = fetch_tick(ticker)
            if tick:
                insert_tick(con, producer, tick)
                total += 1
                log.debug("Inserted tick for %s: close=%.2f", ticker, tick["close"])

        if producer is not None:
            producer.flush()
        log.info("Batch complete — %d ticks written to %s + Redpanda", total, DB_PATH)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
