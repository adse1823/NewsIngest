"""
One-shot price backfill using yfinance — no API key required.

Downloads 1 month of 5-minute OHLCV bars for all 10 tickers.
yfinance supports up to 60 days of 5-minute data for free.

This fills in the price history needed to match the news backfill date range
so pct_change labels are no longer all NULL.

Usage:
    python ingestion/backfill_prices.py
"""

import os
import sqlite3
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
DB_PATH = "./data/raw.db"


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
    # Unique on (ticker, ts) so INSERT OR IGNORE deduplicates
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_price_ticker_ts ON price_ticks (ticker, ts)")
    con.commit()


def backfill_ticker(con: sqlite3.Connection, ticker: str) -> int:
    log.info("Fetching %s (1 month of 5-min bars)...", ticker)
    try:
        df = yf.download(ticker, period="1mo", interval="5m", progress=False, auto_adjust=True)
    except Exception as exc:
        log.warning("  %s: yfinance error — %s", ticker, exc)
        return 0

    if df.empty:
        log.warning("  %s: no data returned", ticker)
        return 0

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"])
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    rows = []
    for ts_idx, row in df.iterrows():
        ts_ms = int(ts_idx.timestamp() * 1000)
        rows.append((
            ticker,
            float(row.get("Open")  or 0),
            float(row.get("High")  or 0),
            float(row.get("Low")   or 0),
            float(row.get("Close") or 0),
            int(row.get("Volume")  or 0),
            ts_ms,
            now_ms,
        ))

    if not rows:
        return 0

    con.executemany(
        """INSERT OR IGNORE INTO price_ticks
           (ticker, open, high, low, close, volume, ts, inserted_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        rows,
    )
    con.commit()
    inserted = con.execute("SELECT changes()").fetchone()[0]
    log.info("  %s: %d new rows inserted (from %d fetched)", ticker, inserted, len(rows))
    return inserted


def main():
    os.makedirs("./data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    before = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    log.info("Starting price backfill. Existing price rows: %d", before)

    total_inserted = 0
    for ticker in TICKERS:
        total_inserted += backfill_ticker(con, ticker)

    after = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    con.close()

    log.info("Price backfill complete.")
    log.info("  New rows inserted : %d", total_inserted)
    log.info("  Total rows in DB  : %d (was %d)", after, before)


if __name__ == "__main__":
    main()
