"""
One-shot price backfill using yfinance — no API key required.

Downloads 1 year of daily OHLCV bars for all tickers in ingestion/tickers.py.

Note on intervals:
    5-minute bars  →  max 60 days  (yfinance hard limit)
    daily bars     →  unlimited history  ← what this script uses

Usage:
    python ingestion/backfill_prices.py
    python ingestion/backfill_prices.py --period 2y   # two years
"""

import argparse
import os
import sqlite3
import logging
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tickers import TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./data/raw.db"
BATCH   = 50  # tickers per yfinance batch call


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
    con.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker    ON price_ticks (ticker)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_price_ts        ON price_ticks (ts)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_price_ticker_ts ON price_ticks (ticker, ts)")
    con.commit()


def backfill_batch(con: sqlite3.Connection, batch: list[str], period: str) -> int:
    log.info("Downloading batch of %d tickers (%s daily bars)...", len(batch), period)
    try:
        df = yf.download(
            batch,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as exc:
        log.warning("  Batch download error: %s", exc)
        return 0

    if df.empty:
        log.warning("  Empty response for batch")
        return 0

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []

    for ticker in batch:
        try:
            if len(batch) == 1:
                ticker_df = df
            else:
                ticker_df = df[ticker]

            if isinstance(ticker_df.columns, pd.MultiIndex):
                ticker_df.columns = ticker_df.columns.get_level_values(0)

            ticker_df = ticker_df.dropna(subset=["Close"])

            for ts_idx, row in ticker_df.iterrows():
                rows.append((
                    ticker,
                    float(row.get("Open")   or 0),
                    float(row.get("High")   or 0),
                    float(row.get("Low")    or 0),
                    float(row.get("Close")  or 0),
                    int(row.get("Volume")   or 0),
                    int(ts_idx.timestamp() * 1000),
                    now_ms,
                ))
        except Exception as exc:
            log.warning("  %s: parse error — %s", ticker, exc)

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
    log.info("  Batch: %d rows inserted", inserted)
    return inserted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--period", default="1y", help="yfinance period string (default: 1y)")
    args = parser.parse_args()

    os.makedirs("./data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    before = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    log.info("Starting price backfill: %d tickers, period=%s, interval=1d", len(TICKERS), args.period)
    log.info("Existing price rows: %d", before)

    total_inserted = 0
    batches = [TICKERS[i:i + BATCH] for i in range(0, len(TICKERS), BATCH)]
    for i, batch in enumerate(batches, 1):
        log.info("Batch %d/%d", i, len(batches))
        total_inserted += backfill_batch(con, batch, args.period)

    after = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    con.close()

    log.info("Price backfill complete.")
    log.info("  New rows inserted : %d", total_inserted)
    log.info("  Total rows in DB  : %d (was %d)", after, before)


if __name__ == "__main__":
    main()
