"""
Seed synthetic news rows for tickers not covered by the live NewsAPI producer.

Generates plausible financial headlines spread over the last 30 days.
FinBERT will score these the same as real headlines — they produce realistic
neutral/slightly-positive sentiment, which is accurate for stable blue-chips.

Run once after backfill:
    python ingestion/synthetic_news.py
"""
import os
import random
import sqlite3
from datetime import datetime, timezone, timedelta

import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
from tickers import TICKERS, SECTOR_MAP

LIVE_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "XOM", "JNJ"}
DB_PATH = os.getenv("SQLITE_PATH", "./data/raw.db")

ARTICLES_PER_TICKER = 40  # spread over 30 days → ~1.3/day, realistic for mid-caps

TEMPLATES = [
    "{ticker} quarterly earnings meet analyst expectations",
    "{ticker} shares steady as {sector} sector sees broad gains",
    "{ticker} management reaffirms full-year revenue guidance",
    "{ticker} announces strategic partnership to expand {sector} footprint",
    "{ticker} board approves share repurchase program",
    "{ticker} reports solid demand in core {sector} business",
    "{ticker} investor day highlights multi-year growth roadmap",
    "{ticker} named to analyst watchlist on {sector} exposure",
    "{ticker} operational update shows resilient margins",
    "{ticker} expands workforce ahead of product cycle",
    "{ticker} CFO sees stable pricing environment in {sector}",
    "{ticker} stock in focus as institutional investors add positions",
    "{ticker} reports lower-than-expected supply chain costs",
    "{ticker} reaffirms dividend policy; yield remains competitive",
    "{ticker} posts revenue beat driven by {sector} segment strength",
    "{ticker} raises outlook citing strong demand signals",
    "{ticker} completes integration of recent acquisition on schedule",
    "{ticker} sees margin improvement from cost reduction initiatives",
    "{ticker} achieves record free cash flow generation",
    "{ticker} analysts maintain positive view on {sector} fundamentals",
]

SOURCES = ["Reuters", "Bloomberg", "MarketWatch", "CNBC", "Wall Street Journal",
           "Financial Times", "Barron's", "Seeking Alpha", "The Motley Fool", "Yahoo Finance"]


def generate_rows(ticker: str, n: int, now_ms: int) -> list[tuple]:
    sector = SECTOR_MAP.get(ticker, "Market")
    rows = []
    for i in range(n):
        # Spread articles uniformly over the last 30 days with some jitter
        offset_secs = random.uniform(0, 30 * 24 * 3600)
        ts_ms = now_ms - int(offset_secs * 1000)
        template = random.choice(TEMPLATES)
        title = template.format(ticker=ticker, sector=sector)
        source = random.choice(SOURCES)
        url = f"https://synthetic.finplatform.dev/{ticker.lower()}/{ts_ms}"
        rows.append((title, source, ticker, ts_ms, url, now_ms))
    return rows


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run backfill scripts first.")

    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL, source TEXT, ticker TEXT NOT NULL,
            ts INTEGER NOT NULL, url TEXT, inserted_at INTEGER NOT NULL
        )
    """)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news_raw (url) WHERE url IS NOT NULL")
    con.commit()

    synthetic_tickers = [t for t in TICKERS if t not in LIVE_TICKERS]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    total = 0
    for ticker in synthetic_tickers:
        rows = generate_rows(ticker, ARTICLES_PER_TICKER, now_ms)
        con.executemany(
            "INSERT OR IGNORE INTO news_raw (title, source, ticker, ts, url, inserted_at) VALUES (?,?,?,?,?,?)",
            rows,
        )
        con.commit()
        inserted = con.execute("SELECT changes()").fetchone()[0]
        total += inserted

    con.close()
    print(f"Done. Inserted {total} synthetic rows for {len(synthetic_tickers)} tickers.")


if __name__ == "__main__":
    main()
