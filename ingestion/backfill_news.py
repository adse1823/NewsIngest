"""
One-shot historical backfill from NewsAPI using weekly date windows.

Instead of fetching "most recent 100" (which are already in the DB), this script
chunks the last 30 days into weekly windows and fetches each separately, giving
access to older articles that the live producer never captured.

NewsAPI free tier: 100 requests/day, up to 30 days back, 100 articles per request.
This script uses 4 weeks × 10 tickers = 40 requests — within the daily limit.

Usage:
    python ingestion/backfill_news.py           # last 28 days in weekly chunks
    python ingestion/backfill_news.py --weeks 2 # last 14 days (20 requests)
"""

import os
import argparse
import sqlite3
import logging
import time
from datetime import datetime, timezone, timedelta

from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from tickers import TICKERS
DB_PATH = "./data/raw.db"


def init_db(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            source      TEXT,
            ticker      TEXT NOT NULL,
            ts          INTEGER NOT NULL,
            url         TEXT,
            inserted_at INTEGER NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_raw (ticker)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_news_ts    ON news_raw (ts)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news_raw (url) WHERE url IS NOT NULL")
    con.commit()


def fetch_window(client: NewsApiClient, ticker: str, from_dt: datetime, to_dt: datetime) -> list[dict]:
    try:
        resp = client.get_everything(
            q=ticker,
            language="en",
            page_size=100,
            sort_by="publishedAt",
            from_param=from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            to=to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        articles = resp.get("articles", [])
        total = resp.get("totalResults", 0)
        log.info("    %s [%s → %s]: %d returned (%d total available)",
                 ticker, from_dt.strftime("%m-%d"), to_dt.strftime("%m-%d"),
                 len(articles), total)
        return articles
    except Exception as exc:
        log.warning("    %s: NewsAPI error — %s", ticker, exc)
        return []


def insert_articles(con: sqlite3.Connection, ticker: str, articles: list[dict]) -> int:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    for article in articles:
        ts_str = article.get("publishedAt") or datetime.now(timezone.utc).isoformat()
        try:
            ts_ms = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            ts_ms = now_ms

        title = (article.get("title") or "").strip()
        if not title or title == "[Removed]":
            continue

        rows.append((
            title[:500],
            (article.get("source", {}).get("name") or "unknown")[:100],
            ticker,
            ts_ms,
            article.get("url"),
            now_ms,
        ))

    if not rows:
        return 0

    con.executemany(
        "INSERT OR IGNORE INTO news_raw (title, source, ticker, ts, url, inserted_at) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    return con.execute("SELECT changes()").fetchone()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weeks", type=int, default=4,
                        help="How many weeks back to fetch (max 4 on free tier, uses weeks×10 API requests)")
    args = parser.parse_args()

    weeks = min(args.weeks, 4)
    now = datetime.now(timezone.utc)

    # Build weekly windows oldest → newest
    windows = []
    for w in range(weeks, 0, -1):
        win_to   = now - timedelta(weeks=w - 1)
        win_from = now - timedelta(weeks=w)
        windows.append((win_from, win_to))

    total_requests = len(windows) * len(TICKERS)

    os.makedirs("./data", exist_ok=True)
    client = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    before = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    log.info("Starting weekly backfill: %d weeks, %d windows × %d tickers = %d API requests",
             weeks, len(windows), len(TICKERS), total_requests)
    log.info("Existing rows in DB: %d", before)

    total_inserted = 0
    req_count = 0

    for w_idx, (win_from, win_to) in enumerate(windows, start=1):
        log.info("Week %d/%d  (%s → %s)", w_idx, len(windows),
                 win_from.strftime("%Y-%m-%d"), win_to.strftime("%Y-%m-%d"))
        for ticker in TICKERS:
            articles = fetch_window(client, ticker, win_from, win_to)
            inserted = insert_articles(con, ticker, articles)
            total_inserted += inserted
            req_count += 1
            # Small pause to be respectful to the API
            time.sleep(0.2)

    after = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    con.close()

    log.info("Backfill complete.  Requests used: %d", req_count)
    log.info("  New rows inserted : %d", total_inserted)
    log.info("  Total rows in DB  : %d (was %d)", after, before)


if __name__ == "__main__":
    main()
