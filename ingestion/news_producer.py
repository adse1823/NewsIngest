import os
import time
import sqlite3
import logging
from datetime import datetime, timezone

from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
POLL_INTERVAL = 900  # 15 min — 10 tickers × 96 polls/day stays under 100 req/day free limit
DB_PATH = "./data/raw.db"


def init_db(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_raw (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            title     TEXT NOT NULL,
            source    TEXT,
            ticker    TEXT NOT NULL,
            ts        INTEGER NOT NULL,
            url       TEXT,
            inserted_at INTEGER NOT NULL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker ON news_raw (ticker)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_news_ts    ON news_raw (ts)")
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news_raw (url) WHERE url IS NOT NULL")
    con.commit()


def fetch_articles(client: NewsApiClient, ticker: str) -> list[dict]:
    try:
        resp = client.get_everything(q=ticker, language="en", page_size=100, sort_by="publishedAt")
        return resp.get("articles", [])
    except Exception as exc:
        log.warning("NewsAPI error for %s: %s", ticker, exc)
        return []


def insert_articles(con: sqlite3.Connection, ticker: str, articles: list[dict]):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    for article in articles:
        ts_str = article.get("publishedAt") or datetime.now(timezone.utc).isoformat()
        try:
            ts_ms = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            ts_ms = now_ms

        rows.append((
            (article.get("title") or "")[:500],
            (article.get("source", {}).get("name") or "unknown")[:100],
            ticker,
            ts_ms,
            article.get("url"),
            now_ms,
        ))

    con.executemany(
        "INSERT OR IGNORE INTO news_raw (title, source, ticker, ts, url, inserted_at) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    inserted = con.execute("SELECT changes()").fetchone()[0]
    return inserted


def main():
    os.makedirs("./data", exist_ok=True)
    news_client = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
    con = sqlite3.connect(DB_PATH)
    init_db(con)

    log.info("News producer started (Phase 1 — writing to SQLite). Polling every %ds.", POLL_INTERVAL)

    while True:
        total = 0
        for ticker in TICKERS:
            articles = fetch_articles(news_client, ticker)
            if articles:
                count = insert_articles(con, ticker, articles)
                total += count
                log.debug("Inserted %d articles for %s", count, ticker)

        log.info("Batch complete — %d articles written to %s", total, DB_PATH)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
