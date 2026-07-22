import io
import json
import os
import sqlite3
import struct
import time
import logging
from datetime import datetime, timezone

import fastavro
import requests
from kafka import KafkaProducer
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
# Free tier = 100 requests/day. Full TICKERS list (300+) exhausts it in one poll.
# Live polling covers only these 10; all other tickers are seeded via synthetic_news.py.
LIVE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "XOM", "JNJ"]

# 10 tickers × 6 polls/day = 60 requests/day (safe under 100 limit)
POLL_INTERVAL = 14400
DB_PATH       = "./data/raw.db"
BROKER        = os.getenv("REDPANDA_BROKERS", "localhost:29092")
REGISTRY_URL  = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
TOPIC         = "news-raw"
SCHEMA_FILE   = os.path.join(os.path.dirname(__file__), "..", "schemas", "news_event_v1.avsc")
SUBJECT       = "news-raw-value"


def _load_avro_schema():
    with open(SCHEMA_FILE) as f:
        return fastavro.parse_schema(json.load(f))


def _get_schema_id(schema_json: str) -> int:
    """Register (or fetch existing) schema ID from Schema Registry."""
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


def fetch_articles(client: NewsApiClient, ticker: str) -> list[dict]:
    try:
        resp = client.get_everything(q=ticker, language="en", page_size=100, sort_by="publishedAt")
        return resp.get("articles", [])
    except Exception as exc:
        log.warning("NewsAPI error for %s: %s", ticker, exc)
        return []


def insert_articles(con: sqlite3.Connection, producer, schema, schema_id, ticker: str, articles: list[dict]):
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    avro_records = []
    for article in articles:
        ts_str = article.get("publishedAt") or datetime.now(timezone.utc).isoformat()
        try:
            ts_ms = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1000)
        except ValueError:
            ts_ms = now_ms

        title  = (article.get("title") or "")[:500]
        source = (article.get("source", {}).get("name") or "unknown")[:100]
        url    = article.get("url")
        rows.append((title, source, ticker, ts_ms, url, now_ms))
        avro_records.append({"title": title, "source": source, "ticker": ticker, "ts": ts_ms, "url": url})

    con.executemany(
        "INSERT OR IGNORE INTO news_raw (title, source, ticker, ts, url, inserted_at) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()

    if producer is not None:
        for record in avro_records:
            try:
                producer.send(TOPIC, key=ticker.encode(), value=_serialize(schema, schema_id, record))
            except Exception as exc:
                log.warning("Kafka publish failed for %s: %s", ticker, exc)

    return con.execute("SELECT changes()").fetchone()[0]


def main():
    os.makedirs("./data", exist_ok=True)
    news_client = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
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

    log.info("News producer started (dual-write: SQLite + Redpanda). Polling every %ds.", POLL_INTERVAL)

    while True:
        total = 0
        for ticker in LIVE_TICKERS:
            articles = fetch_articles(news_client, ticker)
            if articles:
                count = insert_articles(con, producer, schema, schema_id, ticker, articles)
                total += count
                log.debug("Inserted %d articles for %s", count, ticker)

        if producer is not None:
            producer.flush()
        log.info("Batch complete — %d articles written to %s + Redpanda", total, DB_PATH)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
