"""
One-shot replay of SQLite data into Redpanda topics.

Run after schema registration and before starting Spark so the streaming
consumer has historical records to process on first start.

This is idempotent from Spark's perspective: offset tracking means Spark
only processes each message once, even if replay is run multiple times.
"""

import io
import json
import logging
import os
import sqlite3
import struct

import fastavro
import requests
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH      = "./data/raw.db"
BROKER       = os.getenv("REDPANDA_BROKERS", "localhost:29092")
REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")

_SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "schemas")
NEWS_SCHEMA_FILE  = os.path.join(_SCHEMAS_DIR, "news_event_v1.avsc")
PRICE_SCHEMA_FILE = os.path.join(_SCHEMAS_DIR, "price_tick_v1.avsc")


def _load_schema(path: str):
    with open(path) as f:
        return fastavro.parse_schema(json.load(f))


def _get_schema_id(subject: str, schema_json: str) -> int:
    url = f"{REGISTRY_URL}/subjects/{subject}/versions"
    resp = requests.post(url, json={"schemaType": "AVRO", "schema": schema_json}, timeout=10)
    resp.raise_for_status()
    return resp.json()["id"]


def _serialize(schema, schema_id: int, record: dict) -> bytes:
    buf = io.BytesIO()
    fastavro.schemaless_writer(buf, schema, record)
    return b"\x00" + struct.pack(">I", schema_id) + buf.getvalue()


def replay_news(con: sqlite3.Connection, producer: KafkaProducer):
    schema    = _load_schema(NEWS_SCHEMA_FILE)
    schema_id = _get_schema_id("news-raw-value", open(NEWS_SCHEMA_FILE).read())

    rows = con.execute(
        "SELECT title, source, ticker, ts, url FROM news_raw ORDER BY ts"
    ).fetchall()
    log.info("Replaying %d news rows → news-raw", len(rows))

    for title, source, ticker, ts, url in rows:
        record = {
            "title":  title  or "",
            "source": source or "unknown",
            "ticker": ticker,
            "ts":     ts,
            "url":    url,
        }
        try:
            producer.send("news-raw", key=ticker.encode(), value=_serialize(schema, schema_id, record))
        except Exception as exc:
            log.warning("news-raw publish failed (ticker=%s ts=%d): %s", ticker, ts, exc)

    producer.flush()
    log.info("News replay complete.")


def replay_prices(con: sqlite3.Connection, producer: KafkaProducer):
    schema    = _load_schema(PRICE_SCHEMA_FILE)
    schema_id = _get_schema_id("price-ticks-value", open(PRICE_SCHEMA_FILE).read())

    rows = con.execute(
        "SELECT ticker, open, high, low, close, volume, ts FROM price_ticks ORDER BY ts"
    ).fetchall()
    log.info("Replaying %d price rows → price-ticks", len(rows))

    for ticker, open_, high, low, close, volume, ts in rows:
        record = {
            "ticker": ticker,
            "open":   float(open_  or 0.0),
            "high":   float(high   or 0.0),
            "low":    float(low    or 0.0),
            "close":  float(close  or 0.0),
            "volume": int(volume   or 0),
            "ts":     ts,
        }
        try:
            producer.send("price-ticks", key=ticker.encode(), value=_serialize(schema, schema_id, record))
        except Exception as exc:
            log.warning("price-ticks publish failed (ticker=%s ts=%d): %s", ticker, ts, exc)

    producer.flush()
    log.info("Price replay complete.")


def main():
    if not os.path.exists(DB_PATH):
        log.warning("SQLite DB not found at %s — skipping replay.", DB_PATH)
        return

    con = sqlite3.connect(DB_PATH)
    news_count  = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    price_count = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    log.info("SQLite: %d news rows, %d price rows", news_count, price_count)

    if news_count == 0 and price_count == 0:
        log.info("Both tables empty — nothing to replay.")
        con.close()
        return

    try:
        producer = KafkaProducer(bootstrap_servers=BROKER)
    except Exception as exc:
        log.error("Cannot connect to Redpanda at %s: %s — skipping replay.", BROKER, exc)
        con.close()
        return

    if news_count > 0:
        replay_news(con, producer)
    if price_count > 0:
        replay_prices(con, producer)

    producer.close()
    con.close()
    log.info("Kafka replay finished.")


if __name__ == "__main__":
    main()
