import os
import time
import json
import io
import logging
from datetime import datetime, timezone

import fastavro
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
TOPIC = "news-raw"
POLL_INTERVAL = 30

SCHEMA_STR = open(os.path.join(os.path.dirname(__file__), "..", "schemas", "news_event_v1.avsc")).read()


def delivery_report(err, msg):
    if err:
        log.error("Delivery failed for %s: %s", msg.key(), err)
    else:
        log.debug("Delivered to %s [%d] offset %d", msg.topic(), msg.partition(), msg.offset())


def build_serializer(registry_url: str) -> AvroSerializer:
    sr_conf = {"url": registry_url}
    sr_client = SchemaRegistryClient(sr_conf)
    return AvroSerializer(sr_client, SCHEMA_STR)


def fetch_articles(client: NewsApiClient, ticker: str) -> list[dict]:
    try:
        resp = client.get_everything(q=ticker, language="en", page_size=5, sort_by="publishedAt")
        return resp.get("articles", [])
    except Exception as exc:
        log.warning("NewsAPI error for %s: %s", ticker, exc)
        return []


def main():
    news_client = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
    registry_url = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    broker = os.getenv("REDPANDA_BROKERS", "localhost:29092")

    serializer = build_serializer(registry_url)
    producer = Producer({"bootstrap.servers": broker})

    log.info("News producer started. Polling every %ds.", POLL_INTERVAL)

    while True:
        for ticker in TICKERS:
            articles = fetch_articles(news_client, ticker)
            for article in articles:
                ts_str = article.get("publishedAt") or datetime.now(timezone.utc).isoformat()
                try:
                    ts_ms = int(datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp() * 1000)
                except ValueError:
                    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

                record = {
                    "title":  (article.get("title") or "")[:500],
                    "source": (article.get("source", {}).get("name") or "unknown")[:100],
                    "ticker": ticker,
                    "ts":     ts_ms,
                    "url":    article.get("url"),
                }

                try:
                    serialized = serializer(
                        record,
                        SerializationContext(TOPIC, MessageField.VALUE),
                    )
                    producer.produce(
                        topic=TOPIC,
                        key=ticker.encode(),
                        value=serialized,
                        on_delivery=delivery_report,
                    )
                except Exception as exc:
                    log.error("Serialization/produce error: %s", exc)

            producer.poll(0)

        producer.flush()
        log.info("Batch produced. Sleeping %ds.", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
