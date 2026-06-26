import os
import time
import logging
from datetime import datetime, timezone

import yfinance as yf
from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]
TOPIC = "price-ticks"
POLL_INTERVAL = 30

SCHEMA_STR = open(os.path.join(os.path.dirname(__file__), "..", "schemas", "price_tick_v1.avsc")).read()


def delivery_report(err, msg):
    if err:
        log.error("Delivery failed for %s: %s", msg.key(), err)
    else:
        log.debug("Delivered to %s [%d] offset %d", msg.topic(), msg.partition(), msg.offset())


def build_serializer(registry_url: str) -> AvroSerializer:
    sr_client = SchemaRegistryClient({"url": registry_url})
    return AvroSerializer(sr_client, SCHEMA_STR)


def fetch_tick(ticker: str) -> dict | None:
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            return None
        row = df.iloc[-1]
        ts_ms = int(df.index[-1].timestamp() * 1000)
        return {
            "ticker": ticker,
            "open":   float(row["Open"]),
            "high":   float(row["High"]),
            "low":    float(row["Low"]),
            "close":  float(row["Close"]),
            "volume": int(row["Volume"]),
            "ts":     ts_ms,
        }
    except Exception as exc:
        log.warning("yfinance error for %s: %s", ticker, exc)
        return None


def main():
    registry_url = os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")
    broker = os.getenv("REDPANDA_BROKERS", "localhost:29092")

    serializer = build_serializer(registry_url)
    producer = Producer({"bootstrap.servers": broker})

    log.info("Price producer started. Polling every %ds.", POLL_INTERVAL)

    while True:
        for ticker in TICKERS:
            tick = fetch_tick(ticker)
            if tick is None:
                continue

            try:
                serialized = serializer(
                    tick,
                    SerializationContext(TOPIC, MessageField.VALUE),
                )
                producer.produce(
                    topic=TOPIC,
                    key=ticker.encode(),
                    value=serialized,
                    on_delivery=delivery_report,
                )
            except Exception as exc:
                log.error("Serialization/produce error for %s: %s", ticker, exc)

            producer.poll(0)

        producer.flush()
        log.info("Price batch produced. Sleeping %ds.", POLL_INTERVAL)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
