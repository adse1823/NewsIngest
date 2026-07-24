# Layer 2 — Ingestion Layer: Redpanda

## What This Layer Does

Acts as the central message bus. Producers write events into it; consumers read from it at their own pace. Decouples data production from data consumption completely.

---

## Why a Message Broker at All

Without a broker, producers and consumers are directly coupled:

```
WITHOUT BROKER (fragile)

news_producer ──────────────────► Spark consumer
                                       ↑
                              If this goes down,
                              all data is lost
```

With a broker, the producer writes to a durable log. The consumer reads from that log at its own pace. If the consumer restarts, it picks up from its last offset — no data lost.

```
WITH BROKER (resilient)

news_producer ──► Redpanda ──► Spark consumer
                  (log)            ↑
                              Restarts safely;
                              reads from last offset
```

---

## Redpanda Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    REDPANDA CONTAINER                        │
│                     (512 MB RAM cap)                         │
│                                                              │
│   Port 29092 — Kafka API                                     │
│   Port 8081  — Schema Registry API                           │
│                                                              │
│  ┌──────────────────┐      ┌──────────────────────────────┐  │
│  │   news-raw topic │      │   price-ticks topic          │  │
│  │                  │      │                              │  │
│  │  Partition 0     │      │  Partition 0                 │  │
│  │  [AAPL msg]      │      │  [AAPL tick]                 │  │
│  │  [AAPL msg]      │      │  [MSFT tick]                 │  │
│  │  [MSFT msg]      │      │  [AAPL tick]                 │  │
│  │  ...             │      │  ...                         │  │
│  │                  │      │                              │  │
│  │  offset: 0,1,2.. │      │  offset: 0,1,2..             │  │
│  └──────────────────┘      └──────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              Schema Registry                           │  │
│  │   news_event_v1  ·  price_tick_v1  ·  news_event_v2   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Topics and Partitioning

Two topics are created:

| Topic | Source | Key | Content |
|-------|--------|-----|---------|
| `news-raw` | news_producer.py | ticker symbol | Avro-encoded headlines |
| `price-ticks` | price_producer.py | ticker symbol | Avro-encoded OHLCV ticks |

**Partitioned by ticker symbol** — all messages for AAPL land on the same partition, maintaining ordering guarantees per company. This is critical: when you join news and price events for the same ticker in Spark, they arrive in time order.

---

## Message Lifecycle

```
Producer side:
                       ┌──────────────────────────┐
headline dict ────────►│  Avro Serializer          │
                       │  - validate against schema│
                       │  - embed schema ID (5 B)  │
                       │  - encode to binary        │
                       └──────────┬───────────────┘
                                  │
                   SerializationError raised here
                   if field missing or wrong type ◄─── CAUGHT EARLY
                                  │
                                  ▼
                          Redpanda broker
                          (message stored)

Consumer side:
                       ┌──────────────────────────┐
binary message ───────►│  Avro Deserializer        │
                       │  - read schema ID (5 B)   │
                       │  - fetch schema from reg  │
                       │  - decode to dict/struct  │
                       └──────────┬───────────────┘
                                  │
                                  ▼
                           Python dict / Spark Row
```

---

## Redpanda vs Kafka

| Aspect | Kafka | Redpanda |
|--------|-------|----------|
| Runtime | JVM | Native binary (C++) |
| Dependencies | ZooKeeper or KRaft | None |
| RAM usage | 2–3 GB | < 600 MB |
| API compatibility | Reference | Wire-compatible (identical) |
| Schema registry | Confluent (separate container) | Built-in |
| Best for | Large production clusters | Dev, single-node, resource-constrained |

**Key point:** Every producer, consumer, and admin script written for Kafka works against Redpanda without code changes. The concepts — topics, partitions, offsets, consumer groups — are identical.

---

## Offsets Explained

```
Topic: news-raw
──────────────────────────────────────────────────────►  time
offset:  0        1        2        3        4        5

         [AAPL]  [MSFT]  [AAPL]  [GOOGL]  [TSLA]  [AAPL]

Consumer A (Spark):         reads up to offset 3
                            last committed offset = 3
                            on restart: resumes at 4

Consumer B (another app):   could read independently
                            at a completely different offset
```

Each consumer maintains its own offset pointer. Producers and consumers are completely independent — producers don't know or care who is reading.

---

## Docker Configuration

```yaml
# docker-compose.yml (relevant excerpt)
redpanda:
  image: redpandadata/redpanda:latest
  ports:
    - "29092:29092"   # Kafka API
    - "8081:8081"     # Schema Registry
  command:
    - redpanda
    - start
    - --memory 512M   # hard cap: keeps footprint under 600 MB
```

---

## Files in This Layer

| File | Role |
|------|------|
| [ingestion/news_producer.py](../../ingestion/news_producer.py) | NewsAPI → SQLite + Redpanda |
| [ingestion/price_producer.py](../../ingestion/price_producer.py) | yfinance → SQLite + Redpanda |
| [ingestion/kafka_replay.py](../../ingestion/kafka_replay.py) | Replay SQLite history into topics on resume |
| [scripts/register_schemas.py](../../scripts/register_schemas.py) | Register Avro schemas on startup |
