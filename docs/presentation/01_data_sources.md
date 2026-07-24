# Layer 1 — Data Sources

## What This Layer Does

Pulls raw data from two external APIs every 30 seconds and hands it to the ingestion layer.

---

## Data Flow

```
┌────────────────────────────────────────────────────────┐
│                   EXTERNAL WORLD                       │
│                                                        │
│   ┌─────────────────┐        ┌───────────────────┐     │
│   │    NewsAPI       │        │  yfinance /        │     │
│   │                 │        │  AlphaVantage       │     │
│   │  Financial news │        │                    │     │
│   │  headlines      │        │  OHLCV price ticks │     │
│   │  every 30s      │        │  every 30s         │     │
│   └────────┬────────┘        └─────────┬──────────┘     │
└────────────┼─────────────────────────── ┼───────────────┘
             │                            │
             ▼                            ▼
   news_producer.py              price_producer.py
   (ingestion/)                  (ingestion/)
             │                            │
             ▼                            ▼
       Avro serialized             Avro serialized
       → Redpanda                  → Redpanda
         news-raw topic              price-ticks topic
```

---

## Source 1 — NewsAPI

**What it provides:** Financial headlines across sources (Reuters, Bloomberg, CNBC, etc.)

**Fields captured per article:**

| Field | Type | Example |
|-------|------|---------|
| `title` | string | "Apple beats Q3 earnings estimates" |
| `source` | string | "Reuters" |
| `ts` | timestamp-millis | 1720000000000 |
| `url` | string (nullable) | "https://reuters.com/..." |

**Producer file:** [ingestion/news_producer.py](../../ingestion/news_producer.py)

**Key behaviors:**
- Runs on a 30-second poll loop
- Deduplicates by URL before publishing (same headline won't be published twice)
- Writes to SQLite as a local backup (dual-write) before publishing to Redpanda
- Serializes each message as Avro against the `news_event_v1` schema

---

## Source 2 — yfinance / AlphaVantage

**What it provides:** OHLCV price ticks for ~500 configured tickers

**Fields captured per tick:**

| Field | Type | Example |
|-------|------|---------|
| `ticker` | string | "AAPL" |
| `open` | float | 189.45 |
| `high` | float | 191.20 |
| `low` | float | 188.90 |
| `close` | float | 190.10 |
| `volume` | long | 42000000 |
| `ts` | timestamp-millis | 1720000000000 |

**Producer file:** [ingestion/price_producer.py](../../ingestion/price_producer.py)

**Key behaviors:**
- Pulls OHLCV for all tickers in the registry (`ingestion/tickers.py`)
- Same dual-write pattern: SQLite first, then Redpanda
- Serializes as Avro against `price_tick_v1`

---

## Ticker Registry

**File:** [ingestion/tickers.py](../../ingestion/tickers.py)

~500 tickers organized by sector. Used by both producers and the GNN to define the node set for the company graph.

```
TICKERS = {
  "AAPL":  {"sector": "Technology"},
  "TSMC":  {"sector": "Semiconductors"},
  "JPM":   {"sector": "Financials"},
  ...
}
```

---

## Dual-Write Pattern

Both producers write to SQLite **before** Redpanda. This serves two purposes:

```
Producer
   │
   ├──► SQLite (local backup)     ← survives Redpanda restarts
   │                              ← enables kafka_replay.py on resume
   │
   └──► Redpanda topic            ← feeds Spark in real time
```

On resume (`resume.sh`), `kafka_replay.py` replays SQLite history into Redpanda topics so Spark starts with data at offset `earliest` — no data is lost between sessions.

---

## Backfill Scripts

On first run, the system needs historical data to train on. Two scripts handle this:

| Script | What it does |
|--------|-------------|
| [ingestion/backfill_news.py](../../ingestion/backfill_news.py) | Pulls 30 days of historical headlines from NewsAPI |
| [ingestion/backfill_prices.py](../../ingestion/backfill_prices.py) | Pulls 1 year of daily OHLCV from yfinance |
| [ingestion/synthetic_news.py](../../ingestion/synthetic_news.py) | Seeds synthetic headlines for tickers with no live coverage |

These are skipped on subsequent runs if the database already has data.

---

## Why These Sources

| Decision | Reason |
|----------|--------|
| NewsAPI over scraping | Reliable, structured, rate-limited at 100 req/day on free tier — manageable for ~500 tickers at 30s intervals |
| yfinance as primary | Free, no API key needed, covers all major exchanges |
| 30-second poll interval | Balances freshness against API rate limits; fast enough to capture intraday moves |
| SQLite as local buffer | Zero-config, durable, supports replay — critical for a dev environment where Docker restarts frequently |
