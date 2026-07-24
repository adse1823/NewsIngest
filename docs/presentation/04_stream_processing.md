# Layer 4 — Stream Processing: Apache Spark Structured Streaming

## What This Layer Does

Reads continuously from both Redpanda topics, aggregates events into 5-minute windows per ticker, and writes the results as Parquet files for the feature store to consume.

---

## Why Streaming Instead of Batch Polling

```
BATCH POLLING (cron every 5 min)

Time:   0    5    10   15   20   25   30  (minutes)
        │    │    │    │    │    │    │
Poll:   ▼    ▼    ▼    ▼    ▼    ▼    ▼
       read  read  read read read read read
       all   all   all  all  all  all  all

Problem: poll too often → wasted CPU
         poll too slowly → stale data
         clock-based → ignores EVENT timestamps


SPARK STRUCTURED STREAMING (event-driven)

Time:   0    5    10   15   20   25   30  (minutes)
        │                                │
        └────────────────────────────────┘
                 continuous read
                 windows triggered by EVENT timestamps
                 late data handled by watermark
                 no wasted polling
```

---

## Core Concepts Used

### 1. Tumbling Windows

```
Event timestamps (not wall clock):

        │     Window [0:00–0:05)    │   Window [0:05–0:10)  │
        │                           │                        │
  AAPL  ●──────●────●               │         ●──────●       │
  MSFT  │    ●──────────────●       │                        │
        │                           │                        │
        0:00  0:01  0:02  0:03  0:04  0:05  0:06  0:07  0:08  0:09
                                   │                        │
                                  close                    close
                                  window,                  window
                                  emit result              emit result

Result per window:
  {ticker: "AAPL", window_start: "0:00", headline_count: 3, mean_price_change: 0.4}
  {ticker: "MSFT", window_start: "0:00", headline_count: 2, mean_price_change: -0.1}
```

### 2. Watermark (Late Data Handling)

```
Watermark = 10 minutes

            event time
  ─────────────────────────────────────────────────────►
  Window [0:00–0:05) closes at max_event_time - 10min

  Message arrives at 0:14 with event timestamp 0:03:
                                  │
  max_event_seen = 0:14           │
  watermark_threshold = 0:14 - 0:10 = 0:04
                                  │
  message timestamp 0:03 < watermark 0:04 → DROPPED (too late)
  message timestamp 0:09 < watermark 0:04? NO → INCLUDED

  Window stays open until watermark passes its end time.
  After that, late messages are dropped (no unbounded state).
```

### 3. Checkpointing

```
Spark job runs:
  ↓
  Reads offset 0–100 from news-raw
  Writes Parquet files
  Commits checkpoint: "last read offset = 100"
  ↓
  Job crashes (Docker restart, OOM, etc.)
  ↓
  Job restarts
  Reads checkpoint: "last read offset = 100"
  Resumes from offset 101
  No reprocessing. No data loss.

Checkpoint location: ./data/checkpoints/
```

---

## Full Data Flow Through Spark

```
Redpanda                    Spark Job                      File System
──────────                  ─────────                      ───────────

news-raw ──────────────►  deserialize Avro
                           extract: ticker, ts, title
                                │
price-ticks ───────────►  deserialize Avro
                           extract: ticker, ts, close, volume
                                │
                                ▼
                         Apply 10-min watermark
                         on event timestamp (ts)
                                │
                                ▼
                         5-min tumbling window
                         GROUP BY ticker, window
                         AGG: count(headlines),
                              mean(price_change)
                                │
                                ▼
                         Write mode: APPEND          ──────► ./data/windowed/
                         Format: Parquet                      year=2024/
                         Partitioned by date                  month=07/
                                                              day=14/
                                                              *.parquet
```

---

## Output Schema (Windowed Parquet)

```
┌──────────────┬────────────────────┬────────────────────┬───────────────────┬────────────────────┐
│   ticker     │   window_start     │   window_end       │  headline_count   │  mean_price_change  │
├──────────────┼────────────────────┼────────────────────┼───────────────────┼────────────────────┤
│  AAPL        │  2024-07-14 09:00  │  2024-07-14 09:05  │       3           │       0.42          │
│  MSFT        │  2024-07-14 09:00  │  2024-07-14 09:05  │       1           │      -0.11          │
│  TSLA        │  2024-07-14 09:05  │  2024-07-14 09:10  │       5           │       1.23          │
└──────────────┴────────────────────┴────────────────────┴───────────────────┴────────────────────┘
```

These Parquet files are the input to the Feature Store (Layer 5).

---

## Spark vs Flink

| Aspect | Spark Structured Streaming | Apache Flink |
|--------|--------------------------|--------------|
| Python API maturity | High (PySpark, well-documented) | Lower (PyFlink less mature) |
| Local development | `local[2]` mode, no cluster needed | Requires mini-cluster setup |
| Windowing model | Identical | Identical |
| Watermarks | Identical | Identical |
| Stateful transforms | Identical | Identical |
| Best for | Unified batch + stream, Python-first teams | Low-latency streaming, Java/Scala shops |

**Decision:** Same concepts, simpler local setup. Everything learned in Spark transfers to Flink.

---

## Local Mode: `local[2]`

```
Spark local[2] means:
  - 1 driver + 2 executor threads
  - Runs entirely in one JVM process on your laptop
  - No cluster, no YARN, no Kubernetes
  - Good for: dev, testing, processing < ~10 GB
  - Production equivalent: Spark on EMR, Databricks, or GKE
```

---

## Files in This Layer

| File | Role |
|------|------|
| [streaming/spark_consumer.py](../../streaming/spark_consumer.py) | Full Spark Structured Streaming job |
| `./data/windowed/` | Parquet output (created at runtime) |
| `./data/checkpoints/` | Spark checkpoint location (created at runtime) |

---

## Known Issue

> Spark streaming on Windows has a bind-mount write permission issue that blocks Parquet output in Docker. Tracked in roadmap. Local (non-Docker) Spark run works correctly.
