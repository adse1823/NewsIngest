# Layer 5 — Feature Store: DuckDB

## What This Layer Does

Takes raw windowed data from Spark and transforms it into model-ready features. Computes rolling statistics, builds entity tables, and provides a single clean table that the NLP and modeling layers consume.

---

## What a Feature Store Is

```
WITHOUT a feature store:

  Training pipeline:    computes rolling_1h_mean from scratch
  Inference pipeline:   recomputes rolling_1h_mean from scratch (differently)
                                │
                                ▼
                        Training/serving skew:
                        model trained on one version of a feature,
                        served with a slightly different version
                        → silent accuracy degradation

WITH a feature store:

  Training pipeline:    reads rolling_1h_mean FROM feature store
  Inference pipeline:   reads rolling_1h_mean FROM feature store
                                │
                                ▼
                        Single source of truth.
                        Features computed once, reused everywhere.
```

---

## Data Flow Through the Feature Store

```
Data sources:

  ./data/windowed/*.parquet  ──────────────────┐
  (from Spark)                                 │
                                               ▼
  SQLite (news_articles, price_ticks)  ──► DuckDB ELT
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │   raw_news model    │
                                    │   (base table)      │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌────────────────────────────┐
                                    │  features_sentiment model  │
                                    │                            │
                                    │  rolling_1h_mean           │
                                    │  rolling_24h_std           │
                                    │  volume_zscore             │
                                    └──────────┬─────────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  entity_table model  │
                                    │                      │
                                    │  company nodes       │
                                    │  sector mapping      │
                                    └──────────┬───────────┘
                                               │
                                               ▼
                                    features_export.parquet
                                    (used by NLP + modeling)
```

---

## SQL Models

All models live under `feature_store/models/` as `.sql` files.

### raw_news.sql
```sql
-- Base table: windowed headline counts joined with price changes
SELECT
    w.ticker,
    w.window_start,
    w.window_end,
    w.headline_count,
    p.close - p.open AS price_change,
    p.volume
FROM windowed_news w
JOIN price_windows p
    ON w.ticker = p.ticker
    AND w.window_start = p.window_start
```

### features_sentiment.sql
```sql
-- Rolling features using SQL window functions
SELECT
    ticker,
    window_start,
    headline_count,
    price_change,
    volume,

    AVG(headline_count) OVER (
        PARTITION BY ticker
        ORDER BY window_start
        RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
    ) AS rolling_1h_mean,

    STDDEV(headline_count) OVER (
        PARTITION BY ticker
        ORDER BY window_start
        RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
    ) AS rolling_24h_std,

    (volume - AVG(volume) OVER (PARTITION BY ticker)) /
    NULLIF(STDDEV(volume) OVER (PARTITION BY ticker), 0)
        AS volume_zscore

FROM raw_news
```

### entity_table.sql
```sql
-- Company/sector node registry (used by GNN layer)
SELECT DISTINCT
    ticker,
    sector,
    ROW_NUMBER() OVER (ORDER BY ticker) AS node_id
FROM ticker_registry
```

---

## Rolling Feature Explanation

```
Rolling 1h mean (for AAPL):

  Time:          09:00   09:05   09:10   09:15   09:20
  headline_count:  3       1       0       2       4

  rolling_1h_mean at 09:20:
  = mean of all windows in [08:20, 09:20]
  = mean(3, 1, 0, 2, 4) = 2.0

  This captures: "how much news is AAPL generating relative to its normal?"


Volume z-score (for AAPL):

  historical_mean_volume = 45,000,000
  historical_std_volume  =  8,000,000
  current_volume         = 61,000,000

  z-score = (61M - 45M) / 8M = 2.0

  This captures: "is today's trading volume abnormally high?"
  z-score > 2: unusual → often precedes a move
```

---

## DuckDB vs Alternatives

| Tool | Config needed | Parquet native | SQL window fns | Production ready |
|------|--------------|----------------|----------------|-----------------|
| **DuckDB** | None (in-process) | YES | YES | Single-node only |
| Snowflake | Cloud credentials, account | Via Snowpipe | YES | YES (managed) |
| BigQuery | GCP project, credentials | YES | YES | YES (managed) |
| Pandas | None | Via pyarrow | Limited | Not at scale |
| dbt + Postgres | Postgres server | No | YES | YES |

**Why DuckDB for dev:** Zero config. Reads Parquet natively. Supports the same SQL patterns as Snowflake. Iteration speed matters — no server to spin up, no credentials to manage.

**Why Snowflake in prod:** Three capabilities DuckDB cannot replicate:
- **Snowpipe** — continuous ingestion from S3 (trigger-based, not scheduled)
- **Time Travel** — query any table as of any past timestamp (`AT (timestamp => ...)`)
- **Clustering keys** — physical sort order for large tables, eliminates full scans

---

## dbt-Style Pattern

```
dbt flow (production teams):               This project's flow:

  sources.yml                               SQLite + Parquet files
      │                                          │
      ▼                                          ▼
  raw_news.sql                              raw_news.sql
      │                                          │
      ▼                                          ▼
  features_sentiment.sql                    features_sentiment.sql
      │                                          │
      ▼                                          ▼
  entity_table.sql                          entity_table.sql
      │                                          │
      ▼                                          ▼
  dbt run (dependency ordering)             Airflow DAG (dependency ordering)
  dbt test (data quality checks)            export.py → features_export.parquet
```

The SQL model pattern is identical. dbt itself isn't used — Airflow runs the SQL files directly — but the modular SQL + dependency graph concept is the same.

---

## Airflow Orchestration of Feature Store

```
Airflow DAG: fin_pipeline (daily at 2 AM)

  [export_features] ──► [run_nlp] ──► [train_gnn] ──► [train_model] ──► [run_drift]
        │
        └── runs feature_store/export.py
            which executes:
              1. raw_news.sql
              2. features_sentiment.sql  (depends on raw_news)
              3. entity_table.sql
              4. exports features_export.parquet
```

---

## Files in This Layer

| File | Role |
|------|------|
| [feature_store/models/raw_news.sql](../../feature_store/models/raw_news.sql) | Base windowed table |
| [feature_store/models/features_sentiment.sql](../../feature_store/models/features_sentiment.sql) | Rolling features |
| [feature_store/models/entity_table.sql](../../feature_store/models/entity_table.sql) | Company/sector nodes |
| [feature_store/export.py](../../feature_store/export.py) | DuckDB ELT runner → features_export.parquet |
| [dags/fin_pipeline.py](../../dags/fin_pipeline.py) | Airflow DAG orchestrating all steps |
