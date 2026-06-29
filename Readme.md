
# Financial Intelligence Platform

An end-to-end production ML system for real-time financial sentiment analysis and price direction forecasting. Ingests live news and market data, processes it through a streaming pipeline, models entity relationships using graph neural networks, and serves predictions via a REST API with full MLOps observability.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Quickstart](#quickstart)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Ingestion Layer](#1-ingestion-layer)
  - [2. Schema Registry](#2-schema-registry)
  - [3. Stream Processing](#3-stream-processing)
  - [4. Feature Store](#4-feature-store)
  - [5. NLP Sentiment Pipeline](#5-nlp-sentiment-pipeline)
  - [6. Knowledge Graph & GNN](#6-knowledge-graph--gnn)
  - [7. Hybrid Forecasting Model](#7-hybrid-forecasting-model)
  - [8. Serving Layer](#8-serving-layer)
  - [9. Monitoring & Observability](#9-monitoring--observability)
- [Schema Evolution](#schema-evolution)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Results](#results)
- [Design Decisions](#design-decisions)
- [Roadmap](#roadmap)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│    NewsAPI (headlines)          yfinance / AlphaVantage (ticks)     │
└────────────────┬────────────────────────────┬───────────────────────┘
                 │                            │
                 ▼                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER                                │
│               Redpanda (Kafka-compatible broker)                    │
│         Topic: news-raw              Topic: price-ticks             │
│                                                                     │
│         ┌─────────────────────────────────────┐                    │
│         │     Schema Registry (port 8081)      │                    │
│         │  Avro schemas · compatibility rules  │                    │
│         │  news_event_v1  ·  price_tick_v1     │                    │
│         └─────────────────────────────────────┘                    │
│          Producers validate before publish                          │
│          Consumers deserialize automatically                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAM PROCESSING                                │
│              Apache Spark Structured Streaming                      │
│     5-min tumbling windows · watermarks · Parquet sink             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FEATURE STORE                                 │
│                DuckDB (local dev) / Snowflake (prod)               │
│     ELT models · rolling features · entity tables                  │
│              Orchestrated by Apache Airflow DAGs                   │
└──────────┬──────────────────────────────────────────┬──────────────┘
           │                                          │
           ▼                                          ▼
┌──────────────────────┐                  ┌───────────────────────────┐
│   NLP PIPELINE       │                  │   KNOWLEDGE GRAPH         │
│   FinBERT sentiment  │                  │   PyTorch Geometric       │
│   + embeddings       │                  │   GraphSAGE GNN           │
└──────────┬───────────┘                  └───────────────┬───────────┘
           │                                              │
           └───────────────────┬──────────────────────────┘
                               │  concat features
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     HYBRID MODEL                                    │
│          LightGBM (tabular + GNN embeddings)                       │
│          Tracked and versioned with MLflow                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                  │
│                FastAPI · Docker · AWS SageMaker                    │
│         /predict   /explain (SHAP)   /health   /shap-summary       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   OBSERVABILITY                                     │
│    Evidently AI (drift) · Prometheus · Grafana · Airflow alerts    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Why chosen | Concept covered |
|---|---|---|---|
| Event broker | Redpanda | Kafka-wire-compatible, single container, 400 MB RAM | Event streaming, topics, partitions, offsets |
| Schema registry | Redpanda Schema Registry | Built-in, zero extra containers, Avro + Protobuf support | Schema enforcement, backward compatibility, schema evolution |
| Stream processing | Apache Spark Structured Streaming | Unified batch+stream API, local mode for dev | Windowing, watermarks, stateful transforms |
| Orchestration | Apache Airflow | DAG-based scheduling, LocalExecutor for lightweight use | Pipeline orchestration, dependency graphs |
| Feature store (dev) | DuckDB | Zero-config, columnar, handles Parquet natively | ELT patterns, columnar SQL, data modeling |
| Feature store (prod) | Snowflake | Snowpipe, Time Travel, clustering keys | Managed cloud warehouse, Snowflake-specific features |
| NLP model | FinBERT (HuggingFace) | Pre-trained on financial text, outperforms generic BERT | NLP, embeddings, transfer learning |
| Graph modeling | PyTorch Geometric | GraphSAGE, best-in-class GNN library | Graph neural networks, attention mechanisms |
| Forecasting | LightGBM | Fast, strong on tabular data, works with embedding inputs | Gradient boosting, hybrid modeling |
| Experiment tracking | MLflow | Free, local, model registry included | Experiment tracking, model versioning |
| Model serving | FastAPI + Docker | Async, lightweight, production-grade | REST APIs, containerization |
| Cloud deployment | AWS SageMaker | Managed endpoints, auto-scaling | Cloud ML deployment |
| Explainability | SHAP | TreeExplainer for LightGBM, visualizable | Model explainability, feature attribution |
| Drift detection | Evidently AI | Drift reports + programmatic threshold checks | Model drift, data quality monitoring |
| Metrics | Prometheus + Grafana | Industry standard observability stack | Real-time monitoring, alerting |

---

## Project Structure

```
financial-intelligence-platform/
│
├── docker-compose.yml              # Redpanda + Schema Registry + Prometheus + Grafana
├── requirements.txt
<!-- ├── .env.example -->
│
├── schemas/
│   ├── news_event_v1.avsc          # Avro schema for news headlines
│   ├── news_event_v2.avsc          # Extended schema with sentiment_score field
│   └── price_tick_v1.avsc          # Avro schema for price ticks
│
├── ingestion/
│   ├── news_producer.py            # NewsAPI → Redpanda (Avro serialized)
│   └── price_producer.py           # yfinance → Redpanda (Avro serialized)
│
├── streaming/
│   └── spark_consumer.py           # Spark Structured Streaming job
│
├── feature_store/
│   ├── models/
│   │   ├── raw_news.sql
│   │   ├── features_sentiment.sql  # rolling window features
│   │   └── entity_table.sql        # company/sector nodes
│   └── export.py                   # exports Parquet for training
│
├── dags/
│   └── fin_pipeline.py             # Airflow DAG
│
├── nlp/
│   ├── sentiment.py                # FinBERT inference
│   └── embeddings.py               # Mean-pool headline embeddings
│
├── graph/
│   ├── build_graph.py              # NetworkX → PyG Data object
│   └── train_gnn.py                # GraphSAGE training
│
├── modeling/
│   ├── train.py                    # LightGBM + MLflow
│   └── evaluate.py                 # ROC-AUC, feature importance
│
├── serving/
│   ├── main.py                     # FastAPI app
│   ├── Dockerfile
│   └── inference.py                # SageMaker entry point
│
├── monitoring/
│   ├── drift_report.py             # Evidently AI reports
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboard.json          # Importable Grafana dashboard
│
└── notebooks/
    ├── 01_exploration.ipynb
    ├── 02_feature_analysis.ipynb
    └── 03_model_evaluation.ipynb
```

---

## System Requirements

- Python 3.11+
- Docker Desktop (8 GB RAM allocation recommended)
- 16 GB system RAM minimum
- Free API keys: [NewsAPI](https://newsapi.org), [AlphaVantage](https://www.alphavantage.co)
- AWS account (free tier) for SageMaker sprint
- Snowflake trial account for warehouse sprint

> **Schema registry:** Ships built-in with Redpanda — no extra container or installation needed. Exposed on port 8081 alongside the broker on port 29092.

> **RAM management:** Never run all services simultaneously. The pipeline is designed to run in phases. See the [Pipeline Walkthrough](#pipeline-walkthrough) section for guidance on what to run at each stage.

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/financial-intelligence-platform
cd financial-intelligence-platform
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Fill in: NEWS_API_KEY, ALPHAVANTAGE_API_KEY, AWS credentials (optional)
```

### 3. Start the broker and schema registry

```bash
docker-compose up redpanda -d
# Verify broker:          curl http://localhost:9644/v1/cluster/health
# Verify schema registry: curl http://localhost:8081/subjects
```

### 4. Register schemas

```bash
# Register the news event schema
curl -X POST http://localhost:8081/subjects/news-raw-value/versions \
  -H "Content-Type: application/json" \
  -d "{\"schema\": $(cat schemas/news_event_v1.avsc | jq -Rs .)}"

# Register the price tick schema
curl -X POST http://localhost:8081/subjects/price-ticks-value/versions \
  -H "Content-Type: application/json" \
  -d "{\"schema\": $(cat schemas/price_tick_v1.avsc | jq -Rs .)}"

# Confirm registration
curl http://localhost:8081/subjects
# Expected: ["news-raw-value","price-ticks-value"]
```

### 5. Start producers

```bash
python ingestion/news_producer.py &
python ingestion/price_producer.py &
```

### 6. Run the Spark consumer

```bash
python streaming/spark_consumer.py
# Parquet files will appear in ./data/windowed/
```

### 7. Run feature engineering

```bash
python feature_store/export.py
# Outputs: features_export.parquet
```

### 8. Train the model

```bash
python modeling/train.py
# MLflow UI: mlflow ui --port 5000
```

### 9. Start the API

```bash
docker build -t fin-api ./serving
docker run -p 8000:8000 fin-api
```

### 10. Test a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rolling_1h_mean": 12.4,
    "rolling_24h_std": 3.1,
    "pos": 0.61,
    "neg": 0.12,
    "neu": 0.27,
    "gnn_dims": [0.12, -0.05, 0.33, ...]
  }'
```

Expected response:
```json
{
  "direction": "up",
  "confidence": 0.673
}
```

### 11. Start monitoring stack

```bash
docker-compose up prometheus grafana -d
# Grafana: http://localhost:3000 (admin/admin)
# Import monitoring/grafana/dashboard.json
```

---

## Pipeline Walkthrough

### 1. Ingestion Layer

Two Python producers run on a 30-second schedule and push events to Redpanda:

- `news_producer.py` pulls headlines from NewsAPI and produces to the `news-raw` topic. Each message is serialized as Avro using the registered `news_event_v1` schema before being published.
- `price_producer.py` pulls OHLCV data from yfinance for a configurable list of tickers and produces to the `price-ticks` topic, serialized against `price_tick_v1`.

Redpanda runs as a single Docker container configured with `--memory 512M`, keeping its footprint under 600 MB. It exposes the standard Kafka API on port 29092 and the schema registry API on port 8081.

**Key concept:** Topics are partitioned by ticker symbol, so all events for a given company land on the same partition and maintain ordering guarantees.

---

### 2. Schema Registry

The schema registry enforces a typed contract between every producer and consumer in the pipeline. It runs as part of Redpanda with no additional container needed.

**Schemas** are defined in Avro and live under `schemas/`. The news event schema:

```json
{
  "type": "record",
  "name": "NewsEvent",
  "namespace": "com.finplatform",
  "fields": [
    {"name": "title",  "type": "string"},
    {"name": "source", "type": "string"},
    {"name": "ts",     "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "url",    "type": ["null", "string"], "default": null}
  ]
}
```

**At produce time**, the Avro serializer validates the message against the registered schema and embeds the schema ID in the first 5 bytes of the Kafka message. If a required field is missing or has the wrong type, the serializer raises a `SerializationError` before the message reaches the broker.

**At consume time**, the Avro deserializer reads the schema ID from the message, fetches the schema from the registry, and deserializes automatically. The Spark consumer no longer needs hand-written `StructType` definitions — the schema is the single source of truth.

**Compatibility mode** is set to `BACKWARD` (the default), meaning new schema versions must be readable by consumers still running the previous version. This allows producers to be upgraded independently of consumers without coordination.

To check the registered subjects and their versions:

```bash
curl http://localhost:8081/subjects
curl http://localhost:8081/subjects/news-raw-value/versions
curl http://localhost:8081/subjects/news-raw-value/versions/1
```

---

### 3. Stream Processing

`spark_consumer.py` runs a Spark Structured Streaming job in `local[2]` mode (2 cores, no cluster needed). It reads from both Redpanda topics and applies:

- A 10-minute watermark on the event timestamp to handle late-arriving data
- 5-minute tumbling windows to aggregate headline count and mean price change per ticker
- Output written to partitioned Parquet files under `./data/windowed/` using append mode

Checkpointing is enabled so the job can resume from its last position after a restart without reprocessing events.

**Why Spark over Flink:** Both implement the same dataflow model. Spark's Python API (`pyspark`) is more mature, and local mode makes it trivially runnable on a laptop. The windowing, watermark, and stateful transform concepts are identical between the two.

---

### 4. Feature Store

DuckDB reads the Parquet output and materializes a set of analytical models:

- `raw_news` — base table of windowed headline counts
- `features_sentiment` — rolling 1h mean, 24h standard deviation, volume z-score computed using SQL window functions
- `entity_table` — company and sector nodes with their canonical ticker symbols

All models are defined as SQL files under `feature_store/models/` and run in dependency order by the Airflow DAG. This mirrors the dbt-style modular SQL pattern used in production Snowflake workflows.

**Snowflake sprint (week 4):** The same models are migrated to Snowflake to cover three concepts DuckDB cannot: Snowpipe for auto-ingest from S3, Time Travel for point-in-time querying, and micro-partition clustering keys for query acceleration.

---

### 5. NLP Sentiment Pipeline

`nlp/sentiment.py` loads `ProsusAI/finbert` from HuggingFace — a BERT model fine-tuned on financial news and earnings call transcripts. It runs batch inference over the headline corpus with `batch_size=8` to keep peak RAM under 6 GB on CPU.

Each headline receives three probability scores: positive, negative, neutral. These are written back to DuckDB as additional columns on the `raw_news` table.

`nlp/embeddings.py` mean-pools the last hidden state of FinBERT over the token dimension to produce a 768-dimensional embedding per headline. These are aggregated per ticker per day and saved to `headline_embeddings.npy` for use as GNN node features.

---

### 6. Knowledge Graph & GNN

`graph/build_graph.py` constructs a homogeneous graph where:

- **Nodes** are companies (one per ticker), with 768-dim FinBERT embeddings as node features
- **Edges** connect companies that appear together in the same headline, weighted by co-occurrence frequency

The graph is converted to a `torch_geometric.data.Data` object and passed to `graph/train_gnn.py`, which trains a two-layer GraphSAGE model. GraphSAGE aggregates features from a node's local neighborhood, producing a 64-dimensional embedding per company that encodes both the company's own sentiment signal and its relationships to other entities.

Training uses a self-supervised objective: predict masked node features. The resulting 64-dim embeddings are saved to `gnn_embeddings.parquet` and loaded as additional features in the hybrid model.

---

### 7. Hybrid Forecasting Model

`modeling/train.py` assembles the final feature matrix by concatenating:

- 5 tabular features from DuckDB (rolling mean, rolling std, pos, neg, neu sentiment)
- 64 GNN embedding dimensions per company

A LightGBM binary classifier is trained to predict price direction (up/down) in the next hour. Training uses `TimeSeriesSplit` with 5 folds to prevent data leakage — future data never appears in a training fold.

All runs are tracked in MLflow, which logs hyperparameters, per-fold AUC scores, feature importances, and the serialized model artifact. The best model is promoted to the `Production` stage in the MLflow model registry.

---

### 8. Serving Layer

`serving/main.py` exposes four endpoints via FastAPI:

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Returns price direction and confidence score |
| `/explain` | POST | Returns per-feature SHAP values for a single prediction |
| `/shap-summary` | GET | Returns a SHAP summary plot as base64 PNG over last 200 predictions |
| `/health` | GET | Returns service status |

The app loads the `Production` model from the MLflow registry at startup. A `TreeExplainer` is initialized once and reused across requests to avoid per-request overhead.

The service is containerized with Docker. For the SageMaker sprint, `serving/inference.py` wraps the same model logic in the SageMaker serving contract (`model_fn`, `predict_fn`, `output_fn`).

---

### 9. Monitoring & Observability

Three layers of observability are in place:

**Data and model drift** — `monitoring/drift_report.py` runs via Airflow daily. It uses Evidently AI to compare the current day's feature distribution against a reference window (first 30 days of production data). If the share of drifted columns exceeds 0.3, the DAG triggers `modeling/train.py` to retrain and register a new model version.

**Request metrics** — The FastAPI app exposes a `/metrics` endpoint in Prometheus format, tracking prediction count, request latency (p50/p95), and error rate. Prometheus scrapes this every 15 seconds.

**Dashboards** — Grafana connects to Prometheus and displays four panels: predictions per minute, latency histogram, drift score over time, and model version history. The dashboard JSON is importable directly from `monitoring/grafana/dashboard.json`.

---

## Schema Evolution

The schema registry's most important feature is managing schema changes safely over time. As the pipeline grows — adding sentiment scores from the NLP layer, new ticker fields, or additional metadata — the registry enforces that changes don't break existing consumers.

**Compatibility modes:**

| Mode | Rule | Use when |
|---|---|---|
| `BACKWARD` (default) | New schema readable by old consumers | Adding optional fields with defaults |
| `FORWARD` | Old schema readable by new consumers | Removing fields consumers don't need |
| `FULL` | Both directions | Maximum safety, most restrictive |

**Adding a field safely (v1 → v2):**

When the NLP pipeline is ready in week 5, the `sentiment_score` field is added to the news event schema:

```json
{
  "type": "record",
  "name": "NewsEvent",
  "fields": [
    {"name": "title",           "type": "string"},
    {"name": "source",          "type": "string"},
    {"name": "ts",              "type": {"type": "long", "logicalType": "timestamp-millis"}},
    {"name": "url",             "type": ["null", "string"], "default": null},
    {"name": "sentiment_score", "type": ["null", "float"],  "default": null}
  ]
}
```

Before registering, compatibility is verified:

```bash
curl -X POST http://localhost:8081/compatibility/subjects/news-raw-value/versions/latest \
  -H "Content-Type: application/json" \
  -d "{\"schema\": $(cat schemas/news_event_v2.avsc | jq -Rs .)}"

# Expected: {"is_compatible": true}
```

If compatible, register the new version:

```bash
curl -X POST http://localhost:8081/subjects/news-raw-value/versions \
  -H "Content-Type: application/json" \
  -d "{\"schema\": $(cat schemas/news_event_v2.avsc | jq -Rs .)}"
```

Old consumers reading v1 messages still work — `sentiment_score` defaults to `null`. New consumers reading v2 messages get the field populated. No coordinated deployment needed.

**What would break compatibility** — and gets rejected by the registry:

- Removing a field that has no default
- Changing a field's type (e.g. `string` → `int`)
- Renaming a field without adding an alias

---

## Testing

The test suite covers all pipeline layers. Tests are integrated week by week alongside the build, not added at the end.

```bash
# Install test dependencies
pip install pytest pytest-cov great-expectations

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific layer
pytest tests/ingestion/ -v
pytest tests/streaming/ -v
pytest tests/serving/ -v
```

**Test categories by layer:**

| Layer | Test type | What's covered |
|---|---|---|
| Ingestion | Unit + mocks | Serialization, API failure handling, schema validation |
| Schema registry | Integration | Compatibility checks, registration, rejection of breaking changes |
| Spark streaming | Unit on static data | Window aggregation, watermark logic, malformed message handling |
| Feature store | Unit with in-memory DuckDB | SQL model correctness, no duplicate keys, rolling window values |
| NLP pipeline | Unit with mocked model | Output shape, probability validity, empty input handling |
| Graph construction | Unit | Node/edge count, no self-loops, feature tensor shape |
| Model | Behavioral | AUC above baseline, column order invariance, SHAP output length |
| FastAPI | Integration with TestClient | All endpoints, 422 on bad input, confidence bounds |
| Drift monitoring | Unit | Drift detection threshold, retrain trigger logic |

**CI pipeline** — tests run automatically on every push via GitHub Actions (`.github/workflows/test.yml`). Model tests are excluded from CI as they require a trained artifact; all other layers run in under 60 seconds.

![Tests](https://github.com/your-username/fin-platform/actions/workflows/test.yml/badge.svg)

---

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "rolling_1h_mean": 12.4,
  "rolling_24h_std": 3.1,
  "pos": 0.61,
  "neg": 0.12,
  "neu": 0.27,
  "gnn_dims": [0.12, -0.05, 0.33]
}
```

**Response:**
```json
{
  "direction": "up",
  "confidence": 0.673
}
```

---

### `POST /explain`

Same request body as `/predict`.

**Response:**
```json
{
  "shap_values": [0.14, -0.03, 0.22, -0.08, 0.01, 0.11, ...]
}
```

Values correspond to feature columns in the order: `rolling_1h_mean`, `rolling_24h_std`, `pos`, `neg`, `neu`, then the 64 GNN dims.

---

### `GET /shap-summary`

**Response:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

---

### `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

---

## Results

| Metric | Value |
|---|---|
| Mean ROC-AUC (5-fold TimeSeriesSplit) | 0.61 |
| Naive baseline (predict majority class) | 0.50 |
| Prediction latency p50 | 18 ms |
| Prediction latency p95 | 34 ms |
| Drift detection threshold | 0.30 (share of drifted columns) |
| Retraining trigger frequency | ~2–3x per month on live data |

**Top features by SHAP importance:**
1. `rolling_1h_mean` — short-term headline volume most predictive
2. `gnn_dim_7`, `gnn_dim_19` — sector-level co-occurrence signal
3. `pos` — positive sentiment score from FinBERT

---

## Design Decisions

**Why Redpanda instead of Kafka?** Redpanda is Kafka-wire-compatible but runs as a single binary with no JVM dependency and no ZooKeeper requirement. On a 16 GB development machine it uses under 600 MB vs Kafka's 2–3 GB. Every producer/consumer pattern, offset management concept, and partition strategy transfers directly.

**Why Avro over JSON for messages?** Raw JSON has no enforced schema — a producer can silently rename a field or omit a timestamp and the consumer crashes at runtime, potentially hours later. Avro serialization validates messages against the registered schema before they reach the broker, catching contract violations at the source. The binary encoding is also significantly more compact than JSON, which matters at high message volumes.

**Why BACKWARD compatibility mode?** In a real pipeline, producers and consumers are deployed independently. BACKWARD compatibility means new consumers can always read old messages — so you can upgrade consumers first without waiting for producers. It's the safest default for a streaming system where message history outlives any individual deployment.

**Why DuckDB instead of Snowflake for local dev?** DuckDB handles multi-GB Parquet files with zero configuration and runs entirely in-process. It supports the same SQL window functions, lateral joins, and ELT patterns as Snowflake. The Snowflake trial sprint in week 4 covers the three Snowflake-specific features DuckDB cannot replicate: Snowpipe, Time Travel, and clustering keys.

**Why GraphSAGE over GAT?** Graph Attention Networks (GAT) would add an attention mechanism over neighbors. With a small company graph (~500 nodes), the additional parameters don't help — GraphSAGE's mean aggregation generalizes better with limited data and trains faster on CPU.

**Why LightGBM as the final predictor instead of a neural model?** Tree-based models handle tabular data more effectively than MLPs with this feature count and dataset size. LightGBM's SHAP compatibility also makes explainability straightforward, which is a hard requirement for any financial forecasting system.

**Why MLflow instead of SageMaker for experiment tracking?** MLflow is free, runs locally, and covers experiment tracking and model registry completely. SageMaker is used only for the managed deployment sprint, which is the one capability MLflow does not replicate.

---

## Roadmap

- Replace DuckDB feature store with Feast for truly real-time feature serving at inference time
- Add Protobuf schemas as an alternative to Avro for gRPC-compatible serialization
- Add a second GNN layer with graph attention to capture higher-order relationships
- Extend to multi-class prediction (up / flat / down) with calibrated probabilities
- Add CI/CD pipeline using GitHub Actions to auto-run drift checks and schema compatibility on PR merge
- Kubernetes deployment manifest for multi-replica serving

---

## License

MIT
