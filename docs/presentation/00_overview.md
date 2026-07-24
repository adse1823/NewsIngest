# Financial Intelligence Platform — System Overview

## What It Does

Answers one question in near real-time:

> **Given the last few hours of financial news about a company, will its stock price go up or down in the next hour?**

It ingests live news and price data, processes it through a streaming pipeline, enriches it with NLP and graph signals, trains a forecasting model, and serves predictions via a REST API — with full monitoring and auto-retraining.

---

## End-to-End Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│         NewsAPI (headlines)      yfinance (price ticks)         │
└───────────────┬──────────────────────────┬──────────────────────┘
                │                          │
                ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                             │
│              Redpanda  (Kafka-compatible broker)                │
│        Topic: news-raw          Topic: price-ticks              │
│                                                                 │
│        ┌──────────────────────────────────────┐                 │
│        │    Schema Registry  (port 8081)      │                 │
│        │  Avro schemas · compatibility rules  │                 │
│        └──────────────────────────────────────┘                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STREAM PROCESSING                             │
│             Apache Spark Structured Streaming                   │
│      5-min tumbling windows · watermarks · Parquet sink         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FEATURE STORE                              │
│               DuckDB (dev)  /  Snowflake (prod)                 │
│    Rolling features · entity tables · Airflow-orchestrated      │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌──────────────────────┐          ┌───────────────────────────────┐
│    NLP PIPELINE      │          │      KNOWLEDGE GRAPH          │
│  FinBERT sentiment   │          │   PyTorch Geometric           │
│  + 768-dim embeddings│          │   GraphSAGE GNN               │
└──────────┬───────────┘          └──────────────┬────────────────┘
           │                                     │
           └──────────────┬──────────────────────┘
                          │  concat features
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID MODEL                               │
│         LightGBM  (tabular + GNN embeddings)                    │
│         Tracked and versioned with MLflow                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                               │
│            FastAPI · Docker · AWS SageMaker                     │
│    /predict   /explain (SHAP)   /health   /shap-summary         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY                                 │
│  Evidently AI (drift) · Prometheus · Grafana · Airflow alerts   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer Index

| # | Layer | File |
|---|-------|------|
| 1 | Data Sources | [01_data_sources.md](01_data_sources.md) |
| 2 | Ingestion — Redpanda | [02_ingestion_redpanda.md](02_ingestion_redpanda.md) |
| 3 | Schema Registry + Avro | [03_schema_registry.md](03_schema_registry.md) |
| 4 | Stream Processing — Spark | [04_stream_processing.md](04_stream_processing.md) |
| 5 | Feature Store — DuckDB | [05_feature_store.md](05_feature_store.md) |
| 6 | NLP Sentiment — FinBERT | [06_nlp_sentiment.md](06_nlp_sentiment.md) |
| 7 | Knowledge Graph + GNN | [07_knowledge_graph_gnn.md](07_knowledge_graph_gnn.md) |
| 8 | Hybrid Forecasting Model | [08_hybrid_model.md](08_hybrid_model.md) |
| 9 | Serving Layer — FastAPI | [09_serving_layer.md](09_serving_layer.md) |
| 10 | Monitoring & Observability | [10_monitoring_observability.md](10_monitoring_observability.md) |

---

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC (5-fold TimeSeriesSplit) | 0.6721 |
| Naive baseline | 0.50 |
| Training data | 28,267 news articles · 145,214 price rows · 502 tickers |
| Test suite | 130 tests, 1 skipped |
| Drift threshold | 30% of columns drifted → auto-retrain |

---

## What Makes This "Production"

Most ML projects are notebooks that stop at a trained model. This system adds:

- Schema enforcement — bad data rejected before it enters the pipeline
- Streaming with watermarks — late-arriving data handled correctly
- Feature store — features computed once, reused at training and inference
- Model registry — every version tracked; champion promoted explicitly
- Explainability — every prediction has SHAP attribution
- Drift detection + auto-retraining — system heals itself
- Real-time metrics — latency, error rate, prediction volume in Grafana
