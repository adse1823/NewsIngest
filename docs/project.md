# Project Overview

## What This Is

A financial intelligence platform that ingests live news and stock price data, processes it
through a machine learning pipeline, and serves predictions on whether a stock's price will
go up or down in the next 5-minute window.

---

## The Core Idea

Given recent news sentiment about a company and how it relates to other companies in the news,
predict whether its stock price will go up or down in the next 5 minutes.

Three signals are combined to make this prediction:

1. **Headline activity** — how many news articles is this company getting, and how does that
   compare to its recent average?
2. **Sentiment** — are those articles positive, negative, or neutral?
3. **Company relationships** — which other companies are being mentioned alongside this one,
   and what is their sentiment doing?

---

## Tickers Tracked

10 companies across four sectors:

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple | Technology |
| MSFT | Microsoft | Technology |
| GOOGL | Alphabet (Google) | Technology |
| NVDA | NVIDIA | Technology |
| META | Meta (Facebook) | Technology |
| AMZN | Amazon | Consumer Discretionary |
| TSLA | Tesla | Automotive |
| JPM | JPMorgan Chase | Financials |
| BAC | Bank of America | Financials |
| GS | Goldman Sachs | Financials |

---

## Data Sources

| Source | What it provides | Cost |
|---|---|---|
| NewsAPI | Live news headlines, filtered by ticker/company name | Free tier (100 req/day) |
| yfinance | Real-time OHLCV price data from Yahoo Finance | Free, no key needed |

---

## Two Phases

### Phase 1 — Local (current)
Everything runs inside a Python virtual environment. No Docker, no cloud services required.
SQLite stores raw data. DuckDB runs feature engineering. Streamlit provides the dashboard.

### Phase 2 — Production (future)
Replace the SQLite ingestion layer with Redpanda (Kafka-compatible broker) running in WSL2.
Replace Spark for stream processing. Replace Streamlit with Prometheus + Grafana.
Everything from the feature store downward stays identical between phases.

---

## Tools Used

| Tool | What it does in this project |
|---|---|
| **SQLite** | Stores raw headlines and price ticks as they arrive |
| **DuckDB** | Reads SQLite, computes 5-minute windows and rolling features using SQL |
| **FinBERT** | NLP model fine-tuned on financial text — produces sentiment scores and embeddings per headline |
| **PyTorch Geometric / GraphSAGE** | Models company relationships as a graph — tickers mentioned together in headlines get connected |
| **LightGBM** | Gradient boosting classifier — predicts price direction from tabular + graph features |
| **MLflow** | Tracks every training run, versions and stores trained models |
| **FastAPI** | Serves predictions, explanations, and health checks over HTTP |
| **SHAP** | Explains why the model made each prediction (which features mattered most) |
| **Evidently AI** | Detects when input data has drifted far enough from training data to warrant retraining |
| **Streamlit** | Live dashboard showing ingestion counts, price charts, and sentiment scores |

### Phase 2 additions
| Tool | What it replaces / adds |
|---|---|
| **Redpanda** | Replaces SQLite as the message queue between producers and consumers |
| **Avro + Schema Registry** | Enforces typed contracts on every message before it enters the queue |
| **Apache Spark** | Replaces DuckDB windowing for real-time stream aggregation |
| **Apache Airflow** | Orchestrates the daily batch pipeline with dependency tracking |
| **Prometheus + Grafana** | Replaces Streamlit for production-grade metrics and alerting |

---

## Future Extensions Planned

- **Long-term trend model** — predict next week or next month return using 7d/30d/90d rolling features
- **S&P 500 coverage** — expand from 10 tickers to 500 using a config file
- **Historical backfill** — 6 months of data via GDELT or AlphaVantage News
- **Claude router interface** — natural language query that automatically routes to the right model
  (real-time vs long-term) based on the time horizon in the question
