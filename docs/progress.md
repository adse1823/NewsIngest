# Progress Log

<!-- markdownlint-disable MD024 -->

## Session 2 — 2 July 2026

### What Was Fixed / Built

All outstanding issues from Session 1 resolved. Phase 1 is now fully working end-to-end.

---

### Pipeline Status

| Step | Status | Notes |
|---|---|---|
| Virtual environment | Done | Python 3.11.5, venv at `./venv/` |
| Dependencies installed | Done | `pip install -r requirements.txt` |
| `.env` configured | Done | `NEWS_API_KEY` filled in, `MLFLOW_TRACKING_URI` removed (hardcoded to `./mlruns`) |
| News producer | Done | Deduplication added — INSERT OR IGNORE on URL |
| Price producer | Done | |
| News backfill | Done | `ingestion/backfill_news.py` — weekly date windows, 40 API requests, ~1,000 unique articles |
| Price backfill | Done | `ingestion/backfill_prices.py` — 1 month of 5-min bars via yfinance, 16,420 rows |
| Feature export | Done | pct_change fixed to use LAG() between-window return instead of within-window |
| FinBERT sentiment | Done | Fixed numpy.ndarray type bug; deduplication of headline lists added |
| FinBERT embeddings | Done | |
| Graph build | Done | |
| GNN training | Done | |
| LightGBM training | Done | AUC 0.6721 across 2 folds — real, meaningful score above 0.5 baseline |
| FastAPI | Done | `/predict` returns direction + confidence; fixed Booster vs LGBMClassifier mismatch |
| Streamlit dashboard | Done | Sentiment chart fixed; headline row limit now user-controlled (default 200) |
| Pipeline runner | Done | `run_pipeline.py` — progress bar, per-step skip flags |
| MLflow (local) | Done | Runs without server — uses `./mlruns` file storage |

---

### Decisions Made

**MLflow without a server**
Changed `MLFLOW_TRACKING_URI` from `http://localhost:5000` to `Path("./mlruns").resolve().as_uri()`
in both `modeling/train.py` and `serving/main.py`. MLflow writes experiment data to `./mlruns/`
locally. No server process needed. Run `mlflow ui --backend-store-uri ./mlruns` to browse runs.

**pct_change as between-window return (LAG)**
Original formula used `LAST(close) - FIRST(close)` within a 5-minute bucket. With 5-minute
interval price data from yfinance, each bucket has exactly one row → always 0. Fixed by using
`LAG(avg_close) OVER (PARTITION BY ticker ORDER BY window_start)` to compute return vs the
previous window instead. This gives a real distribution of positive and negative returns.

**News deduplication at the DB level**
Added `UNIQUE INDEX ON news_raw(url)` and changed `INSERT INTO` to `INSERT OR IGNORE`.
Previously the same article was re-inserted every 30-second poll, giving 2,009 duplicates
from 2,100 rows (only 91 unique). Same fix applied to price_ticks on (ticker, ts).

**NewsAPI poll interval changed to 15 minutes**
Free tier allows 100 requests/day. 10 tickers × 96 polls/day would exceed the limit at
30-second polling. Changed `POLL_INTERVAL` to 900 seconds (15 minutes): 10 × 96 = stays under.

**Backfill uses weekly date windows**
Simple "last 30 days" backfill returned 0 new rows because the producer had already fetched
the most recent 100 articles for each ticker. Fixed by splitting the 30-day range into 4
weekly windows — each window returns different, older articles. 4 weeks × 10 tickers = 40 requests.

**LightGBM Booster vs LGBMClassifier**
`mlflow.lightgbm.log_model(final_model.booster_)` saves the raw Booster, which only has
`predict()` (returns probabilities directly), not `predict_proba()`. Fixed serving/main.py
to call `model.predict(x)[0]` instead of `model.predict_proba(x)[0, 1]`.

**Pipeline runner with skip flags**
`run_pipeline.py` runs all 7 steps with a progress bar. Each step has a `--skip-*` flag so
expensive steps (FinBERT ~5 min) can be reused across runs. Typical re-run after new data:
`python run_pipeline.py --skip-backfill --skip-sentiment --skip-embeddings --skip-graph --skip-gnn`

---

### Bugs Fixed This Session

| Bug | Root Cause | Fix |
|---|---|---|
| Streamlit sentiment chart not rendering | DuckDB connection kept open across Streamlit reruns | `query_duckdb()` helper opens, queries, and closes per call |
| FinBERT scores all NULL | `isinstance(row["titles"], list)` always False — DuckDB returns `numpy.ndarray` for LIST columns | Changed to `isinstance(raw, (list, np.ndarray))` |
| Duplicate headlines (19× per row) | News producer re-inserts same articles every 30s poll | `UNIQUE INDEX` on url + `INSERT OR IGNORE` |
| AUC = 0.0, all folds skipped | Only 91 unique articles → 48 windows → single class in every fold | News + price backfill scripts to populate historical data |
| pct_change all NULL | No price windows matched news windows (5-day price range vs 30-day news range) | Added `ingestion/backfill_prices.py` with `period="1mo"` |
| pct_change all exactly 0.0 | 5-min interval bars give 1 row per bucket → LAST = FIRST = 0 | Changed SQL to LAG-based between-window return |
| MLflow connecting to localhost:5000 | `.env` had `MLFLOW_TRACKING_URI=http://localhost:5000` overriding code default | Hardcoded `Path("./mlruns").resolve().as_uri()` in both train.py and serving/main.py |
| MLflow artifact URI scheme error | Relative path `"./mlruns"` leaves URI scheme empty | Used `Path.resolve().as_uri()` to get `file:///` absolute URI |
| `predict_proba` AttributeError | Booster saved via `final_model.booster_` has no `predict_proba` | Changed serving to call `model.predict(x)[0]` |
| Dashboard headlines capped at 50 | Hardcoded `LIMIT 50` in SQL query | Added `number_input` widget; default 200, max 5000 |
| Price backfill unique index failure | Existing price_ticks had duplicate (ticker, ts) rows | Deduped price_ticks first, then created unique index |

---

### Data Volume (end of session)

- 2,449 news rows in SQLite (91 pre-backfill + ~1,000 from weekly backfill + ongoing)
- 16,420 price rows in SQLite (1 month of 5-min bars for all 10 tickers)
- 2,132 feature windows in DuckDB
- Model v7 in Production — AUC 0.6721

---

### Saved State (persists between sessions)

| File | Contents |
|---|---|
| `./data/raw.db` | All news + price data |
| `./data/feature_store.duckdb` | Features, sentiment scores |
| `./data/headline_embeddings.npy` | FinBERT embeddings |
| `./data/graph.pt` | Co-occurrence graph |
| `./data/gnn_embeddings.parquet` | GNN node embeddings |
| `./mlruns/` | MLflow experiments + model registry |
| `~/.cache/huggingface/` | FinBERT weights (never re-downloaded) |

---

### Known Issues Remaining

**MLflow deprecation warnings**
`get_latest_versions` and `transition_model_version_stage` deprecated since MLflow 2.9.0.
Functional for now. Needs migration to the aliases API for Phase 2.

**Drift report needs more data**
`monitoring/drift_report.py` is implemented but splits data into "old" vs "new" across
a 30-day boundary. Not useful until data is collected over at least 30 days.

---

### Next Session Priorities

1. Begin Phase 2 planning — Redpanda + Spark via WSL2
2. Migrate MLflow stage API to aliases (fix deprecation warnings)
3. Run drift report once 30 days of data has accumulated
4. Consider expanding to S&P 500 tickers via config file

---

## Session 3 — 8 July 2026

### What Was Built

Phase 2 streaming infrastructure started. MLflow deprecated APIs removed. Redpanda dual-write working.

---

### Pipeline Status

| Step | Status | Notes |
|---|---|---|
| MLflow aliases migration | Done | Replaced deprecated Stages API with Aliases — model URI now `models:/fin-platform-lgbm@champion` |
| Redpanda container | Done | Running via Docker on port 29092 |
| Kafka topics created | Done | `news-raw` and `price-ticks` (1 partition, 1 replica each) |
| kafka-python installed | Done | v3.0.7 — pure Python, no compiler needed |
| News producer dual-write | Done | Publishes JSON to `news-raw` keyed by ticker — code written, not yet verified |
| Price producer dual-write | Done | Publishes JSON to `price-ticks` keyed by ticker — verified via `rpk topic consume` |

---

### Decisions Made

**MLflow aliases over Stages**
Migrated from deprecated `get_latest_versions` + `transition_model_version_stage` to
`search_model_versions` + `set_registered_model_alias`. Alias name is `champion` (conventional).
Model URI changed from `models:/fin-platform-lgbm/Production` to `models:/fin-platform-lgbm@champion`.
`scripts/set_champion.py` created to backfill alias on existing v8; v9 is now champion.

**Dual-write instead of replacing SQLite**
Both producers continue writing to SQLite so the Phase 1 pipeline works unchanged. Kafka publish
is fire-and-forget — failures are logged as warnings but never crash the producer.

**kafka-python over confluent-kafka**
Pure Python, no C compiler or build tools needed on Windows. `confluent-kafka` is faster but
requires Confluent's C library which is difficult to install on Windows without WSL2.

**Producers keyed by ticker**
`producer.send(topic, key=ticker.encode(), value=payload)` guarantees all messages for the same
ticker land on the same partition. This gives the Spark consumer an ordering guarantee per ticker.

**Phase 2 infrastructure runs in Docker, not WSL2**
Original plan noted WSL2 for Spark. Decided to run all Phase 2 infrastructure (Redpanda,
Spark, Airflow, Prometheus, Grafana) as Docker containers instead. Simpler setup, no OS changes.

---

### Bugs Fixed This Session

| Bug | Root Cause | Fix |
|---|---|---|
| `AttributeError: 'ModelInfo' has no attribute 'registered_model_version'` | MLflow version installed doesn't expose that attribute | Use `search_model_versions` and pick max version by number |
| `sqlite3.IntegrityError: UNIQUE constraint failed: price_ticks.ticker, price_ticks.ts` | `insert_tick` used plain `INSERT` — crashes when same candle re-polled within same 1-minute bar | Changed to `INSERT OR IGNORE` |

---

### Data Volume (end of session)

- 2,449+ news rows, 16,420+ price rows in SQLite
- Model v9 champion — AUC 0.6721
- `price-ticks` Redpanda topic: 10 messages (1 per ticker, first poll)

---

### Next Session Priorities

1. Verify news producer (`rpk topic consume news-raw --num 5 --offset oldest`)
2. Bring up Prometheus + Grafana: `docker compose --profile monitoring up -d`
3. Add Spark to `docker-compose.yml` and submit `streaming/spark_consumer.py`
4. Add Airflow to `docker-compose.yml`, init DB, mount `dags/`

---

## Session 1 — 29 June 2026

### What Was Built

Complete Phase 1 pipeline from scratch. Every file was created this session.

---

### Pipeline Status (Session 1)

| Step | Status | Notes |
|---|---|---|
| Virtual environment | Done | Python 3.11.5, venv at `./venv/` |
| Dependencies installed | Done | `pip install -r requirements.txt` |
| `.env` configured | Done | `NEWS_API_KEY` filled in |
| News producer | Working | Confirmed writing to SQLite |
| Price producer | Working | Fixed yfinance `period="5d"` for closed market hours |
| Feature export | Working | 48 rows exported from 950 headlines |
| FinBERT sentiment | Working | 48 sentiment rows written to DuckDB |
| FinBERT embeddings | Working | 10×768 matrix saved to `./data/headline_embeddings.npy` |
| Graph build | Working | 10 nodes, 3 edges — sparse but functional |
| GNN training | Working | Loss 0.002 → 0.000005 over 100 epochs, 10×64 embeddings saved |
| LightGBM training | Working | Model v1 registered in MLflow, promoted to Production |
| FastAPI | Working | `/health` returns 200 |
| Streamlit dashboard | Partial | Tables visible, price chart broken, sentiment chart not rendering |

---

### Decisions Made (Session 1)

**Phase 1 vs Phase 2 architecture**
Chose to build a simplified local pipeline first (Phase 1) before adding production
infrastructure (Redpanda, Spark, Prometheus/Grafana). This lets the ML pipeline get
validated before spending time on infrastructure.

**SQLite instead of Redpanda**
Producers write directly to SQLite. DuckDB reads SQLite via the sqlite extension.
This replaces the Redpanda → Spark → Parquet flow entirely for local development.

**DuckDB does the windowing**
The 5-minute tumbling windows that Spark handled in Phase 2 are now done by DuckDB SQL
using `time_bucket(INTERVAL '5 minutes', ...)`. No Spark dependency needed.

**Streamlit instead of Prometheus + Grafana**
Pure Python dashboard. No Docker needed. Swapped back to Prometheus + Grafana in Phase 2.

**Company name keywords for graph co-occurrence**
Headlines don't use ticker symbols — they use company names. The original code searched
for "AAPL" in headlines and found nothing. Fixed with a keyword mapping (e.g. AAPL →
["apple", "aapl"], TSLA → ["tesla", "elon musk"]).

---

### Bugs Fixed (Session 1)

| Bug | Root Cause | Fix |
|---|---|---|
| yfinance `JSONDecodeError` | US market closed (12:55 PM IST = pre-market) | Changed `period="1d"` to `period="5d"` |
| `pyarrow==16.0.0` conflict | mlflow 2.13.0 requires `pyarrow<16` | Downgraded to `pyarrow==15.0.2` |
| `ModuleNotFoundError: prometheus_client` | Moved to Phase 2 requirements by mistake | Moved back to Phase 1 requirements |
| Graph had 0 edges | Code searched for ticker symbols ("AAPL") in headlines | Added company name keyword mapping |
| `FileNotFoundError: embedding_meta.json` | `nlp/embeddings.py` was skipped | Must run embeddings.py before build_graph.py |
| LightGBM `Only one class in y_true` | 48 rows too few for 5-fold TimeSeriesSplit | Reduced to 2 folds, added skip logic for single-class folds |
| Streamlit sentiment chart missing | Single try/except block — one failure silences everything after it | Split into per-section try/except blocks |
| Streamlit price chart NaN warning | yfinance returns NaN closes when market is closed | Added `.dropna(subset=["close"])` before plotting |
