# Progress Log

## Session 1 — 29 June 2026

### What Was Built

Complete Phase 1 pipeline from scratch. Every file was created this session.

---

### Pipeline Status

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

### Decisions Made

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

### Bugs Fixed This Session

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

---

### Known Issues Remaining

**Streamlit sentiment chart still not rendering**
- Data confirmed present: `SELECT COUNT(*) FROM sentiment_scores` returns 48
- Tables visible in dashboard: entity_table, features_sentiment, raw_news, sentiment_scores
- Price chart NaN issue patched but not yet confirmed fixed
- To investigate next session

**Model CV is meaningless**
- Only 48 feature rows available — both folds skipped (single class in validation set)
- Mean AUC logged as 0.0
- Not a bug — will resolve itself once data accumulates over days

**Graph is very sparse**
- Only 3 edges across 10 nodes after 1 hour of data collection
- Headlines rarely mention two tracked companies by name in one title
- Will improve as more data accumulates

**MLflow deprecation warnings**
- `get_latest_versions` and `transition_model_version_stage` deprecated since MLflow 2.9.0
- Functional for now, needs migration to aliases API eventually

---

### Next Session Priorities

1. Fix Streamlit sentiment chart rendering
2. Let producers run overnight to accumulate meaningful data volume
3. Retrain model with full data — confirm AUC above 0.50 baseline
4. Begin planning long-term trend model (next week return prediction)
5. Investigate historical backfill options (GDELT / AlphaVantage News)
