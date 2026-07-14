# News Ingest Progress Log

---

### 2026-07-13 — Session: Spark merge, Grafana provisioning, Avro registration

**What was built / changed**

#### Feature export — Spark streaming integration
- `feature_store/export.py`: Full rewrite. Now builds `raw_news` and `price_windows_merged` tables directly in Python before running SQL models. If Spark-produced parquet files exist in `./data/windowed/news/` or `./data/windowed/prices/`, rows are merged via UNION (SQLite rows take priority; Spark rows fill in any windows not already covered). Falls back to SQLite-only when Spark hasn't run. The `_has_parquet()` helper checks for actual `.parquet` files so there's no crash on an empty/missing dir.
- `feature_store/models/features_sentiment.sql`: Removed the `price_windows_raw` CTE that read directly from `sqlite_db.price_ticks` — now reads from `price_windows_merged` (pre-created by export.py). This decouples the SQL from the data source.
- `feature_store/models/raw_news.sql`: No longer invoked (replaced by inline Python in export.py). File kept for reference.
- Pipeline order is now: build merged sources → `features_sentiment.sql` → `entity_table.sql` → parquet export.

#### Grafana — auto-provisioning
- `monitoring/grafana/provisioning/datasources/prometheus.yml`: Datasource config — wires Prometheus at `http://prometheus:9090` with uid `prometheus`.
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`: Dashboard provider config — points Grafana at `/var/lib/grafana/dashboards` and polls for changes every 30s.
- `monitoring/grafana/dashboard.json`: Added `datasource` field to all 4 panels (required by Grafana 10.x provisioning).
- `docker-compose.yml`: Added two volume mounts to grafana service — provisioning dir and dashboard JSON. Dashboard now loads automatically on `docker compose --profile monitoring up`.

#### Avro schema registration
- `scripts/register_schemas.py`: New script. POSTs `schemas/news_event_v1.avsc` → `news-raw-value` and `schemas/price_tick_v1.avsc` → `price-ticks-value` to the Redpanda Schema Registry. Skips subjects that already exist (use `--force` to re-register). Exits non-zero on any failure. Run once after `docker compose up redpanda` before starting producers.

**Bugs avoided / design decisions**
- Spark windowed parquet uses `window_start` as a datetime column; DuckDB's `read_parquet()` handles this natively.
- UNION deduplication uses `WHERE (ticker, window_start) NOT IN (SELECT ...)` — correct for 5-min buckets since each (ticker, window) is unique per source.
- Grafana 10.x requires `datasource` object (not string) in panel targets — fixed in dashboard.json.

**Suggested next steps**
1. Switch producers from JSON to Avro serialization (use `fastavro` + Confluent magic bytes prefix, update `spark_consumer.py` to deserialize Avro instead of `from_json`)
2. Add `spark_consumer` as a Docker Compose service under the `spark` profile so it starts with `docker compose --profile spark up`
3. End-to-end smoke test: run producers → Spark consumer → export.py → confirm Spark rows appear in `features_export.parquet`

---

### 2026-07-12 — Session: Pipeline hardening + full test suite

**What was built / changed**

#### Phase 1 — Ingestion reliability
- `ingestion/news_producer.py`: Fixed KafkaProducer crash on startup — wrapped init in try/except, producer falls back to `None` (SQLite-only mode). Added `if producer is not None:` guard before all sends. Moved `producer.flush()` out of the per-article loop to after the ticker loop.
- `ingestion/price_producer.py`: Same crash-safe pattern applied.

#### Phase 2 — Docker Compose services
- Added `spark-master`, `spark-worker` (apache/spark:3.5.0, profile `spark`)
- Added `airflow-init`, `airflow-webserver`, `airflow-scheduler` (apache/airflow:2.9.1, profile `airflow`, SequentialExecutor + SQLite backend)
- Added `serving-api` (profile `serving`, port 8000, mounts ./mlruns and ./data)
- Named volumes: `grafana-storage`, `spark-ivy`, `airflow-data`

#### Phase 3 — Serving API
- `serving/main.py`: Rewrote model loading to use `_find_artifact()` — glob-scans MLRUNS_DIR by run_id instead of relying on stored absolute paths (which break in Docker).
- `serving/Dockerfile`: Added `libgomp1` (required by LightGBM on python:3.11-slim).
- Fixed tracking URI: `Path(MLRUNS_DIR).resolve().as_uri()` (Windows requires this — `f"file://{path}"` is invalid).
- `modeling/evaluate.py`: Fixed alias from `/Production` → `@champion`.

#### Phase 4 — Airflow DAG
- `dags/fin_pipeline.py`: All 7 BashOperator commands changed from `cd /opt/airflow` → `cd /opt/airflow/project` to match the volume mount path.

#### Phase 5 — Monitoring dashboard
- `monitoring/dashboard.py`: Full rewrite. DuckDB feature join, MLflow champion info, sentiment trend with pos/neg/neu checkboxes, headline volume line chart, Evidently drift report via `streamlit.components.v1.html`.
- `monitoring/drift_report.py`: Fixed reference window (30→14 days), fixed subprocess Python path (`sys.executable`).
- Fixed UnicodeDecodeError on drift HTML: opened with `encoding="utf-8"`.
- Fixed `AttributeError: module 'streamlit.components' has no attribute 'v1'`: added explicit `import streamlit.components.v1 as components`.

#### Phase 6 — Test suite (32 tests, all passing)

**Serving API — `tests/serving/test_api.py` (14 tests)**
- Root cause of `ModuleNotFoundError: No module named 'serving.main'` fixed: load module via `importlib.util.spec_from_file_location`, bypassing pytest's package resolution.
- State isolation: `sm._recent_X.clear()` called at start of each `_make_client()` to prevent bleed between tests.
- Coverage: `/health`, `/predict` happy path, direction boundary (prob=0.5 → "up"), up/down classification, `/explain` SHAP values, `/shap-summary` 404 before prediction, `/shap-summary` after prediction (image in response), missing required field (422), `pos`/`neg`/`neu` out of range (422), `gnn_dims` auto-padding, `/metrics` Prometheus text.

**News producer — `tests/ingestion/test_news_producer.py` (11 tests)**
- `init_db`: table + all 3 indexes present.
- `insert_articles`: writes rows, deduplicates by URL (UNIQUE INDEX), sends to Kafka (call count + key bytes), producer=None guard, Kafka failure resilience (DB still written), missing `publishedAt` fallback, invalid `publishedAt` fallback to now, title truncated at 500 chars, null-URL articles both inserted (partial index `WHERE url IS NOT NULL` tested).
- `fetch_articles`: returns `[]` on API error.

**Price producer — `tests/ingestion/test_price_producer.py` (7 tests)**
- `init_db`: table + 2 indexes present.
- `insert_tick`: writes correct field values, sends to Kafka with ticker as key bytes, producer=None guard, Kafka failure resilience.
- `fetch_tick`: returns `None` on exception, returns `None` on empty DataFrame (market-closed branch).

**Bugs hit and resolved**
- bitnami/spark:3.5.1 not found → switched to apache/spark:3.5.0
- Airflow LocalExecutor incompatible with SQLite → switched to SequentialExecutor
- Ivy cache permission error in spark-submit → added `--conf spark.jars.ivy=/tmp/.ivy2`
- MLflow artifact paths stored as Windows absolute paths → fixed with filesystem glob in `_find_artifact()`
- `libgomp.so.1` missing → added `apt-get install libgomp1` to Dockerfile
- `venv\Scripts\streamlit run` fails (stale venv path) → must use `venv\Scripts\python.exe -m streamlit run`
- REFERENCE_DAYS=30 gave empty reference set (only 27 days of data) → changed to 14

**Suggested next steps**
1. Wire Spark streaming parquet output into the feature export pipeline
2. Add Grafana dashboards for serving API metrics (using `fin_predict_requests_total`, `fin_predict_latency_seconds`)
3. Register Avro schemas, switch producers from JSON to Avro
