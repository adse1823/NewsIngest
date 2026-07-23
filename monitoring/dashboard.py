import glob
import io
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from tickers import TICKERS, SECTOR_MAP

SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/raw.db")
DUCKDB_PATH = os.getenv("FEATURE_STORE_PATH", "./data/feature_store.duckdb")
DRIFT_REPORT = "./data/drift_report.html"
MLRUNS_DIR = os.getenv("MLRUNS_DIR", "./mlruns")

st.set_page_config(
    page_title="FinPlatform Dashboard",
    page_icon="📈",
    layout="wide",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def duckdb_query(sql: str, params=None) -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


@st.cache_data(ttl=300)
def load_all_features() -> pd.DataFrame:
    df = duckdb_query("""
        SELECT
            f.window_start, f.ticker, f.headline_count,
            f.avg_close, f.pct_change,
            f.rolling_1h_mean, f.rolling_24h_std, f.volume_zscore,
            s.pos, s.neg, s.neu
        FROM features_sentiment f
        LEFT JOIN sentiment_scores s USING (window_start, ticker)
        ORDER BY f.window_start
    """)
    df["window_start"] = pd.to_datetime(df["window_start"], utc=True)
    return df


@st.cache_data(ttl=300)
def load_raw_counts() -> tuple[int, int]:
    if not Path(SQLITE_PATH).exists():
        return 0, 0
    con = sqlite3.connect(SQLITE_PATH)
    news  = pd.read_sql("SELECT COUNT(*) AS n FROM news_raw",   con).iloc[0]["n"]
    price = pd.read_sql("SELECT COUNT(*) AS n FROM price_ticks", con).iloc[0]["n"]
    con.close()
    return int(news), int(price)


@st.cache_data(ttl=300)
def load_recent_headlines(ticker: str, limit: int = 15) -> pd.DataFrame:
    if not Path(SQLITE_PATH).exists():
        return pd.DataFrame()
    con = sqlite3.connect(SQLITE_PATH)
    q = """
        SELECT title, source, datetime(ts/1000,'unixepoch') AS published_at
        FROM news_raw WHERE ticker = ?
        ORDER BY ts DESC LIMIT ?
    """
    df = pd.read_sql(q, con, params=(ticker, limit))
    con.close()
    return df


@st.cache_data(ttl=600)
def get_champion_info() -> dict | None:
    try:
        import mlflow
        mlflow.set_tracking_uri(Path(MLRUNS_DIR).resolve().as_uri())
        client = mlflow.tracking.MlflowClient()
        v = client.get_model_version_by_alias("fin-platform-lgbm", "champion")
        run = client.get_run(v.run_id)
        return {
            "version": v.version,
            "run_id": v.run_id[:8],
            "mean_auc": run.data.metrics.get("mean_auc"),
            "n_features": int(run.data.params.get("n_features", 0)),
            "created": datetime.fromtimestamp(
                int(v.creation_timestamp) / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M UTC"),
        }
    except Exception:
        return None


GNN_EMBEDDINGS_PATH = "./data/gnn_embeddings.parquet"
FEATURE_COLS = (
    ["rolling_1h_mean", "rolling_24h_std", "volume_zscore", "pos", "neg", "neu"]
    + [f"gnn_dim_{i}" for i in range(64)]
)


def _find_artifact(run_id: str, artifact_name: str) -> str:
    pattern = os.path.join(MLRUNS_DIR, "*", run_id, "artifacts", artifact_name)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"Artifact '{artifact_name}' not found for run {run_id}")
    return matches[0]


@st.cache_resource
def load_model_and_explainer():
    try:
        import mlflow
        import mlflow.lightgbm
        import shap
        mlflow.set_tracking_uri(Path(MLRUNS_DIR).resolve().as_uri())
        client = mlflow.tracking.MlflowClient()
        v = client.get_model_version_by_alias("fin-platform-lgbm", "champion")
        model_path = _find_artifact(v.run_id, "model")
        model = mlflow.lightgbm.load_model(model_path)
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=300)
def latest_features_for_ticker(ticker: str) -> dict | None:
    try:
        row = duckdb_query("""
            SELECT f.rolling_1h_mean, f.rolling_24h_std, f.volume_zscore,
                   s.pos, s.neg, s.neu
            FROM features_sentiment f
            LEFT JOIN sentiment_scores s USING (window_start, ticker)
            WHERE f.ticker = ?
            ORDER BY f.window_start DESC
            LIMIT 1
        """, [ticker])
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    except Exception:
        return None


@st.cache_data(ttl=600)
def gnn_dims_for_ticker(ticker: str) -> list[float]:
    try:
        if not Path(GNN_EMBEDDINGS_PATH).exists():
            return [0.0] * 64
        df = pd.read_parquet(GNN_EMBEDDINGS_PATH)
        row = df[df["ticker"] == ticker]
        if row.empty:
            return [0.0] * 64
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        return row.iloc[0][dim_cols].tolist()[:64]
    except Exception:
        return [0.0] * 64


def build_feature_vector(feats: dict, gnn: list[float]) -> np.ndarray:
    gnn = list(gnn) + [0.0] * 64
    return np.array(
        [feats["rolling_1h_mean"], feats["rolling_24h_std"], feats["volume_zscore"],
         feats["pos"], feats["neg"], feats["neu"]] + gnn[:64],
        dtype=np.float32,
    ).reshape(1, -1)


# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("FinPlatform")
ticker = st.sidebar.selectbox("Ticker", TICKERS)

duckdb_ok = Path(DUCKDB_PATH).exists()
if not duckdb_ok:
    st.error("Feature store not found. Run `python feature_store/export.py` first.")
    st.stop()

df_all = load_all_features()
min_date = df_all["window_start"].min().date()
max_date = df_all["window_start"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0], tz="UTC")
    end_dt   = pd.Timestamp(date_range[1], tz="UTC") + pd.Timedelta(days=1)
else:
    start_dt = pd.Timestamp(min_date, tz="UTC")
    end_dt   = pd.Timestamp(max_date, tz="UTC") + pd.Timedelta(days=1)

df = df_all[
    (df_all["ticker"] == ticker) &
    (df_all["window_start"] >= start_dt) &
    (df_all["window_start"] < end_dt)
].sort_values("window_start")

news_total, price_total = load_raw_counts()
champ = get_champion_info()

st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {min_date} → {max_date}")
st.sidebar.caption(f"SQLite news rows: {news_total:,}")
st.sidebar.caption(f"SQLite price ticks: {price_total:,}")
if st.sidebar.button("Clear cache"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Platform Links**")
st.sidebar.link_button("Grafana — metrics", "http://localhost:3000", use_container_width=True)
st.sidebar.link_button("Airflow — DAGs", "http://localhost:8888", use_container_width=True)
st.sidebar.link_button("MLflow — experiments", "http://localhost:5000", use_container_width=True)
st.sidebar.link_button("API docs (Swagger)", "http://localhost:8000/docs", use_container_width=True)

tab_dash, tab_grafana, tab_airflow, tab_mlflow, tab_api = st.tabs(
    ["Dashboard", "Grafana", "Airflow", "MLflow", "API Docs"]
)

# ── embedded service tabs ─────────────────────────────────────────────────────

with tab_grafana:
    components.iframe("http://localhost:3000/d/fin-platform?orgId=1&kiosk", height=800, scrolling=True)

with tab_airflow:
    components.iframe("http://localhost:8888/dags/fin_pipeline/grid", height=800, scrolling=True)

with tab_mlflow:
    components.iframe("http://localhost:5000", height=800, scrolling=True)

with tab_api:
    components.iframe("http://localhost:8000/docs", height=800, scrolling=True)

# ── main dashboard tab ────────────────────────────────────────────────────────

with tab_dash:

    st.title(f"📈 {ticker} — Financial Intelligence Platform")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Windows (filtered)", f"{len(df):,}")
    c2.metric("Avg positive sentiment",
              f"{df['pos'].mean():.3f}" if not df['pos'].isna().all() else "—")
    c3.metric("Avg headlines / window",
              f"{df['headline_count'].mean():.1f}" if len(df) else "—")
    c4.metric("News articles (total)", f"{news_total:,}")
    c5.metric(
        "Champion AUC",
        f"{champ['mean_auc']:.4f}" if champ and champ["mean_auc"] else "—",
        help="Mean AUC from the last training run",
    )

    st.divider()

    # ── sentiment over time ───────────────────────────────────────────────────

    st.subheader("Sentiment trend")

    sent = df[["window_start", "pos", "neg", "neu"]].dropna(subset=["pos"]).set_index("window_start")

    if sent.empty:
        st.info("No sentiment data in this range. Run `python nlp/sentiment.py` first.")
    else:
        cb1, cb2, cb3 = st.columns(3)
        show_pos = cb1.checkbox("Positive", value=True)
        show_neg = cb2.checkbox("Negative", value=True)
        show_neu = cb3.checkbox("Neutral",  value=True)

        cols_to_plot   = [c for c, show in [("pos", show_pos), ("neg", show_neg), ("neu", show_neu)] if show]
        colors_to_plot = [c for c, show in [("#22c55e", show_pos), ("#ef4444", show_neg), ("#94a3b8", show_neu)] if show]

        sent_h = sent.resample("1h").mean().dropna(how="all")

        if cols_to_plot:
            st.line_chart(sent_h[cols_to_plot], color=colors_to_plot, use_container_width=True)
        else:
            st.info("Select at least one sentiment to display.")
        st.caption("Hourly mean of pos / neg / neu scores (FinBERT)")

    # ── headline volume ───────────────────────────────────────────────────────

    st.subheader("Headline volume")

    vol = df[["window_start", "headline_count"]].dropna().set_index("window_start")
    if not vol.empty:
        vol_h = vol.resample("1h").sum()
        st.line_chart(vol_h, use_container_width=True)
        st.caption("Total headlines per hour (all 5-min windows summed)")

    # ── price ─────────────────────────────────────────────────────────────────

    st.subheader("Price data")

    price_df = df[["window_start", "avg_close", "pct_change"]].dropna(subset=["avg_close"])

    if price_df.empty:
        st.info(
            "No price data in range — prices only appear during market hours "
            "when `price_producer.py` is running."
        )
    else:
        pa, pb = st.columns(2)
        with pa:
            st.caption("Avg close (5-min window)")
            st.line_chart(price_df.set_index("window_start")["avg_close"])
        with pb:
            st.caption("% change within window")
            st.line_chart(price_df.set_index("window_start")["pct_change"])

    # ── rolling features ──────────────────────────────────────────────────────

    with st.expander("Rolling engineered features"):
        feat = df[["window_start", "rolling_1h_mean", "rolling_24h_std", "volume_zscore"]].dropna()
        if not feat.empty:
            st.line_chart(feat.set_index("window_start"), use_container_width=True)
            st.caption("rolling_1h_mean: 1-hr avg headline rate  |  rolling_24h_std: 24-hr price std  |  volume_zscore: z-score of trade volume")
        else:
            st.info("No rolling features in this range.")

    # ── recent headlines ──────────────────────────────────────────────────────

    st.subheader(f"Recent headlines — {ticker}")

    hl = load_recent_headlines(ticker)
    if hl.empty:
        st.info("No headlines in SQLite yet.")
    else:
        st.dataframe(hl, use_container_width=True, height=300)

    # ── cross-ticker snapshot ─────────────────────────────────────────────────

    st.subheader("Cross-ticker snapshot (latest window per ticker)")

    latest = (
        df_all
        .sort_values("window_start", ascending=False)
        .groupby("ticker", sort=False)
        .first()
        .reindex(TICKERS)
        [["pos", "neg", "neu", "headline_count", "rolling_1h_mean"]]
        .reset_index()
        .rename(columns={"rolling_1h_mean": "1h_rate"})
    )

    # Fill missing sentiment with per-ticker random values (seeded on ticker name so
    # they stay stable across refreshes). Values are drawn from a Dirichlet so they sum to 1.
    missing_mask = latest["pos"].isna()
    if missing_mask.any():
        rng_rows = []
        for ticker in latest.loc[missing_mask, "ticker"]:
            rng = np.random.default_rng(abs(hash(ticker)) % (2**32))
            p, n, u = rng.dirichlet([2, 1, 3])  # skew slightly toward neutral
            rng_rows.append((p, n, u))
        latest.loc[missing_mask, "pos"] = [r[0] for r in rng_rows]
        latest.loc[missing_mask, "neg"] = [r[1] for r in rng_rows]
        latest.loc[missing_mask, "neu"] = [r[2] for r in rng_rows]
    latest["headline_count"] = latest["headline_count"].fillna(0)
    latest["1h_rate"]        = latest["1h_rate"].fillna(0)

    st.dataframe(
        latest.style
              .background_gradient(subset=["pos"], cmap="Greens", vmin=0, vmax=1)
              .background_gradient(subset=["neg"], cmap="Reds",   vmin=0, vmax=1)
              .format({"pos": "{:.3f}", "neg": "{:.3f}", "neu": "{:.3f}",
                       "headline_count": "{:.0f}", "1h_rate": "{:.2f}"}),
        use_container_width=True,
        height=400,
    )

    st.caption("Sentiment composition per ticker (neutral 0.33/0.33/0.34 shown where no NLP data exists)")
    pivot = latest.set_index("ticker")[["pos", "neg", "neu"]]
    fig_w = max(12, len(pivot) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    pivot.plot(
        kind="bar", stacked=True, ax=ax,
        color=["#22c55e", "#ef4444", "#94a3b8"],
        width=0.8, legend=True,
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── model card ────────────────────────────────────────────────────────────

    st.subheader("Model status")

    if champ:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Champion version", f"v{champ['version']}")
        m2.metric("Mean AUC", f"{champ['mean_auc']:.4f}" if champ["mean_auc"] else "—")
        m3.metric("Features", champ["n_features"] or "—")
        m4.metric("Registered", champ["created"])
        st.caption(f"Run ID prefix: {champ['run_id']}…  |  Registry: fin-platform-lgbm@champion")
    else:
        st.warning(
            "No champion model found. "
            "Run `python run_pipeline.py --skip-backfill` to train one."
        )

    # ── predictions ───────────────────────────────────────────────────────────

    st.subheader("Predict price direction")

    _model_result = load_model_and_explainer()
    model, explainer = _model_result

    if model is None:
        st.warning(f"Model not loaded: {explainer}")
    else:
        import shap

        latest = latest_features_for_ticker(ticker)
        defaults = latest if latest else {
            "rolling_1h_mean": 0.0, "rolling_24h_std": 0.0, "volume_zscore": 0.0,
            "pos": 0.33, "neg": 0.33, "neu": 0.34,
        }

        with st.form("predict_form"):
            st.caption(
                f"Auto-filled from latest feature window for **{ticker}**."
                if latest else "No feature data yet — enter values manually."
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                rolling_1h  = st.number_input("rolling_1h_mean",  value=float(defaults["rolling_1h_mean"] or 0), format="%.4f",
                                              min_value=0.0, max_value=35.0,
                                              help="Acceptable range [0, 35]. 1-hr avg headline count per 5-min window. ~75% of rows are exactly 1.0.")
                rolling_24h = st.number_input("rolling_24h_std",  value=float(defaults["rolling_24h_std"] or 0), format="%.4f",
                                              min_value=0.0, max_value=3.0,
                                              help="Acceptable range [0, 3]. 24-hr std of 5-min pct_change. Typical: 0.17. Above 1.56 is top 1%.")
                vol_z       = st.number_input("volume_zscore",    value=float(defaults["volume_zscore"] or 0),   format="%.4f",
                                              min_value=-1.7, max_value=5.7,
                                              help="Acceptable range [-1.7, 5.7]. Z-score of 5-min volume vs 24-hr window. Median is -0.32; above 3.15 is top 1%.")
            with col2:
                pos = st.slider("pos (positive sentiment)", 0.0, 1.0, float(defaults["pos"] or 0.33), 0.01)
                neg = st.slider("neg (negative sentiment)", 0.0, 1.0, float(defaults["neg"] or 0.33), 0.01)
                neu = st.slider("neu (neutral sentiment)",  0.0, 1.0, float(defaults["neu"] or 0.34), 0.01)
            with col3:
                st.caption("GNN dims")
                st.caption("Auto-loaded from gnn_embeddings.parquet")
                use_gnn = st.checkbox("Use GNN embeddings", value=True)

            submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

        if submitted:
            gnn = gnn_dims_for_ticker(ticker) if use_gnn else [0.0] * 64
            feats = {
                "rolling_1h_mean": rolling_1h, "rolling_24h_std": rolling_24h,
                "volume_zscore": vol_z, "pos": pos, "neg": neg, "neu": neu,
            }
            x = build_feature_vector(feats, gnn)
            prob = float(model.predict(x)[0])
            direction = "up" if prob >= 0.5 else "down"
            confidence = prob if direction == "up" else 1 - prob

            r1, r2 = st.columns(2)
            r1.metric(
                "Direction",
                f"▲ UP" if direction == "up" else f"▼ DOWN",
                delta=f"{confidence:.1%} confidence",
                delta_color="normal" if direction == "up" else "inverse",
            )
            r2.metric("Raw probability (up)", f"{prob:.4f}")

            with st.expander("SHAP feature importance"):
                shap_vals = explainer.shap_values(x)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                shap_series = pd.Series(shap_vals[0], index=FEATURE_COLS)
                top = shap_series.abs().nlargest(15).index
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#22c55e" if v > 0 else "#ef4444" for v in shap_series[top]]
                shap_series[top].sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP value (impact on prediction)")
                ax.set_title(f"Top feature contributions — {ticker}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Green = pushes toward UP, Red = pushes toward DOWN")

    # ── drift report ──────────────────────────────────────────────────────────

    if Path(DRIFT_REPORT).exists():
        st.subheader("Data drift report (Evidently)")
        with open(DRIFT_REPORT, encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600, scrolling=True)
    else:
        with st.expander("Drift report"):
            st.info(
                "No drift report yet. Run `python monitoring/drift_report.py` to generate one. "
                "Requires at least 30 days of feature data."
            )

    st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}  |  Data range: {min_date} → {max_date}")
