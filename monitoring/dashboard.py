import os
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/raw.db")
DUCKDB_PATH = os.getenv("FEATURE_STORE_PATH", "./data/feature_store.duckdb")
DRIFT_REPORT = "./data/drift_report.html"
MLRUNS_DIR = os.getenv("MLRUNS_DIR", "./mlruns")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]

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


# ── header metrics ────────────────────────────────────────────────────────────

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


# ── sentiment over time ───────────────────────────────────────────────────────

st.subheader("Sentiment trend")

sent = df[["window_start", "pos", "neg", "neu"]].dropna(subset=["pos"]).set_index("window_start")

if sent.empty:
    st.info("No sentiment data in this range. Run `python nlp/sentiment.py` first.")
else:
    # Resample to hourly so the chart isn't too dense
    sent_h = sent.resample("1h").mean().dropna(how="all")
    st.line_chart(sent_h, color=["#22c55e", "#ef4444", "#94a3b8"],
                  use_container_width=True)
    st.caption("Hourly mean of pos / neg / neu scores (FinBERT)")


# ── headline volume ────────────────────────────────────────────────────────────

st.subheader("Headline volume")

vol = df[["window_start", "headline_count"]].dropna().set_index("window_start")
if not vol.empty:
    vol_h = vol.resample("1h").sum()
    st.bar_chart(vol_h, use_container_width=True)
    st.caption("Total headlines per hour (all 5-min windows summed)")


# ── price ─────────────────────────────────────────────────────────────────────

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


# ── rolling features ──────────────────────────────────────────────────────────

with st.expander("Rolling engineered features"):
    feat = df[["window_start", "rolling_1h_mean", "rolling_24h_std", "volume_zscore"]].dropna()
    if not feat.empty:
        st.line_chart(feat.set_index("window_start"), use_container_width=True)
        st.caption("rolling_1h_mean: 1-hr avg headline rate  |  rolling_24h_std: 24-hr price std  |  volume_zscore: z-score of trade volume")
    else:
        st.info("No rolling features in this range.")


# ── recent headlines ──────────────────────────────────────────────────────────

st.subheader(f"Recent headlines — {ticker}")

hl = load_recent_headlines(ticker)
if hl.empty:
    st.info("No headlines in SQLite yet.")
else:
    st.dataframe(hl, use_container_width=True, height=300)


# ── cross-ticker snapshot ─────────────────────────────────────────────────────

st.subheader("Cross-ticker snapshot (latest window per ticker)")

latest = (
    df_all[df_all["pos"].notna()]
    .sort_values("window_start", ascending=False)
    .groupby("ticker", sort=False)
    .first()
    .reindex(TICKERS)
    [["pos", "neg", "neu", "headline_count", "rolling_1h_mean"]]
    .reset_index()
    .rename(columns={"rolling_1h_mean": "1h_rate"})
)

if not latest["pos"].isna().all():
    st.dataframe(
        latest.style
              .background_gradient(subset=["pos"], cmap="Greens", vmin=0, vmax=1)
              .background_gradient(subset=["neg"], cmap="Reds",   vmin=0, vmax=1)
              .format({"pos": "{:.3f}", "neg": "{:.3f}", "neu": "{:.3f}",
                       "headline_count": "{:.0f}", "1h_rate": "{:.2f}"}),
        use_container_width=True,
    )

    # Stacked bar across all tickers
    st.caption("Sentiment composition per ticker")
    pivot = latest.set_index("ticker")[["pos", "neg", "neu"]].dropna()
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        pivot.plot(
            kind="bar", stacked=True, ax=ax,
            color=["#22c55e", "#ef4444", "#94a3b8"],
            width=0.6, legend=True,
        )
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("Run `python nlp/sentiment.py` to populate sentiment data.")


# ── model card ────────────────────────────────────────────────────────────────

st.subheader("Model status")

if champ:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Champion version", f"v{champ['version']}")
    m2.metric("Mean AUC", f"{champ['mean_auc']:.4f}" if champ["mean_auc"] else "—")
    m3.metric("Features", champ["n_features"] or "—")
    m4.metric("Registered", champ["created"])
    st.caption(f"Run ID prefix: {champ['run_id']}…  |  Registry: fin-platform-lgbm@champion")
    st.caption("Predict endpoint: `POST http://localhost:8000/predict`")
else:
    st.warning(
        "No champion model found. "
        "Run `python run_pipeline.py --skip-backfill` to train one."
    )


# ── drift report ──────────────────────────────────────────────────────────────

if Path(DRIFT_REPORT).exists():
    st.subheader("Data drift report (Evidently)")
    with open(DRIFT_REPORT) as f:
        html = f.read()
    st.components.v1.html(html, height=600, scrolling=True)
else:
    with st.expander("Drift report"):
        st.info(
            "No drift report yet. Run `python monitoring/drift_report.py` to generate one. "
            "Requires at least 30 days of feature data."
        )

st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}  |  Data range: {min_date} → {max_date}")
