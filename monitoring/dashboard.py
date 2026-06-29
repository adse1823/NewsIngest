import os
import sqlite3
import duckdb
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

SQLITE_PATH = "./data/raw.db"
DUCKDB_PATH = "./data/feature_store.duckdb"

st.set_page_config(page_title="Financial Intelligence Platform", layout="wide")
st.title("Financial Intelligence Platform — Dashboard")

if not os.path.exists(SQLITE_PATH):
    st.error("No data yet. Run the producers first: `python ingestion/news_producer.py`")
    st.stop()


# ── Raw data stats ────────────────────────────────────────────────────────────
con_sq = sqlite3.connect(SQLITE_PATH)

news_count  = pd.read_sql("SELECT COUNT(*) AS n FROM news_raw",  con_sq).iloc[0]["n"]
price_count = pd.read_sql("SELECT COUNT(*) AS n FROM price_ticks", con_sq).iloc[0]["n"]

col1, col2 = st.columns(2)
col1.metric("Total news articles ingested", f"{news_count:,}")
col2.metric("Total price ticks ingested",   f"{price_count:,}")


# ── Recent headlines per ticker ───────────────────────────────────────────────
st.subheader("Recent Headlines")

recent_news = pd.read_sql("""
    SELECT ticker, title, source,
           datetime(ts / 1000, 'unixepoch') AS published_at
    FROM news_raw
    ORDER BY ts DESC
    LIMIT 50
""", con_sq)

ticker_filter = st.selectbox("Filter by ticker", ["All"] + sorted(recent_news["ticker"].unique().tolist()))
if ticker_filter != "All":
    recent_news = recent_news[recent_news["ticker"] == ticker_filter]

st.dataframe(recent_news, use_container_width=True)


# ── Price chart ───────────────────────────────────────────────────────────────
st.subheader("Price History")

price_data = pd.read_sql("""
    SELECT ticker, close,
           datetime(ts / 1000, 'unixepoch') AS ts
    FROM price_ticks
    ORDER BY ts DESC
    LIMIT 500
""", con_sq)
con_sq.close()

if not price_data.empty:
    price_ticker = st.selectbox("Select ticker for price chart",
                                sorted(price_data["ticker"].unique().tolist()))
    chart_data = (
        price_data[price_data["ticker"] == price_ticker]
        .sort_values("ts")
        .dropna(subset=["close"])
    )
    if not chart_data.empty:
        st.line_chart(chart_data.set_index("ts")["close"])
    else:
        st.info("No valid price data yet for this ticker.")


# ── Feature store stats ───────────────────────────────────────────────────────
st.subheader("Feature Store")

if os.path.exists(DUCKDB_PATH):
    con_dq = duckdb.connect(DUCKDB_PATH, read_only=True)
    tables = con_dq.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).df()
    table_names = tables["table_name"].tolist()
    st.write("Tables in DuckDB:", table_names)

    if "features_sentiment" in table_names:
        try:
            features = con_dq.execute("""
                SELECT ticker,
                       COUNT(*)              AS windows,
                       AVG(rolling_1h_mean)  AS avg_1h_headline_rate,
                       AVG(rolling_24h_std)  AS avg_24h_price_std
                FROM features_sentiment
                GROUP BY ticker
                ORDER BY ticker
            """).df()
            st.dataframe(features, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not load features_sentiment: {exc}")

    if "sentiment_scores" in table_names:
        st.subheader("Sentiment Scores")
        try:
            sentiment = con_dq.execute("""
                SELECT ticker,
                       ROUND(AVG(pos), 3) AS avg_positive,
                       ROUND(AVG(neg), 3) AS avg_negative,
                       ROUND(AVG(neu), 3) AS avg_neutral
                FROM sentiment_scores
                GROUP BY ticker
                ORDER BY avg_positive DESC
            """).df()
            st.dataframe(sentiment, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            x = range(len(sentiment))
            ax.bar(x, sentiment["avg_positive"], label="Positive", color="#4CAF50", alpha=0.8)
            ax.bar(x, sentiment["avg_negative"], label="Negative", color="#F44336", alpha=0.8,
                   bottom=sentiment["avg_positive"])
            ax.bar(x, sentiment["avg_neutral"],  label="Neutral",  color="#9E9E9E", alpha=0.8,
                   bottom=sentiment["avg_positive"] + sentiment["avg_negative"])
            ax.set_xticks(list(x))
            ax.set_xticklabels(sentiment["ticker"])
            ax.set_ylabel("Sentiment proportion")
            ax.legend()
            st.pyplot(fig)
        except Exception as exc:
            st.warning(f"Could not load sentiment_scores: {exc}")

    con_dq.close()
else:
    st.info("Feature store not built yet. Run `python feature_store/export.py` first.")


# ── Refresh ───────────────────────────────────────────────────────────────────
st.caption("Refresh the page to update. Run producers continuously to see data grow.")
