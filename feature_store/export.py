import os
import logging
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SQLITE_PATH = "./data/raw.db"
DUCKDB_PATH = "./data/feature_store.duckdb"
OUTPUT_PATH = "./data/features_export.parquet"
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")


def run_sql_file(con: duckdb.DuckDBPyConnection, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    sql  = open(path).read()
    log.info("Running %s", filename)
    con.execute(sql)


def main():
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(
            f"SQLite database not found at {SQLITE_PATH}. "
            "Run ingestion/news_producer.py and ingestion/price_producer.py first."
        )

    con = duckdb.connect(DUCKDB_PATH)

    # Install and load the SQLite extension so DuckDB can read raw.db directly
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")
    con.execute(f"ATTACH '{SQLITE_PATH}' AS sqlite_db (TYPE sqlite);")

    row_counts = con.execute(
        "SELECT COUNT(*) FROM sqlite_db.news_raw"
    ).fetchone()[0]
    log.info("Found %d rows in news_raw", row_counts)

    if row_counts == 0:
        raise ValueError("news_raw is empty — let the producers run for a few minutes first.")

    run_sql_file(con, "raw_news.sql")
    run_sql_file(con, "features_sentiment.sql")
    run_sql_file(con, "entity_table.sql")

    df: pd.DataFrame = con.execute("""
        SELECT
            f.window_start,
            f.ticker,
            f.rolling_1h_mean,
            f.rolling_24h_std,
            f.volume_zscore,
            f.pct_change,
            e.sector,
            e.node_id
        FROM features_sentiment f
        LEFT JOIN entity_table e USING (ticker)
        WHERE f.rolling_1h_mean IS NOT NULL
        ORDER BY f.ticker, f.window_start
    """).df()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Exported %d rows to %s", len(df), OUTPUT_PATH)

    con.close()


if __name__ == "__main__":
    main()
