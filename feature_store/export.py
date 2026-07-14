import glob
import os
import logging
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SQLITE_PATH     = "./data/raw.db"
DUCKDB_PATH     = "./data/feature_store.duckdb"
OUTPUT_PATH     = "./data/features_export.parquet"
MODELS_DIR      = os.path.join(os.path.dirname(__file__), "models")
SPARK_NEWS_DIR  = "./data/windowed/news"
SPARK_PRICE_DIR = "./data/windowed/prices"


def _has_parquet(directory: str) -> bool:
    return bool(glob.glob(os.path.join(directory, "**", "*.parquet"), recursive=True))


def run_sql_file(con: duckdb.DuckDBPyConnection, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    sql  = open(path).read()
    log.info("Running %s", filename)
    con.execute(sql)


def _create_raw_news(con: duckdb.DuckDBPyConnection):
    """Create raw_news table: 5-min windowed news from SQLite, merged with Spark parquet if available."""
    sqlite_sql = """
        SELECT
            time_bucket(INTERVAL '5 minutes', to_timestamp(CAST(ts AS DOUBLE) / 1000)) AS window_start,
            time_bucket(INTERVAL '5 minutes', to_timestamp(CAST(ts AS DOUBLE) / 1000))
                + INTERVAL '5 minutes'                                                  AS window_end,
            ticker,
            COUNT(*)   AS headline_count,
            LIST(title) AS titles
        FROM sqlite_db.news_raw
        WHERE ticker IS NOT NULL AND title IS NOT NULL
        GROUP BY window_start, window_end, ticker
    """
    use_spark = _has_parquet(SPARK_NEWS_DIR)
    if use_spark:
        log.info("Spark news parquet found — merging with SQLite news")
        spark_path = SPARK_NEWS_DIR.replace("\\", "/") + "/**/*.parquet"
        sql = f"""
            CREATE OR REPLACE TABLE raw_news AS
            WITH sqlite_base AS ({sqlite_sql}),
            spark_extra AS (
                SELECT window_start, window_end, ticker, headline_count, titles
                FROM read_parquet('{spark_path}')
                WHERE (ticker, window_start) NOT IN (
                    SELECT ticker, window_start FROM sqlite_base
                )
            )
            SELECT * FROM sqlite_base
            UNION ALL
            SELECT * FROM spark_extra
        """
    else:
        log.info("No Spark news parquet — using SQLite only")
        sql = f"CREATE OR REPLACE TABLE raw_news AS {sqlite_sql}"
    con.execute(sql)
    count = con.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0]
    log.info("raw_news: %d rows (spark_merge=%s)", count, use_spark)


def _create_price_windows(con: duckdb.DuckDBPyConnection):
    """Create price_windows_merged: 5-min price windows from SQLite, merged with Spark parquet if available."""
    sqlite_sql = """
        SELECT
            time_bucket(INTERVAL '5 minutes', to_timestamp(CAST(ts AS DOUBLE) / 1000)) AS window_start,
            ticker,
            AVG(close)  AS avg_close,
            AVG(volume) AS avg_volume
        FROM sqlite_db.price_ticks
        WHERE ticker IS NOT NULL
        GROUP BY window_start, ticker
    """
    use_spark = _has_parquet(SPARK_PRICE_DIR)
    if use_spark:
        log.info("Spark price parquet found — merging with SQLite ticks")
        spark_path = SPARK_PRICE_DIR.replace("\\", "/") + "/**/*.parquet"
        sql = f"""
            CREATE OR REPLACE TABLE price_windows_merged AS
            WITH sqlite_base AS ({sqlite_sql}),
            spark_extra AS (
                SELECT window_start, ticker, avg_close, avg_volume
                FROM read_parquet('{spark_path}')
                WHERE (ticker, window_start) NOT IN (
                    SELECT ticker, window_start FROM sqlite_base
                )
            )
            SELECT * FROM sqlite_base
            UNION ALL
            SELECT * FROM spark_extra
        """
    else:
        log.info("No Spark price parquet — using SQLite only")
        sql = f"CREATE OR REPLACE TABLE price_windows_merged AS {sqlite_sql}"
    con.execute(sql)
    count = con.execute("SELECT COUNT(*) FROM price_windows_merged").fetchone()[0]
    log.info("price_windows_merged: %d rows (spark_merge=%s)", count, use_spark)


def main():
    if not os.path.exists(SQLITE_PATH):
        raise FileNotFoundError(
            f"SQLite database not found at {SQLITE_PATH}. "
            "Run ingestion/news_producer.py and ingestion/price_producer.py first."
        )

    con = duckdb.connect(DUCKDB_PATH)

    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")
    con.execute(f"ATTACH '{SQLITE_PATH}' AS sqlite_db (TYPE sqlite);")

    row_counts = con.execute("SELECT COUNT(*) FROM sqlite_db.news_raw").fetchone()[0]
    log.info("Found %d rows in news_raw", row_counts)

    if row_counts == 0:
        raise ValueError("news_raw is empty — let the producers run for a few minutes first.")

    # Build merged source tables (SQLite + Spark parquet when available)
    _create_raw_news(con)
    _create_price_windows(con)

    # Run downstream feature models (entity_table still derives from raw_news)
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
