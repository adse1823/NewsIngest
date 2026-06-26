import os
import logging
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./data/feature_store.duckdb"
OUTPUT_PATH = "./features_export.parquet"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def run_sql_file(con: duckdb.DuckDBPyConnection, filename: str):
    path = os.path.join(MODELS_DIR, filename)
    sql = open(path).read()
    log.info("Running %s", filename)
    con.execute(sql)


def main():
    os.makedirs("./data", exist_ok=True)
    con = duckdb.connect(DB_PATH)

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

    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("Exported %d rows to %s", len(df), OUTPUT_PATH)
    con.close()


if __name__ == "__main__":
    main()
