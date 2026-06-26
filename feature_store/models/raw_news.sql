-- Base table: aggregate windowed news Parquet into a clean relation
CREATE TABLE IF NOT EXISTS raw_news AS
SELECT
    window_start,
    window_end,
    ticker,
    headline_count,
    titles
FROM read_parquet('./data/windowed/news/**/*.parquet', hive_partitioning = false)
WHERE ticker IS NOT NULL
  AND window_start IS NOT NULL;
