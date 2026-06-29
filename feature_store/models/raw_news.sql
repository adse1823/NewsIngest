-- Read from SQLite news_raw, bucket into 5-minute windows per ticker
CREATE OR REPLACE TABLE raw_news AS
SELECT
    time_bucket(INTERVAL '5 minutes', to_timestamp(CAST(ts AS DOUBLE) / 1000)) AS window_start,
    time_bucket(INTERVAL '5 minutes', to_timestamp(CAST(ts AS DOUBLE) / 1000))
        + INTERVAL '5 minutes'                                                  AS window_end,
    ticker,
    COUNT(*)          AS headline_count,
    LIST(title)       AS titles
FROM sqlite_db.news_raw
WHERE ticker IS NOT NULL
  AND title  IS NOT NULL
GROUP BY window_start, window_end, ticker;
