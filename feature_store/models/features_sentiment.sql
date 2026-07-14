-- Join news windows with price windows and compute rolling features.
-- price_windows_merged is pre-created by export.py (SQLite ticks + Spark streaming parquet).
-- raw_news is pre-created by export.py (SQLite news_raw + Spark streaming parquet).
CREATE OR REPLACE TABLE features_sentiment AS
WITH price_windows AS (
    SELECT
        window_start,
        ticker,
        avg_close,
        avg_volume,
        (avg_close - LAG(avg_close) OVER (PARTITION BY ticker ORDER BY window_start))
            / NULLIF(LAG(avg_close) OVER (PARTITION BY ticker ORDER BY window_start), 0) * 100
            AS pct_change
    FROM price_windows_merged
),
joined AS (
    SELECT
        n.window_start,
        n.ticker,
        n.headline_count,
        p.avg_close,
        p.avg_volume,
        p.pct_change
    FROM raw_news n
    LEFT JOIN price_windows p
        ON n.ticker = p.ticker AND n.window_start = p.window_start
),
rolling AS (
    SELECT
        window_start,
        ticker,
        headline_count,
        avg_close,
        avg_volume,
        pct_change,
        AVG(headline_count) OVER (
            PARTITION BY ticker
            ORDER BY window_start
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS rolling_1h_mean,
        STDDEV(pct_change) OVER (
            PARTITION BY ticker
            ORDER BY window_start
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS rolling_24h_std,
        (avg_volume - AVG(avg_volume) OVER (
            PARTITION BY ticker
            ORDER BY window_start
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        )) / NULLIF(STDDEV(avg_volume) OVER (
            PARTITION BY ticker
            ORDER BY window_start
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ), 0) AS volume_zscore
    FROM joined
)
SELECT * FROM rolling;
