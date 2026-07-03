-- Window price ticks into 5-minute buckets then join with news windows
-- This replaces what Spark was doing in Phase 2
CREATE OR REPLACE TABLE features_sentiment AS
WITH price_windows_raw AS (
    SELECT
        time_bucket(INTERVAL '5 minutes', to_timestamp(CAST(ts AS DOUBLE) / 1000)) AS window_start,
        ticker,
        AVG(close)  AS avg_close,
        AVG(volume) AS avg_volume
    FROM sqlite_db.price_ticks
    WHERE ticker IS NOT NULL
    GROUP BY window_start, ticker
),
price_windows AS (
    SELECT
        window_start,
        ticker,
        avg_close,
        avg_volume,
        -- between-window return: how much did price move vs previous 5-min window
        (avg_close - LAG(avg_close) OVER (PARTITION BY ticker ORDER BY window_start))
            / NULLIF(LAG(avg_close) OVER (PARTITION BY ticker ORDER BY window_start), 0) * 100
            AS pct_change
    FROM price_windows_raw
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
