-- Rolling window features joined with price data
CREATE TABLE IF NOT EXISTS features_sentiment AS
WITH news AS (
    SELECT
        window_start,
        ticker,
        headline_count
    FROM raw_news
),
prices AS (
    SELECT
        window_start,
        ticker,
        avg_close,
        avg_volume,
        pct_change
    FROM read_parquet('./data/windowed/prices/**/*.parquet', hive_partitioning = false)
    WHERE ticker IS NOT NULL
),
joined AS (
    SELECT
        n.window_start,
        n.ticker,
        n.headline_count,
        p.avg_close,
        p.avg_volume,
        p.pct_change
    FROM news n
    LEFT JOIN prices p
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
