import duckdb
import sqlite3
import pandas as pd
import numpy as np

print("=== SQLite ===")
con_sq = sqlite3.connect('./data/raw.db')
print(f"news_raw:    {con_sq.execute('SELECT COUNT(*) FROM news_raw').fetchone()[0]} rows")
print(f"price_ticks: {con_sq.execute('SELECT COUNT(*) FROM price_ticks').fetchone()[0]} rows")
con_sq.close()

print("\n=== DuckDB tables ===")
con = duckdb.connect('./data/feature_store.duckdb', read_only=True)
for t in ['raw_news', 'features_sentiment', 'sentiment_scores']:
    n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {n} rows")

print("\n=== features_sentiment sample ===")
df = con.execute("""
    SELECT f.window_start, f.ticker, f.rolling_1h_mean, f.rolling_24h_std,
           f.volume_zscore, f.pct_change, s.pos, s.neg, s.neu
    FROM features_sentiment f
    LEFT JOIN sentiment_scores s ON f.ticker = s.ticker AND f.window_start = s.window_start
    WHERE f.rolling_1h_mean IS NOT NULL
    ORDER BY f.ticker, f.window_start
""").df()
con.close()

print(f"Total feature rows: {len(df)}")
print(f"pct_change non-null: {df['pct_change'].notna().sum()}")
print(f"pct_change null: {df['pct_change'].isna().sum()}")

df['label'] = (df.groupby('ticker')['pct_change'].shift(-1) > 0).astype('Int64')
df_valid = df.dropna(subset=['label'])
print(f"\nAfter label creation: {len(df_valid)} valid rows")

if len(df_valid) > 0:
    counts = df_valid['label'].value_counts()
    print(f"Label=0 (down): {counts.get(0, 0)}")
    print(f"Label=1 (up):   {counts.get(1, 0)}")
    print(f"\npct_change stats:\n{df['pct_change'].describe()}")
    print(f"\nRows per ticker:\n{df_valid.groupby('ticker').size()}")
