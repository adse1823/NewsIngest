import sqlite3
import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingestion"))

import pandas as pd
from tickers import TICKERS

con = sqlite3.connect("./data/raw.db")

total   = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM price_ticks").fetchone()[0]
dates   = con.execute("SELECT MIN(ts), MAX(ts) FROM price_ticks").fetchone()

min_dt = datetime.datetime.fromtimestamp(dates[0]/1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
max_dt = datetime.datetime.fromtimestamp(dates[1]/1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d")

print(f"Total rows      : {total:,}")
print(f"Distinct tickers: {tickers} / {len(TICKERS)} expected")
print(f"Date range      : {min_dt}  ->  {max_dt}")

df = pd.read_sql("SELECT ticker, COUNT(*) as rows FROM price_ticks GROUP BY ticker ORDER BY rows DESC", con)

print(f"\nTop 10 by row count:")
print(df.head(10).to_string(index=False))

print(f"\nBottom 10 by row count:")
print(df.tail(10).to_string(index=False))

print(f"\nSample AAPL (5 most recent):")
sample = pd.read_sql(
    "SELECT ticker, open, high, low, close, volume, datetime(ts/1000,'unixepoch') as date "
    "FROM price_ticks WHERE ticker='AAPL' ORDER BY ts DESC LIMIT 5",
    con
)
print(sample.to_string(index=False))

present = set(df["ticker"].values)
missing = [t for t in TICKERS if t not in present]
print(f"\nMissing tickers ({len(missing)}): {missing if missing else 'none'}")

con.close()
