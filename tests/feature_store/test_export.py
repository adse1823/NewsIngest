import importlib.util
import sqlite3
import sys
from pathlib import Path

import duckdb
import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_export():
    name = "feature_store.export"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "feature_store" / "export.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EX = _load_export()

TS_BASE = 1_700_000_000_000  # epoch ms — 2023-11-14


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sqlite_path(tmp_path):
    """Temp SQLite file with news_raw and price_ticks tables."""
    db_path = str(tmp_path / "raw.db")
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE news_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, title TEXT, source TEXT, ts INTEGER, url TEXT
        )
    """)
    con.execute("""
        CREATE TABLE price_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, ts INTEGER
        )
    """)
    con.executemany(
        "INSERT INTO news_raw (ticker, title, source, ts, url) VALUES (?,?,?,?,?)",
        [
            ("AAPL", "Apple earnings beat",   "Reuters",   TS_BASE,          "https://a.com/1"),
            ("AAPL", "Apple product launch",  "Bloomberg", TS_BASE + 60_000, "https://a.com/2"),
            ("MSFT", "Microsoft cloud surge", "CNBC",      TS_BASE,          "https://m.com/1"),
        ],
    )
    con.executemany(
        "INSERT INTO price_ticks (ticker, open, high, low, close, volume, ts) VALUES (?,?,?,?,?,?,?)",
        [
            ("AAPL", 150.0, 155.0, 148.0, 153.0, 1_000_000, TS_BASE),
            ("AAPL", 153.0, 156.0, 152.0, 154.0,   900_000, TS_BASE + 60_000),
            ("MSFT", 300.0, 305.0, 298.0, 302.0,   500_000, TS_BASE),
        ],
    )
    con.commit()
    con.close()
    return db_path


@pytest.fixture
def duck_con(sqlite_path):
    con = duckdb.connect(":memory:")
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")
    con.execute(f"ATTACH '{sqlite_path}' AS sqlite_db (TYPE sqlite);")
    yield con
    con.close()


# ── tests: _has_parquet ───────────────────────────────────────────────────────

def test_has_parquet_empty_dir(tmp_path):
    assert not EX._has_parquet(str(tmp_path))


def test_has_parquet_with_file(tmp_path):
    (tmp_path / "data.parquet").write_text("")
    assert EX._has_parquet(str(tmp_path))


def test_has_parquet_nested(tmp_path):
    sub = tmp_path / "partition=1"
    sub.mkdir()
    (sub / "part.parquet").write_text("")
    assert EX._has_parquet(str(tmp_path))


def test_has_parquet_ignores_non_parquet(tmp_path):
    (tmp_path / "data.csv").write_text("")
    assert not EX._has_parquet(str(tmp_path))


# ── tests: _create_raw_news ───────────────────────────────────────────────────

def test_create_raw_news_creates_table(duck_con):
    EX._create_raw_news(duck_con)
    count = duck_con.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0]
    # 2 AAPL rows land in the same 5-min bucket → 1 AAPL window + 1 MSFT window
    assert count == 2


def test_create_raw_news_expected_columns(duck_con):
    EX._create_raw_news(duck_con)
    cols = {row[0] for row in duck_con.execute("DESCRIBE raw_news").fetchall()}
    assert {"window_start", "window_end", "ticker", "headline_count", "titles"} <= cols


def test_create_raw_news_aggregates_headlines_in_same_window(duck_con):
    EX._create_raw_news(duck_con)
    row = duck_con.execute(
        "SELECT headline_count FROM raw_news WHERE ticker = 'AAPL'"
    ).fetchone()
    assert row is not None
    assert row[0] == 2  # both AAPL headlines fall in the same 5-min window


def test_create_raw_news_no_spark_uses_sqlite_only(duck_con, monkeypatch):
    monkeypatch.setattr(EX, "_has_parquet", lambda _: False)
    EX._create_raw_news(duck_con)
    assert duck_con.execute("SELECT COUNT(*) FROM raw_news").fetchone()[0] > 0


# ── tests: _create_price_windows ─────────────────────────────────────────────

def test_create_price_windows_creates_table(duck_con):
    EX._create_price_windows(duck_con)
    count = duck_con.execute("SELECT COUNT(*) FROM price_windows_merged").fetchone()[0]
    assert count == 2  # AAPL + MSFT, each in one 5-min bucket


def test_create_price_windows_expected_columns(duck_con):
    EX._create_price_windows(duck_con)
    cols = {row[0] for row in duck_con.execute("DESCRIBE price_windows_merged").fetchall()}
    assert {"window_start", "ticker", "avg_close", "avg_volume"} <= cols


def test_create_price_windows_averages_close(duck_con):
    EX._create_price_windows(duck_con)
    row = duck_con.execute(
        "SELECT avg_close FROM price_windows_merged WHERE ticker = 'AAPL'"
    ).fetchone()
    assert row is not None
    assert abs(row[0] - 153.5) < 0.01  # avg(153.0, 154.0)


def test_create_price_windows_no_spark_uses_sqlite_only(duck_con, monkeypatch):
    monkeypatch.setattr(EX, "_has_parquet", lambda _: False)
    EX._create_price_windows(duck_con)
    assert duck_con.execute("SELECT COUNT(*) FROM price_windows_merged").fetchone()[0] > 0


# ── tests: run_sql_file (features_sentiment + entity_table) ──────────────────

def test_features_sentiment_sql_creates_table(duck_con):
    EX._create_raw_news(duck_con)
    EX._create_price_windows(duck_con)
    EX.run_sql_file(duck_con, "features_sentiment.sql")
    cols = {row[0] for row in duck_con.execute("DESCRIBE features_sentiment").fetchall()}
    assert {"window_start", "ticker", "rolling_1h_mean", "rolling_24h_std", "volume_zscore"} <= cols


def test_entity_table_sql_creates_table(duck_con):
    EX._create_raw_news(duck_con)
    EX.run_sql_file(duck_con, "entity_table.sql")
    cols = {row[0] for row in duck_con.execute("DESCRIBE entity_table").fetchall()}
    assert {"ticker", "sector", "node_id"} <= cols


def test_entity_table_sql_correct_sector_for_aapl(duck_con):
    EX._create_raw_news(duck_con)
    EX.run_sql_file(duck_con, "entity_table.sql")
    row = duck_con.execute(
        "SELECT sector FROM entity_table WHERE ticker = 'AAPL'"
    ).fetchone()
    assert row is not None
    assert row[0] == "Technology"


def test_entity_table_sql_node_ids_are_unique(duck_con):
    EX._create_raw_news(duck_con)
    EX.run_sql_file(duck_con, "entity_table.sql")
    total = duck_con.execute("SELECT COUNT(*) FROM entity_table").fetchone()[0]
    distinct = duck_con.execute("SELECT COUNT(DISTINCT node_id) FROM entity_table").fetchone()[0]
    assert total == distinct
