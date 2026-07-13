import importlib.util
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import pytest


# ── load module by file path ──────────────────────────────────────────────────

def _load_price_producer():
    name = "ingestion.price_producer"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "ingestion" / "price_producer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PP = _load_price_producer()


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    PP.init_db(c)
    yield c
    c.close()


# ── sample data ───────────────────────────────────────────────────────────────

TICK = {
    "ticker": "AAPL",
    "open":   150.0,
    "high":   155.0,
    "low":    148.0,
    "close":  153.0,
    "volume": 1_000_000,
    "ts":     1_700_000_000_000,
}


# ── tests: init_db ────────────────────────────────────────────────────────────

def test_init_db_creates_table_and_indexes():
    c = sqlite3.connect(":memory:")
    PP.init_db(c)
    objects = {r[0] for r in c.execute("SELECT name FROM sqlite_master").fetchall()}
    assert "price_ticks" in objects
    assert "idx_price_ticker" in objects
    assert "idx_price_ts" in objects
    c.close()


# ── tests: insert_tick ────────────────────────────────────────────────────────

def test_insert_tick_writes_correct_values_to_db(con):
    PP.insert_tick(con, None, TICK)
    row = con.execute("SELECT ticker, open, high, low, close, volume FROM price_ticks").fetchone()
    assert row == ("AAPL", 150.0, 155.0, 148.0, 153.0, 1_000_000)


def test_insert_tick_sends_to_kafka_with_ticker_key(con):
    producer = MagicMock()
    PP.insert_tick(con, producer, TICK)
    producer.send.assert_called_once()
    _, kwargs = producer.send.call_args
    assert kwargs["key"] == b"AAPL"


def test_insert_tick_skips_kafka_when_producer_none(con):
    PP.insert_tick(con, None, TICK)  # must not raise
    count = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    assert count == 1


def test_insert_tick_kafka_error_does_not_raise(con):
    producer = MagicMock()
    producer.send.side_effect = Exception("broker down")
    PP.insert_tick(con, producer, TICK)  # must not raise
    # DB write still completed
    count = con.execute("SELECT COUNT(*) FROM price_ticks").fetchone()[0]
    assert count == 1


# ── tests: fetch_tick ─────────────────────────────────────────────────────────

def test_fetch_tick_returns_none_on_error():
    with patch.object(PP, "yf") as mock_yf:
        mock_yf.download.side_effect = Exception("network error")
        result = PP.fetch_tick("AAPL")
    assert result is None


def test_fetch_tick_returns_none_when_dataframe_empty():
    with patch.object(PP, "yf") as mock_yf:
        mock_yf.download.return_value = pd.DataFrame()
        result = PP.fetch_tick("AAPL")
    assert result is None
