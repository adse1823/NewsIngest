"""
Tests for ingestion/kafka_replay.py.

All tests use mocked Kafka and SQLite — no live broker needed.
"""

import importlib.util
import sqlite3
import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_kafka_replay():
    name = "ingestion.kafka_replay"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "ingestion" / "kafka_replay.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KR = _load_kafka_replay()

TS_BASE = 1_700_000_000_000  # epoch ms — 2023-11-14


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sqlite_db(tmp_path):
    db_path = tmp_path / "raw.db"
    con = sqlite3.connect(str(db_path))
    con.execute("""
        CREATE TABLE news_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, source TEXT, ticker TEXT,
            ts INTEGER, url TEXT, inserted_at INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE price_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, open REAL, high REAL, low REAL,
            close REAL, volume INTEGER, ts INTEGER, inserted_at INTEGER
        )
    """)
    con.executemany(
        "INSERT INTO news_raw (title, source, ticker, ts, url, inserted_at) VALUES (?,?,?,?,?,?)",
        [
            ("Apple rises", "Reuters", "AAPL", TS_BASE,       "http://a.com", TS_BASE),
            ("MSFT beats",  "Bloomberg", "MSFT", TS_BASE + 1000, "http://b.com", TS_BASE),
        ],
    )
    con.executemany(
        "INSERT INTO price_ticks (ticker, open, high, low, close, volume, ts, inserted_at) VALUES (?,?,?,?,?,?,?,?)",
        [
            ("AAPL", 180.0, 182.0, 179.0, 181.0, 50000000, TS_BASE,        TS_BASE),
            ("MSFT", 370.0, 372.0, 369.0, 371.0, 30000000, TS_BASE + 1000, TS_BASE),
        ],
    )
    con.commit()
    con.close()
    return str(db_path)


# ── _serialize ────────────────────────────────────────────────────────────────

def test_serialize_starts_with_magic_byte():
    schema = MagicMock()
    with patch("ingestion.kafka_replay.fastavro.schemaless_writer"):
        result = KR._serialize(schema, schema_id=7, record={})
    assert result[0:1] == b"\x00"


def test_serialize_encodes_schema_id_big_endian():
    schema = MagicMock()
    with patch("ingestion.kafka_replay.fastavro.schemaless_writer"):
        result = KR._serialize(schema, schema_id=42, record={})
    schema_id_bytes = result[1:5]
    assert struct.unpack(">I", schema_id_bytes)[0] == 42


def test_serialize_total_length_is_at_least_5():
    schema = MagicMock()
    with patch("ingestion.kafka_replay.fastavro.schemaless_writer"):
        result = KR._serialize(schema, schema_id=1, record={})
    assert len(result) >= 5


# ── replay_news ───────────────────────────────────────────────────────────────

def test_replay_news_sends_one_message_per_row(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=1), \
         patch.object(KR, "_serialize", return_value=b"\x00" + b"\x00" * 4):
        KR.replay_news(con, producer)

    assert producer.send.call_count == 2
    con.close()


def test_replay_news_uses_correct_topic(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=1), \
         patch.object(KR, "_serialize", return_value=b"\x00" * 5):
        KR.replay_news(con, producer)

    for c in producer.send.call_args_list:
        assert c.args[0] == "news-raw"
    con.close()


def test_replay_news_uses_ticker_as_key(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=1), \
         patch.object(KR, "_serialize", return_value=b"\x00" * 5):
        KR.replay_news(con, producer)

    keys = {c.kwargs["key"] for c in producer.send.call_args_list}
    assert keys == {b"AAPL", b"MSFT"}
    con.close()


def test_replay_news_calls_flush(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=1), \
         patch.object(KR, "_serialize", return_value=b"\x00" * 5):
        KR.replay_news(con, producer)

    producer.flush.assert_called_once()
    con.close()


# ── replay_prices ─────────────────────────────────────────────────────────────

def test_replay_prices_sends_one_message_per_row(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=2), \
         patch.object(KR, "_serialize", return_value=b"\x00" * 5):
        KR.replay_prices(con, producer)

    assert producer.send.call_count == 2
    con.close()


def test_replay_prices_uses_correct_topic(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=2), \
         patch.object(KR, "_serialize", return_value=b"\x00" * 5):
        KR.replay_prices(con, producer)

    for c in producer.send.call_args_list:
        assert c.args[0] == "price-ticks"
    con.close()


def test_replay_prices_calls_flush(sqlite_db):
    producer = MagicMock()
    con = sqlite3.connect(sqlite_db)

    with patch.object(KR, "_load_schema", return_value=MagicMock()), \
         patch.object(KR, "_get_schema_id", return_value=2), \
         patch.object(KR, "_serialize", return_value=b"\x00" * 5):
        KR.replay_prices(con, producer)

    producer.flush.assert_called_once()
    con.close()


# ── main ─────────────────────────────────────────────────────────────────────

def test_main_skips_when_db_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(KR, "DB_PATH", str(tmp_path / "nonexistent.db"))
    # Should return without error
    KR.main()


def test_main_skips_when_both_tables_empty(tmp_path, monkeypatch):
    db_path = tmp_path / "raw.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE news_raw (id INTEGER PRIMARY KEY, title TEXT, source TEXT, ticker TEXT, ts INTEGER, url TEXT, inserted_at INTEGER)")
    con.execute("CREATE TABLE price_ticks (id INTEGER PRIMARY KEY, ticker TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, ts INTEGER, inserted_at INTEGER)")
    con.commit()
    con.close()

    monkeypatch.setattr(KR, "DB_PATH", str(db_path))
    with patch("ingestion.kafka_replay.KafkaProducer") as mock_kp:
        KR.main()
    mock_kp.assert_not_called()


def test_main_skips_when_kafka_unreachable(sqlite_db, monkeypatch):
    monkeypatch.setattr(KR, "DB_PATH", sqlite_db)
    with patch("ingestion.kafka_replay.KafkaProducer", side_effect=Exception("connection refused")):
        KR.main()  # should not raise


def test_main_calls_both_replay_functions(sqlite_db, monkeypatch):
    monkeypatch.setattr(KR, "DB_PATH", sqlite_db)
    mock_producer = MagicMock()

    with patch("ingestion.kafka_replay.KafkaProducer", return_value=mock_producer), \
         patch.object(KR, "replay_news")  as mock_news, \
         patch.object(KR, "replay_prices") as mock_prices:
        KR.main()

    mock_news.assert_called_once()
    mock_prices.assert_called_once()
    mock_producer.close.assert_called_once()
