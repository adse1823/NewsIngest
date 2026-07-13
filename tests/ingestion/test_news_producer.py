import importlib.util
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── load module by file path ──────────────────────────────────────────────────

def _load_news_producer():
    name = "ingestion.news_producer"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "ingestion" / "news_producer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


NP = _load_news_producer()


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    NP.init_db(c)
    yield c
    c.close()


# ── sample data ───────────────────────────────────────────────────────────────

ARTICLES = [
    {
        "title": "Apple earnings beat expectations",
        "source": {"name": "Reuters"},
        "publishedAt": "2024-01-15T10:30:00Z",
        "url": "https://example.com/1",
    },
    {
        "title": "Apple launches new product",
        "source": {"name": "Bloomberg"},
        "publishedAt": "2024-01-15T11:00:00Z",
        "url": "https://example.com/2",
    },
]


# ── tests: init_db ────────────────────────────────────────────────────────────

def test_init_db_creates_table_and_indexes():
    c = sqlite3.connect(":memory:")
    NP.init_db(c)
    objects = {r[0] for r in c.execute("SELECT name FROM sqlite_master").fetchall()}
    assert "news_raw" in objects
    assert "idx_news_ticker" in objects
    assert "idx_news_ts" in objects
    assert "idx_news_url" in objects
    c.close()


# ── tests: insert_articles ────────────────────────────────────────────────────

def test_insert_articles_writes_rows_to_db(con):
    NP.insert_articles(con, None, "AAPL", ARTICLES)
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 2


def test_insert_articles_deduplicates_by_url(con):
    NP.insert_articles(con, None, "AAPL", ARTICLES[:1])
    NP.insert_articles(con, None, "AAPL", ARTICLES[:1])  # same URL second time
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 1


def test_insert_articles_sends_to_kafka_for_each_article(con):
    producer = MagicMock()
    NP.insert_articles(con, producer, "AAPL", ARTICLES)
    assert producer.send.call_count == len(ARTICLES)
    _, kwargs = producer.send.call_args
    assert kwargs["key"] == b"AAPL"


def test_insert_articles_skips_kafka_when_producer_none(con):
    NP.insert_articles(con, None, "AAPL", ARTICLES)  # must not raise
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 2


def test_insert_articles_kafka_error_does_not_raise(con):
    producer = MagicMock()
    producer.send.side_effect = Exception("broker down")
    NP.insert_articles(con, producer, "AAPL", ARTICLES)  # must not raise
    # DB write still completed despite Kafka failure
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 2


def test_insert_articles_handles_missing_published_at(con):
    article = {"title": "Test", "source": {"name": "Test"}, "url": "https://example.com/3"}
    NP.insert_articles(con, None, "AAPL", [article])  # must not raise
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 1


def test_insert_articles_truncates_long_title(con):
    article = {
        "title": "X" * 600,
        "source": {"name": "Test"},
        "publishedAt": "2024-01-15T10:00:00Z",
        "url": "https://example.com/long",
    }
    NP.insert_articles(con, None, "AAPL", [article])
    title = con.execute("SELECT title FROM news_raw").fetchone()[0]
    assert len(title) == 500


# ── tests: fetch_articles ─────────────────────────────────────────────────────

def test_fetch_articles_returns_empty_on_api_error():
    client = MagicMock()
    client.get_everything.side_effect = Exception("API down")
    assert NP.fetch_articles(client, "AAPL") == []


def test_insert_articles_null_url_articles_both_inserted(con):
    # Partial index (WHERE url IS NOT NULL) must not deduplicate NULL-url rows
    articles = [
        {"title": "A", "source": {"name": "X"}, "publishedAt": "2024-01-15T10:00:00Z", "url": None},
        {"title": "B", "source": {"name": "X"}, "publishedAt": "2024-01-15T11:00:00Z", "url": None},
    ]
    NP.insert_articles(con, None, "AAPL", articles)
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 2


def test_insert_articles_invalid_published_at_falls_back_to_now(con):
    article = {
        "title": "Bad date",
        "source": {"name": "X"},
        "publishedAt": "not-a-date",
        "url": "https://example.com/baddate",
    }
    NP.insert_articles(con, None, "AAPL", [article])  # must not raise
    count = con.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
    assert count == 1
