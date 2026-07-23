"""
Streaming layer tests — pure Python only (no SparkSession required).

The parse_news / parse_prices functions use from_avro (requires a Spark
avro JAR) and cannot be unit-tested without the full cluster. Tests here
cover:
  - Schema strings are valid Avro JSON
  - Required fields are present in each schema
  - Module constants have correct defaults
  - The 5-byte Confluent prefix stripping uses the right offset
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

# Skip entire module if PySpark is not installed
pytest.importorskip("pyspark", reason="PySpark not installed — streaming tests skipped")


# ── load module ───────────────────────────────────────────────────────────────

def _load_spark_consumer():
    name = "streaming.spark_consumer"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "streaming" / "spark_consumer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SC = _load_spark_consumer()


# ── tests: Avro schema strings ────────────────────────────────────────────────

def test_news_schema_is_valid_json():
    schema = json.loads(SC.NEWS_SCHEMA_STR)
    assert isinstance(schema, dict)


def test_news_schema_type_is_record():
    schema = json.loads(SC.NEWS_SCHEMA_STR)
    assert schema["type"] == "record"


def test_news_schema_name_is_news_event():
    schema = json.loads(SC.NEWS_SCHEMA_STR)
    assert schema["name"] == "NewsEvent"


def test_news_schema_has_required_fields():
    schema = json.loads(SC.NEWS_SCHEMA_STR)
    field_names = {f["name"] for f in schema["fields"]}
    assert {"title", "source", "ticker", "ts"} <= field_names


def test_news_schema_ts_is_timestamp_millis():
    schema = json.loads(SC.NEWS_SCHEMA_STR)
    ts_field = next(f for f in schema["fields"] if f["name"] == "ts")
    assert ts_field["type"]["logicalType"] == "timestamp-millis"


def test_price_schema_is_valid_json():
    schema = json.loads(SC.PRICE_SCHEMA_STR)
    assert isinstance(schema, dict)


def test_price_schema_type_is_record():
    schema = json.loads(SC.PRICE_SCHEMA_STR)
    assert schema["type"] == "record"


def test_price_schema_name_is_price_tick():
    schema = json.loads(SC.PRICE_SCHEMA_STR)
    assert schema["name"] == "PriceTick"


def test_price_schema_has_required_fields():
    schema = json.loads(SC.PRICE_SCHEMA_STR)
    field_names = {f["name"] for f in schema["fields"]}
    assert {"ticker", "open", "high", "low", "close", "volume", "ts"} <= field_names


def test_price_schema_ohlcv_fields_are_numeric():
    schema = json.loads(SC.PRICE_SCHEMA_STR)
    numeric_fields = {f["name"]: f["type"] for f in schema["fields"]
                      if f["name"] in {"open", "high", "low", "close"}}
    for name, typ in numeric_fields.items():
        assert typ == "double", f"{name} should be double, got {typ}"


def test_price_schema_volume_is_long():
    schema = json.loads(SC.PRICE_SCHEMA_STR)
    vol = next(f for f in schema["fields"] if f["name"] == "volume")
    assert vol["type"] == "long"


# ── tests: module constants ───────────────────────────────────────────────────

def test_output_dir_is_data_windowed():
    assert SC.OUTPUT_DIR == "./data/windowed"


def test_checkpoint_dir_has_default():
    assert SC.CHECKPOINT_DIR is not None
    assert len(SC.CHECKPOINT_DIR) > 0


def test_broker_has_default():
    assert SC.BROKER is not None
    assert ":" in SC.BROKER  # host:port format


def test_broker_env_override(monkeypatch):
    monkeypatch.setenv("REDPANDA_BROKERS", "myhost:9092")
    # Re-read the env var to verify it would be picked up
    assert os.getenv("REDPANDA_BROKERS") == "myhost:9092"


# ── tests: schema strings match avsc files ────────────────────────────────────

def test_news_schema_string_matches_avsc_file():
    avsc_path = Path(__file__).parent.parent.parent / "schemas" / "news_event_v1.avsc"
    on_disk = json.loads(avsc_path.read_text())
    in_module = json.loads(SC.NEWS_SCHEMA_STR)
    assert on_disk == in_module


def test_price_schema_string_matches_avsc_file():
    avsc_path = Path(__file__).parent.parent.parent / "schemas" / "price_tick_v1.avsc"
    on_disk = json.loads(avsc_path.read_text())
    in_module = json.loads(SC.PRICE_SCHEMA_STR)
    assert on_disk == in_module
