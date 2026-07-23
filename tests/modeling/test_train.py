import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_train():
    name = "modeling.train"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "modeling" / "train.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TRAIN = _load_train()


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pct_df():
    """Two tickers, 4 rows each, alternating pct_change signs."""
    return pd.DataFrame({
        "ticker":       ["AAPL"] * 4 + ["MSFT"] * 4,
        "window_start": pd.date_range("2024-01-01", periods=8, freq="5min"),
        "pct_change":   [1.0, -0.5, 2.0, -1.0,  0.3, 0.0, -0.8, 1.5],
    })


@pytest.fixture
def gnn_parquet(tmp_path):
    path = str(tmp_path / "gnn.parquet")
    pd.DataFrame({
        "ticker":    ["AAPL", "MSFT", "GOOGL"],
        "gnn_dim_0": [0.1,    0.2,    0.3],
        "gnn_dim_1": [0.4,    0.5,    0.6],
    }).to_parquet(path, index=False)
    return path


# ── tests: create_label ───────────────────────────────────────────────────────

def test_create_label_values_are_binary(pct_df):
    result = TRAIN.create_label(pct_df)
    assert set(result["label"].unique()).issubset({0, 1})


def test_create_label_positive_next_pct_gives_label_1():
    df = pd.DataFrame({
        "ticker":     ["AAPL", "AAPL"],
        "pct_change": [0.5,     1.0],   # row 0 label = (1.0 > 0) = 1
    })
    result = TRAIN.create_label(df)
    assert result.iloc[0]["label"] == 1


def test_create_label_negative_next_pct_gives_label_0():
    df = pd.DataFrame({
        "ticker":     ["AAPL", "AAPL"],
        "pct_change": [0.5,    -1.0],   # row 0 label = (-1.0 > 0) = 0
    })
    result = TRAIN.create_label(df)
    assert result.iloc[0]["label"] == 0


def test_create_label_last_row_per_ticker_has_zero_label():
    # shift(-1) on last row → NaN → NaN > 0 → False → label 0
    df = pd.DataFrame({
        "ticker":     ["AAPL", "AAPL", "AAPL"],
        "pct_change": [1.0,     2.0,    3.0],
    })
    result = TRAIN.create_label(df)
    assert result.iloc[-1]["label"] == 0


def test_create_label_does_not_bleed_across_tickers():
    # AAPL's last row must not use MSFT's pct_change
    df = pd.DataFrame({
        "ticker":     ["AAPL", "MSFT"],
        "pct_change": [1.0,     99.0],   # MSFT next row must not affect AAPL label
    })
    result = TRAIN.create_label(df)
    aapl_label = result[result["ticker"] == "AAPL"]["label"].values[0]
    assert aapl_label == 0  # AAPL has no next row in its own group


def test_create_label_multiple_tickers_correct_count(pct_df):
    result = TRAIN.create_label(pct_df)
    # pct_change is never truly NaN so no rows are dropped by dropna
    assert len(result) == len(pct_df)


# ── tests: attach_gnn_embeddings ──────────────────────────────────────────────

def test_attach_gnn_embeddings_adds_gnn_columns(gnn_parquet):
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "value": [1.0, 2.0]})
    with patch.object(TRAIN, "GNN_PATH", gnn_parquet):
        result = TRAIN.attach_gnn_embeddings(df)
    assert "gnn_dim_0" in result.columns
    assert "gnn_dim_1" in result.columns


def test_attach_gnn_embeddings_preserves_row_count(gnn_parquet):
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "value": [1.0, 2.0]})
    with patch.object(TRAIN, "GNN_PATH", gnn_parquet):
        result = TRAIN.attach_gnn_embeddings(df)
    assert len(result) == 2


def test_attach_gnn_embeddings_unknown_ticker_gives_nan(gnn_parquet):
    df = pd.DataFrame({"ticker": ["AAPL", "TSLA"], "value": [1.0, 2.0]})
    with patch.object(TRAIN, "GNN_PATH", gnn_parquet):
        result = TRAIN.attach_gnn_embeddings(df)
    tsla_gnn = result[result["ticker"] == "TSLA"]["gnn_dim_0"].values[0]
    assert pd.isna(tsla_gnn)


def test_attach_gnn_embeddings_correct_values(gnn_parquet):
    df = pd.DataFrame({"ticker": ["AAPL"], "value": [1.0]})
    with patch.object(TRAIN, "GNN_PATH", gnn_parquet):
        result = TRAIN.attach_gnn_embeddings(df)
    assert abs(result.iloc[0]["gnn_dim_0"] - 0.1) < 1e-6
    assert abs(result.iloc[0]["gnn_dim_1"] - 0.4) < 1e-6


# ── tests: LGB_PARAMS sanity ──────────────────────────────────────────────────

def test_lgb_params_binary_objective():
    assert TRAIN.LGB_PARAMS["objective"] == "binary"


def test_lgb_params_auc_metric():
    assert TRAIN.LGB_PARAMS["metric"] == "auc"


def test_lgb_params_random_state_set():
    assert "random_state" in TRAIN.LGB_PARAMS
