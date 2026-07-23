import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── load module ───────────────────────────────────────────────────────────────

def _load_drift():
    name = "monitoring.drift_report"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parent.parent.parent / "monitoring" / "drift_report.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DR = _load_drift()
FEATURE_COLS = DR.FEATURE_COLS


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def full_df():
    """DataFrame spanning 20 days, 1 row per day — straddles the 14-day cutoff."""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame({
        "window_start":    dates,
        "rolling_1h_mean": range(20),
        "rolling_24h_std": [0.1] * 20,
        "volume_zscore":   [0.5] * 20,
        "pos":             [0.6] * 20,
        "neg":             [0.2] * 20,
        "neu":             [0.2] * 20,
    })


def _ref_and_cur():
    ref = pd.DataFrame({c: [float(i) for i in range(10)] for c in FEATURE_COLS})
    cur = pd.DataFrame({c: [float(i) for i in range(5)]  for c in FEATURE_COLS})
    return ref, cur


# ── tests: data splitting logic ───────────────────────────────────────────────

def test_reference_plus_current_covers_all_rows(full_df):
    cutoff = full_df["window_start"].max() - pd.Timedelta(days=DR.REFERENCE_DAYS)
    ref = full_df[full_df["window_start"] <= cutoff][FEATURE_COLS].dropna()
    cur = full_df[full_df["window_start"] >  cutoff][FEATURE_COLS].dropna()
    assert len(ref) + len(cur) == len(full_df)


def test_current_window_is_most_recent_days(full_df):
    cutoff = full_df["window_start"].max() - pd.Timedelta(days=DR.REFERENCE_DAYS)
    cur = full_df[full_df["window_start"] > cutoff]
    assert len(cur) == DR.REFERENCE_DAYS


def test_reference_is_older_than_current(full_df):
    cutoff = full_df["window_start"].max() - pd.Timedelta(days=DR.REFERENCE_DAYS)
    ref_max = full_df[full_df["window_start"] <= cutoff]["window_start"].max()
    cur_min = full_df[full_df["window_start"] >  cutoff]["window_start"].min()
    assert ref_max < cur_min


# ── tests: trigger_retrain ────────────────────────────────────────────────────

def test_trigger_retrain_calls_subprocess_run():
    with patch.object(DR, "subprocess") as mock_sub:
        DR.trigger_retrain()
    mock_sub.run.assert_called_once()


def test_trigger_retrain_invokes_train_script():
    with patch.object(DR, "subprocess") as mock_sub:
        DR.trigger_retrain()
    args = mock_sub.run.call_args[0][0]
    assert "modeling/train.py" in args


def test_trigger_retrain_uses_check_true():
    with patch.object(DR, "subprocess") as mock_sub:
        DR.trigger_retrain()
    _, kwargs = mock_sub.run.call_args
    assert kwargs.get("check") is True


# ── tests: main — threshold logic ─────────────────────────────────────────────

def test_main_does_not_retrain_below_threshold():
    ref, cur = _ref_and_cur()
    with patch.object(DR, "load_data", return_value=(ref, cur)), \
         patch.object(DR, "compute_drift", return_value=DR.DRIFT_THRESHOLD - 0.01), \
         patch.object(DR, "trigger_retrain") as mock_retrain, \
         patch.object(DR, "Report"), \
         patch("os.makedirs"):
        DR.main()
    mock_retrain.assert_not_called()


def test_main_triggers_retrain_above_threshold():
    ref, cur = _ref_and_cur()
    with patch.object(DR, "load_data", return_value=(ref, cur)), \
         patch.object(DR, "compute_drift", return_value=DR.DRIFT_THRESHOLD + 0.01), \
         patch.object(DR, "trigger_retrain") as mock_retrain, \
         patch.object(DR, "Report"), \
         patch("os.makedirs"):
        DR.main()
    mock_retrain.assert_called_once()


def test_main_at_exact_threshold_does_not_retrain():
    # drift_share > DRIFT_THRESHOLD (strict greater-than), so exact match → no retrain
    ref, cur = _ref_and_cur()
    with patch.object(DR, "load_data", return_value=(ref, cur)), \
         patch.object(DR, "compute_drift", return_value=DR.DRIFT_THRESHOLD), \
         patch.object(DR, "trigger_retrain") as mock_retrain, \
         patch.object(DR, "Report"), \
         patch("os.makedirs"):
        DR.main()
    mock_retrain.assert_not_called()


def test_main_empty_reference_returns_early():
    ref = pd.DataFrame(columns=FEATURE_COLS)
    cur = pd.DataFrame({c: [1.0] for c in FEATURE_COLS})
    with patch.object(DR, "load_data", return_value=(ref, cur)), \
         patch.object(DR, "trigger_retrain") as mock_retrain:
        DR.main()
    mock_retrain.assert_not_called()


def test_main_empty_current_returns_early():
    ref = pd.DataFrame({c: [1.0, 2.0] for c in FEATURE_COLS})
    cur = pd.DataFrame(columns=FEATURE_COLS)
    with patch.object(DR, "load_data", return_value=(ref, cur)), \
         patch.object(DR, "trigger_retrain") as mock_retrain:
        DR.main()
    mock_retrain.assert_not_called()


# ── tests: DRIFT_THRESHOLD constant ──────────────────────────────────────────

def test_drift_threshold_is_float():
    assert isinstance(DR.DRIFT_THRESHOLD, float)


def test_drift_threshold_is_between_0_and_1():
    assert 0.0 < DR.DRIFT_THRESHOLD < 1.0
