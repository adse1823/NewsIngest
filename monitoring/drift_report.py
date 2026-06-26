import os
import logging
import subprocess
import duckdb
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./data/feature_store.duckdb"
DRIFT_THRESHOLD = 0.30
FEATURE_COLS = ["rolling_1h_mean", "rolling_24h_std", "volume_zscore", "pos", "neg", "neu"]
REFERENCE_DAYS = 30


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT f.window_start, f.rolling_1h_mean, f.rolling_24h_std, f.volume_zscore,
               s.pos, s.neg, s.neu
        FROM features_sentiment f
        LEFT JOIN sentiment_scores s ON f.ticker = s.ticker AND f.window_start = s.window_start
        WHERE f.rolling_1h_mean IS NOT NULL
        ORDER BY f.window_start
    """).df()
    con.close()

    cutoff = df["window_start"].max() - pd.Timedelta(days=REFERENCE_DAYS)
    reference = df[df["window_start"] <= cutoff][FEATURE_COLS].dropna()
    current = df[df["window_start"] > cutoff][FEATURE_COLS].dropna()
    return reference, current


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    return drift_share


def trigger_retrain():
    log.info("Drift threshold exceeded — triggering retrain.")
    subprocess.run(["python", "modeling/train.py"], check=True)


def main():
    reference, current = load_data()

    if reference.empty or current.empty:
        log.warning("Insufficient data for drift check (reference=%d, current=%d rows).",
                    len(reference), len(current))
        return

    drift_share = compute_drift(reference, current)
    log.info("Drift share: %.3f (threshold: %.2f)", drift_share, DRIFT_THRESHOLD)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    os.makedirs("./data", exist_ok=True)
    report.save_html("./data/drift_report.html")
    log.info("Drift report saved to ./data/drift_report.html")

    if drift_share > DRIFT_THRESHOLD:
        trigger_retrain()
    else:
        log.info("No retrain needed.")


if __name__ == "__main__":
    main()
