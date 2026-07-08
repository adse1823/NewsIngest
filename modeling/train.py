import os
import logging
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./data/feature_store.duckdb"
GNN_PATH = "./data/gnn_embeddings.parquet"
N_SPLITS = 2
MLFLOW_EXPERIMENT = "fin-platform"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


def load_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT
            f.window_start,
            f.ticker,
            f.rolling_1h_mean,
            f.rolling_24h_std,
            f.volume_zscore,
            s.pos,
            s.neg,
            s.neu,
            f.pct_change
        FROM features_sentiment f
        LEFT JOIN sentiment_scores s
            ON f.ticker = s.ticker AND f.window_start = s.window_start
        WHERE f.rolling_1h_mean IS NOT NULL
        ORDER BY f.ticker, f.window_start
    """).df()
    con.close()
    return df


def attach_gnn_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    gnn = pd.read_parquet(GNN_PATH)
    return df.merge(gnn, on="ticker", how="left")


def create_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = (df.groupby("ticker")["pct_change"].shift(-1) > 0).astype(int)
    return df.dropna(subset=["label"])


def main():
    mlflow.set_tracking_uri(Path("./mlruns").resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_features()
    df = attach_gnn_embeddings(df)
    df = create_label(df)

    feature_cols = (
        ["rolling_1h_mean", "rolling_24h_std", "volume_zscore", "pos", "neg", "neu"]
        + [c for c in df.columns if c.startswith("gnn_dim_")]
    )
    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    auc_scores = []

    with mlflow.start_run(run_name="lightgbm-timeseries-cv"):
        mlflow.log_params(LGB_PARAMS)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("n_features", len(feature_cols))

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if len(np.unique(y_val)) < 2:
                log.warning("Fold %d skipped — only one class in validation set", fold)
                continue

            model = lgb.LGBMClassifier(**LGB_PARAMS)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            preds = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, preds)
            auc_scores.append(auc)
            mlflow.log_metric(f"fold_{fold}_auc", auc)
            log.info("Fold %d AUC: %.4f", fold, auc)

        if not auc_scores:
            log.warning("No valid folds — skipping AUC logging (need more data for meaningful CV)")
            mean_auc = 0.0
        else:
            mean_auc = float(np.mean(auc_scores))
            log.info("Mean AUC across %d folds: %.4f", len(auc_scores), mean_auc)
        mlflow.log_metric("mean_auc", mean_auc)

        final_model = lgb.LGBMClassifier(**LGB_PARAMS)
        final_model.fit(X, y)

        mlflow.lightgbm.log_model(
            final_model.booster_,
            artifact_path="model",
            registered_model_name="fin-platform-lgbm",
        )

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions("name='fin-platform-lgbm'")
        latest_version = max(versions, key=lambda v: int(v.version)).version
        client.set_registered_model_alias(
            name="fin-platform-lgbm",
            alias="champion",
            version=latest_version,
        )
        log.info("Set alias 'champion' on model version %s", latest_version)

        log.info("Training complete. Mean AUC: %.4f", mean_auc)


if __name__ == "__main__":
    main()
