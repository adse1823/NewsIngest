import os
import logging
import mlflow
import mlflow.lightgbm
import duckdb
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./data/feature_store.duckdb"
GNN_PATH = "./data/gnn_embeddings.parquet"


def load_production_model():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_uri = "models:/fin-platform-lgbm/Production"
    return mlflow.lightgbm.load_model(model_uri)


def load_features() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    df = con.execute("""
        SELECT f.*, s.pos, s.neg, s.neu
        FROM features_sentiment f
        LEFT JOIN sentiment_scores s ON f.ticker = s.ticker AND f.window_start = s.window_start
        WHERE f.rolling_1h_mean IS NOT NULL
        ORDER BY f.ticker, f.window_start
    """).df()
    con.close()
    gnn = pd.read_parquet(GNN_PATH)
    df = df.merge(gnn, on="ticker", how="left")
    df["label"] = (df.groupby("ticker")["pct_change"].shift(-1) > 0).astype(int)
    return df.dropna(subset=["label"])


def main():
    model = load_production_model()
    df = load_features()

    feature_cols = (
        ["rolling_1h_mean", "rolling_24h_std", "volume_zscore", "pos", "neg", "neu"]
        + [c for c in df.columns if c.startswith("gnn_dim_")]
    )
    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    preds = model.predict(X)
    probs = model.predict(X, raw_score=False)

    auc = roc_auc_score(y, probs)
    log.info("ROC-AUC: %.4f", auc)
    print(classification_report(y, preds))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:200])

    plt.figure()
    shap.summary_plot(shap_values, X[:200], feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig("./data/shap_summary.png", dpi=150)
    log.info("SHAP summary saved to ./data/shap_summary.png")


if __name__ == "__main__":
    main()
