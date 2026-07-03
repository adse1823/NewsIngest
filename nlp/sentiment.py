import os
import logging
import numpy as np
import duckdb
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 8
DB_PATH = "./data/feature_store.duckdb"
LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}


def load_model():
    log.info("Loading FinBERT from HuggingFace (first run will download ~400 MB)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def batch_inference(tokenizer, model, texts: list[str]) -> list[dict]:
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).tolist()
        for p in probs:
            results.append({"pos": p[0], "neg": p[1], "neu": p[2]})
    return results


def main():
    con = duckdb.connect(DB_PATH)

    rows = con.execute("""
        SELECT window_start, ticker, titles
        FROM raw_news
        WHERE titles IS NOT NULL
    """).fetchdf()

    if rows.empty:
        log.warning("No rows in raw_news. Run feature_store/export.py first.")
        con.close()
        return

    tokenizer, model = load_model()

    records = []
    for _, row in rows.iterrows():
        raw = row["titles"]
        if isinstance(raw, (list, np.ndarray)) and len(raw) > 0:
            titles = list(dict.fromkeys(str(t) for t in raw))  # deduplicate, preserve order
        else:
            titles = []
        if not titles:
            records.append({
                "window_start": row["window_start"],
                "ticker": row["ticker"],
                "pos": None, "neg": None, "neu": None,
            })
            continue

        sentiments = batch_inference(tokenizer, model, [str(t) for t in titles])
        avg_pos = sum(s["pos"] for s in sentiments) / len(sentiments)
        avg_neg = sum(s["neg"] for s in sentiments) / len(sentiments)
        avg_neu = sum(s["neu"] for s in sentiments) / len(sentiments)
        records.append({
            "window_start": row["window_start"],
            "ticker": row["ticker"],
            "pos": avg_pos,
            "neg": avg_neg,
            "neu": avg_neu,
        })

    df = pd.DataFrame(records)
    con.execute("DROP TABLE IF EXISTS sentiment_scores")
    con.execute("CREATE TABLE sentiment_scores AS SELECT * FROM df")
    log.info("Wrote %d sentiment rows to DuckDB.", len(df))
    con.close()


if __name__ == "__main__":
    main()
