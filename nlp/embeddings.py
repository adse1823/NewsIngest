import os
import logging
import duckdb
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 8
DB_PATH = "./data/feature_store.duckdb"
OUTPUT_PATH = "./data/headline_embeddings.npy"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "BAC", "GS"]


def load_model():
    log.info("Loading FinBERT encoder...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed_texts(tokenizer, model, texts: list[str]) -> np.ndarray:
    all_embeds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeds = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        all_embeds.append(embeds.numpy())
    return np.vstack(all_embeds) if all_embeds else np.zeros((0, 768))


def main():
    con = duckdb.connect(DB_PATH)

    rows = con.execute("""
        SELECT ticker, titles
        FROM raw_news
        WHERE titles IS NOT NULL
    """).fetchdf()

    if rows.empty:
        log.warning("No rows in raw_news.")
        con.close()
        return

    tokenizer, model = load_model()
    ticker_embeddings = {}

    for ticker in TICKERS:
        subset = rows[rows["ticker"] == ticker]
        all_titles = []
        for titles in subset["titles"]:
            if isinstance(titles, list):
                all_titles.extend([str(t) for t in titles if t])

        if not all_titles:
            ticker_embeddings[ticker] = np.zeros(768)
            continue

        embeds = embed_texts(tokenizer, model, all_titles)
        ticker_embeddings[ticker] = embeds.mean(axis=0)
        log.info("Embedded %d titles for %s", len(all_titles), ticker)

    matrix = np.stack([ticker_embeddings[t] for t in TICKERS])
    os.makedirs("./data", exist_ok=True)
    np.save(OUTPUT_PATH, matrix)
    log.info("Saved embedding matrix %s to %s", matrix.shape, OUTPUT_PATH)

    meta = {"tickers": TICKERS, "shape": list(matrix.shape)}
    import json
    with open("./data/embedding_meta.json", "w") as f:
        json.dump(meta, f)

    con.close()


if __name__ == "__main__":
    main()
