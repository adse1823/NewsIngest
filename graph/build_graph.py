import os
import json
import logging
import duckdb
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = "./data/feature_store.duckdb"
EMBEDDINGS_PATH = "./data/headline_embeddings.npy"
META_PATH = "./data/embedding_meta.json"
GRAPH_PATH = "./data/graph.pt"

# Headlines use company names, not ticker symbols — search for all of these
TICKER_KEYWORDS = {
    "AAPL":  ["apple", "aapl"],
    "MSFT":  ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN":  ["amazon", "amzn"],
    "NVDA":  ["nvidia", "nvda"],
    "META":  ["meta", "facebook", "instagram"],
    "TSLA":  ["tesla", "tsla", "elon musk"],
    "JPM":   ["jpmorgan", "jp morgan", "jpm", "jamie dimon"],
    "BAC":   ["bank of america", "bofa", "bac"],
    "GS":    ["goldman sachs", "goldman", "gs"],
}


def build_cooccurrence(titles_by_window: list[list[str]], ticker_to_idx: dict) -> dict:
    co = defaultdict(int)
    for titles in titles_by_window:
        tickers_present = set()
        for title in titles:
            title_lower = title.lower()
            for ticker, idx in ticker_to_idx.items():
                keywords = TICKER_KEYWORDS.get(ticker, [ticker.lower()])
                if any(kw in title_lower for kw in keywords):
                    tickers_present.add(idx)
        for a in tickers_present:
            for b in tickers_present:
                if a != b:
                    co[(min(a, b), max(a, b))] += 1
    return co


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    tickers: list[str] = meta["tickers"]
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    n_nodes = len(tickers)

    embeddings = np.load(EMBEDDINGS_PATH)
    x = torch.tensor(embeddings, dtype=torch.float)

    con = duckdb.connect(DB_PATH)

    # Flatten all titles across every ticker and window into one list
    title_rows = con.execute("""
        SELECT UNNEST(titles) AS title
        FROM raw_news
        WHERE titles IS NOT NULL
    """).fetchdf()
    con.close()

    all_titles = [str(t) for t in title_rows["title"].tolist() if t]
    log.info("Total headlines to scan for co-occurrence: %d", len(all_titles))

    # Log a few samples so we can see what the text looks like
    for sample in all_titles[:3]:
        log.info("Sample title: %s", sample)

    # Treat every title as its own window — any title mentioning 2+ companies creates an edge
    all_titles_per_window = [[t] for t in all_titles]

    co = build_cooccurrence(all_titles_per_window, ticker_to_idx)

    if co:
        edges = list(co.keys())
        weights = [co[e] for e in edges]
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(weights + weights, dtype=torch.float).unsqueeze(1)
        log.info("Built graph: %d nodes, %d edges (undirected)", n_nodes, len(edges))
    else:
        log.warning("No co-occurrences found — creating fully disconnected graph")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.tickers = tickers

    torch.save(data, GRAPH_PATH)
    log.info("Saved graph to %s", GRAPH_PATH)


if __name__ == "__main__":
    main()
