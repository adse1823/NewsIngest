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


def build_cooccurrence(titles_by_window: list[list[str]], ticker_to_idx: dict) -> dict:
    co = defaultdict(int)
    for titles in titles_by_window:
        tickers_present = set()
        for title in titles:
            for ticker, idx in ticker_to_idx.items():
                if ticker.lower() in title.lower():
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
    rows = con.execute("SELECT titles FROM raw_news WHERE titles IS NOT NULL").fetchdf()
    con.close()

    all_titles_per_window = []
    for titles in rows["titles"]:
        if isinstance(titles, list):
            all_titles_per_window.append([str(t) for t in titles if t])

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
