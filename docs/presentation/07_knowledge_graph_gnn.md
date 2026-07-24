# Layer 7 — Knowledge Graph + GNN: PyTorch Geometric / GraphSAGE

## What This Layer Does

Builds a graph where companies are nodes and edges connect companies that appear together in headlines. Trains a Graph Neural Network to produce 64-dim embeddings per company that encode both the company's own news sentiment and its relationships to other companies.

---

## Why a Graph? The Problem With Treating Companies Independently

```
STANDARD TABULAR APPROACH (no graph)

  AAPL features: rolling_mean, pos, neg, neu → predict AAPL direction
  TSMC features: rolling_mean, pos, neg, neu → predict TSMC direction
  AMD  features: rolling_mean, pos, neg, neu → predict AMD direction

  Problem: these companies are interconnected.
  A chip supply shock at TSMC affects AMD, NVIDIA, Apple — all at once.
  A standard model sees each company in isolation and MISSES this signal.


GRAPH APPROACH

  AAPL ──────────── TSMC  (headline: "Apple orders TSMC 3nm chips")
   │  \             /
   │   \           /
  MSFT  \──── AMD ────── NVDA  (headline: "AMD gains share from Intel")
                 │
                INTC

  When TSMC has negative news, the GNN propagates that signal
  to AAPL, AMD, NVDA through the edges.
  Their embeddings reflect the shock even if they had no direct headlines.
```

---

## Graph Construction

```
graph/build_graph.py

Input: all (headline, tickers mentioned) pairs

Step 1 — Parse co-occurrences:
  Headline: "Apple and TSMC announce new chip partnership"
  Tickers mentioned: [AAPL, TSMC]
  → Add edge (AAPL, TSMC) with weight += 1

  Headline: "TSMC and Samsung compete for AMD orders"
  Tickers: [TSMC, SSNLF, AMD]
  → Add edge (TSMC, SSNLF) += 1
  → Add edge (TSMC, AMD)   += 1
  → Add edge (SSNLF, AMD)  += 1

Step 2 — Build adjacency:
  Co-occurrence matrix (symmetric):
         AAPL  TSMC  AMD  NVDA  INTC
  AAPL [  0     12    3    5    2  ]
  TSMC [ 12      0    8    6    4  ]
  AMD  [  3      8    0   15    9  ]
  NVDA [  5      6   15    0    7  ]
  INTC [  2      4    9    7    0  ]

Step 3 — Create PyG Data object:
  x      = node features (768-dim FinBERT embeddings per company)
  edge_index = [source_nodes, target_nodes]  ← COO format
  edge_attr  = co-occurrence weights
```

---

## Graph Structure Visualization

```
                    ┌─────────────────────────────────────────────┐
                    │           COMPANY CO-OCCURRENCE GRAPH        │
                    │                                             │
                    │     (TECH)                                  │
                    │   ┌──────┐                                  │
                    │   │ AAPL │──────────────────┐               │
                    │   └──┬───┘  12              │               │
                    │      │ 3              ┌──────┴───┐          │
                    │      │           (5)  │  TSMC    │          │
                    │   ┌──▼───┐            └──────┬───┘          │
                    │   │ AMD  │◄───────────────(8)┘              │
                    │   └──┬───┘                                  │
                    │   (15)│ (9)                                  │
                    │   ┌──▼───┐    ┌──────┐                      │
                    │   │ NVDA │────│ INTC │                      │
                    │   └──────┘ (7)└──────┘                      │
                    │                                             │
                    │   Edge weights = co-occurrence frequency    │
                    └─────────────────────────────────────────────┘
```

---

## GraphSAGE: How It Works

```
GraphSAGE = "Graph SAmple and aggreGatE"

Goal: produce a 64-dim embedding for each node that captures:
  1. The node's own features (its FinBERT embeddings)
  2. The features of its neighbors (companies it co-occurs with)

Layer 1 — Aggregate 1-hop neighbors:

  h_AAPL^(1) = σ( W · CONCAT(
                    h_AAPL^(0),                          ← own features
                    MEAN(h_TSMC^(0), h_AMD^(0), ...)    ← neighbor mean
                  ))

Layer 2 — Aggregate 2-hop neighbors:

  h_AAPL^(2) = σ( W · CONCAT(
                    h_AAPL^(1),
                    MEAN(h_TSMC^(1), h_AMD^(1), ...)
                  ))

  Now h_AAPL^(2) encodes:
    - AAPL's own headlines
    - TSMC's headlines (1-hop)
    - AMD's headlines (1-hop)
    - NVDA's headlines (2-hop, through AMD)
    → Cross-company signal propagated through graph structure
```

---

## GraphSAGE vs GAT (Graph Attention Network)

```
GraphSAGE (MEAN aggregation):
  h_v = MEAN(h_u for u in neighbors(v))
  All neighbors contribute equally.
  Fast. Generalizes well with limited data.

GAT (attention aggregation):
  h_v = Σ α_vu · h_u
  where α_vu = learned attention weight for each neighbor

  Some neighbors contribute more than others.
  More expressive. But needs more data to learn good attention weights.

With ~500 nodes and sparse co-occurrence data:
  GAT's extra parameters HURT (overfitting)
  GraphSAGE's simplicity HELPS (better generalization)

Decision: GraphSAGE
```

---

## Training Objective (Self-Supervised)

```
No labeled data needed for GNN training.

Self-supervised objective: PREDICT MASKED NODE FEATURES

  Input: graph with node features x
         randomly mask 20% of features → x_masked

  GNN forward pass:
         x_masked → GNN → h (embeddings) → linear head → x_reconstructed

  Loss: MSE(x_reconstructed, x_original)

  The GNN must learn to reconstruct missing features
  by aggregating information from neighbors.
  This forces it to learn meaningful structural representations.

After training:
  Discard the reconstruction head.
  Keep the 64-dim embeddings per node.
  Save to gnn_embeddings.parquet.
```

---

## Output: 64-dim Company Embeddings

```
gnn_embeddings.parquet:

┌─────────┬──────────────────────────────────────────────────────────────┐
│ ticker  │  gnn_embedding (64 floats)                                   │
├─────────┼──────────────────────────────────────────────────────────────┤
│  AAPL   │  [0.12, -0.05, 0.33, 0.71, ..., -0.22]                      │
│  TSMC   │  [0.14, -0.03, 0.31, 0.68, ..., -0.19]  ← similar to AAPL  │
│  AMZN   │  [-0.42, 0.18, -0.11, 0.33, ..., 0.55]  ← different sector │
└─────────┴──────────────────────────────────────────────────────────────┘

Companies with similar news profiles and network positions
have similar embeddings → the model can exploit this.
```

These 64 dimensions are concatenated with the 5 tabular features as input to LightGBM.

---

## Files in This Layer

| File | Role |
|------|------|
| [graph/build_graph.py](../../graph/build_graph.py) | Co-occurrence graph → PyG Data object |
| [graph/train_gnn.py](../../graph/train_gnn.py) | GraphSAGE training → gnn_embeddings.parquet |
