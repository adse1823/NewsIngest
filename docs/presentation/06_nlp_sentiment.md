# Layer 6 — NLP Sentiment Pipeline: FinBERT

## What This Layer Does

Runs every headline through a financial-domain BERT model to produce:
1. Three sentiment probabilities per headline (positive / negative / neutral)
2. A 768-dimensional embedding per headline — a dense numeric representation of its meaning

Both outputs feed into the hybrid model (Layer 8). The embeddings also provide node features for the GNN (Layer 7).

---

## Why Sentiment Signals Matter

```
Raw price model (no news):

  Input: rolling_1h_mean, rolling_24h_std, volume_zscore
  Predicts: up / down
  ROC-AUC: ~0.55

With sentiment signals added:

  Input: rolling_1h_mean, rolling_24h_std, volume_zscore,
         pos, neg, neu
  Predicts: up / down
  ROC-AUC: 0.67

The news headline is carrying causal information about price direction.
```

---

## What FinBERT Is

```
Generic BERT (trained on Wikipedia + BookCorpus):
  knows: "positive" = happy, "beat" = defeat, "miss" = absent

FinBERT (fine-tuned on financial text + earnings calls):
  knows: "beat expectations" = POSITIVE (earnings beat)
         "missed guidance"   = NEGATIVE (guidance miss)
         "flat"              = NEUTRAL  (no move)
         "raised outlook"    = POSITIVE (upward revision)
         "impairment charge" = NEGATIVE (write-down)

Domain-specific fine-tuning is critical. Generic BERT gets these wrong.
```

---

## Architecture: BERT Internals (simplified)

```
Input headline: "Apple beats Q3 earnings estimates"

Tokenization:
  [CLS] Apple beats Q3 earnings estimates [SEP]
     ↓     ↓     ↓    ↓    ↓        ↓       ↓

Token embeddings (768-dim each):
  ┌───┐ ┌─────┐ ┌─────┐ ┌────┐ ┌────────┐ ┌─────────┐ ┌───┐
  │CLS│ │Apple│ │beats│ │ Q3 │ │earnings│ │estimates│ │SEP│
  └───┘ └─────┘ └─────┘ └────┘ └────────┘ └─────────┘ └───┘
    ↓       ↓       ↓      ↓        ↓           ↓        ↓

12 Transformer layers (self-attention):
  Each token attends to every other token
  "beats" now understands it relates to "earnings" and "estimates"
  "Apple" now understands it's the subject of a positive event

  ↓

Last hidden state: 768-dim vector per token

                 ┌─── used for sentiment classification
  [CLS] vector ──┤
                 └─── (pooled with others for embeddings)

  Mean pool over all tokens ──► 768-dim headline embedding
```

---

## Sentiment Classification Output

```
Input headline → FinBERT → softmax over 3 classes

  "Apple beats Q3 earnings estimates"
                          │
                          ▼
                    ┌───────────────┐
                    │  pos:  0.82   │ ← 82% confident this is positive news
                    │  neg:  0.05   │
                    │  neu:  0.13   │
                    └───────────────┘

  "Fed signals further rate hikes amid inflation fears"
                          │
                          ▼
                    ┌───────────────┐
                    │  pos:  0.04   │
                    │  neg:  0.79   │ ← 79% confident this is negative news
                    │  neu:  0.17   │
                    └───────────────┘

All three sum to 1.0 (softmax output).
```

---

## Batch Inference Flow

```
nlp/sentiment.py:

All headlines in DuckDB
         │
         ▼
  Load in batches of 8        ← batch_size=8 keeps peak RAM under 6 GB on CPU
         │
         ▼
  Tokenize (HuggingFace tokenizer)
         │
         ▼
  FinBERT forward pass
         │
         ▼
  Softmax → (pos, neg, neu)
         │
         ▼
  Write back to DuckDB
  sentiment_scores table:
  ┌─────────┬──────────┬──────┬──────┬──────┐
  │ article │  ticker  │ pos  │ neg  │ neu  │
  ├─────────┼──────────┼──────┼──────┼──────┤
  │  1042   │  AAPL    │ 0.82 │ 0.05 │ 0.13 │
  │  1043   │  FED     │ 0.04 │ 0.79 │ 0.17 │
  └─────────┴──────────┴──────┴──────┴──────┘
```

---

## Embedding Generation Flow

```
nlp/embeddings.py:

All headlines
         │
         ▼
  FinBERT forward pass
         │
         ▼
  Last hidden state: shape (batch, seq_len, 768)
         │
         ▼
  Mean pool over seq_len dim: shape (batch, 768)
         │                        ↑
         │               one 768-dim vector
         │               per headline
         ▼
  Aggregate per ticker per day:
  mean of all headline embeddings for AAPL on 2024-07-14
         │
         ▼
  headline_embeddings.npy
  shape: (n_tickers, 768)

  Used as node features in the GNN (Layer 7)
```

---

## Why 768 Dimensions

BERT-base produces 768-dim hidden states — this is the model architecture, not a design choice. Each dimension encodes some learned aspect of the text's meaning. After mean-pooling, the 768-dim vector is a semantic fingerprint of the headline:
- Headlines about the same topic cluster together
- Positive and negative headlines are separated in the space
- Domain-specific financial concepts have learned representations

---

## FinBERT vs Alternatives

| Model | Domain | Size | Cost to run | Quality on finance |
|-------|--------|------|-------------|-------------------|
| **FinBERT** | Financial | 110M params | CPU-feasible | HIGH (fine-tuned) |
| BERT-base | General | 110M params | CPU-feasible | MEDIUM |
| RoBERTa | General | 125M params | CPU-feasible | MEDIUM |
| GPT-4 (API) | General | Unknown | $$$ per call | HIGH (few-shot) |
| DistilBERT | General | 66M params | Very fast | LOWER |

**Decision:** FinBERT gives the best quality/cost tradeoff for financial text classification. GPT-4 would work but costs $$ per inference call and requires internet access in production.

---

## Files in This Layer

| File | Role |
|------|------|
| [nlp/sentiment.py](../../nlp/sentiment.py) | FinBERT batch inference → DuckDB sentiment_scores |
| [nlp/embeddings.py](../../nlp/embeddings.py) | Mean-pool embeddings → headline_embeddings.npy |
