# Layer 8 — Hybrid Forecasting Model: LightGBM + MLflow

## What This Layer Does

Combines tabular features from the feature store with GNN embeddings, trains a binary classifier to predict price direction (up/down in the next hour), and tracks every experiment in MLflow for versioning and reproducibility.

---

## Why "Hybrid"

The model takes two fundamentally different types of input and combines them:

```
┌─────────────────────────────┐     ┌────────────────────────────────┐
│   TABULAR FEATURES (5)      │     │   GNN EMBEDDINGS (64)          │
│                             │     │                                │
│  rolling_1h_mean            │     │  Encode company position       │
│  rolling_24h_std            │     │  in the co-occurrence graph.   │
│  volume_zscore              │     │  Captures sector-level and     │
│  pos (FinBERT)              │     │  cross-company news signals.   │
│  neg (FinBERT)              │     │                                │
│  neu (FinBERT)              │     │  gnn_dim_0 ... gnn_dim_63      │
└─────────────────────────────┘     └────────────────────────────────┘
                │                                   │
                └─────────────┬─────────────────────┘
                              │  CONCAT
                              ▼
                     Feature vector: (69,)
                              │
                              ▼
                         LightGBM
                       binary classifier
                              │
                              ▼
                    P(price goes UP) in next hour
```

---

## Feature Matrix Assembly

```
modeling/train.py

features_export.parquet (from DuckDB):
  ticker │ window_start │ rolling_1h_mean │ rolling_24h_std │ volume_zscore │ pos │ neg │ neu
  ───────┼──────────────┼─────────────────┼─────────────────┼───────────────┼─────┼─────┼────
  AAPL   │ 2024-07-14   │      2.0        │      0.45       │     1.82      │0.82 │0.05 │0.13

gnn_embeddings.parquet:
  ticker │ gnn_dim_0 │ gnn_dim_1 │ ... │ gnn_dim_63
  ───────┼───────────┼───────────┼─────┼───────────
  AAPL   │   0.12    │  -0.05    │ ... │   -0.22

JOIN on ticker → final feature matrix (n_samples × 69)

Label: price_change_next_1h > 0 → label = 1 (UP)
       price_change_next_1h ≤ 0 → label = 0 (DOWN)
```

---

## TimeSeriesSplit: Why It Matters

```
WRONG: Standard K-Fold (causes data leakage)

  Fold 1:  [Jan Feb Mar Apr | May Jun]   ← trains on Apr, validates on Jan
  Fold 2:  [Jan Feb May Jun | Mar Apr]   ← trains on future, validates on past

  LEAKAGE: the model sees future data during training
  → ROC-AUC looks great on CV, terrible in production

RIGHT: TimeSeriesSplit (respects time order)

  Train window always ends BEFORE validation window starts.

  Fold 1:  ████████░░░░░░░░░░░░░░░░░░
           Jan-Feb  validate: Mar-Apr

  Fold 2:  ████████████░░░░░░░░░░░░░░
           Jan-Apr  validate: May-Jun

  Fold 3:  ████████████████░░░░░░░░░░
           Jan-Jun  validate: Jul-Aug

  Fold 4:  ████████████████████░░░░░░
           Jan-Aug  validate: Sep-Oct

  Fold 5:  ████████████████████████░░
           Jan-Oct  validate: Nov-Dec

  Future data NEVER appears in a training fold.
  CV score reflects real-world performance.
```

---

## Why LightGBM and Not a Neural Net

```
Options considered:

  MLP (neural net):
  ✗ Needs larger datasets to beat trees on tabular data
  ✗ Slower to train and tune
  ✗ Black box — hard to explain predictions
  ✗ No native SHAP support

  XGBoost:
  ✓ Good option, similar performance
  ✗ Slower than LightGBM on large datasets
  ✗ Leaf-wise vs depth-wise growth (LightGBM usually wins)

  LightGBM:
  ✓ Fastest gradient boosting on tabular data
  ✓ Handles mixed feature types well
  ✓ Native TreeExplainer support → SHAP values for free
  ✓ Works well with GNN embedding inputs (64 floats per row)
  ✓ TimeSeriesSplit + early stopping prevents overfitting
```

---

## MLflow Tracking

Every training run logs:

```
MLflow run:
  ┌────────────────────────────────────────────────────────────┐
  │  Run ID: a3f9c2b1...                                       │
  │                                                            │
  │  PARAMETERS                                                │
  │    n_estimators: 300                                       │
  │    max_depth: 6                                            │
  │    learning_rate: 0.05                                     │
  │    num_leaves: 31                                          │
  │                                                            │
  │  METRICS                                                   │
  │    fold_1_auc: 0.651                                       │
  │    fold_2_auc: 0.678                                       │
  │    fold_3_auc: 0.681                                       │
  │    fold_4_auc: 0.669                                       │
  │    fold_5_auc: 0.682                                       │
  │    mean_auc: 0.6722                                        │
  │                                                            │
  │  ARTIFACTS                                                 │
  │    model.pkl (serialized LightGBM)                         │
  │    feature_importance.png                                  │
  │    shap_summary.png                                        │
  │                                                            │
  │  TAGS                                                      │
  │    run_type: production                                     │
  └────────────────────────────────────────────────────────────┘
```

---

## Model Registry and Champion Promotion

```
MLflow Model Registry:

  Version 1  (trained 2024-07-01)  mean_auc: 0.651  [Archived]
  Version 2  (trained 2024-07-07)  mean_auc: 0.662  [Archived]
  Version 3  (trained 2024-07-14)  mean_auc: 0.672  [Production ← champion]

  Alias: "champion" → points to Version 3

  scripts/set_champion.py promotes a specific version:
    python set_champion.py --version 3

  FastAPI serving loads at startup:
    client.get_model_version_by_alias("FinPlatform", "champion")
    → loads Version 3 automatically
```

The "champion" alias is the handoff point between training and serving. No hardcoded version numbers anywhere in the serving code.

---

## SHAP: Feature Importance

```
Top features by SHAP importance (mean |SHAP value| over test set):

  Rank  Feature                Mean |SHAP|
  ────  ───────────────────────────────────
  1     gnn_dim_14             0.031   ← sector-level co-occurrence signal
  2     gnn_dim_31             0.029
  3     gnn_dim_7              0.027
  4     pos (FinBERT)          0.024   ← direct sentiment
  5     neg (FinBERT)          0.021
  6     volume_zscore          0.018   ← abnormal volume precedes moves
  7     rolling_1h_mean        0.009   ← low variance (usually 1.0)
  ...
  69    gnn_dim_52             0.002

GNN dimensions dominate → cross-company signal is the strongest predictor.
Sentiment scores second → direct news sentiment matters.
Volume z-score third → abnormal activity precedes moves.
```

---

## Results

| Metric | Value |
|--------|-------|
| Mean ROC-AUC (5-fold TimeSeriesSplit) | 0.6721 |
| Naive baseline (predict majority class) | 0.50 |
| Training samples | 28,267 news articles |
| Price rows | 145,214 |
| Tickers | 502 |

0.67 AUC means the model correctly ranks a positive sample above a negative sample 67% of the time. Not alpha-generating on its own, but a valid proof of concept — and significantly above random.

---

## Files in This Layer

| File | Role |
|------|------|
| [modeling/train.py](../../modeling/train.py) | Feature assembly, TimeSeriesSplit, LightGBM, MLflow logging, champion promotion |
| [modeling/evaluate.py](../../modeling/evaluate.py) | ROC-AUC, feature importance plots |
| [scripts/set_champion.py](../../scripts/set_champion.py) | Manually promote a model version to champion alias |
