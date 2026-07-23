FROM apache/airflow:2.9.1

# ── system build deps ──────────────────────────────────────────────────────────
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# ── light deps (fast, ~200 MB) ─────────────────────────────────────────────────
# Separate layer so Docker doesn't re-download torch when these change.
RUN pip install --no-cache-dir \
    duckdb==0.10.3 \
    pandas==2.2.2 \
    pyarrow==15.0.2 \
    numpy==1.26.4 \
    lightgbm==4.3.0 \
    "mlflow==2.13.0" \
    scikit-learn==1.5.0 \
    evidently==0.4.30 \
    python-dotenv==1.0.1 \
    transformers==4.41.2 \
    sentencepiece==0.2.0 \
    huggingface-hub==0.23.0

# ── torch CPU (~800 MB, separate layer so it stays cached) ────────────────────
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── PyTorch Geometric + sparse backends ───────────────────────────────────────
RUN pip install --no-cache-dir \
    torch_geometric==2.5.3 \
    torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
