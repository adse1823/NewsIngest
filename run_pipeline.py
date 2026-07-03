"""
Full pipeline runner — runs every Phase 1 step in order.

Usage:
    python run_pipeline.py                        # full pipeline including backfill
    python run_pipeline.py --skip-backfill        # skip news fetch (data already fetched today)
    python run_pipeline.py --skip-backfill --skip-sentiment --skip-embeddings
                                                  # jump straight to graph + model (fastest re-run)

Saved state on disk (these persist between runs — skip flags let you reuse them):
    ./data/feature_store.duckdb    sentiment_scores table
    ./data/headline_embeddings.npy FinBERT embeddings per ticker
    ./data/graph.pt                co-occurrence graph
    ./data/gnn_embeddings.parquet  GNN node embeddings
    ./mlruns/                      MLflow model registry (no server needed)

FinBERT weights are cached by HuggingFace at:
    C:/Users/<you>/.cache/huggingface/   (never re-downloaded)
"""

import sys
import subprocess
import argparse
import time
from datetime import datetime


WIDTH = 50


def bar(done: int, total: int) -> str:
    filled = int(WIDTH * done / total)
    pct = int(100 * done / total)
    return f"[{'█' * filled}{'░' * (WIDTH - filled)}] {pct:3d}%"


def header(step: int, total: int, name: str):
    print()
    print("-" * 70)
    print(f"  STEP {step}/{total}  {name}")
    print(f"  Overall progress: {bar(step - 1, total)}")
    print("-" * 70)


def footer(step: int, total: int, name: str, elapsed: float, ok: bool):
    status = "DONE" if ok else "FAILED"
    print(f"\n  [{status}]  {name}  ({elapsed:.1f}s)")
    print(f"  Overall progress: {bar(step, total)}")


def run(cmd: list[str], step: int, total: int, name: str) -> bool:
    header(step, total, name)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print("  | " + line, end="")
    proc.wait()
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    footer(step, total, name, elapsed, ok)
    return ok


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    py = sys.executable
    steps = []

    if not args.skip_backfill:
        steps.append(("News backfill (last 30 days)",          [py, "ingestion/backfill_news.py"]))
        steps.append(("Price backfill (1 month 5-min bars)",   [py, "ingestion/backfill_prices.py"]))

    if not args.skip_features:
        steps.append(("Feature export (DuckDB windowing)", [py, "feature_store/export.py"]))

    if not args.skip_sentiment:
        steps.append(("Sentiment scoring (FinBERT)  [slow ~3-5 min]", [py, "nlp/sentiment.py"]))

    if not args.skip_embeddings:
        steps.append(("Embeddings (FinBERT mean-pool) [slow ~1-2 min]", [py, "nlp/embeddings.py"]))

    if not args.skip_graph:
        steps.append(("Build co-occurrence graph", [py, "graph/build_graph.py"]))

    if not args.skip_gnn:
        steps.append(("Train GNN (GraphSAGE 100 epochs)", [py, "graph/train_gnn.py"]))

    if not args.skip_model:
        steps.append(("Train forecasting model (LightGBM)", [py, "modeling/train.py"]))

    return steps


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_pipeline.py                              full run (first time)
  python run_pipeline.py --skip-backfill              reuse today's news fetch
  python run_pipeline.py --skip-backfill \\
      --skip-sentiment --skip-embeddings              reuse FinBERT results, retrain model only
  python run_pipeline.py --skip-backfill \\
      --skip-sentiment --skip-embeddings \\
      --skip-graph --skip-gnn                         retrain model only (fastest)
        """
    )
    parser.add_argument("--skip-backfill",  action="store_true", help="Skip NewsAPI fetch")
    parser.add_argument("--skip-features",  action="store_true", help="Skip DuckDB feature export")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip FinBERT sentiment scoring (reuse saved scores)")
    parser.add_argument("--skip-embeddings",action="store_true", help="Skip FinBERT embeddings (reuse saved .npy)")
    parser.add_argument("--skip-graph",     action="store_true", help="Skip graph build (reuse saved graph.pt)")
    parser.add_argument("--skip-gnn",       action="store_true", help="Skip GNN training (reuse saved gnn_embeddings.parquet)")
    parser.add_argument("--skip-model",     action="store_true", help="Skip LightGBM training")
    args = parser.parse_args()

    steps = build_steps(args)
    if not steps:
        print("All steps skipped — nothing to do.")
        sys.exit(0)

    total = len(steps)

    print()
    print("=" * 70)
    print("  Financial Intelligence Platform  --  Pipeline Runner")
    print(f"  {total} steps  |  started {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    print(f"\n  {bar(0, total)}  starting...\n")

    failed_at = None
    t_start = time.time()

    for i, (name, cmd) in enumerate(steps, start=1):
        ok = run(cmd, i, total, name)
        if not ok:
            failed_at = name
            break

    wall = time.time() - t_start
    print()
    print("=" * 70)
    if failed_at:
        print(f"  PIPELINE FAILED at: {failed_at}")
        print("  Fix the error shown above and re-run.")
        print("  Tip: use --skip-* flags to jump past steps that already succeeded.")
    else:
        print("  PIPELINE COMPLETE")
        print(f"  {bar(total, total)}")
        print(f"  Total time: {wall:.0f}s")
        print()
        print("  Next:")
        print("    streamlit run monitoring/dashboard.py   <- view updated dashboard")
        print("    uvicorn serving.main:app --port 8000    <- start prediction API")
    print("=" * 70)
    print()

    sys.exit(0 if not failed_at else 1)


if __name__ == "__main__":
    main()
