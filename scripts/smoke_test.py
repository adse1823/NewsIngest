"""
End-to-end smoke test for the Financial Intelligence Platform.

Verifies that data flows through every stage of the pipeline:
  SQLite data → feature export → model train → serving API health

Does NOT require Redpanda/Spark to be running — tests the SQLite-only path,
which is what run_pipeline.py uses locally. Spin up Docker first for the
full Kafka + Spark path.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-train   # fastest re-check (reuse model)
"""

import argparse
import os
import subprocess
import sys
import time

WIDTH = 50
PYTHON = sys.executable
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bar(done: int, total: int) -> str:
    filled = int(WIDTH * done / total)
    return f"[{'█' * filled}{'░' * (WIDTH - filled)}] {int(100 * done / total):3d}%"


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n  ▶  {label}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"     ✓  {label}  ({elapsed:.1f}s)")
        return True
    print(f"     ✗  {label}  ({elapsed:.1f}s)")
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines()[-10:]:
            print(f"        | {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines()[-10:]:
            print(f"        ! {line}")
    return False


def _check_file(label: str, path: str) -> bool:
    full = os.path.join(ROOT, path)
    exists = os.path.isfile(full)
    size   = os.path.getsize(full) if exists else 0
    if exists and size > 0:
        print(f"     ✓  {label}  ({size:,} bytes)")
        return True
    print(f"     ✗  {label}  — {'missing' if not exists else 'empty'}: {path}")
    return False


def _check_sqlite_rows(label: str, table: str, min_rows: int = 1) -> bool:
    import sqlite3
    db = os.path.join(ROOT, "data", "raw.db")
    if not os.path.isfile(db):
        print(f"     ✗  {label}  — raw.db not found")
        return False
    con = sqlite3.connect(db)
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.close()
    if count >= min_rows:
        print(f"     ✓  {label}  ({count:,} rows in {table})")
        return True
    print(f"     ✗  {label}  — {count} rows in {table} (need ≥ {min_rows})")
    return False


def _check_parquet_rows(label: str, path: str, min_rows: int = 1) -> bool:
    import pandas as pd
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        print(f"     ✗  {label}  — missing: {path}")
        return False
    df = pd.read_parquet(full)
    if len(df) >= min_rows:
        print(f"     ✓  {label}  ({len(df):,} rows)")
        return True
    print(f"     ✗  {label}  — {len(df)} rows (need ≥ {min_rows})")
    return False


def _check_spark_parquet(label: str) -> bool:
    import glob
    news_files  = glob.glob(os.path.join(ROOT, "data", "windowed", "news",   "**", "*.parquet"), recursive=True)
    price_files = glob.glob(os.path.join(ROOT, "data", "windowed", "prices", "**", "*.parquet"), recursive=True)
    if news_files and price_files:
        print(f"     ✓  {label}  ({len(news_files)} news files, {len(price_files)} price files)")
        return True
    missing = []
    if not news_files:
        missing.append("data/windowed/news/")
    if not price_files:
        missing.append("data/windowed/prices/")
    print(f"     ~  {label}  — no Spark parquet yet ({', '.join(missing)}) — SQLite-only path will run")
    return True  # not a hard failure; export.py gracefully falls back


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-train", action="store_true", help="Skip model training (reuse existing model)")
    args = parser.parse_args()

    checks = []
    passed = 0

    print()
    print("=" * 70)
    print("  Financial Intelligence Platform — End-to-End Smoke Test")
    print("=" * 70)

    # ── Stage 1: Raw data ──────────────────────────────────────────────────
    print("\n── Stage 1: Raw data (SQLite) ──")
    r = _check_sqlite_rows("news_raw has data",   "news_raw",   min_rows=10)
    checks.append(r); passed += r
    r = _check_sqlite_rows("price_ticks has data", "price_ticks", min_rows=10)
    checks.append(r); passed += r

    # ── Stage 2: Spark parquet (optional) ─────────────────────────────────
    print("\n── Stage 2: Spark streaming output (optional) ──")
    r = _check_spark_parquet("Spark windowed parquet")
    checks.append(r); passed += r

    # ── Stage 3: Feature export ────────────────────────────────────────────
    print("\n── Stage 3: Feature export ──")
    r = _run("feature_store/export.py", [PYTHON, "feature_store/export.py"])
    checks.append(r); passed += r
    if r:
        r2 = _check_parquet_rows("features_export.parquet", "data/features_export.parquet")
        checks.append(r2); passed += r2

    # ── Stage 4: Sentiment + embeddings ───────────────────────────────────
    print("\n── Stage 4: NLP (sentiment + embeddings) ──")
    r = _run("nlp/sentiment.py", [PYTHON, "nlp/sentiment.py"])
    checks.append(r); passed += r
    r = _run("nlp/embeddings.py", [PYTHON, "nlp/embeddings.py"])
    checks.append(r); passed += r
    if r:
        r2 = _check_file("headline_embeddings.npy", "data/headline_embeddings.npy")
        checks.append(r2); passed += r2

    # ── Stage 5: Graph ────────────────────────────────────────────────────
    print("\n── Stage 5: Knowledge graph ──")
    r = _run("graph/build_graph.py",   [PYTHON, "graph/build_graph.py"])
    checks.append(r); passed += r
    r = _run("graph/train_gnn.py",     [PYTHON, "graph/train_gnn.py"])
    checks.append(r); passed += r
    if r:
        r2 = _check_file("gnn_embeddings.parquet", "data/gnn_embeddings.parquet")
        checks.append(r2); passed += r2

    # ── Stage 6: Model training ────────────────────────────────────────────
    if not args.skip_train:
        print("\n── Stage 6: Model training ──")
        r = _run("modeling/train.py", [PYTHON, "modeling/train.py"])
        checks.append(r); passed += r

    # ── Stage 7: Serving API health ────────────────────────────────────────
    print("\n── Stage 7: Serving API ──")
    r = _run("serving/main.py import check", [PYTHON, "-c", "import serving.main; print('ok')"])
    checks.append(r); passed += r

    # ── Summary ────────────────────────────────────────────────────────────
    total = len(checks)
    print()
    print("=" * 70)
    print(f"  {_bar(passed, total)}")
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print()
        print("  PIPELINE OK — all stages passed.")
        print()
        print("  To run the full live pipeline:")
        print("    python run_pipeline.py --skip-backfill --skip-sentiment --skip-embeddings")
        print("  To start the serving API:")
        print("    uvicorn serving.main:app --port 8000")
        print("  To view the dashboard:")
        print("    venv\\Scripts\\python.exe -m streamlit run monitoring/dashboard.py")
    else:
        print()
        print("  PIPELINE FAILURES — fix the ✗ stages above and re-run.")
        print("  Tip: stages are independent; fix each one individually.")
    print("=" * 70)
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
