"""DuckDB-based TSV ingestion and normalisation.

Reads raw TSV files, applies SQL normalisation (matching the logic in
``normalization.py``), and writes Parquet files to the intermediate
directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

from src.config import (
    INTERMEDIATE_DIR,
    TRAIN_GT,
    TRAIN_SOURCES,
    TEST_SOURCES,
    norm_parquet,
)
from src.normalization import build_normalization_query


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ingest_source(
    con: duckdb.DuckDBPyConnection,
    tsv_path: Path,
    source_name: str,
    split: str,
) -> Path:
    """Read a single source TSV, normalise, and write Parquet.

    Returns the output Parquet path.
    """
    out_path = norm_parquet(split, source_name)
    _ensure_dir(out_path.parent)

    print(f"  Ingesting {source_name} from {tsv_path.name} ...")
    t0 = time.perf_counter()

    query = build_normalization_query(tsv_path)

    safe_in = str(tsv_path).replace(chr(92), '/')
    check_sql = f"""
        SELECT column_name
        FROM (DESCRIBE SELECT * FROM read_csv('{safe_in}',
              delim='\t', header=true, all_varchar=true, null_padding=true))
    """
    cols = [row[0] for row in con.execute(check_sql).fetchall()]
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(cols)
    if missing:
        raise ValueError(
            f"{tsv_path.name} is missing columns: {missing}. "
            f"Found: {cols}"
        )

    safe_out = str(out_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            {query}
        ) TO '{safe_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    row_count = con.execute(
        f"SELECT count(*) FROM read_parquet('{safe_out}')"
    ).fetchone()[0]
    elapsed = time.perf_counter() - t0
    print(f"    -> {row_count:,} rows written to {out_path.name}  ({elapsed:.1f}s)")
    return out_path


def ingest_ground_truth(
    con: duckdb.DuckDBPyConnection,
    gt_path: Path,
    split: str,
) -> Path:
    """Read ground-truth TSV and write Parquet (no normalisation needed).

    Expands the comma-separated ``matched_entity_ids`` into separate
    rows for easier joining during evaluation, while keeping the
    aggregated form too.
    """
    out_path = INTERMEDIATE_DIR / split / "ground_truth.parquet"
    _ensure_dir(out_path.parent)

    safe_in = str(gt_path).replace("\\", "/")
    safe_out = str(out_path).replace("\\", "/")

    print(f"  Ingesting ground truth from {gt_path.name} ...")
    t0 = time.perf_counter()

    con.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                COALESCE(matched_entity_ids, '') AS matched_entity_ids
            FROM read_csv('{safe_in}',
                          delim='\t', header=true, all_varchar=true,
                          null_padding=true)
        ) TO '{safe_out}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    row_count = con.execute(
        f"SELECT count(*) FROM read_parquet('{safe_out}')"
    ).fetchone()[0]
    elapsed = time.perf_counter() - t0
    print(f"    -> {row_count:,} rows  ({elapsed:.1f}s)")
    return out_path


def run_ingestion(split: str = "train") -> dict[str, Path]:
    """Ingest all sources for a split and return {name: parquet_path}."""
    sources = TRAIN_SOURCES if split == "train" else TEST_SOURCES
    results: dict[str, Path] = {}

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '8GB'")

    print(f"\n{'='*60}")
    print(f"  INGESTION -- {split} split")
    print(f"{'='*60}")

    for source_name, tsv_path in sources.items():
        if not tsv_path.exists():
            print(f"  [!] {tsv_path} not found -- skipping")
            continue
        results[source_name] = ingest_source(con, tsv_path, source_name, split)

    if split == "train" and TRAIN_GT.exists():
        results["ground_truth"] = ingest_ground_truth(con, TRAIN_GT, split)

    con.close()
    print(f"\n  Ingestion complete. Files in {INTERMEDIATE_DIR / split}/")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest TSV sources into normalised Parquet files."
    )
    parser.add_argument(
        "--split", choices=["train", "test"], default="train",
        help="Which data split to process (default: train)",
    )
    args = parser.parse_args()

    try:
        run_ingestion(args.split)
    except Exception as exc:
        print(f"\n  [x] Ingestion failed: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
