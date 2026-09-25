"""Submission generation and validation.

Generates:
  1. output/matching_results.tsv (required, scored on leaderboard)
  2. output/candidate_pairs.tsv (blocking candidates)

Ensures every S1 entity appears in both files, and validates output
using utils/validate_submission.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import duckdb

from src.config import (
    INTERMEDIATE_DIR,
    OUTPUT_DIR,
    TEST_DIR,
    candidate_parquet,
    norm_parquet,
)


def export_candidate_pairs(split: str = "test") -> Path:
    """Generate candidate_pairs.tsv for submission."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tsv = OUTPUT_DIR / "candidate_pairs.tsv"

    s1_path = norm_parquet(split, "source1")
    s2_cand = candidate_parquet(split, "source2")
    s3_cand = candidate_parquet(split, "source3")

    if not s1_path.exists():
        raise FileNotFoundError(f"Normalized source1 not found: {s1_path}")
    if not s2_cand.exists() or not s3_cand.exists():
        raise FileNotFoundError("Candidate parquet files not found for source2/source3")

    s1_safe = str(s1_path).replace("\\", "/")
    s2_safe = str(s2_cand).replace("\\", "/")
    s3_safe = str(s3_cand).replace("\\", "/")
    out_safe = str(out_tsv).replace("\\", "/")

    print(f"\nGenerating candidate_pairs.tsv [{split}] ...")
    t0 = time.perf_counter()

    con = duckdb.connect()
    con.execute("SET threads TO 6")
    con.execute("SET memory_limit = '6GB'")
    con.execute("SET preserve_insertion_order = false")

    s2_g_path = INTERMEDIATE_DIR / split / "grouped_cand_s2.parquet"
    s3_g_path = INTERMEDIATE_DIR / split / "grouped_cand_s3.parquet"
    s2_g_safe = str(s2_g_path).replace("\\", "/")
    s3_g_safe = str(s3_g_path).replace("\\", "/")

    print("  Aggregating S2 candidates ...")
    con.execute(f"""
        COPY (
            SELECT s1_id, string_agg(sx_id, ',') AS cands2
            FROM (SELECT DISTINCT s1_id, sx_id FROM read_parquet('{s2_safe}'))
            GROUP BY s1_id
        ) TO '{s2_g_safe}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print("  Aggregating S3 candidates ...")
    con.execute(f"""
        COPY (
            SELECT s1_id, string_agg(sx_id, ',') AS cands3
            FROM (SELECT DISTINCT s1_id, sx_id FROM read_parquet('{s3_safe}'))
            GROUP BY s1_id
        ) TO '{s3_g_safe}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    print("  Writing combined candidate_pairs.tsv ...")
    con.execute(f"""
        COPY (
            SELECT
                s1.entity_id AS source1_entity_id,
                CASE
                    WHEN s2.cands2 IS NOT NULL AND s3.cands3 IS NOT NULL THEN s2.cands2 || ',' || s3.cands3
                    WHEN s2.cands2 IS NOT NULL THEN s2.cands2
                    WHEN s3.cands3 IS NOT NULL THEN s3.cands3
                    ELSE ''
                END AS candidate_entity_ids
            FROM read_parquet('{s1_safe}') s1
            LEFT JOIN read_parquet('{s2_g_safe}') s2 ON s1.entity_id = s2.s1_id
            LEFT JOIN read_parquet('{s3_g_safe}') s3 ON s1.entity_id = s3.s1_id
        ) TO '{out_safe}' (DELIMITER '\t', HEADER true, QUOTE '')
    """)

    con.close()
    elapsed = time.perf_counter() - t0
    size_mb = out_tsv.stat().st_size / (1024 * 1024)
    print(f"  -> candidate_pairs.tsv written ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return out_tsv


def export_matching_results(
    matches_s2_parquet: Path,
    matches_s3_parquet: Path,
    split: str = "test",
) -> Path:
    """Generate matching_results.tsv for submission."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tsv = OUTPUT_DIR / "matching_results.tsv"

    s1_path = norm_parquet(split, "source1")
    if not s1_path.exists():
        raise FileNotFoundError(f"Normalized source1 not found: {s1_path}")

    s1_safe = str(s1_path).replace("\\", "/")
    s2_safe = str(matches_s2_parquet).replace("\\", "/")
    s3_safe = str(matches_s3_parquet).replace("\\", "/")
    out_safe = str(out_tsv).replace("\\", "/")

    print(f"\nGenerating matching_results.tsv [{split}] ...")
    t0 = time.perf_counter()

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '8GB'")

    con.execute(f"""
        COPY (
            WITH s2_m AS (
                SELECT s1_id, string_agg(sx_id, ',') AS m2
                FROM (SELECT DISTINCT s1_id, sx_id FROM read_parquet('{s2_safe}'))
                GROUP BY s1_id
            ),
            s3_m AS (
                SELECT s1_id, string_agg(sx_id, ',') AS m3
                FROM (SELECT DISTINCT s1_id, sx_id FROM read_parquet('{s3_safe}'))
                GROUP BY s1_id
            )
            SELECT
                s1.entity_id AS source1_entity_id,
                CASE
                    WHEN s2.m2 IS NOT NULL AND s3.m3 IS NOT NULL THEN s2.m2 || ',' || s3.m3
                    WHEN s2.m2 IS NOT NULL THEN s2.m2
                    WHEN s3.m3 IS NOT NULL THEN s3.m3
                    ELSE ''
                END AS matched_entity_ids
            FROM read_parquet('{s1_safe}') s1
            LEFT JOIN s2_m s2 ON s1.entity_id = s2.s1_id
            LEFT JOIN s3_m s3 ON s1.entity_id = s3.s1_id
            ORDER BY s1.entity_id
        ) TO '{out_safe}' (DELIMITER '\t', HEADER true, QUOTE '')
    """)

    con.close()
    elapsed = time.perf_counter() - t0
    size_mb = out_tsv.stat().st_size / (1024 * 1024)
    print(f"  -> matching_results.tsv written ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return out_tsv


def validate_submission_files(
    matching_tsv: Path,
    candidate_tsv: Optional[Path] = None,
    test_dir: Path = TEST_DIR,
) -> bool:
    """Run utils/validate_submission.py to check submission formatting."""
    print(f"\n{'='*60}")
    print("  VALIDATING SUBMISSION FILES")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(matching_tsv),
        "--test-dir", str(test_dir),
    ]
    if candidate_tsv and candidate_tsv.exists():
        cmd.extend(["--candidate", str(candidate_tsv)])

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)

    if res.returncode == 0:
        print("  -> VALIDATION PASSED! Ready for submission.")
        return True
    else:
        print("  [x] VALIDATION FAILED! Check issues above.", file=sys.stderr)
        return False
