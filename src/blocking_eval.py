"""Blocking recall evaluation.

Creates a deterministic validation split by S1 entity, then measures
blocking recall — the fraction of true matches found by the candidate
set — before any matcher or fuzzy scoring runs.

Reports per-pass and union recall, candidate counts, and distributions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field

import duckdb

from src.config import (
    INTERMEDIATE_DIR,
    VAL_FRACTION,
    VAL_SEED,
    candidate_parquet,
    norm_parquet,
)


@dataclass
class RecallReport:
    """Recall evaluation results for one blocking run."""
    target_source: str
    total_s1_val: int = 0
    total_true_pairs: int = 0   # only pairs involving the target source
    true_pairs_found: int = 0
    true_pairs_missed: int = 0
    blocking_recall: float = 0.0
    candidate_count: int = 0
    avg_candidates_per_s1: float = 0.0
    p95_candidates_per_s1: float = 0.0
    max_candidates_per_s1: int = 0
    s1_with_candidates: int = 0
    s1_with_true_matches: int = 0
    elapsed_seconds: float = 0.0


def _create_val_split(
    con: duckdb.DuckDBPyConnection,
    gt_parquet: str,
    fraction: float = VAL_FRACTION,
    seed: int = VAL_SEED,
) -> int:
    """Create a deterministic S1 validation split.

    Uses hash-based splitting so the result is fully reproducible.
    Returns the number of S1 entities in the validation set.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gt AS
        SELECT * FROM read_parquet('{gt_parquet}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _val_s1 AS
        SELECT source1_entity_id
        FROM _gt
        WHERE abs(hash(source1_entity_id || '{seed}')) % 1000
              < {int(fraction * 1000)}
    """)

    val_count = con.execute("SELECT count(*) FROM _val_s1").fetchone()[0]
    return val_count


def _expand_ground_truth(
    con: duckdb.DuckDBPyConnection,
    target_source: str,
) -> int:
    """Expand the comma-separated ground truth into individual rows,
    filtered to the validation split and target source.

    Returns the count of true pairs.
    """
    prefix = "S2-" if target_source == "source2" else "S3-"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gt_expanded AS
        SELECT
            gt.source1_entity_id AS s1_id,
            trim(unnest(string_split(gt.matched_entity_ids, ','))) AS true_match_id
        FROM _gt gt
        WHERE gt.source1_entity_id IN (SELECT source1_entity_id FROM _val_s1)
          AND gt.matched_entity_ids != ''
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _gt_target AS
        SELECT s1_id, true_match_id
        FROM _gt_expanded
        WHERE true_match_id LIKE '{prefix}%'
    """)

    count = con.execute("SELECT count(*) FROM _gt_target").fetchone()[0]
    return count


def evaluate_blocking(
    split: str = "train",
    target_source: str = "source2",
    val_fraction: float = VAL_FRACTION,
    val_seed: int = VAL_SEED,
) -> RecallReport:
    """Evaluate blocking recall for S1 → target_source."""
    gt_path = INTERMEDIATE_DIR / split / "ground_truth.parquet"
    cand_path = candidate_parquet(split, target_source)

    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth Parquet not found: {gt_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate Parquet not found: {cand_path}")

    gt_safe = str(gt_path).replace("\\", "/")
    cand_safe = str(cand_path).replace("\\", "/")

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '8GB'")

    t0 = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  BLOCKING RECALL -- {split}, S1 -> {target_source}")
    print(f"{'='*60}")

    val_count = _create_val_split(con, gt_safe, val_fraction, val_seed)
    print(f"  Validation S1 entities: {val_count:,} ({val_fraction*100:.0f}%)")

    true_pair_count = _expand_ground_truth(con, target_source)
    print(f"  True {target_source} pairs in val: {true_pair_count:,}")

    if true_pair_count == 0:
        print("  [!] No true pairs for this source -- nothing to evaluate.")
        con.close()
        return RecallReport(target_source=target_source, total_s1_val=val_count)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _candidates AS
        SELECT s1_id, sx_id
        FROM read_parquet('{cand_safe}')
        WHERE s1_id IN (SELECT source1_entity_id FROM _val_s1)
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE _found AS
        SELECT gt.s1_id, gt.true_match_id
        FROM _gt_target gt
        JOIN _candidates c
          ON gt.s1_id = c.s1_id
         AND gt.true_match_id = c.sx_id
    """)

    found_count = con.execute("SELECT count(*) FROM _found").fetchone()[0]
    missed_count = true_pair_count - found_count
    recall = found_count / true_pair_count if true_pair_count > 0 else 0.0

    cand_stats = con.execute("""
        SELECT
            sum(cnt)                AS total,
            count(*)                AS s1_count,
            avg(cnt)                AS avg_cnt,
            percentile_disc(0.95) WITHIN GROUP (ORDER BY cnt) AS p95,
            max(cnt)                AS max_cnt
        FROM (
            SELECT s1_id, count(*) AS cnt
            FROM _candidates
            GROUP BY s1_id
        ) t
    """).fetchone()

    # S1 entities that have at least one true match in this source
    s1_with_matches = con.execute("""
        SELECT count(DISTINCT s1_id) FROM _gt_target
    """).fetchone()[0]

    elapsed = time.perf_counter() - t0
    con.close()

    report = RecallReport(
        target_source=target_source,
        total_s1_val=val_count,
        total_true_pairs=true_pair_count,
        true_pairs_found=found_count,
        true_pairs_missed=missed_count,
        blocking_recall=round(recall, 6),
        candidate_count=cand_stats[0] or 0,
        avg_candidates_per_s1=round(cand_stats[2] or 0, 2),
        p95_candidates_per_s1=round(cand_stats[3] or 0, 2),
        max_candidates_per_s1=cand_stats[4] or 0,
        s1_with_candidates=cand_stats[1] or 0,
        s1_with_true_matches=s1_with_matches,
        elapsed_seconds=round(elapsed, 2),
    )

    print(f"\n  +------------------------------------------+")
    print(f"  | Blocking Recall:   {recall:>10.4%}")
    print(f"  | True pairs found:  {found_count:>10,} / {true_pair_count:,}")
    print(f"  | True pairs missed: {missed_count:>10,}")
    print(f"  | Candidates (val):  {report.candidate_count:>10,}")
    print(f"  | S1 with candidates:{report.s1_with_candidates:>10,}")
    print(f"  | S1 with true match:{s1_with_matches:>10,}")
    print(f"  | Avg cands/S1:      {report.avg_candidates_per_s1:>10.1f}")
    print(f"  | P95 cands/S1:      {report.p95_candidates_per_s1:>10.0f}")
    print(f"  | Max cands/S1:      {report.max_candidates_per_s1:>10,}")
    print(f"  | Elapsed:           {elapsed:>10.1f}s")
    print(f"  +------------------------------------------+")

    report_path = (
        INTERMEDIATE_DIR / split / f"blocking_recall_{target_source}.json"
    )
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"  Report saved to {report_path}")

    if missed_count > 0:
        con2 = duckdb.connect()
        con2.execute(f"""
            CREATE TABLE _gt_target AS
            SELECT * FROM read_parquet('{gt_safe}')
        """)
        con2.close()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate blocking recall against ground truth."
    )
    parser.add_argument(
        "--split", choices=["train"], default="train",
        help="Data split (ground truth only available for train)",
    )
    parser.add_argument(
        "--target", choices=["source2", "source3"], default=None,
        help="Target source (default: both)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=VAL_FRACTION,
    )
    parser.add_argument(
        "--val-seed", type=int, default=VAL_SEED,
    )
    args = parser.parse_args()

    targets = [args.target] if args.target else ["source2", "source3"]
    for target in targets:
        evaluate_blocking(
            split=args.split,
            target_source=target,
            val_fraction=args.val_fraction,
            val_seed=args.val_seed,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
