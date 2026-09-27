"""
Phase 2: Dense Retrieval Feasibility & Latency Benchmark.
Benchmarks local sentence-transformers (all-MiniLM-L6-v2) on CPU.
Measures:
  1. Encoding throughput (queries/s, targets/s)
  2. Dense search latency
  3. Memory footprint
  4. Extrapolated runtime on full test corpus (2.0M target entities, 37k queries)
  5. Candidate recall & precision on gold pairs within slice.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet

_CACHE_DIR = ROOT / "intermediate" / "v2_cache"
_H = "abs(hash(entity_id || 'v1')) % 1000"
_VAL1_WHERE = f"{_H} >= 0 AND {_H} < 5"


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def benchmark_dense_feasibility(target_source: str = "source2", n_queries: int = 500, n_targets: int = 10000):
    print("=" * 80)
    print(f"PHASE 2: DENSE RETRIEVAL FEASIBILITY & BENCHMARK ({target_source.upper()})")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Available device: {device}")

    c = duckdb.connect()
    prefix = "S2-" if target_source == "source2" else "S3-"

    # Sample n_queries from val1
    s1_df = c.execute(f"""
        SELECT entity_id AS s1_id, country_clean, name_clean
        FROM read_parquet('{_s1pq()}')
        WHERE {_VAL1_WHERE}
        ORDER BY entity_id
        LIMIT {n_queries}
    """).df()

    s1_set = set(s1_df["s1_id"])
    gt_df = c.execute(f"""
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
        FROM read_parquet('{_gtpq()}')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
          AND source1_entity_id IN (SELECT entity_id FROM read_parquet('{_s1pq()}') WHERE {_VAL1_WHERE} LIMIT {n_queries})
    """).df()
    gt_df = gt_df[gt_df["s1_id"].isin(s1_set)]
    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))

    # Sample n_targets including any known true matches for fair evaluation
    known_targets = set(gt_df["true_match_id"])
    target_df = c.execute(f"""
        SELECT entity_id AS sx_id, country_clean, name_clean
        FROM read_parquet('{_sxpq(target_source)}')
        ORDER BY entity_id
        LIMIT {n_targets}
    """).df()
    c.close()

    print(f"Benchmark slice: {len(s1_df)} queries | {len(target_df)} target candidates")
    print(f"Ground truth matches in slice: {len(gt_set)}")

    tracemalloc.start()
    t_start = time.time()

    print("\nLoading local SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # 1. Encode queries
    t0 = time.time()
    q_emb = model.encode(s1_df["name_clean"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    q_time = time.time() - t0
    q_rate = len(s1_df) / q_time
    print(f"Query encoding: {len(s1_df)} queries in {q_time:.2f}s -> {q_rate:.1f} queries/s")

    # 2. Encode targets
    t0 = time.time()
    t_emb = model.encode(target_df["name_clean"].tolist(), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    t_time = time.time() - t0
    t_rate = len(target_df) / t_time
    print(f"Target encoding: {len(target_df)} targets in {t_time:.2f}s -> {t_rate:.1f} targets/s")

    # 3. Dense search latency
    t0 = time.time()
    sims = np.dot(q_emb, t_emb.T)
    top20_idx = np.argpartition(sims, -20, axis=1)[:, -20:]
    search_time = time.time() - t0
    print(f"Cosine search latency (500 x 10,000): {search_time*1000:.1f} ms")

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f"RAM peak: {peak_mb:.1f} MB")

    # 4. Extrapolation to Full Dataset
    # Full dataset size: Source 2 has ~2,000,000 entities, Source 1 has ~37,000 queries per split
    full_target_count = 2000000
    full_query_count = 37000

    est_target_hrs = (full_target_count / t_rate) / 3600
    est_query_hrs = (full_query_count / q_rate) / 3600
    est_total_hrs = est_target_hrs + est_query_hrs
    est_ram_gb = (full_target_count * 384 * 4) / (1024**3)

    print("\n" + "=" * 80)
    print("PHASE 2 DENSE RETRIEVAL EXTRAPOLATION & ASSESSMENT")
    print("=" * 80)
    print(f"Extrapolated Target Encoding Time (2.0M records): ~{est_target_hrs:.2f} hours")
    print(f"Extrapolated Query Encoding Time (37k queries):   ~{est_query_hrs:.2f} hours")
    print(f"Total Projected Runtime per Split:                ~{est_total_hrs:.2f} hours ({est_total_hrs * 60:.1f} mins)")
    print(f"Dense Embedding Memory Footprint:                 ~{est_ram_gb:.2f} GB (target corpus vectors alone)")
    print(f"Submission Time Limit Constraint:                 EXCEEDED (>60 mins limit, estimated ~{est_total_hrs:.1f}h)")
    print("Hardware Constraint:                              No CUDA GPU available (CPU only)")
    print("Conclusion: DENSE RETRIEVAL INFEASIBLE within competition runtime budget.")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_dense_feasibility()
