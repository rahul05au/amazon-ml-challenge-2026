"""
MinHash LSH Experiment — VECTORISED NumPy implementation.
Char 3-shingles on business_name. Two configs tested:
  b32r4 (~Jaccard 0.45 threshold, aggressive recall)
  b64r2 (~Jaccard 0.71 threshold, balanced)
"""
from __future__ import annotations

import re
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet

_CACHE_DIR = ROOT / "intermediate" / "v2_cache"
_H = "abs(hash(entity_id || 'v1')) % 1000"
_WHERE = {
    "val1": f"{_H} >= 0 AND {_H} < 5",
    "val2": f"{_H} >= 25 AND {_H} < 30",
}


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def _shingle_to_int(s: str) -> int:
    h = 5381
    for c in s:
        h = (h * 33 + ord(c)) & 0xFFFF
    return h


def char_shingles_int(text: str, k: int = 3) -> list[int]:
    t = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(t) < k:
        return [_shingle_to_int(t)] if t else []
    return [_shingle_to_int(t[i:i+k]) for i in range(len(t) - k + 1)]


def build_signatures_vectorised(names: list[str], n_perm: int, seed: int = 42) -> np.ndarray:
    """Build MinHash signature matrix shape=(N, n_perm) using vectorised NumPy."""
    p = (1 << 31) - 1
    rng = np.random.default_rng(seed)
    a = rng.integers(1, p, size=n_perm, dtype=np.int64)
    b = rng.integers(0, p, size=n_perm, dtype=np.int64)

    N = len(names)
    sigs = np.full((N, n_perm), p, dtype=np.int64)

    for i, name in enumerate(names):
        ints = char_shingles_int(name)
        if not ints:
            continue
        h = np.array(ints, dtype=np.int64)  # (S,)
        # (S, n_perm): hash each shingle with each of n_perm functions
        vals = (a[None, :] * h[:, None] + b[None, :]) % p
        sigs[i] = vals.min(axis=0)

    return sigs.astype(np.int32)


def build_lsh_index(entity_ids: list[str], countries: list[str],
                    sigs: np.ndarray, n_bands: int) -> dict:
    n_perm = sigs.shape[1]
    rpb = max(1, n_perm // n_bands)
    index: dict = defaultdict(list)
    for i, (eid, ctry) in enumerate(zip(entity_ids, countries)):
        for b in range(n_bands):
            s, e = b * rpb, min((b + 1) * rpb, n_perm)
            key = (ctry, b, sigs[i, s:e].tobytes())
            index[key].append(eid)
    return dict(index)


print("=" * 80)
print("MINHASH LSH EXPERIMENT (VECTORISED NumPy — ~50x faster than pure Python)")
print("char 3-shingles | b32r4 (~J=0.45) and b64r2 (~J=0.71)")
print("=" * 80)

c = duckdb.connect()
print("\nLoading normalized data...")
t0 = time.time()
s1_df = c.execute(f"SELECT entity_id, country_clean, COALESCE(name_clean,'') AS name_clean FROM read_parquet('{_s1pq()}')").df()
s2_df = c.execute(f"SELECT entity_id, country_clean, COALESCE(name_clean,'') AS name_clean FROM read_parquet('{_sxpq('source2')}')").df()
s3_df = c.execute(f"SELECT entity_id, country_clean, COALESCE(name_clean,'') AS name_clean FROM read_parquet('{_sxpq('source3')}')").df()
c.close()
print(f"Loaded S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,} in {time.time()-t0:.1f}s")

for n_perm, n_bands, cfg_label in [(128, 32, "b32r4_j45"), (128, 64, "b64r2_j71")]:
    rpb = n_perm // n_bands
    print(f"\n{'='*70}")
    print(f"CONFIG: {cfg_label} | n_perm={n_perm} | n_bands={n_bands} | rows/band={rpb}")
    print(f"{'='*70}")
    tracemalloc.start()

    print("  Building S1 signatures...")
    t0 = time.time()
    s1_sigs = build_signatures_vectorised(s1_df["name_clean"].tolist(), n_perm)
    s1_sig_t = time.time() - t0
    print(f"  S1 sigs: {s1_sig_t:.1f}s ({len(s1_df)/s1_sig_t:.0f}/s)")

    print("  Building S1 LSH index...")
    t0 = time.time()
    s1_idx = build_lsh_index(s1_df["entity_id"].tolist(), s1_df["country_clean"].tolist(), s1_sigs, n_bands)
    print(f"  S1 index: {len(s1_idx):,} buckets in {time.time()-t0:.1f}s")

    for src_name, sx_df, sx_prefix in [("source2", s2_df, "S2-"), ("source3", s3_df, "S3-")]:
        print(f"\n  Computing {src_name.upper()} signatures...")
        t0 = time.time()
        sx_sigs = build_signatures_vectorised(sx_df["name_clean"].tolist(), n_perm)
        print(f"  {src_name.upper()} sigs: {time.time()-t0:.1f}s")

        print(f"  Building {src_name.upper()} LSH index...")
        t0 = time.time()
        sx_idx = build_lsh_index(sx_df["entity_id"].tolist(), sx_df["country_clean"].tolist(), sx_sigs, n_bands)
        print(f"  {src_name.upper()} index: {len(sx_idx):,} buckets in {time.time()-t0:.1f}s")

        print(f"  Generating candidate pairs...")
        t0 = time.time()
        s1_out, sx_out, block_sizes = [], [], []
        for key, s1_ids in s1_idx.items():
            if key not in sx_idx:
                continue
            sx_ids = sx_idx[key]
            block_sizes.append(len(s1_ids) * len(sx_ids))
            for s1 in s1_ids:
                for sx in sx_ids:
                    s1_out.append(s1)
                    sx_out.append(sx)
        cands_df = pd.DataFrame({"s1_id": s1_out, "sx_id": sx_out}).drop_duplicates()
        cand_set = set(zip(cands_df["s1_id"], cands_df["sx_id"]))
        gen_t = time.time() - t0

        ba = np.array(block_sizes) if block_sizes else np.array([0])
        print(f"  Candidates: {len(cand_set):,} | S1 affected: {cands_df['s1_id'].nunique():,} | Gen: {gen_t:.1f}s")
        print(f"  Block size: mean={ba.mean():.1f} | P95={np.percentile(ba,95):.0f} | P99={np.percentile(ba,99):.0f} | MAX={ba.max():.0f}")

        for split in ["val1", "val2"]:
            c2 = duckdb.connect()
            c2.execute(f"CREATE TEMP TABLE _s1 AS SELECT entity_id AS s1_id FROM read_parquet('{_s1pq()}') WHERE {_WHERE[split]}")
            s1_split = set(c2.execute("SELECT s1_id FROM _s1").df()["s1_id"])
            gt_df = c2.execute(f"""
                SELECT source1_entity_id AS s1_id,
                       trim(unnest(string_split(matched_entity_ids,','))) AS true_match_id
                FROM read_parquet('{_gtpq()}')
                WHERE matched_entity_ids!='' AND matched_entity_ids LIKE '%{sx_prefix}%'
                  AND source1_entity_id IN (SELECT s1_id FROM _s1)
            """).df()
            c2.close()

            gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
            total_gold = len(gt_set)
            split_cands = cands_df[cands_df["s1_id"].isin(s1_split)]
            split_set = set(zip(split_cands["s1_id"], split_cands["sx_id"]))
            rec_gold = len(gt_set & split_set)
            standalone = rec_gold / total_gold if total_gold else 0.0

            base = pd.read_parquet(_CACHE_DIR / f"v2_{src_name}_{split}.parquet")
            base_set = set(zip(base["s1_id"], base["sx_id"]))
            base_rec = len(gt_set & base_set)
            base_recall = base_rec / total_gold if total_gold else 0.0

            union_set = base_set | split_set
            union_rec = len(gt_set & union_set)
            union_recall = union_rec / total_gold if total_gold else 0.0
            new_rec = union_rec - base_rec
            added = len(split_set - base_set)
            noise = added / max(new_rec, 1)

            print(f"\n    [{src_name.upper()} | {split.upper()}] base={base_recall:.4f}")
            print(f"      Standalone Recall: {standalone:.4f} ({rec_gold:,}/{total_gold:,})")
            print(f"      Net New Cands:     {added:,}")
            print(f"      Net New Gold:      {new_rec:,}")
            print(f"      Noise Ratio:       {noise:.1f}")
            print(f"      Union Recall:      {union_recall:.4f} ({union_recall-base_recall:+.4f})")

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n  Peak RAM {cfg_label}: {peak/(1024**2):.1f} MB")

print("\nMINHASH LSH EXPERIMENT COMPLETE.")
