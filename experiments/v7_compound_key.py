"""
Compound Key Experiment:
Key = Soundex(first token of business_address) + first 3 normalized chars of business_name.
Measures standalone recall, false-candidate ratio, and block-size tail.
Kept ISOLATED — not merged into main pool.
"""
from __future__ import annotations

import re
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np
import duckdb
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


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def soundex(name: str) -> str:
    """American Soundex."""
    if not name:
        return ""
    name = re.sub(r"[^a-zA-Z]", "", name).upper()
    if not name:
        return ""
    keep = name[0]
    mapping = str.maketrans("BFPVCGJKQSXZDTLMNR", "111122222222334556")
    coded = name.translate(mapping)
    result = keep
    prev = coded[0] if coded else ""
    for c in coded[1:]:
        if c in "12345678" and c != prev:
            result += c
        prev = c if not c.isdigit() else c
        if len(result) == 4:
            break
    return result.ljust(4, "0")


def first_addr_token(addr: str) -> str:
    """Extract the first meaningful token from address (skip leading numbers)."""
    if not addr:
        return ""
    addr = addr.strip()
    tokens = re.split(r"[\s,]+", addr)
    noise = {"PO", "BOX", "UNIT", "SUITE", "APT", "FL", "STE", "A", "AN", "THE"}
    for t in tokens:
        t_clean = re.sub(r"[^a-zA-Z]", "", t).upper()
        if t_clean and t_clean not in noise and not t_clean.isdigit():
            return t_clean
    return ""


def name_prefix3(name: str) -> str:
    """First 3 lowercase alphanum chars of business_name."""
    cleaned = re.sub(r"[^a-z0-9]", "", name.lower())
    return cleaned[:3]


def build_compound_key(country: str, name: str, addr: str) -> str | None:
    """Compound key = country + name_prefix_3 + soundex(first_addr_token)."""
    if not country:
        return None
    prefix3 = name_prefix3(name)
    if len(prefix3) < 2:
        return None
    token = first_addr_token(addr)
    if not token or len(token) < 2:
        return None
    sdx = soundex(token)
    return f"{country.lower()}|{prefix3}|{sdx}"


print("=" * 80)
print("COMPOUND KEY EXPERIMENT")
print("Key = country_clean + name_prefix_3 + Soundex(first_addr_token)")
print("Kept ISOLATED from main pool — oracle check after all three experiments complete")
print("=" * 80)

c = duckdb.connect()

print("\nLoading normalized data...")
t0 = time.time()
s1_df = c.execute(f"""
    SELECT entity_id, country_clean,
           COALESCE(name_clean, '') AS name_clean,
           COALESCE(addr_clean, '') AS addr_clean
    FROM read_parquet('{_s1pq()}')
""").df()

src_dfs = {}
for src_name in ["source2", "source3"]:
    pq = str(norm_parquet("train", src_name)).replace("\\", "/")
    src_dfs[src_name] = c.execute(f"""
        SELECT entity_id, country_clean,
               COALESCE(name_clean, '') AS name_clean,
               COALESCE(addr_clean, '') AS addr_clean
        FROM read_parquet('{pq}')
    """).df()

c.close()
print(f"Loaded in {time.time()-t0:.1f}s")

# Build key maps
print("\nBuilding compound key maps...")
t0 = time.time()

def compute_key_map(df: pd.DataFrame) -> dict[str, list[str]]:
    km = defaultdict(list)
    for row in df.itertuples(index=False):
        k = build_compound_key(row.country_clean, row.name_clean, row.addr_clean)
        if k:
            km[k].append(row.entity_id)
    return dict(km)

tracemalloc.start()
s1_key_map = compute_key_map(s1_df)
print(f"S1 keys: {len(s1_key_map):,} (computed in {time.time()-t0:.1f}s)")

for src_name in ["source2", "source3"]:
    t_sx = time.time()
    sx_key_map = compute_key_map(src_dfs[src_name])
    sx_prefix = "S2-" if src_name == "source2" else "S3-"
    print(f"{src_name.upper()} keys: {len(sx_key_map):,} (computed in {time.time()-t_sx:.1f}s)")

    # Generate candidates
    t_gen = time.time()
    s1_ids_out = []
    sx_ids_out = []
    block_sizes = []
    MAX_BLOCK = 200  # Hard cap per key to prevent combinatorial explosion

    skipped_blocks = 0
    for key, s1_ids in s1_key_map.items():
        if key not in sx_key_map:
            continue
        sx_ids = sx_key_map[key]
        block_sz = len(s1_ids) * len(sx_ids)
        block_sizes.append(block_sz)
        if len(s1_ids) > MAX_BLOCK or len(sx_ids) > MAX_BLOCK:
            skipped_blocks += 1
            continue
        for s1 in s1_ids:
            for sx in sx_ids:
                s1_ids_out.append(s1)
                sx_ids_out.append(sx)

    cands_df = pd.DataFrame({"s1_id": s1_ids_out, "sx_id": sx_ids_out}).drop_duplicates()
    cand_set = set(zip(cands_df["s1_id"], cands_df["sx_id"]))
    gen_time = time.time() - t_gen

    block_arr = np.array(block_sizes) if block_sizes else np.array([0])
    print(f"\n  {'='*60}")
    print(f"  {src_name.upper()} COMPOUND KEY RESULTS")
    print(f"  {'='*60}")
    print(f"  Total Candidate Pairs:    {len(cand_set):,}")
    print(f"  S1 Entities Covered:      {cands_df['s1_id'].nunique():,}")
    print(f"  Generation Time:          {gen_time:.1f}s")
    print(f"  Skipped Large Blocks:     {skipped_blocks:,} (> {MAX_BLOCK} per side)")
    print(f"  Block Size: mean={block_arr.mean():.1f} | P50={np.percentile(block_arr,50):.0f} | P90={np.percentile(block_arr,90):.0f} | P95={np.percentile(block_arr,95):.0f} | P99={np.percentile(block_arr,99):.0f} | MAX={block_arr.max():.0f}")

    for split in ["val1", "val2"]:
        c2 = duckdb.connect()
        c2.execute(f"""
            CREATE TEMP TABLE _s1 AS
            SELECT entity_id AS s1_id FROM read_parquet('{_s1pq()}') WHERE {_WHERE[split]}
        """)
        s1_all_split = set(c2.execute("SELECT s1_id FROM _s1").df()["s1_id"])
        gt_df = c2.execute(f"""
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{_gtpq()}')
            WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{sx_prefix}%'
              AND source1_entity_id IN (SELECT s1_id FROM _s1)
        """).df()
        c2.close()

        gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
        total_gold = len(gt_set)

        split_cands = cands_df[cands_df["s1_id"].isin(s1_all_split)]
        split_cand_set = set(zip(split_cands["s1_id"], split_cands["sx_id"]))
        recovered_gold = len(gt_set & split_cand_set)
        standalone_recall = recovered_gold / total_gold if total_gold > 0 else 0.0

        base_cache = pd.read_parquet(_CACHE_DIR / f"v2_{src_name}_{split}.parquet")
        base_set = set(zip(base_cache["s1_id"], base_cache["sx_id"]))
        base_present = gt_set & base_set
        base_recall = len(base_present) / total_gold if total_gold > 0 else 0.0

        union_set = base_set | split_cand_set
        union_recovered = len(gt_set & union_set)
        union_recall = union_recovered / total_gold if total_gold > 0 else 0.0
        newly_recovered = union_recovered - len(base_present)
        added_cands = len(split_cand_set - base_set)
        noise_ratio = added_cands / max(newly_recovered, 1)

        # FLAG if noise ratio is near Phase 1's critical failure threshold (~1000)
        noise_flag = " *** PHASE-1-LEVEL NOISE — DISCARD ***" if noise_ratio > 500 else ""
        print(f"\n  [{src_name.upper()} | {split.upper()}] baseline_recall={base_recall:.4f}")
        print(f"    Standalone Recall:  {standalone_recall:.4f} ({recovered_gold:,}/{total_gold:,})")
        print(f"    Net New Cands:      {added_cands:,}")
        print(f"    Net New Gold:       {newly_recovered:,}")
        print(f"    Noise Ratio:        {noise_ratio:.1f} false cands/recovered gold{noise_flag}")
        print(f"    Union Recall:       {union_recall:.4f} ({union_recall - base_recall:+.4f})")

_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nPeak RAM: {peak/(1024**2):.1f} MB")
print("\nCOMPOUND KEY EXPERIMENT COMPLETE.")
