"""
Step 3: Phonetic Address Key Blocking Experiment.
Key = country_clean + postal_code + street_number + Soundex(first_street_name_token).
Measures standalone recall, noise ratio, and oracle ceiling impact on val1/val2 for S2 and S3.
"""
from __future__ import annotations

import re
import sys
import time
import tracemalloc
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet
from src.matcher import calculate_macro_f05

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


# Soundex implementation (American Soundex)
def soundex(name: str) -> str:
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
        if not c.isdigit():
            prev = c
        else:
            prev = c
        if len(result) == 4:
            break
    return result.ljust(4, "0")


def extract_address_parts(addr: str) -> tuple[str, str]:
    """Extract street number and first street name token from address string."""
    if not addr:
        return "", ""
    addr = addr.strip()
    # Extract leading street number
    num_match = re.match(r"^(\d+[\w/\-]*)", addr)
    street_num = num_match.group(1) if num_match else ""
    # Remove number prefix and take first token
    rest = addr[len(street_num):].strip().lstrip(",.-").strip()
    tokens = re.split(r"[\s,]+", rest)
    first_token = tokens[0].upper() if tokens else ""
    # Remove pure noise tokens
    noise = {"THE", "A", "AN", "PO", "BOX", "UNIT", "SUITE", "APT", "FL", "STE"}
    if first_token in noise and len(tokens) > 1:
        first_token = tokens[1].upper()
    return street_num, first_token


def build_phonetic_address_key(entity_id: str, country: str, postal: str, addr: str) -> str | None:
    """Build blocking key: country + postal_code + street_num + Soundex(first_street_token)."""
    if not country:
        return None
    street_num, first_token = extract_address_parts(addr)
    if not first_token or len(first_token) < 2:
        return None
    sdx = soundex(first_token)
    # Require at least postal code or street number
    if not postal and not street_num:
        return None
    key_parts = [country.lower(), postal.strip()[:6] if postal else "", street_num[:6] if street_num else "", sdx]
    return "|".join(key_parts)


print("=" * 80)
print("STEP 3: PHONETIC ADDRESS KEY BLOCKING")
print("Key = country_clean + postal_code[:6] + street_number[:6] + Soundex(first_street_token)")
print("=" * 80)

c = duckdb.connect()

# Build address key lookup for all sources in memory
print("\nLoading and building address keys for Source 1...")
tracemalloc.start()
t0 = time.time()

s1_rows = c.execute(f"""
    SELECT entity_id, country_clean, COALESCE(postal_code, '') AS postal_code,
           COALESCE(addr_clean, '') AS addr_clean
    FROM read_parquet('{_s1pq()}')
""").df()

s2_rows = c.execute(f"""
    SELECT entity_id, country_clean, COALESCE(postal_code, '') AS postal_code,
           COALESCE(addr_clean, '') AS addr_clean
    FROM read_parquet('{_s1pq().replace("source1", "source2")}')
""").df()

s3_rows = c.execute(f"""
    SELECT entity_id, country_clean, COALESCE(postal_code, '') AS postal_code,
           COALESCE(addr_clean, '') AS addr_clean
    FROM read_parquet('{_s1pq().replace("source1", "source3")}')
""").df()
c.close()

load_time = time.time() - t0
print(f"Loaded S1={len(s1_rows):,} S2={len(s2_rows):,} S3={len(s3_rows):,} in {load_time:.1f}s")


def compute_keys(df: pd.DataFrame) -> dict[str, list[str]]:
    """Compute phonetic address keys. Returns key -> [entity_ids]."""
    key_map: dict[str, list[str]] = {}
    for row in df.itertuples(index=False):
        k = build_phonetic_address_key(row.entity_id, row.country_clean, row.postal_code, row.addr_clean)
        if k:
            key_map.setdefault(k, []).append(row.entity_id)
    return key_map


print("Computing address keys for S1, S2, S3...")
t0 = time.time()
s1_key_map = compute_keys(s1_rows)
s2_key_map = compute_keys(s2_rows)
s3_key_map = compute_keys(s3_rows)
key_time = time.time() - t0
print(f"Keys computed in {key_time:.1f}s | S1 unique keys: {len(s1_key_map):,} | S2: {len(s2_key_map):,} | S3: {len(s3_key_map):,}")

# Generate candidate pairs by key collision
def generate_candidates(s1_map: dict, sx_map: dict, max_block: int = 500) -> pd.DataFrame:
    s1_ids_out = []
    sx_ids_out = []
    skipped_blocks = 0
    block_sizes = []
    for key, s1_ids in s1_map.items():
        if key not in sx_map:
            continue
        sx_ids = sx_map[key]
        block_sz = len(s1_ids) * len(sx_ids)
        block_sizes.append(block_sz)
        if block_sz > max_block * max_block:
            skipped_blocks += 1
            continue
        for s1 in s1_ids:
            for sx in sx_ids:
                s1_ids_out.append(s1)
                sx_ids_out.append(sx)
    return pd.DataFrame({"s1_id": s1_ids_out, "sx_id": sx_ids_out}), block_sizes, skipped_blocks


for src_name, sx_key_map, sx_prefix in [("source2", s2_key_map, "S2-"), ("source3", s3_key_map, "S3-")]:
    print(f"\n{'='*70}")
    print(f"SOURCE: {src_name.upper()}")
    print(f"{'='*70}")

    t_gen = time.time()
    cands_df, block_sizes, skipped = generate_candidates(s1_key_map, sx_key_map)
    gen_time = time.time() - t_gen

    if len(cands_df) == 0:
        print("  No candidates generated.")
        continue

    cands_df = cands_df.drop_duplicates()
    cand_set = set(zip(cands_df["s1_id"], cands_df["sx_id"]))
    s1_affected = cands_df["s1_id"].nunique()

    import numpy as np
    block_sizes = [b for b in block_sizes if b > 0]
    block_arr = np.array(block_sizes) if block_sizes else np.array([0])

    print(f"  Candidates generated: {len(cand_set):,} | S1 affected: {s1_affected:,}")
    print(f"  Runtime: {gen_time:.1f}s | Skipped blocks (too large): {skipped}")
    print(f"  Block size stats: mean={block_arr.mean():.1f} | P50={np.percentile(block_arr, 50):.0f} | P95={np.percentile(block_arr, 95):.0f} | P99={np.percentile(block_arr, 99):.0f} | MAX={block_arr.max():.0f}")

    for split in ["val1", "val2"]:
        # Ground truth
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

        # Restrict to val split S1 entities
        split_cands = cands_df[cands_df["s1_id"].isin(s1_all_split)]
        split_cand_set = set(zip(split_cands["s1_id"], split_cands["sx_id"]))
        recovered_gold = len(gt_set & split_cand_set)
        standalone_recall = recovered_gold / total_gold if total_gold > 0 else 0.0

        # Baseline pool
        base_cache = pd.read_parquet(_CACHE_DIR / f"v2_{src_name}_{split}.parquet")
        base_set = set(zip(base_cache["s1_id"], base_cache["sx_id"]))
        base_present = gt_set & base_set
        base_recall = len(base_present) / total_gold if total_gold > 0 else 0.0

        # Union
        union_set = base_set | split_cand_set
        union_recovered = len(gt_set & union_set)
        union_recall = union_recovered / total_gold if total_gold > 0 else 0.0
        newly_recovered = union_recovered - len(base_present)
        added_cands = len(split_cand_set - base_set)
        noise_ratio = added_cands / max(newly_recovered, 1)

        print(f"\n  [{src_name.upper()} | {split.upper()}]")
        print(f"    Total Gold: {total_gold:,} | Baseline Recall: {base_recall:.4f} ({len(base_present):,}/{total_gold:,})")
        print(f"    Standalone Recall: {standalone_recall:.4f} ({recovered_gold:,}/{total_gold:,})")
        print(f"    Added Candidates (net new to pool): {added_cands:,}")
        print(f"    Newly Recovered Gold (net new): {newly_recovered:,}")
        print(f"    False Candidates per Recovered Gold: {noise_ratio:.1f}")
        print(f"    Union Recall: {union_recall:.4f} (+{union_recall - base_recall:+.4f} vs baseline)")

_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nPeak RAM: {peak / (1024**2):.1f} MB")
print("\nSTEP 3 COMPLETE.")
