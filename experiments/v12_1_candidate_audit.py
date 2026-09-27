"""V12.1 STEP 2: Candidate-Miss Audit on Untouched Validation Splits.

Quantifies every true match missing from the frozen V2 candidate pool and classifies:
- Indic/non-Latin script
- Transliteration mismatch
- Abbreviation/domain variant
- Address variation
- House-number variation
- Postal variation
- Word-order variation
- Legal suffix variation
- Short-name ambiguity
- Same-name/different-location
- Other
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from rapidfuzz import distance, fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet

NUM_RE = re.compile(r"\b\d+\b")
POSTAL_RE = re.compile(r"\b(\d{5,6})\b")
DEV_RE = re.compile(r"[\u0900-\u097F]")  # Devanagari script


def s1_norm_path() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def sx_norm_path(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def gt_path() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def get_missed_pairs(src: str, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find all true gold pairs and identify those absent from the cached candidate pool."""
    prefix = "S2-" if src == "source2" else "S3-"
    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == "val1":
        where_cond = f"{h_expr} >= 0 AND {h_expr} < 5"
    elif split_name == "val2":
        where_cond = f"{h_expr} >= 25 AND {h_expr} < 30"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    con = duckdb.connect()
    con.execute(f"""
        CREATE TEMP TABLE s1_split AS
        SELECT entity_id AS s1_id
        FROM read_parquet('{s1_norm_path()}')
        WHERE {where_cond}
    """)

    gold_df = con.execute(f"""
        WITH unnested AS (
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{gt_path()}')
            WHERE matched_entity_ids != ''
              AND source1_entity_id IN (SELECT s1_id FROM s1_split)
        )
        SELECT s1_id, true_match_id AS sx_id
        FROM unnested
        WHERE true_match_id LIKE '{prefix}%'
    """).df()

    # Join metadata
    con.register("gold_pairs", gold_df)
    meta_df = con.execute(f"""
        SELECT g.s1_id, g.sx_id,
               s1.raw_name AS s1_raw_name, s1.name_clean AS s1_name,
               s1.raw_address AS s1_raw_addr, s1.addr_clean AS s1_addr,
               s1.postal_code AS s1_post, s1.country_clean AS s1_country,
               sx.raw_name AS sx_raw_name, sx.name_clean AS sx_name,
               sx.raw_address AS sx_raw_addr, sx.addr_clean AS sx_addr,
               sx.postal_code AS sx_post, sx.country_clean AS sx_country
        FROM gold_pairs g
        JOIN read_parquet('{s1_norm_path()}') s1 ON g.s1_id = s1.entity_id
        JOIN read_parquet('{sx_norm_path(src)}') sx ON g.sx_id = sx.entity_id
    """).df()
    con.close()

    # Load cached candidate pool
    cache_path = ROOT / f"intermediate/v2_cache/v2_{src}_{split_name}.parquet"
    cand_df = pd.read_parquet(cache_path, columns=["s1_id", "sx_id"])
    cand_set = set(zip(cand_df["s1_id"], cand_df["sx_id"]))

    meta_df["in_candidates"] = [(s, x) in cand_set for s, x in zip(meta_df["s1_id"], meta_df["sx_id"])]
    missed_df = meta_df[~meta_df["in_candidates"]].copy()

    return meta_df, missed_df


def classify_miss(row: pd.Series) -> str:
    """Classify the root cause of why a gold pair was missed by candidate blocking."""
    s1_raw = str(row["s1_raw_name"])
    sx_raw = str(row["sx_raw_name"])
    s1_n = str(row["s1_name"])
    sx_n = str(row["sx_name"])
    s1_a = str(row["s1_addr"])
    sx_a = str(row["sx_addr"])

    # 1. Indic / Non-Latin script
    has_indic = bool(DEV_RE.search(s1_raw) or DEV_RE.search(sx_raw))
    is_non_latin = any(ord(c) > 127 for c in s1_raw + sx_raw)
    if has_indic:
        return "indic_devanagari_script"

    # 2. Transliteration mismatch (non-Latin or phonetic divergence)
    s1_ascii = unicodedata.normalize("NFKD", s1_raw).encode("ascii", "ignore").decode("ascii").lower()
    sx_ascii = unicodedata.normalize("NFKD", sx_raw).encode("ascii", "ignore").decode("ascii").lower()
    if is_non_latin and (s1_ascii == sx_ascii or fuzz.ratio(s1_ascii, sx_ascii) >= 80):
        return "transliteration_mismatch"

    # 3. Short-name ambiguity (< 5 characters)
    if len(s1_n) <= 4 or len(sx_n) <= 4:
        return "short_name_ambiguity"

    # 4. Word-order variation (high token sort ratio, lower ratio)
    tok_sort = fuzz.token_sort_ratio(s1_n, sx_n)
    raw_ratio = fuzz.ratio(s1_n, sx_n)
    if tok_sort >= 85 and raw_ratio < 65:
        return "word_order_variation"

    # 5. Same name, different location / address variation
    s1_toks = set(s1_n.split())
    sx_toks = set(sx_n.split())
    if s1_n == sx_n or (len(s1_toks & sx_toks) / max(len(s1_toks | sx_toks), 1) >= 0.75):
        # Name is nearly identical, so blocker missed due to address / location constraint
        s1_nums = set(NUM_RE.findall(s1_a))
        sx_nums = set(NUM_RE.findall(sx_a))
        if s1_nums and sx_nums and not (s1_nums & sx_nums):
            return "house_number_variation"
        if row["s1_post"] and row["sx_post"] and row["s1_post"] != row["sx_post"]:
            return "postal_variation"
        return "address_variation"

    # 6. Abbreviation / Domain variant (e.g. initials, acronyms)
    s1_acronym = "".join(w[0] for w in s1_n.split() if w)
    sx_acronym = "".join(w[0] for w in sx_n.split() if w)
    if (len(s1_acronym) >= 2 and s1_acronym == sx_n) or (len(sx_acronym) >= 2 and sx_acronym == s1_n):
        return "abbreviation_variant"

    # 7. Low lexical overlap
    if fuzz.token_set_ratio(s1_n, sx_n) >= 70:
        return "fuzzy_name_variation"

    return "other_heavy_divergence"


def run_audit():
    print("=" * 80)
    print("V12.1 STEP 2: CANDIDATE-MISS ERROR AUDIT ON UNTOUCHED VAL1 AND VAL2")
    print("=" * 80)

    for src in ["source2", "source3"]:
        print(f"\nAUDIT FOR {src.upper()}:")
        all_gold_list = []
        all_missed_list = []

        for split in ["val1", "val2"]:
            gold_df, missed_df = get_missed_pairs(src, split)
            all_gold_list.append(gold_df)
            all_missed_list.append(missed_df)

        total_gold = pd.concat(all_gold_list, ignore_index=True)
        total_missed = pd.concat(all_missed_list, ignore_index=True)

        recall = (len(total_gold) - len(total_missed)) / len(total_gold)
        print(f"  Total reachable gold pairs: {len(total_gold):,}")
        print(f"  Captured in V2 pool:        {len(total_gold) - len(total_missed):,} ({recall*100:.2f}%)")
        print(f"  Missed gold pairs:          {len(total_missed):,} ({(1-recall)*100:.2f}%)")

        total_missed["miss_category"] = [classify_miss(row) for _, row in total_missed.iterrows()]
        counts = Counter(total_missed["miss_category"])

        print("\n  Miss Root Causes:")
        for cat, cnt in counts.most_common():
            pct = cnt / len(total_missed) * 100
            print(f"    - {cat:<28}: {cnt:>4} ({pct:>5.1f}%)")

        # Print 5 concrete examples
        print("\n  Sample Missed Pairs:")
        for idx, row in total_missed.head(5).iterrows():
            print(f"    [{row['miss_category']}]")
            print(f"      S1: {row['s1_raw_name']} | {row['s1_raw_addr']} ({row['s1_country']})")
            print(f"      SX: {row['sx_raw_name']} | {row['sx_raw_addr']} ({row['sx_country']})")


if __name__ == "__main__":
    run_audit()
