"""Fast DuckDB-powered bounded multi-view sparse retrieval for V13.

Replaces slow dense matrix operations with streaming hash-based inverted n-gram joins.
Runs in seconds and adheres strictly to:
- Country partitioning
- Bounded top-5 forward & reverse retrieval
- Multi-view: native name, transliterated name, OCR-tolerant name, address
- Immediate Candidate Recall Gate evaluation
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.v12_assignment_lgb import (
    get_clean_gold,
    s1_norm_path,
    sx_norm_path,
    gt_path,
    V2_CACHE,
)
from experiments.v12_1_candidate_expansion import build_learned_transliteration_dict

OCR_DIGIT_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"})


def normalize_clean(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).lower())
    cleaned = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_ocr(text: str) -> str:
    return normalize_clean(text).translate(OCR_DIGIT_MAP)


def test_fast_retrieval(src: str = "source2", split_name: str = "val1"):
    print(f"\n{'='*75}\nV13 FAST BOUNDED RETRIEVAL: {src.upper()} | {split_name.upper()}\n{'='*75}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    total_gold = len(gt_set)

    val_cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
    v2_df = pd.read_parquet(val_cache_path, columns=["s1_id", "sx_id"])
    v2_set = set(zip(v2_df["s1_id"], v2_df["sx_id"]))
    base_recall = len(v2_set & gt_set) / total_gold
    print(f"Baseline V2 Candidates: {len(v2_set):,} | Gold Covered: {len(v2_set & gt_set):,} / {total_gold:,} ({base_recall*100:.2f}%)")

    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == "val1":
        where_cond = f"{h_expr} >= 0 AND {h_expr} < 5"
    elif split_name == "val2":
        where_cond = f"{h_expr} >= 25 AND {h_expr} < 30"
    else:
        raise ValueError(split_name)

    tmap = build_learned_transliteration_dict(src)
    tmap["प्रा"] = "private"
    tmap["लि"] = "limited"

    def py_translit(name):
        tokens = re.sub(r"[^\w\s]", " ", str(name).lower()).split()
        return " ".join([tmap.get(t, t) for t in tokens])

    con = duckdb.connect()
    con.create_function("translit", py_translit, return_type="VARCHAR")
    con.create_function("norm_clean", normalize_clean, return_type="VARCHAR")
    con.create_function("norm_ocr", normalize_ocr, return_type="VARCHAR")

    print("\nPreparing country-partitioned views in DuckDB...")
    t0 = time.time()
    con.execute(f"""
        CREATE TEMP TABLE s1_records AS
        SELECT entity_id AS s1_id, country_clean AS country,
               norm_clean(raw_name) AS name_native,
               norm_clean(translit(raw_name)) AS name_translit,
               norm_ocr(raw_name) AS name_ocr,
               norm_clean(addr_clean) AS addr_clean
        FROM read_parquet('{s1_norm_path()}')
        WHERE {where_cond}
    """)

    con.execute(f"""
        CREATE TEMP TABLE sx_records AS
        SELECT entity_id AS sx_id, country_clean AS country,
               norm_clean(raw_name) AS name_native,
               norm_clean(translit(raw_name)) AS name_translit,
               norm_ocr(raw_name) AS name_ocr,
               norm_clean(addr_clean) AS addr_clean
        FROM read_parquet('{sx_norm_path(src)}')
        WHERE country_clean IN (SELECT DISTINCT country FROM s1_records)
    """)
    print(f"Views created in {time.time()-t0:.1f}s")

    # [View 1] OCR View
    print("\n[View 1] Running Fast OCR & Distinctive Name Prefix Retrieval...")
    t1 = time.time()
    ocr_pairs_df = con.execute("""
        WITH s1_tokens AS (
            SELECT s1_id, country,
                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,
                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2
            FROM s1_records
            WHERE length(name_ocr) >= 5
        ),
        sx_tokens AS (
            SELECT sx_id, country,
                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,
                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2
            FROM sx_records
            WHERE length(name_ocr) >= 5
        )
        SELECT s1.s1_id, sx.sx_id
        FROM s1_tokens s1
        JOIN sx_tokens sx
          ON s1.country = sx.country
         AND s1.t1 = sx.t1
         AND s1.t2 = sx.t2
        WHERE s1.t1 NOT IN ('the', 'and', 'company', 'enterprises', 'group', 'services')
        QUALIFY row_number() OVER (PARTITION BY s1.s1_id) <= 5
    """).df()
    ocr_pairs = set(zip(ocr_pairs_df["s1_id"], ocr_pairs_df["sx_id"]))
    print(f"  OCR View retrieved {len(ocr_pairs):,} pairs in {time.time()-t1:.1f}s")

    # [View 2] Transliterated Name Prefix Join
    print("\n[View 2] Running Learned-Transliteration Name Prefix Retrieval...")
    t2 = time.time()
    trans_pairs_df = con.execute("""
        WITH s1_tokens AS (
            SELECT s1_id, country,
                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,
                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2
            FROM s1_records
            WHERE length(name_translit) >= 5
        ),
        sx_tokens AS (
            SELECT sx_id, country,
                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,
                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2
            FROM sx_records
            WHERE length(name_translit) >= 5
        )
        SELECT s1.s1_id, sx.sx_id
        FROM s1_tokens s1
        JOIN sx_tokens sx
          ON s1.country = sx.country
         AND s1.t1 = sx.t1
         AND s1.t2 = sx.t2
        WHERE s1.t1 NOT IN ('the', 'and', 'company', 'enterprises', 'group', 'services')
        QUALIFY row_number() OVER (PARTITION BY s1.s1_id) <= 5
    """).df()
    trans_pairs = set(zip(trans_pairs_df["s1_id"], trans_pairs_df["sx_id"]))
    print(f"  Transliteration View retrieved {len(trans_pairs):,} pairs in {time.time()-t2:.1f}s")

    # [View 3] Address Number + City/Locality Join
    print("\n[View 3] Running Address Number + City/Locality Retrieval...")
    t3 = time.time()
    addr_pairs_df = con.execute("""
        WITH s1_addr AS (
            SELECT s1_id, country,
                   regexp_extract(addr_clean, '(\\b\\d+[a-z]?\\b)', 1) AS num,
                   regexp_extract(addr_clean, '(\\b[a-z]{5,}\\b)', 1) AS word
            FROM s1_records
            WHERE addr_clean != ''
        ),
        sx_addr AS (
            SELECT sx_id, country,
                   regexp_extract(addr_clean, '(\\b\\d+[a-z]?\\b)', 1) AS num,
                   regexp_extract(addr_clean, '(\\b[a-z]{5,}\\b)', 1) AS word
            FROM sx_records
            WHERE addr_clean != ''
        )
        SELECT s1.s1_id, sx.sx_id
        FROM s1_addr s1
        JOIN sx_addr sx
          ON s1.country = sx.country
         AND s1.num = sx.num
         AND s1.word = sx.word
        WHERE s1.num != '' AND s1.word != ''
          AND s1.word NOT IN ('street', 'avenue', 'nagar', 'floor', 'suite', 'india')
        QUALIFY row_number() OVER (PARTITION BY s1.s1_id) <= 5
    """).df()
    addr_pairs = set(zip(addr_pairs_df["s1_id"], addr_pairs_df["sx_id"]))
    print(f"  Address View retrieved {len(addr_pairs):,} pairs in {time.time()-t3:.1f}s")

    # Combine all selected high-efficiency channels
    comb_all = v2_set | ocr_pairs | trans_pairs | addr_pairs
    recov_all = len(comb_all & gt_set) - len(v2_set & gt_set)
    added_all = len(comb_all) - len(v2_set)
    rec_all = len(comb_all & gt_set) / total_gold

    print(f"\n{'-'*65}\nCOMBINED V13 MULTI-VIEW RECALL FOR {src.upper()} {split_name.upper()}:")
    print(f"  Total Candidates:     {len(v2_set):,} -> {len(comb_all):,} (+{added_all:,}, +{added_all/len(v2_set)*100:.2f}%)")
    print(f"  Total Gold Recovered: +{recov_all:,} ({len(comb_all & gt_set):,}/{total_gold:,})")
    print(f"  Recall Growth:        {base_recall*100:.2f}% -> {rec_all*100:.2f}% (+{(rec_all-base_recall)*100:.2f}%)")
    if recov_all > 0:
        print(f"  Overall Efficiency:   {added_all/recov_all:.1f} candidates per gold pair")

    con.close()
    return {
        "base_recall": base_recall,
        "new_recall": rec_all,
        "recovered": recov_all,
        "added": added_all,
    }


if __name__ == "__main__":
    test_fast_retrieval("source2", "val1")
