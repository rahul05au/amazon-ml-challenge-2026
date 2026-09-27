"""V13 Multi-View Sparse Bounded Retrieval & Candidate Recall Expansion.

Implements:
- View A: Mark-safe Unicode normalization (native-script preserved)
- View B: Learned transliteration mined strictly from training positives
- View C: Country-partitioned character 3-gram TF-IDF (bounded top-k forward + reverse)
- View D: Country-partitioned address token TF-IDF
- View E: OCR / digit-tolerant character normalization
- Candidate Recall Gate & Efficiency Evaluation
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Common OCR digit-to-letter confusions identified in audit (e.g. East Fide1ity -> Fidelity)
OCR_DIGIT_MAP = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "5": "s",
    "8": "b",
})


def normalize_mark_safe(text: str) -> str:
    """NFC normalization preserving native scripts with clean punctuation."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text).lower())
    # Replace punctuation with space while preserving native unicode characters
    cleaned = re.sub(r"[^\w\s]", " ", norm)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_ocr_tolerant(text: str) -> str:
    """Character normalization mapping common OCR digits to letters."""
    norm = normalize_mark_safe(text)
    return norm.translate(OCR_DIGIT_MAP)


def run_bounded_tfidf_retrieval(
    s1_df: pd.DataFrame,
    sx_df: pd.DataFrame,
    s1_col: str,
    sx_col: str,
    country_col: str = "country",
    analyzer: str = "char",
    ngram_range: tuple[int, int] = (3, 3),
    top_k: int = 5,
    min_sim: float = 0.50,
    include_reverse: bool = True,
) -> set[tuple[str, str]]:
    """Sparse country-partitioned bounded top-k retrieval (Forward + Reverse)."""
    retrieved_pairs: set[tuple[str, str]] = set()

    # Partition by country
    countries = set(s1_df[country_col].dropna().unique()) & set(sx_df[country_col].dropna().unique())

    for c in countries:
        s1_c = s1_df[s1_df[country_col] == c]
        sx_c = sx_df[sx_df[country_col] == c]

        if len(s1_c) == 0 or len(sx_c) == 0:
            continue

        s1_texts = s1_c[s1_col].fillna("").tolist()
        sx_texts = sx_c[sx_col].fillna("").tolist()
        s1_ids = s1_c["s1_id"].tolist()
        sx_ids = sx_c["sx_id"].tolist()

        # Fit TF-IDF on combined country texts
        vec = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.8,
            norm="l2",
            dtype=np.float32,
        )

        try:
            vec.fit(s1_texts + sx_texts)
            X_s1 = vec.transform(s1_texts)
            X_sx = vec.transform(sx_texts)
        except ValueError:
            continue

        # Forward retrieval: S1 -> Top-k SX
        # Dot product in chunks to manage memory
        chunk_size = 1000
        for i in range(0, X_s1.shape[0], chunk_size):
            s1_chunk = X_s1[i : i + chunk_size]
            sims = s1_chunk.dot(X_sx.T).toarray()  # (chunk, n_sx)

            for local_idx, row_sims in enumerate(sims):
                global_s1_idx = i + local_idx
                # Get top_k indices
                top_indices = np.argpartition(-row_sims, min(top_k, len(row_sims) - 1))[:top_k]
                for sx_idx in top_indices:
                    sim = row_sims[sx_idx]
                    if sim >= min_sim:
                        retrieved_pairs.add((s1_ids[global_s1_idx], sx_ids[sx_idx]))

        # Reverse retrieval: SX -> Top-k S1
        if include_reverse:
            for j in range(0, X_sx.shape[0], chunk_size):
                sx_chunk = X_sx[j : j + chunk_size]
                sims_rev = sx_chunk.dot(X_s1.T).toarray()  # (chunk, n_s1)

                for local_idx, row_sims in enumerate(sims_rev):
                    global_sx_idx = j + local_idx
                    top_indices = np.argpartition(-row_sims, min(top_k, len(row_sims) - 1))[:top_k]
                    for s1_idx in top_indices:
                        sim = row_sims[s1_idx]
                        if sim >= min_sim:
                            retrieved_pairs.add((s1_ids[s1_idx], sx_ids[global_sx_idx]))

    return retrieved_pairs


def evaluate_retrieval_channels(src: str, split_name: str):
    """Benchmark each retrieval channel independently and measure candidate recall & efficiency."""
    print(f"\n{'='*75}\nEVALUATING RETRIEVAL CHANNELS: {src.upper()} | {split_name.upper()}\n{'='*75}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    total_gold = len(gt_set)

    # Load baseline V2 candidate pool
    val_cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
    v2_df = pd.read_parquet(val_cache_path, columns=["s1_id", "sx_id"])
    v2_set = set(zip(v2_df["s1_id"], v2_df["sx_id"]))
    base_recall = len(v2_set & gt_set) / total_gold
    print(f"Baseline Candidate Pool: {len(v2_set):,} candidates | Recall: {base_recall*100:.2f}% ({len(v2_set & gt_set):,}/{total_gold:,})")

    # Load normalized text records
    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == "val1":
        where_cond = f"{h_expr} >= 0 AND {h_expr} < 5"
    elif split_name == "val2":
        where_cond = f"{h_expr} >= 25 AND {h_expr} < 30"
    else:
        raise ValueError(split_name)

    con = duckdb.connect()
    s1_df = con.execute(f"""
        SELECT entity_id AS s1_id, raw_name AS s1_raw, name_clean AS s1_name,
               addr_clean AS s1_addr, country_clean AS country
        FROM read_parquet('{s1_norm_path()}')
        WHERE {where_cond}
    """).df()

    val_countries = set(s1_df["country"].dropna().unique())
    country_filter = "','".join(val_countries)

    sx_df = con.execute(f"""
        SELECT entity_id AS sx_id, raw_name AS sx_raw, name_clean AS sx_name,
               addr_clean AS sx_addr, country_clean AS country
        FROM read_parquet('{sx_norm_path(src)}')
        WHERE country_clean IN ('{country_filter}')
    """).df()
    con.close()

    print(f"Loaded {len(s1_df):,} S1 queries and {len(sx_df):,} target candidates in active countries.")

    # 1. Prepare Normalized Views
    tmap = build_learned_transliteration_dict(src)
    tmap["प्रा"] = "private"
    tmap["लि"] = "limited"

    def py_translit(name):
        tokens = re.sub(r"[^\w\s]", " ", str(name).lower()).split()
        return " ".join([tmap.get(t, t) for t in tokens])

    s1_df["name_native"] = s1_df["s1_raw"].apply(normalize_mark_safe)
    sx_df["name_native"] = sx_df["sx_raw"].apply(normalize_mark_safe)

    s1_df["name_translit"] = s1_df["s1_raw"].apply(py_translit).apply(normalize_mark_safe)
    sx_df["name_translit"] = sx_df["sx_raw"].apply(py_translit).apply(normalize_mark_safe)

    s1_df["name_ocr"] = s1_df["s1_raw"].apply(normalize_ocr_tolerant)
    sx_df["name_ocr"] = sx_df["sx_raw"].apply(normalize_ocr_tolerant)

    s1_df["addr_tokens"] = s1_df["s1_addr"].fillna("").apply(normalize_mark_safe)
    sx_df["addr_tokens"] = sx_df["sx_addr"].fillna("").apply(normalize_mark_safe)

    # 2. Test Channel 1: Native Name Character 3-Gram TF-IDF (top_k=5, min_sim=0.60)
    print("\n[Channel 1] Testing Native Name Char 3-Gram TF-IDF (k=5, sim>=0.60)...")
    t0 = time.time()
    c1_pairs = run_bounded_tfidf_retrieval(
        s1_df, sx_df, "name_native", "name_native",
        analyzer="char", ngram_range=(3, 3), top_k=5, min_sim=0.60, include_reverse=True
    )
    comb1 = v2_set | c1_pairs
    recov1 = len(comb1 & gt_set) - len(v2_set & gt_set)
    added1 = len(comb1) - len(v2_set)
    rec1 = len(comb1 & gt_set) / total_gold
    print(f"  Pairs retrieved: {len(c1_pairs):,} in {time.time()-t0:.1f}s")
    print(f"  Recall: {base_recall*100:.2f}% -> {rec1*100:.2f}% (+{(rec1-base_recall)*100:.2f}%)")
    print(f"  Gold recovered: +{recov1:,} | Added cands: +{added1:,}")
    if recov1 > 0:
        print(f"  Efficiency: {added1/recov1:.1f} candidates per gold pair")

    # 3. Test Channel 2: Transliterated Name Char 3-Gram TF-IDF (top_k=5, min_sim=0.60)
    print("\n[Channel 2] Testing Transliterated Name Char 3-Gram TF-IDF (k=5, sim>=0.60)...")
    t0 = time.time()
    c2_pairs = run_bounded_tfidf_retrieval(
        s1_df, sx_df, "name_translit", "name_translit",
        analyzer="char", ngram_range=(3, 3), top_k=5, min_sim=0.60, include_reverse=True
    )
    comb2 = v2_set | c2_pairs
    recov2 = len(comb2 & gt_set) - len(v2_set & gt_set)
    added2 = len(comb2) - len(v2_set)
    rec2 = len(comb2 & gt_set) / total_gold
    print(f"  Pairs retrieved: {len(c2_pairs):,} in {time.time()-t0:.1f}s")
    print(f"  Recall: {base_recall*100:.2f}% -> {rec2*100:.2f}% (+{(rec2-base_recall)*100:.2f}%)")
    print(f"  Gold recovered: +{recov2:,} | Added cands: +{added2:,}")
    if recov2 > 0:
        print(f"  Efficiency: {added2/recov2:.1f} candidates per gold pair")

    # 4. Test Channel 3: OCR-Tolerant Name Char 3-Gram TF-IDF (top_k=5, min_sim=0.65)
    print("\n[Channel 3] Testing OCR-Tolerant Name Char 3-Gram TF-IDF (k=5, sim>=0.65)...")
    t0 = time.time()
    c3_pairs = run_bounded_tfidf_retrieval(
        s1_df, sx_df, "name_ocr", "name_ocr",
        analyzer="char", ngram_range=(3, 3), top_k=5, min_sim=0.65, include_reverse=True
    )
    comb3 = v2_set | c3_pairs
    recov3 = len(comb3 & gt_set) - len(v2_set & gt_set)
    added3 = len(comb3) - len(v2_set)
    rec3 = len(comb3 & gt_set) / total_gold
    print(f"  Pairs retrieved: {len(c3_pairs):,} in {time.time()-t0:.1f}s")
    print(f"  Recall: {base_recall*100:.2f}% -> {rec3*100:.2f}% (+{(rec3-base_recall)*100:.2f}%)")
    print(f"  Gold recovered: +{recov3:,} | Added cands: +{added3:,}")
    if recov3 > 0:
        print(f"  Efficiency: {added3/recov3:.1f} candidates per gold pair")

    # 5. Combined Selected High-Efficiency Channels Union
    comb_all = v2_set | c1_pairs | c2_pairs | c3_pairs
    recov_all = len(comb_all & gt_set) - len(v2_set & gt_set)
    added_all = len(comb_all) - len(v2_set)
    rec_all = len(comb_all & gt_set) / total_gold
    print(f"\n{'-'*60}\nFINAL V13 CANDIDATE POOL UNION FOR {src.upper()} {split_name.upper()}:")
    print(f"  Total Candidates:     {len(v2_set):,} -> {len(comb_all):,} (+{added_all:,})")
    print(f"  Total Gold Recovered: +{recov_all:,} ({len(comb_all & gt_set):,}/{total_gold:,})")
    print(f"  Recall Growth:        {base_recall*100:.2f}% -> {rec_all*100:.2f}% (+{(rec_all-base_recall)*100:.2f}%)")
    if recov_all > 0:
        print(f"  Overall Efficiency:   {added_all/recov_all:.1f} candidates per recovered gold")


def main():
    evaluate_retrieval_channels("source2", "val1")


if __name__ == "__main__":
    main()
