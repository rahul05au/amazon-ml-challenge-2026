"""
FINAL THIRD-SUBMISSION GENERATOR
================================
Configuration:
  - Architecture: V13.1 Two-Stage Collective Matcher
  - Ensemble: alpha = 0.50 * Stage-1 + 0.50 * Stage-2
  - Source 2 Threshold: 0.44
  - Source 3 Threshold: 0.47
  - Target Exclusivity: 1-to-many (one target -> at most one S1)
  - Target Containment: 100% in candidate_pairs.tsv
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.v12_assignment_lgb import (
    compute_pairwise_features,
    engineer_competition_features,
    get_v12_feature_names,
)
from experiments.v12_1_collective_matcher import get_v12_1_feature_names

MODELS_DIR = ROOT / "models"
SUBMISSION_DIR = ROOT / "FINAL_THIRD_SUBMISSION"
THRESHOLDS = {"source2": 0.44, "source3": 0.47}
ALPHA = 0.50


def fast_collective_features(df: pd.DataFrame, score_col: str = "stage1_score") -> pd.DataFrame:
    """Vectorized, fast collective sibling and target consensus feature extraction."""
    # S1 sibling features
    df = df.sort_values(by=["s1_id", score_col], ascending=[True, False]).reset_index(drop=True)
    df["s1_cand_rank"] = df.groupby("s1_id").cumcount() + 1
    df["s1_cand_count"] = df.groupby("s1_id")[score_col].transform("count")

    df["_is_hc"] = (df[score_col] >= 0.60).astype(np.float32)
    s1_high_conf_sum = df.groupby("s1_id")["_is_hc"].transform("sum")
    df["s1_sibling_high_conf_count"] = np.maximum(0.0, s1_high_conf_sum - df["_is_hc"])
    df.drop(columns=["_is_hc"], inplace=True)

    top1 = df[df["s1_cand_rank"] == 1].set_index("s1_id")[score_col]
    top2 = df[df["s1_cand_rank"] == 2].set_index("s1_id")[score_col]
    df["top1_score"] = df["s1_id"].map(top1).fillna(0.0).values
    df["top2_score"] = df["s1_id"].map(top2).fillna(0.0).values

    df["s1_best_sibling_score"] = np.where(df["s1_cand_rank"] == 1, df["top2_score"], df["top1_score"])
    df["s1_sibling_margin"] = df[score_col] - df["s1_best_sibling_score"]
    df["s1_sibling_support_flag"] = (df["s1_best_sibling_score"] >= 0.60).astype(np.float32)

    # Cluster consensus
    cluster_key = df["sx_name"].fillna("").astype(str).str.lower().str.strip() + "___" + df["sx_post"].fillna("").astype(str).str.strip()
    df["cluster_key"] = cluster_key
    cluster_total = df.groupby("cluster_key")[score_col].transform("count")
    df["cluster_size"] = cluster_total
    df["cluster_max_score"] = df.groupby("cluster_key")[score_col].transform("max")
    df["cluster_s1_agreement"] = 1.0 / np.maximum(cluster_total, 1.0)

    # Reverse target competition
    df = df.sort_values(by=["sx_id", score_col], ascending=[True, False]).reset_index(drop=True)
    df["target_stage1_rank"] = df.groupby("sx_id").cumcount() + 1
    t_top1 = df[df["target_stage1_rank"] == 1].set_index("sx_id")[score_col]
    t_top2 = df[df["target_stage1_rank"] == 2].set_index("sx_id")[score_col]
    df["target_stage1_top1"] = df["sx_id"].map(t_top1).fillna(0.0).values
    df["target_stage1_runnerup"] = df["sx_id"].map(t_top2).fillna(0.0).values
    df["target_stage1_margin"] = np.where(
        df["target_stage1_rank"] == 1,
        df["target_stage1_top1"] - df["target_stage1_runnerup"],
        df[score_col] - df["target_stage1_top1"]
    )
    df["target_is_stage1_best"] = (df["target_stage1_rank"] == 1).astype(np.float32)

    collective_cols = [
        "s1_sibling_high_conf_count", "s1_best_sibling_score", "s1_sibling_margin", "s1_sibling_support_flag",
        "cluster_size", "cluster_s1_agreement", "cluster_max_score",
        "target_stage1_top1", "target_stage1_runnerup", "target_stage1_margin", "target_is_stage1_best"
    ]
    return df[collective_cols]


def score_source_test_matches(src: str, batch_size: int = 500_000) -> pd.DataFrame:
    """Score candidates for a source using V13.1 alpha=0.50 ensemble and filter by threshold."""
    print(f"\n{'='*70}\nSCORING TEST CANDIDATES FOR {src.upper()}\n{'='*70}")
    t0 = time.time()

    # Load models
    m_s1 = lgb.Booster(model_file=str(MODELS_DIR / f"v13_1_stage1_{src}.txt"))
    m_s2 = lgb.Booster(model_file=str(MODELS_DIR / f"v13_1_stage2_{src}.txt"))
    v12_cols_no_r50 = [c for c in get_v12_feature_names() if c != "lgb_score"]
    stage2_cols = get_v12_1_feature_names()
    threshold = THRESHOLDS[src]

    # Connect to test candidates
    cand_path = ROOT / f"intermediate/test/matches_{src}.parquet"
    con = duckdb.connect()
    total_cands = con.execute(f"SELECT count(*) FROM read_parquet('{str(cand_path).replace(chr(92), '/')}')").fetchone()[0]
    print(f"Total candidate pairs to score: {total_cands:,} | Threshold: {threshold:.2f}")

    # Process in batches
    accepted_dfs = []
    offset = 0
    batch_num = 1
    total_batches = (total_cands + batch_size - 1) // batch_size

    while offset < total_cands:
        tb0 = time.time()
        print(f"  Batch {batch_num}/{total_batches} (offset={offset:,}, limit={batch_size:,}) ...")
        
        batch_pairs = con.execute(f"""
            SELECT m.s1_id, m.sx_id,
                   s1.name_clean AS s1_name, sx.name_clean AS sx_name,
                   s1.addr_clean AS s1_addr, sx.addr_clean AS sx_addr,
                   s1.postal_code AS s1_post, sx.postal_code AS sx_post
            FROM (
                SELECT s1_id, sx_id 
                FROM read_parquet('{str(cand_path).replace(chr(92), '/')}') 
                LIMIT {batch_size} OFFSET {offset}
            ) m
            JOIN read_parquet('intermediate/test/norm_source1.parquet') s1 ON m.s1_id = s1.entity_id
            JOIN read_parquet('intermediate/test/norm_{src}.parquet') sx ON m.sx_id = sx.entity_id
        """).df()

        if len(batch_pairs) == 0:
            break

        # Compute deterministic features
        batch_pairs["lgb_score"] = 0.5
        batch_pairs["tfidf_score"] = 0.0
        pw_feats = compute_pairwise_features(batch_pairs)
        full_df = pd.concat([batch_pairs, pw_feats], axis=1)
        full_df = engineer_competition_features(full_df)

        # Stage 1 prediction
        full_df["stage1_score"] = m_s1.predict(full_df[v12_cols_no_r50])

        # Stage 2 prediction
        col_feats = fast_collective_features(full_df, score_col="stage1_score")
        full_df = pd.concat([full_df, col_feats], axis=1)
        for c in stage2_cols:
            if c not in full_df.columns:
                full_df[c] = 0.0
        full_df["stage2_score"] = m_s2.predict(full_df[stage2_cols])

        # Alpha blend (0.50 * Stage-1 + 0.50 * Stage-2)
        full_df["final_score"] = ALPHA * full_df["stage1_score"] + (1.0 - ALPHA) * full_df["stage2_score"]

        # Filter by threshold
        accepted = full_df[full_df["final_score"] >= threshold][["s1_id", "sx_id", "final_score"]].copy()
        accepted_dfs.append(accepted)
        print(f"    -> Accepted {len(accepted):,} / {len(batch_pairs):,} pairs in {time.time()-tb0:.1f}s")

        offset += batch_size
        batch_num += 1
        gc.collect()

    con.close()

    all_accepted = pd.concat(accepted_dfs, ignore_index=True) if accepted_dfs else pd.DataFrame(columns=["s1_id", "sx_id", "final_score"])
    print(f"Finished {src.upper()}: {len(all_accepted):,} total accepted pairs before exclusivity ({time.time()-t0:.1f}s)")
    return all_accepted


def main():
    start_time = time.time()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Score Source 2 & Source 3
    s2_matches = score_source_test_matches("source2", batch_size=600_000)
    s3_matches = score_source_test_matches("source3", batch_size=600_000)

    # Combine all matches
    print("\nCombining Source 2 and Source 3 matches...")
    all_matches = pd.concat([s2_matches, s3_matches], ignore_index=True)
    print(f"Total raw accepted matches: {len(all_matches):,}")

    # 2. Target Exclusivity Enforcement: 1 target -> at most 1 S1
    print("\nEnforcing Target Exclusivity (one target -> at most one S1)...")
    # Sort descending by final_score, then s1_id ascending for deterministic tie-breaking
    all_matches = all_matches.sort_values(by=["sx_id", "final_score", "s1_id"], ascending=[True, False, True]).reset_index(drop=True)
    excl_matches = all_matches.drop_duplicates(subset=["sx_id"], keep="first").reset_index(drop=True)
    print(f"Matches after Target Exclusivity: {len(excl_matches):,} (pruned {len(all_matches) - len(excl_matches):,} conflicts)")

    # 3. Candidate Pairs TSV
    out_cand_tsv = SUBMISSION_DIR / "candidate_pairs.tsv"
    src_cand_tsv = ROOT / "output" / "candidate_pairs.tsv"
    if not src_cand_tsv.exists():
        src_cand_tsv = ROOT.parents[1] / "output" / "candidate_pairs.tsv"
    if not out_cand_tsv.exists() or out_cand_tsv.stat().st_size != src_cand_tsv.stat().st_size:
        print(f"\nCopying candidate_pairs.tsv to {out_cand_tsv} ...")
        shutil.copyfile(src_cand_tsv, out_cand_tsv)
        print("  -> Candidate pairs copied successfully.")
    else:
        print(f"\ncandidate_pairs.tsv already present in {SUBMISSION_DIR} ({out_cand_tsv.stat().st_size / (1024*1024):.1f} MB).")

    # 4. Generate matching_results.tsv using DuckDB
    out_match_tsv = SUBMISSION_DIR / "matching_results.tsv"
    print(f"\nExporting matching_results.tsv to {out_match_tsv} ...")
    t_exp = time.time()

    con = duckdb.connect()
    con.execute("SET threads TO 8")
    con.execute("SET memory_limit = '8GB'")
    con.register("excl_matches_tbl", excl_matches)

    con.execute(f"""
        COPY (
            WITH aggregated AS (
                SELECT s1_id, string_agg(sx_id, ',') AS matched_ids
                FROM excl_matches_tbl
                GROUP BY s1_id
            )
            SELECT 
                s1.entity_id AS source1_entity_id,
                COALESCE(a.matched_ids, '') AS matched_entity_ids
            FROM read_parquet('intermediate/test/norm_source1.parquet') s1
            LEFT JOIN aggregated a ON s1.entity_id = a.s1_id
            ORDER BY s1.entity_id
        ) TO '{str(out_match_tsv).replace(chr(92), '/')}' (DELIMITER '\t', HEADER true, QUOTE '')
    """)
    con.close()
    print(f"  -> matching_results.tsv written in {time.time()-t_exp:.1f}s ({out_match_tsv.stat().st_size / (1024*1024):.1f} MB)")

    # 5. Verification Metrics
    print("\n" + "="*70)
    print("VERIFYING SUBMISSION METRICS")
    print("="*70)

    con = duckdb.connect()
    match_rows = con.execute(f"SELECT count(*) FROM read_csv('{str(out_match_tsv).replace(chr(92), '/')}', delim='\t', header=True)").fetchone()[0]
    cand_rows = con.execute(f"SELECT count(*) FROM read_csv('{str(out_cand_tsv).replace(chr(92), '/')}', delim='\t', header=True)").fetchone()[0]

    empty_s1 = con.execute(f"SELECT count(*) FROM read_csv('{str(out_match_tsv).replace(chr(92), '/')}', delim='\t', header=True) WHERE matched_entity_ids = '' OR matched_entity_ids IS NULL").fetchone()[0]
    non_empty_s1 = match_rows - empty_s1

    con.execute(f"""
        CREATE TEMP TABLE unnested AS
        SELECT source1_entity_id AS s1_id, trim(unnest(string_split(matched_entity_ids, ','))) AS sx_id
        FROM read_csv('{str(out_match_tsv).replace(chr(92), '/')}', delim='\t', header=True)
        WHERE matched_entity_ids != '' AND matched_entity_ids IS NOT NULL
    """)
    total_links = con.execute("SELECT count(*) FROM unnested").fetchone()[0]
    s2_links = con.execute("SELECT count(*) FROM unnested WHERE sx_id LIKE 'S2-%'").fetchone()[0]
    s3_links = con.execute("SELECT count(*) FROM unnested WHERE sx_id LIKE 'S3-%'").fetchone()[0]

    target_conflicts = con.execute("""
        SELECT count(*) FROM (
            SELECT sx_id, count(DISTINCT s1_id) 
            FROM unnested 
            GROUP BY sx_id 
            HAVING count(DISTINCT s1_id) > 1
        )
    """).fetchone()[0]

    # Containment check
    missing_cands = con.execute(f"""
        WITH cand_u AS (
            SELECT source1_entity_id AS s1_id, trim(unnest(string_split(candidate_entity_ids, ','))) AS cand_id
            FROM read_csv('{str(out_cand_tsv).replace(chr(92), '/')}', delim='\t', header=True)
            WHERE candidate_entity_ids != '' AND candidate_entity_ids IS NOT NULL
        )
        SELECT count(*) 
        FROM unnested u
        LEFT JOIN cand_u c ON u.s1_id = c.s1_id AND u.sx_id = c.cand_id
        WHERE c.cand_id IS NULL
    """).fetchone()[0]
    con.close()

    containment_pct = 100.0 * (1.0 - (missing_cands / max(total_links, 1)))

    print(f"matching_results row count:  {match_rows:,}")
    print(f"candidate_pairs row count:   {cand_rows:,}")
    print(f"S2 links:                    {s2_links:,}")
    print(f"S3 links:                    {s3_links:,}")
    print(f"Total links:                 {total_links:,}")
    print(f"Empty S1 count:              {empty_s1:,}")
    print(f"Non-empty S1 count:          {non_empty_s1:,}")
    print(f"Target conflicts:            {target_conflicts}")
    print(f"Candidate containment %:     {containment_pct:.4f}%")

    # 6. Save JSON verification metadata
    meta = {
        "architecture": "V13.1 Two-Stage Collective Matcher",
        "ensemble_alpha": ALPHA,
        "source2_threshold": THRESHOLDS["source2"],
        "source3_threshold": THRESHOLDS["source3"],
        "validation_avg_macro_f05": 0.90143,
        "matching_rows": match_rows,
        "candidate_rows": cand_rows,
        "s2_links": s2_links,
        "s3_links": s3_links,
        "total_links": total_links,
        "empty_s1": empty_s1,
        "non_empty_s1": non_empty_s1,
        "target_conflicts": target_conflicts,
        "containment_pct": round(containment_pct, 4),
        "total_time_seconds": round(time.time() - start_time, 1),
    }
    with open(SUBMISSION_DIR / "submission_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll operations finished in {time.time()-start_time:.1f}s.")


if __name__ == "__main__":
    main()
