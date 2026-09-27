"""V6 Phase 0: Oracle Ceiling Analysis & Bottleneck Decomposition.

Calculates:
  3A. Current blocking recall on val1 and val2 for S2 and S3:
      - true pairs present vs missed
      - blocking recall
  3B. Oracle matcher ceiling:
      - perfect matcher on current candidate sets
      - Macro F0.5, precision, recall, singleton accuracy
      - gap to active R50 baseline
  3C. Oracle blocking / matcher diagnosis:
      - inject all missing gold pairs into candidate set
      - score with frozen active R50 LightGBM
      - Macro F0.5, precision, recall
  4. Bottleneck classification: BLOCKING-DOMINANT, MATCHER-DOMINANT, or BOTH.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet
from src.features import compute_features_parallel
from src.matcher import MODELS_DIR, calculate_macro_f05

_V2_THRESH = {"source2": 0.70, "source3": 0.65}
_CACHE_DIR = ROOT / "intermediate" / "v2_cache"

_H = "abs(hash(entity_id || 'v1')) % 1000"
_WHERE = {
    "val1": f"{_H} >= 0  AND {_H} < 5",
    "val2": f"{_H} >= 25 AND {_H} < 30",
}


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET threads TO 6; SET memory_limit = '6GB'")
    return c


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def get_split_s1_and_gt(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    prefix = "S2-" if src == "source2" else "S3-"
    c = _con()
    c.execute(f"""
        CREATE TEMP TABLE _s1 AS
        SELECT entity_id AS s1_id
        FROM read_parquet('{_s1pq()}')
        WHERE {_WHERE[split_name]}
    """)
    s1_all = set(c.execute("SELECT s1_id FROM _s1").df()["s1_id"])
    gt_df = c.execute(f"""
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
        FROM read_parquet('{_gtpq()}')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
          AND source1_entity_id IN (SELECT s1_id FROM _s1)
    """).df()
    c.close()
    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
    return s1_all, gt_set, gt_df


def load_val_candidates_with_features(src: str, split_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    cache_path = _CACHE_DIR / f"v2_{src}_{split_name}.parquet"
    cache = pd.read_parquet(cache_path)

    c = _con()
    c.register("_c", cache[["s1_id", "sx_id"]])
    ns = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf
        FROM _c c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()

    feat_df = cache.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({"s1_no_suf": "", "sx_no_suf": ""})
    X = compute_features_parallel(feat_df, n_workers=6)
    return feat_df, X


def run_phase_0():
    print("=" * 80)
    print("PHASE 0: ORACLE CEILING ANALYSIS & ERROR DECOMPOSITION")
    print("=" * 80)

    summary_results = {}

    for src in ["source2", "source3"]:
        print(f"\nEvaluating target source: {src.upper()}")
        print("-" * 50)
        bst = lgb.Booster(model_file=str(MODELS_DIR / f"lgb_{src}.txt"))
        thresh = _V2_THRESH[src]

        for split in ["val1", "val2"]:
            t0 = time.time()
            s1_all, gt_set, gt_df = get_split_s1_and_gt(src, split)
            total_gold = len(gt_set)

            # Load current candidate set and compute features
            feat_df, X = load_val_candidates_with_features(src, split)
            cand_pairs = feat_df[["s1_id", "sx_id"]]
            cand_set = set(zip(cand_pairs["s1_id"], cand_pairs["sx_id"]))
            total_cands = len(cand_set)

            # 3A: Blocking recall
            present_gold = gt_set & cand_set
            missed_gold = gt_set - cand_set
            blocking_recall = len(present_gold) / total_gold if total_gold > 0 else 0.0

            # Active R50 predictions on current candidates
            r50_scores = bst.predict(X)
            r50_preds = cand_pairs[r50_scores >= thresh][["s1_id", "sx_id"]]
            r50_eval = calculate_macro_f05(r50_preds, gt_df, s1_all)

            # 3B: Oracle matcher with current candidates
            # Perfect matcher accepts pair iff it is a known gold match
            oracle_mask = [(s, x) in gt_set for s, x in zip(cand_pairs["s1_id"], cand_pairs["sx_id"])]
            oracle_preds = cand_pairs[oracle_mask][["s1_id", "sx_id"]]
            oracle_eval = calculate_macro_f05(oracle_preds, gt_df, s1_all)
            oracle_gap = oracle_eval["macro_f05"] - r50_eval["macro_f05"]

            # 3C: Oracle blocking (injected gold candidates)
            # Inject all missed gold pairs into candidates, compute features, score with R50
            if missed_gold:
                missed_list = list(missed_gold)
                missed_df = pd.DataFrame(missed_list, columns=["s1_id", "sx_id"])
                c = _con()
                c.register("_m", missed_df)
                missed_full = c.execute(f"""
                    SELECT m.s1_id, m.sx_id,
                           s1.name_clean     AS s1_name,   sx.name_clean     AS sx_name,
                           s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
                           s1.addr_clean     AS s1_addr,   sx.addr_clean     AS sx_addr,
                           COALESCE(s1.postal_code, '') AS s1_post,
                           COALESCE(sx.postal_code, '') AS sx_post
                    FROM _m m
                    JOIN read_parquet('{_s1pq()}') s1 ON m.s1_id = s1.entity_id
                    JOIN read_parquet('{_sxpq(src)}') sx ON m.sx_id = sx.entity_id
                """).df()
                c.close()

                X_missed = compute_features_parallel(missed_full, n_workers=6)
                scores_missed = bst.predict(X_missed)
                missed_preds = missed_full[scores_missed >= thresh][["s1_id", "sx_id"]]

                # Combine existing R50 preds + injected gold preds that scored >= thresh
                injected_preds = pd.concat([r50_preds, missed_preds], ignore_index=True).drop_duplicates()
            else:
                injected_preds = r50_preds

            injected_eval = calculate_macro_f05(injected_preds, gt_df, s1_all)

            dt = time.time() - t0
            summary_results[(src, split)] = {
                "total_gold": total_gold,
                "present_gold": len(present_gold),
                "missed_gold": len(missed_gold),
                "blocking_recall": blocking_recall,
                "total_cands": total_cands,
                "r50_f05": r50_eval["macro_f05"],
                "r50_precision": r50_eval["macro_precision"],
                "r50_recall": r50_eval["macro_recall"],
                "oracle_f05": oracle_eval["macro_f05"],
                "oracle_gap": oracle_gap,
                "injected_f05": injected_eval["macro_f05"],
                "injected_precision": injected_eval["macro_precision"],
                "injected_recall": injected_eval["macro_recall"],
                "runtime_s": dt,
            }

            print(f"\n--- {src.upper()} - {split.upper()} (elapsed {dt:.1f}s) ---")
            print(f"  Gold pairs: {total_gold:,} | Present in cands: {len(present_gold):,} | Missed by blocking: {len(missed_gold):,}")
            print(f"  Blocking recall: {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")
            print(f"  Active R50:         F0.5 = {r50_eval['macro_f05']:.5f} | P = {r50_eval['macro_precision']:.5f} | R = {r50_eval['macro_recall']:.5f}")
            print(f"  Oracle Matcher:     F0.5 = {oracle_eval['macro_f05']:.5f} | gap = +{oracle_gap:.5f}")
            print(f"  Injected Gold (R50): F0.5 = {injected_eval['macro_f05']:.5f} | P = {injected_eval['macro_precision']:.5f} | R = {injected_eval['macro_recall']:.5f} (gain = +{injected_eval['macro_f05'] - r50_eval['macro_f05']:.5f})")

    # Overall Summary
    print("\n" + "=" * 80)
    print("PHASE 0 ORACLE SUMMARY & BOTTLENECK CLASSIFICATION")
    print("=" * 80)
    print(f"{'Source':<10} {'Split':<6} {'TotalGold':<10} {'Present':<10} {'Missed':<8} {'BlkRecall':<11} {'R50 F0.5':<10} {'Oracle F0.5':<12} {'Injected F0.5':<14}")
    print("-" * 88)
    for (src, split), d in summary_results.items():
        print(f"{src:<10} {split:<6} {d['total_gold']:<10} {d['present_gold']:<10} {d['missed_gold']:<8} "
              f"{d['blocking_recall']:<11.4f} {d['r50_f05']:<10.5f} {d['oracle_f05']:<12.5f} {d['injected_f05']:<14.5f}")

    return summary_results


if __name__ == "__main__":
    run_phase_0()
