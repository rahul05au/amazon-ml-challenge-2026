"""V2 Experiment 1C: Selective pruning of additional TF-IDF candidates."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import candidate_parquet, norm_parquet
from src.features import compute_features_parallel
from src.lexical_blocking import retrieve_tfidf_candidates
from src.matcher import MODELS_DIR, calculate_macro_f05

FROZEN_THRESHOLDS: dict[str, float] = {
    "source2": 0.70,
    "source3": 0.65,
}


def load_candidate_features(
    cand_df: pd.DataFrame,
    target_source: str,
    split: str = "train",
) -> pd.DataFrame:
    """Join candidate pairs with normalized entity text attributes."""
    s1_safe = str(norm_parquet(split, "source1")).replace("\\", "/")
    sx_safe = str(norm_parquet(split, target_source)).replace("\\", "/")

    has_score = "tfidf_score" in cand_df.columns
    score_col = "c.tfidf_score," if has_score else ""

    con = duckdb.connect()
    con.register("_cands", cand_df)
    feat_input = con.execute(f"""
        SELECT
            c.s1_id, c.sx_id, {score_col}
            s1.name_clean as s1_name, sx.name_clean as sx_name,
            s1.name_no_suffix as s1_no_suf, sx.name_no_suffix as sx_no_suf,
            s1.addr_clean as s1_addr, sx.addr_clean as sx_addr,
            COALESCE(s1.postal_code, '') as s1_post, COALESCE(sx.postal_code, '') as sx_post
        FROM _cands c
        JOIN read_parquet('{s1_safe}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{sx_safe}') sx ON c.sx_id = sx.entity_id
    """).df()
    con.close()
    return feat_input


def evaluate_target_pruning(
    target_source: str = "source2",
    k: int = 20,
) -> dict[str, Any]:
    """Run selective pruning evaluation across multiple TF-IDF score thresholds."""
    prefix = "S2-" if target_source == "source2" else "S3-"
    lgb_threshold = FROZEN_THRESHOLDS[target_source]
    model_path = MODELS_DIR / f"lgb_{target_source}.txt"
    bst = lgb.Booster(model_file=str(model_path))

    con = duckdb.connect()
    con.execute("SET threads TO 6")
    con.execute("SET memory_limit = '6GB'")

    # Clustered validation split (exact V1 baseline definition)
    con.execute("""
        CREATE TEMP TABLE s1_split AS
        SELECT entity_id as s1_id,
               CASE WHEN abs(hash(entity_id || 'v1')) % 1000 < 5 THEN 'val'
                    ELSE 'other' END as split
        FROM read_parquet('intermediate/train/norm_source1.parquet')
    """)

    con.execute(f"""
        CREATE TEMP TABLE gt_target AS
        SELECT source1_entity_id as s1_id, trim(unnest(string_split(matched_entity_ids, ','))) as true_id
        FROM read_parquet('intermediate/train/ground_truth.parquet')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
    """)

    val_s1_all = set(con.execute("SELECT s1_id FROM s1_split WHERE split = 'val'").df()["s1_id"])
    val_gt_pairs = con.execute("""
        SELECT g.s1_id, g.true_id as true_match_id
        FROM gt_target g
        JOIN (SELECT s1_id FROM s1_split WHERE split = 'val') s ON g.s1_id = s.s1_id
    """).df()

    # Ground truth for target blocking evaluation (matches specific to target source)
    target_gt_set = set(
        con.execute(f"""
            SELECT g.s1_id, g.true_id
            FROM gt_target g
            JOIN (SELECT s1_id FROM s1_split WHERE split = 'val') s ON g.s1_id = s.s1_id
            WHERE g.true_id LIKE '{prefix}%'
        """).df().itertuples(index=False, name=None)
    )
    total_target_gt = len(target_gt_set)

    # V1 candidates for validation S1
    cand_safe = str(candidate_parquet("train", target_source)).replace("\\", "/")
    v1_cands = con.execute(f"""
        SELECT c.s1_id, c.sx_id
        FROM read_parquet('{cand_safe}') c
        JOIN (SELECT s1_id FROM s1_split WHERE split = 'val') s ON c.s1_id = s.s1_id
    """).df().drop_duplicates(["s1_id", "sx_id"])
    con.close()

    # Retrieve TF-IDF candidates with cosine similarity scores
    tfidf_cands = retrieve_tfidf_candidates(
        target_source=target_source,
        split="train",
        k=k,
        s1_ids=val_s1_all,
        return_scores=True,
    )

    # Separate candidates that are already in V1 vs purely new
    v1_set = set(zip(v1_cands["s1_id"], v1_cands["sx_id"]))
    tfidf_cands["is_in_v1"] = [p in v1_set for p in zip(tfidf_cands["s1_id"], tfidf_cands["sx_id"])]

    # Distribution of TF-IDF scores on retrieved candidates
    all_scores = tfidf_cands["tfidf_score"].values
    score_p25 = float(np.percentile(all_scores, 25))
    score_p50 = float(np.percentile(all_scores, 50))
    score_p75 = float(np.percentile(all_scores, 75))
    score_p90 = float(np.percentile(all_scores, 90))

    print(f"\n--- {target_source.upper()} TF-IDF Score Distribution (N={len(all_scores):,}) ---")
    print(f"Min: {all_scores.min():.4f} | P25: {score_p25:.4f} | P50 (Median): {score_p50:.4f} | P75: {score_p75:.4f} | P90: {score_p90:.4f} | Max: {all_scores.max():.4f}")

    # Thresholds: unpruned (0.00) plus 4 empirical percentiles
    test_thresholds = [0.0, round(score_p25, 2), round(score_p50, 2), round(score_p75, 2), round(score_p90, 2)]

    # Baseline blocking metrics
    v1_matches_covered = len(v1_set & target_gt_set)
    v1_block_recall = v1_matches_covered / total_target_gt if total_target_gt > 0 else 0.0

    # True singletons set (for target source)
    true_singletons = val_s1_all - set(val_gt_pairs["s1_id"])

    # Feature computation & scoring: V1 pairs (guaranteed 1-to-1 row alignment)
    print(f"Computing LightGBM scores for {len(v1_cands):,} V1 candidates...")
    v1_feat = load_candidate_features(v1_cands, target_source)
    X_v1 = compute_features_parallel(v1_feat, n_workers=6)
    v1_feat["lgb_score"] = bst.predict(X_v1)

    # Feature computation & scoring: newly added TF-IDF candidates only
    new_tfidf = tfidf_cands[~tfidf_cands["is_in_v1"]].copy()
    print(f"Computing LightGBM scores for {len(new_tfidf):,} new TF-IDF candidates...")
    if len(new_tfidf) > 0:
        new_feat = load_candidate_features(new_tfidf, target_source)
        X_new = compute_features_parallel(new_feat, n_workers=6)
        new_feat["lgb_score"] = bst.predict(X_new)
    else:
        new_feat = pd.DataFrame(columns=["s1_id", "sx_id", "tfidf_score", "lgb_score"])

    # Unpruned union set for blocking reference
    all_new_set = set(zip(new_feat["s1_id"], new_feat["sx_id"]))
    unpruned_union_set = v1_set | all_new_set
    unpruned_matches_covered = len(unpruned_union_set & target_gt_set)

    # Evaluate each threshold
    threshold_results = []
    n_s1 = len(val_s1_all)

    for tau in test_thresholds:
        # Prune only the ADDITIONAL TF-IDF candidates. V1 candidates are ALWAYS kept.
        filtered_new = new_feat[new_feat["tfidf_score"] >= tau]

        # Final candidate set: V1 union filtered additional TF-IDF
        current_cands = pd.concat([
            v1_feat[["s1_id", "sx_id", "lgb_score"]],
            filtered_new[["s1_id", "sx_id", "lgb_score"]],
        ], ignore_index=True)

        cand_count = len(current_cands)
        cands_per_s1 = pd.Series(0, index=list(val_s1_all))
        cands_per_s1.update(current_cands.groupby("s1_id").size())
        avg_cands = cand_count / n_s1
        p95_cands = float(np.percentile(cands_per_s1.values, 95))

        # Blocking metrics against target ground truth
        filtered_new_set = set(zip(filtered_new["s1_id"], filtered_new["sx_id"]))
        curr_cand_set = v1_set | filtered_new_set
        curr_covered = len(curr_cand_set & target_gt_set)
        block_recall = curr_covered / total_target_gt if total_target_gt > 0 else 0.0
        matches_recovered = curr_covered - v1_matches_covered
        matches_lost_vs_unpruned = unpruned_matches_covered - curr_covered
        added_cands = cand_count - len(v1_cands)
        cands_per_recovered = added_cands / matches_recovered if matches_recovered > 0 else 0.0
        pct_tfidf_retained = (len(filtered_new) / len(new_feat) * 100.0) if len(new_feat) > 0 else 100.0
        cands_removed_from_union = len(new_feat) - len(filtered_new)

        # Matching metrics (frozen LightGBM threshold)
        pred_pairs = current_cands[current_cands["lgb_score"] >= lgb_threshold][["s1_id", "sx_id"]]
        matching_eval = calculate_macro_f05(pred_pairs, val_gt_pairs, val_s1_all)

        # Entity-level safety metrics for singletons
        fp_singletons = matching_eval["total_singletons"] - matching_eval["singletons_correct"]
        new_matched = filtered_new[filtered_new["lgb_score"] >= lgb_threshold]
        singletons_hit_by_tfidf = len(set(new_matched[new_matched["s1_id"].isin(true_singletons)]["s1_id"]))

        res = {
            "threshold": tau,
            "block_recall": block_recall,
            "true_matches_recovered": matches_recovered,
            "matches_lost_vs_unpruned": matches_lost_vs_unpruned,
            "cands_per_recovered": cands_per_recovered,
            "total_candidates": cand_count,
            "avg_candidates": avg_cands,
            "p95_candidates": p95_cands,
            "cands_removed_from_union": cands_removed_from_union,
            "pct_tfidf_retained": pct_tfidf_retained,
            "macro_f05": matching_eval["macro_f05"],
            "precision": matching_eval["macro_precision"],
            "recall": matching_eval["macro_recall"],
            "singleton_accuracy": matching_eval["singleton_accuracy"],
            "zero_pred_singletons": matching_eval["singletons_correct"],
            "fp_singletons": fp_singletons,
            "singletons_hit_by_tfidf": singletons_hit_by_tfidf,
            "total_singletons": matching_eval["total_singletons"],
            "predicted_matches": matching_eval["predicted_matches"],
        }
        threshold_results.append(res)

    return {
        "target": target_source,
        "v1_candidates": len(v1_cands),
        "v1_block_recall": v1_block_recall,
        "total_gt": total_target_gt,
        "results": threshold_results,
    }


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("=" * 65)
    print("V2 EXPERIMENT 1C: SELECTIVE PRUNING OF ADDITIONAL TF-IDF CANDIDATES")
    print("=" * 65)

    print("\n--- Evaluating S1 -> Source 2 ---")
    s2_data = evaluate_target_pruning("source2")

    print("\n--- Evaluating S1 -> Source 3 ---")
    s3_data = evaluate_target_pruning("source3")

    total_time = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def print_table(data: dict[str, Any], label: str) -> None:
        print(f"\n{label}")
        print("---")
        header = f"{'Threshold':<11}| {'Block Recall':<13}| {'F0.5':<7}| {'Precision':<10}| {'Recall':<8}| {'Singleton Acc':<14}| {'Candidates':<11}| {'P95':<5}"
        print(header)
        print("-" * len(header))
        for r in data["results"]:
            th_str = "0.00 (raw)" if r["threshold"] == 0.0 else f"{r['threshold']:.2f}"
            print(
                f"{th_str:<11}| "
                f"{r['block_recall'] * 100:>10.2f}% | "
                f"{r['macro_f05']:>5.4f} | "
                f"{r['precision'] * 100:>7.2f}% | "
                f"{r['recall'] * 100:>5.2f}% | "
                f"{r['singleton_accuracy'] * 100:>11.2f}% | "
                f"{r['total_candidates']:>10,d} | "
                f"{r['p95_candidates']:>4.0f}"
            )

    print("\n" + "=" * 65)
    print("V2 EXPERIMENT 1C — TF-IDF PRUNING RESULTS")
    print("=" * 65)

    print_table(s2_data, "S2")
    print_table(s3_data, "S3")

    # Safety metrics detail
    print("\n--- DETAILED SAFETY & PRUNING METRICS ---")
    for data, label in [(s2_data, "S2"), (s3_data, "S3")]:
        print(f"\nTarget {label}:")
        print(f"{'Threshold':<10}{'Removed':<10}{'% Retained':<12}{'Matches Lost':<14}{'FP Singletons':<15}{'Singletons Hit by TFIDF':<25}")
        for r in data["results"]:
            th_str = "0.00" if r["threshold"] == 0.0 else f"{r['threshold']:.2f}"
            print(
                f"{th_str:<10}"
                f"{r['cands_removed_from_union']:<10,d}"
                f"{r['pct_tfidf_retained']:<12.1f}%"
                f"{r['matches_lost_vs_unpruned']:<14,d}"
                f"{r['fp_singletons']:<15,d}"
                f"{r['singletons_hit_by_tfidf']:<25,d}"
            )

    print(f"\nRuntime: {total_time:.2f}s | Peak memory: {peak_bytes / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
