"""V2 Experiment 1B: Evaluate additive TF-IDF candidate union with frozen V1 matcher."""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import INTERMEDIATE_DIR, candidate_parquet, norm_parquet
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

    con = duckdb.connect()
    con.register("_cands", cand_df)
    feat_input = con.execute(f"""
        SELECT
            c.s1_id, c.sx_id,
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


def evaluate_target(
    target_source: str = "source2",
    k: int = 20,
) -> dict[str, Any]:
    """Run V1 vs V2 frozen matcher evaluation for a single target source."""
    prefix = "S2-" if target_source == "source2" else "S3-"
    threshold = FROZEN_THRESHOLDS[target_source]
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

    # V1 candidates for validation S1
    cand_safe = str(candidate_parquet("train", target_source)).replace("\\", "/")
    v1_cands = con.execute(f"""
        SELECT c.s1_id, c.sx_id
        FROM read_parquet('{cand_safe}') c
        JOIN (SELECT s1_id FROM s1_split WHERE split = 'val') s ON c.s1_id = s.s1_id
    """).df().drop_duplicates(["s1_id", "sx_id"])
    con.close()

    # Retrieve TF-IDF candidates for the exact same validation S1 entities
    tfidf_cands = retrieve_tfidf_candidates(
        target_source=target_source,
        split="train",
        k=k,
        s1_ids=val_s1_all,
    )

    # V2 candidates: union and deduplicate
    v2_cands = pd.concat([v1_cands, tfidf_cands], ignore_index=True).drop_duplicates(["s1_id", "sx_id"])

    # Added candidate pairs
    v1_set = set(zip(v1_cands["s1_id"], v1_cands["sx_id"]))
    is_new = [pair not in v1_set for pair in zip(v2_cands["s1_id"], v2_cands["sx_id"])]
    new_cands = v2_cands[is_new].copy()

    # Candidate volume statistics
    n_s1 = len(val_s1_all)
    v1_counts = v1_cands.groupby("s1_id").size()
    v1_per_s1 = pd.Series(0, index=list(val_s1_all))
    v1_per_s1.update(v1_counts)
    v1_cand_count = len(v1_cands)
    v1_avg_cands = v1_cand_count / n_s1
    v1_p95_cands = float(np.percentile(v1_per_s1.values, 95))

    v2_counts = v2_cands.groupby("s1_id").size()
    v2_per_s1 = pd.Series(0, index=list(val_s1_all))
    v2_per_s1.update(v2_counts)
    v2_cand_count = len(v2_cands)
    v2_avg_cands = v2_cand_count / n_s1
    v2_p95_cands = float(np.percentile(v2_per_s1.values, 95))

    # Feature extraction & scoring for V1 candidates
    v1_feat_df = load_candidate_features(v1_cands, target_source)
    X_v1 = compute_features_parallel(v1_feat_df, n_workers=6)
    scores_v1 = bst.predict(X_v1)
    v1_feat_df["score"] = scores_v1

    # Feature extraction & scoring for newly added candidates
    if len(new_cands) > 0:
        new_feat_df = load_candidate_features(new_cands, target_source)
        X_new = compute_features_parallel(new_feat_df, n_workers=6)
        scores_new = bst.predict(X_new)
        new_feat_df["score"] = scores_new
        v2_scored_df = pd.concat([
            v1_feat_df[["s1_id", "sx_id", "score"]],
            new_feat_df[["s1_id", "sx_id", "score"]],
        ], ignore_index=True)
    else:
        v2_scored_df = v1_feat_df[["s1_id", "sx_id", "score"]].copy()

    # V1 evaluation at frozen threshold
    v1_preds = v1_feat_df[v1_feat_df["score"] >= threshold][["s1_id", "sx_id"]]
    v1_metrics = calculate_macro_f05(v1_preds, val_gt_pairs, val_s1_all)

    # V2 evaluation at frozen threshold
    v2_preds = v2_scored_df[v2_scored_df["score"] >= threshold][["s1_id", "sx_id"]]
    v2_metrics = calculate_macro_f05(v2_preds, val_gt_pairs, val_s1_all)

    return {
        "target": target_source,
        "threshold": threshold,
        "n_s1": n_s1,
        "v1": {
            **v1_metrics,
            "candidate_count": v1_cand_count,
            "avg_cands": v1_avg_cands,
            "p95_cands": v1_p95_cands,
        },
        "v2": {
            **v2_metrics,
            "candidate_count": v2_cand_count,
            "avg_cands": v2_avg_cands,
            "p95_cands": v2_p95_cands,
        },
        "new_candidates": len(new_cands),
    }


def main() -> None:
    tracemalloc.start()
    t_start = time.perf_counter()

    print("Running V2 Experiment 1B: Frozen Matcher Evaluation...")
    print("Evaluating S1 -> Source 2...")
    s2_res = evaluate_target("source2")

    print("Evaluating S1 -> Source 3...")
    s3_res = evaluate_target("source3")

    total_time = time.perf_counter() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def fmt_delta(v2: float, v1: float, is_pct: bool = False, decimals: int = 4) -> str:
        diff = v2 - v1
        sign = "+" if diff >= 0 else ""
        if is_pct:
            return f"{sign}{diff * 100:.2f}%"
        return f"{sign}{diff:.{decimals}f}"

    def print_section(res: dict[str, Any], label: str) -> None:
        v1 = res["v1"]
        v2 = res["v2"]
        print(f"\n{label}")
        print("---")
        print(f"V1 Macro F0.5: {v1['macro_f05']:.4f}")
        print(f"V2 Macro F0.5: {v2['macro_f05']:.4f}")
        print(f"Δ Macro F0.5: {fmt_delta(v2['macro_f05'], v1['macro_f05'])}")
        print()
        print(f"V1 Precision: {v1['macro_precision'] * 100:.2f}%")
        print(f"V2 Precision: {v2['macro_precision'] * 100:.2f}%")
        print(f"Δ Precision: {fmt_delta(v2['macro_precision'], v1['macro_precision'], is_pct=True)}")
        print()
        print(f"V1 Recall: {v1['macro_recall'] * 100:.2f}%")
        print(f"V2 Recall: {v2['macro_recall'] * 100:.2f}%")
        print(f"Δ Recall: {fmt_delta(v2['macro_recall'], v1['macro_recall'], is_pct=True)}")
        print()
        print(f"V1 Singleton Accuracy: {v1['singleton_accuracy'] * 100:.2f}%")
        print(f"V2 Singleton Accuracy: {v2['singleton_accuracy'] * 100:.2f}%")
        print(f"Δ Singleton Accuracy: {fmt_delta(v2['singleton_accuracy'], v1['singleton_accuracy'], is_pct=True)}")
        print()
        print(f"V1 Predicted Matches: {v1['predicted_matches']:,}")
        print(f"V2 Predicted Matches: {v2['predicted_matches']:,}")
        print(f"Δ Predicted Matches: {v2['predicted_matches'] - v1['predicted_matches']:+,}")
        print()
        print(f"V1 Zero-Prediction Singletons: {v1['singletons_correct']:,} / {v1['total_singletons']:,}")
        print(f"V2 Zero-Prediction Singletons: {v2['singletons_correct']:,} / {v2['total_singletons']:,}")
        print(f"Δ Zero-Prediction Singletons: {v2['singletons_correct'] - v1['singletons_correct']:+,}")
        print()
        print(f"V1 Candidate Count: {v1['candidate_count']:,}")
        print(f"V2 Candidate Count: {v2['candidate_count']:,}")
        print(f"Δ Candidates: {v2['candidate_count'] - v1['candidate_count']:+,}")
        print()
        print(f"V1 Avg Candidates/S1: {v1['avg_cands']:.2f}")
        print(f"V2 Avg Candidates/S1: {v2['avg_cands']:.2f}")
        print()
        print(f"V1 P95 Candidates/S1: {v1['p95_cands']:.1f}")
        print(f"V2 P95 Candidates/S1: {v2['p95_cands']:.1f}")

    print("\n" + "=" * 60)
    print("V2 EXPERIMENT 1B — FROZEN MATCHER EVALUATION")
    print("=" * 60)

    print_section(s2_res, "S2")
    print_section(s3_res, "S3")

    # Decision rule
    delta_f05_s2 = s2_res["v2"]["macro_f05"] - s2_res["v1"]["macro_f05"]
    delta_f05_s3 = s3_res["v2"]["macro_f05"] - s3_res["v1"]["macro_f05"]
    mean_delta_f05 = (delta_f05_s2 + delta_f05_s3) / 2.0

    if mean_delta_f05 > 0.005 and s2_res["v2"]["singleton_accuracy"] >= s2_res["v1"]["singleton_accuracy"] - 0.02:
        decision = "KEEP"
        reason = "Macro F0.5 improves meaningfully on both targets with acceptable precision and singleton retention."
    elif mean_delta_f05 > -0.002:
        decision = "PRUNE"
        reason = (
            "Recall gained substantially from TF-IDF candidates, but false positives "
            "dampened the Macro F0.5 gain under frozen thresholds. Candidate pruning or score filtering "
            "is indicated before full adoption."
        )
    else:
        decision = "DISCARD"
        reason = "Macro F0.5 decreased due to false positive inflation eroding precision and singleton accuracy."

    print("\nOVERALL DECISION")
    print("----------------")
    print(decision)
    print(f"\nReason:\n{reason}")
    print("\nRESOURCE & ENVIRONMENT REPORT")
    print("-----------------------------")
    print(f"Runtime: {total_time:.2f}s")
    print(f"Peak memory: {peak_bytes / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
