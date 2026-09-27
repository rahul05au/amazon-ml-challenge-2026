"""
V9: True Groupwise Ranking for Multi-Match Entity Resolution.

Evaluates LambdaRank groupwise ranking vs R50 baseline vs Phase-3 reranker
on untouched validation splits (val1 and val2) for Source 2 and Source 3.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet
from src.features import FEATURE_NAMES, compute_features_parallel
from src.matcher import calculate_macro_f05

_CACHE_DIR = ROOT / "intermediate" / "v2_cache"
_MODELS_DIR = ROOT / "models"
_OPERATIONAL_THRESHOLDS = {"source2": 0.70, "source3": 0.65}

_H = "abs(hash(entity_id || 'v1')) % 1000"
_SPLIT_WHERE = {
    "val1": f"{_H} >= 0 AND {_H} < 5",
    "val2": f"{_H} >= 25 AND {_H} < 30",
}


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def get_clean_gold(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    """Retrieve clean, unpolluted ground truth for the target source."""
    prefix = "S2-" if src == "source2" else "S3-"
    where_clause = _SPLIT_WHERE[split_name]

    c = duckdb.connect()
    c.execute(f"""
        CREATE TEMP TABLE _s1 AS
        SELECT entity_id AS s1_id
        FROM read_parquet('{_s1pq()}')
        WHERE {where_clause}
    """)
    s1_all = set(c.execute("SELECT s1_id FROM _s1").df()["s1_id"])

    gt_df = c.execute(f"""
        WITH unnested AS (
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{_gtpq()}')
            WHERE matched_entity_ids != ''
              AND source1_entity_id IN (SELECT s1_id FROM _s1)
        )
        SELECT s1_id, true_match_id
        FROM unnested
        WHERE true_match_id LIKE '{prefix}%'
    """).df()
    c.close()

    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
    return s1_all, gt_set, gt_df


def load_cached_df_with_features(parquet_path: Path, src: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load cached pairs, join name_no_suffix, and compute the 15 base RapidFuzz features."""
    cache = pd.read_parquet(parquet_path)
    c = duckdb.connect()
    c.register("_c", cache[["s1_id", "sx_id"]])
    ns = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf
        FROM _c c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()

    feat_df = cache.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({
        "s1_no_suf": "", "sx_no_suf": ""
    })

    X = compute_features_parallel(feat_df, n_workers=6)
    for f_idx, fname in enumerate(FEATURE_NAMES):
        feat_df[fname] = X[:, f_idx]

    bst = lgb.Booster(model_file=str(_MODELS_DIR / f"lgb_{src}.txt"))
    feat_df["r50_score"] = bst.predict(X)
    return feat_df, X


def engineer_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline Phase-3 and new V9 groupwise contextual features."""
    # Ensure sorted by s1_id and descending R50 score
    df = df.sort_values(by=["s1_id", "r50_score"], ascending=[True, False]).reset_index(drop=True)

    # 1. Baseline Phase-3 Features
    df["cand_rank"] = df.groupby("s1_id").cumcount() + 1
    df["cand_count"] = df.groupby("s1_id")["r50_score"].transform("count")

    top1_scores = df.groupby("s1_id")["r50_score"].transform("first")
    df["top1_score"] = top1_scores
    df["score_diff_top1"] = df["r50_score"] - top1_scores
    df["score_ratio_top1"] = df["r50_score"] / (top1_scores + 1e-6)

    mean_scores = df.groupby("s1_id")["r50_score"].transform("mean")
    df["score_diff_mean"] = df["r50_score"] - mean_scores

    # 2. New V9 Contextual Features
    # A. Score percentile within S1: 0 = lowest, 1 = highest
    df["score_percentile"] = np.where(
        df["cand_count"] > 1,
        (df["cand_count"] - df["cand_rank"]) / (df["cand_count"] - 1),
        1.0
    )

    # B & C. Score gaps to previous and next candidate in rank order
    prev_scores = df.groupby("s1_id")["r50_score"].shift(1).fillna(df["r50_score"])
    next_scores = df.groupby("s1_id")["r50_score"].shift(-1).fillna(df["r50_score"])
    df["gap_to_prev_rank"] = prev_scores - df["r50_score"]
    df["gap_to_next_rank"] = df["r50_score"] - next_scores

    # D. Local score density (vectorized fast group sum)
    df["_ge_050"] = (df["r50_score"] >= 0.50).astype(np.int32)
    df["_ge_070"] = (df["r50_score"] >= 0.70).astype(np.int32)
    df["_ge_085"] = (df["r50_score"] >= 0.85).astype(np.int32)
    df["_ge_090"] = (df["r50_score"] >= 0.90).astype(np.int32)

    df["count_ge_050"] = df.groupby("s1_id")["_ge_050"].transform("sum")
    df["count_ge_070"] = df.groupby("s1_id")["_ge_070"].transform("sum")
    df["count_ge_085"] = df.groupby("s1_id")["_ge_085"].transform("sum")
    df["count_ge_090"] = df.groupby("s1_id")["_ge_090"].transform("sum")
    df.drop(columns=["_ge_050", "_ge_070", "_ge_085", "_ge_090"], inplace=True)

    # E. Feature contrast against top-ranked candidate
    for f in ["name_wratio", "name_token_set", "addr_ratio", "name_addr_mean"]:
        top1_f = df.groupby("s1_id")[f].transform("first")
        df[f"{f}_diff_top1"] = df[f] - top1_f

    return df


def get_v9_feature_names() -> list[str]:
    return [
        # 15 Pairwise base features
        *FEATURE_NAMES,
        # Baseline contextual features
        "r50_score", "cand_rank", "cand_count", "top1_score",
        "score_diff_top1", "score_ratio_top1", "score_diff_mean",
        # New V9 features
        "score_percentile", "gap_to_prev_rank", "gap_to_next_rank",
        "count_ge_050", "count_ge_070", "count_ge_085", "count_ge_090",
        "name_wratio_diff_top1", "name_token_set_diff_top1",
        "addr_ratio_diff_top1", "name_addr_mean_diff_top1",
    ]


def train_models_for_source(src: str) -> tuple[lgb.Booster, lgb.Booster]:
    """Train Phase-3 Reranker and V9 LambdaRank on the disjoint train_gate cache."""
    train_cache = _CACHE_DIR / f"v2_{src}_train_gate.parquet"
    print(f"\nLoading training cache: {train_cache.name}...")
    train_df, _ = load_cached_df_with_features(train_cache, src)

    # Get clean ground truth for training set
    prefix = "S2-" if src == "source2" else "S3-"
    c = duckdb.connect()
    gt_df = c.execute(f"""
        WITH unnested AS (
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{_gtpq()}')
            WHERE matched_entity_ids != ''
        )
        SELECT s1_id, true_match_id
        FROM unnested
        WHERE true_match_id LIKE '{prefix}%'
    """).df()
    c.close()
    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))

    train_df["label"] = [(1 if (s, x) in gt_set else 0) for s, x in zip(train_df["s1_id"], train_df["sx_id"])]
    print(f"Total training pairs: {len(train_df):,} | True matches: {train_df['label'].sum():,}")

    train_df = engineer_context_features(train_df)

    # Ensure strict sorting by s1_id for LightGBM grouping
    train_df = train_df.sort_values(by="s1_id").reset_index(drop=True)
    group_sizes = train_df.groupby("s1_id", sort=False).size().values

    # Group sanity checks
    print(f"Group Sanity Checks:")
    print(f"  Sum of groups == rows: {sum(group_sizes) == len(train_df)} ({sum(group_sizes):,} == {len(train_df):,})")
    print(f"  Number of groups == distinct S1: {len(group_sizes) == train_df['s1_id'].nunique()} ({len(group_sizes):,} == {train_df['s1_id'].nunique():,})")

    # 1. Train Phase-3 Reranker (Control)
    phase3_feats = ["r50_score", "cand_rank", "cand_count", "top1_score", "score_diff_top1", "score_ratio_top1", "score_diff_mean"]
    dtrain_p3 = lgb.Dataset(train_df[phase3_feats], label=train_df["label"])
    p3_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 20,
        "scale_pos_weight": 3.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 6,
    }
    print("Training Phase-3 Control Reranker...")
    p3_model = lgb.train(p3_params, dtrain_p3)

    # 2. Train V9 LambdaRank Model
    v9_feats = get_v9_feature_names()
    dtrain_v9 = lgb.Dataset(train_df[v9_feats], label=train_df["label"], group=group_sizes)
    v9_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 6,
    }
    print("Training V9 LambdaRank Groupwise Model...")
    v9_model = lgb.train(v9_params, dtrain_v9)

    return p3_model, v9_model


def evaluate_split(src: str, split_name: str, p3_model: lgb.Booster, v9_model: lgb.Booster) -> dict:
    print(f"\n{'-'*60}")
    print(f"EVALUATING: {src.upper()} | {split_name.upper()}")
    print(f"{'-'*60}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    total_gold = len(gt_set)
    base_thresh = _OPERATIONAL_THRESHOLDS[src]

    cache_path = _CACHE_DIR / f"v2_{src}_{split_name}.parquet"
    val_df, _ = load_cached_df_with_features(cache_path, src)
    val_df = engineer_context_features(val_df)
    val_df["is_gold"] = [(s, x) in gt_set for s, x in zip(val_df["s1_id"], val_df["sx_id"])]

    # Scores
    phase3_feats = ["r50_score", "cand_rank", "cand_count", "top1_score", "score_diff_top1", "score_ratio_top1", "score_diff_mean"]
    v9_feats = get_v9_feature_names()

    val_df["p3_score"] = p3_model.predict(val_df[phase3_feats])
    val_df["v9_score"] = v9_model.predict(val_df[v9_feats])

    # -------------------------------------------------------------
    # EXPERIMENT 1: RANKING QUALITY COMPARISON
    # -------------------------------------------------------------
    # Compute rank under R50, Phase 3, and V9
    val_df["r50_rank"] = val_df.groupby("s1_id")["r50_score"].rank(ascending=False, method="min")
    val_df["p3_rank"] = val_df.groupby("s1_id")["p3_score"].rank(ascending=False, method="min")
    val_df["v9_rank"] = val_df.groupby("s1_id")["v9_score"].rank(ascending=False, method="min")

    gold_rows = val_df[val_df["is_gold"]]

    def get_topk(ranks_series: pd.Series, k: int) -> float:
        return float((ranks_series <= k).sum() / total_gold)

    rank_metrics = {
        "r50_top1": get_topk(gold_rows["r50_rank"], 1),
        "r50_top3": get_topk(gold_rows["r50_rank"], 3),
        "r50_top5": get_topk(gold_rows["r50_rank"], 5),
        "p3_top1": get_topk(gold_rows["p3_rank"], 1),
        "p3_top3": get_topk(gold_rows["p3_rank"], 3),
        "p3_top5": get_topk(gold_rows["p3_rank"], 5),
        "v9_top1": get_topk(gold_rows["v9_rank"], 1),
        "v9_top3": get_topk(gold_rows["v9_rank"], 3),
        "v9_top5": get_topk(gold_rows["v9_rank"], 5),
    }

    print("Ranking Quality:")
    print(f"  R50:    Top-1 = {rank_metrics['r50_top1']:.4f}, Top-3 = {rank_metrics['r50_top3']:.4f}, Top-5 = {rank_metrics['r50_top5']:.4f}")
    print(f"  Phase3: Top-1 = {rank_metrics['p3_top1']:.4f}, Top-3 = {rank_metrics['p3_top3']:.4f}, Top-5 = {rank_metrics['p3_top5']:.4f}")
    print(f"  V9:     Top-1 = {rank_metrics['v9_top1']:.4f}, Top-3 = {rank_metrics['v9_top3']:.4f}, Top-5 = {rank_metrics['v9_top5']:.4f}")

    # -------------------------------------------------------------
    # EXPERIMENT 2: ENTITY DECISION & MACRO F0.5
    # -------------------------------------------------------------
    # 1. Baseline R50 evaluation
    r50_preds = val_df[val_df["r50_score"] >= base_thresh][["s1_id", "sx_id"]]
    r50_eval = calculate_macro_f05(r50_preds, gt_df, s1_all)

    # 2. Phase 3 Reranker evaluation (validated threshold ~0.50 on reranker output)
    p3_thresh = 0.50
    p3_preds = val_df[(val_df["r50_score"] >= base_thresh) & (val_df["p3_score"] >= p3_thresh)][["s1_id", "sx_id"]]
    p3_eval = calculate_macro_f05(p3_preds, gt_df, s1_all)

    # 3. V9 LambdaRank Acceptance Strategies (3 targeted configs)
    # LambdaRank scores are unbounded real logits.
    # Config A: High-scoring pairs (R50 >= base_thresh) where LambdaRank ranks within top-5 (multi-match safe)
    v9_cfg_a_preds = val_df[(val_df["r50_score"] >= base_thresh) & (val_df["v9_rank"] <= 5)][["s1_id", "sx_id"]]
    v9_cfg_a_eval = calculate_macro_f05(v9_cfg_a_preds, gt_df, s1_all)

    # Config B: Ranker score threshold sweep (e.g. ranker logit >= P50 of accepted pairs)
    v9_median_accepted = val_df[val_df["r50_score"] >= base_thresh]["v9_score"].median()
    v9_cfg_b_preds = val_df[(val_df["r50_score"] >= base_thresh) & (val_df["v9_score"] >= v9_median_accepted)][["s1_id", "sx_id"]]
    v9_cfg_b_eval = calculate_macro_f05(v9_cfg_b_preds, gt_df, s1_all)

    # Config C: R50 accepted pairs pruned if V9 margin vs top1 exceeds threshold
    val_df["v9_top1_score"] = val_df.groupby("s1_id")["v9_score"].transform("max")
    val_df["v9_margin_top1"] = val_df["v9_top1_score"] - val_df["v9_score"]
    v9_cfg_c_preds = val_df[(val_df["r50_score"] >= base_thresh) & (val_df["v9_margin_top1"] <= 1.5)][["s1_id", "sx_id"]]
    v9_cfg_c_eval = calculate_macro_f05(v9_cfg_c_preds, gt_df, s1_all)

    print("\nDecision Evaluations:")
    print(f"  R50 Baseline:     F0.5 = {r50_eval['macro_f05']:.5f} | P = {r50_eval['macro_precision']:.5f} | R = {r50_eval['macro_recall']:.5f} | SingAcc = {r50_eval['singletons_correct']}/{r50_eval['total_singletons']}")
    print(f"  Phase-3 Control:  F0.5 = {p3_eval['macro_f05']:.5f} | P = {p3_eval['macro_precision']:.5f} | R = {p3_eval['macro_recall']:.5f} | SingAcc = {p3_eval['singletons_correct']}/{p3_eval['total_singletons']}")
    print(f"  V9 Config A (Top-5 Rank):  F0.5 = {v9_cfg_a_eval['macro_f05']:.5f} | P = {v9_cfg_a_eval['macro_precision']:.5f} | R = {v9_cfg_a_eval['macro_recall']:.5f}")
    print(f"  V9 Config B (Score Thresh): F0.5 = {v9_cfg_b_eval['macro_f05']:.5f} | P = {v9_cfg_b_eval['macro_precision']:.5f} | R = {v9_cfg_b_eval['macro_recall']:.5f}")
    print(f"  V9 Config C (Top1 Margin):  F0.5 = {v9_cfg_c_eval['macro_f05']:.5f} | P = {v9_cfg_c_eval['macro_precision']:.5f} | R = {v9_cfg_c_eval['macro_recall']:.5f}")

    # Best V9 config
    v9_configs = [("Config A (Top-5)", v9_cfg_a_eval, v9_cfg_a_preds),
                  ("Config B (Score Thresh)", v9_cfg_b_eval, v9_cfg_b_preds),
                  ("Config C (Margin)", v9_cfg_c_eval, v9_cfg_c_preds)]
    best_v9_name, best_v9_eval, best_v9_preds = max(v9_configs, key=lambda x: x[1]["macro_f05"])

    # -------------------------------------------------------------
    # MULTI-MATCH & FP RANK DIAGNOSTICS
    # -------------------------------------------------------------
    # Multi-match entities (S1 with >= 2 true matches)
    multi_s1 = set(gt_df.groupby("s1_id").filter(lambda g: len(g) >= 2)["s1_id"])
    secondary_gold = gold_rows[gold_rows["s1_id"].isin(multi_s1) & (gold_rows["r50_rank"] > 1)]
    total_secondary_gold = len(secondary_gold)

    v9_secondary_retained = len(secondary_gold[secondary_gold["v9_rank"] <= 5])
    gold_sec_recall = v9_secondary_retained / total_secondary_gold if total_secondary_gold else 0.0

    # FP ranks before vs after
    r50_fps = val_df[(val_df["r50_score"] >= base_thresh) & (~val_df["is_gold"])]
    fp_rank_before = r50_fps["r50_rank"].median()

    v9_fps = val_df[(val_df["s1_id"].isin(best_v9_preds["s1_id"])) & (val_df["sx_id"].isin(best_v9_preds["sx_id"])) & (~val_df["is_gold"])]
    fp_rank_after = v9_fps["v9_rank"].median() if len(v9_fps) else 0.0

    return {
        "src": src,
        "split": split_name,
        "rank_metrics": rank_metrics,
        "r50_eval": r50_eval,
        "p3_eval": p3_eval,
        "v9_eval": best_v9_eval,
        "v9_best_name": best_v9_name,
        "gold_sec_recall": gold_sec_recall,
        "fp_rank_before": fp_rank_before,
        "fp_rank_after": fp_rank_after,
        "fp_count_before": len(r50_fps),
        "fp_count_after": len(v9_fps),
    }


def main():
    tracemalloc.start()
    t_start = time.time()

    print("=" * 80)
    print("V9: TRUE GROUPWISE RANKING EXPERIMENT (LAMBDARANK)")
    print("Multi-Match Entity Resolution Contextual Evaluation")
    print("=" * 80)

    results = {}
    for src in ["source2", "source3"]:
        print(f"\n{'#'*70}")
        print(f"PROCESSING TARGET SOURCE: {src.upper()}")
        print(f"{'#'*70}")
        p3_model, v9_model = train_models_for_source(src)

        for split in ["val1", "val2"]:
            results[(src, split)] = evaluate_split(src, split, p3_model, v9_model)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print("V9 EXPERIMENT COMPLETED")
    print(f"Total Runtime: {total_time:.1f}s | Peak RAM: {peak / 1024 / 1024:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
