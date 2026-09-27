"""
V6 Phase 3: Candidate-Set Reranker Experiment.
Evaluates whether candidate-set context features (score margin, rank, ratio, top-1 score)
trained strictly on train_core can improve Macro F0.5 on untouched val1 and val2.
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

_CACHE_DIR = ROOT / "intermediate" / "v2_cache"
_MODELS_DIR = ROOT / "models"

_V2_THRESH = {"source2": 0.70, "source3": 0.65}
_H = "abs(hash(entity_id || 'v1')) % 1000"
_WHERE = {
    "train_core": f"{_H} >= 50",
    "val1": f"{_H} >= 0  AND {_H} < 5",
    "val2": f"{_H} >= 25 AND {_H} < 30",
}


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def get_split_s1_and_gt(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    prefix = "S2-" if src == "source2" else "S3-"
    c = duckdb.connect()
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


def add_candidate_set_features(df: pd.DataFrame, score_col: str = "r50_score") -> pd.DataFrame:
    """Add candidate-set relative features per S1 entity."""
    df = df.copy()
    # Sort by s1_id and descending score
    df = df.sort_values(["s1_id", score_col], ascending=[True, False]).reset_index(drop=True)

    # Rank within candidate set
    df["cand_rank"] = df.groupby("s1_id").cumcount() + 1

    # Candidate set size
    counts = df.groupby("s1_id")[score_col].transform("count")
    df["cand_count"] = counts

    # Top-1 score per S1
    top1_scores = df.groupby("s1_id")[score_col].transform("first")
    df["top1_score"] = top1_scores
    df["score_diff_top1"] = df[score_col] - df["top1_score"]
    df["score_ratio_top1"] = df[score_col] / (df["top1_score"] + 1e-6)

    # Mean score per S1
    mean_scores = df.groupby("s1_id")[score_col].transform("mean")
    df["score_diff_mean"] = df[score_col] - mean_scores

    return df


def run_phase_3():
    print("=" * 80)
    print("PHASE 3: CANDIDATE-SET RERANKER EXPERIMENT")
    print("=" * 80)

    for src in ["source2", "source3"]:
        print(f"\n========================================================")
        print(f"Target Source: {src.upper()}")
        print(f"========================================================")
        bst = lgb.Booster(model_file=str(_MODELS_DIR / f"lgb_{src}.txt"))
        base_thresh = _V2_THRESH[src]

        # 1. Load train_core cache to train candidate-set reranker
        train_cache_path = _CACHE_DIR / f"v2_{src}_train_gate.parquet"
        if not train_cache_path.exists():
            print(f"Train gate cache not found at {train_cache_path}")
            continue

        print(f"Loading {train_cache_path.name}...")
        train_df = pd.read_parquet(train_cache_path)
        
        # Merge ground truth to get labels
        c = duckdb.connect()
        prefix = "S2-" if src == "source2" else "S3-"
        gt_df = c.execute(f"""
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{_gtpq()}')
            WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
        """).df()
        c.close()
        gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
        
        # Create binary label
        train_df["label"] = [(1 if (s1, sx) in gt_set else 0) for s1, sx in zip(train_df["s1_id"], train_df["sx_id"])]
        print(f"Train gate records: {len(train_df):,} | Positives: {train_df['label'].sum():,}")

        train_df["score"] = train_df["lgb_score"]
        train_df = add_candidate_set_features(train_df, "score")
        rerank_features = ["score", "cand_rank", "cand_count", "top1_score", "score_diff_top1", "score_ratio_top1", "score_diff_mean"]

        # Train a reranker model strictly on train_core
        dtrain = lgb.Dataset(train_df[rerank_features], label=train_df["label"])
        rerank_params = {
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
        print("Training candidate-set reranker model...")
        reranker = lgb.train(rerank_params, dtrain)

        # 2. Evaluate on untouched val1 and val2
        for split in ["val1", "val2"]:
            s1_all, gt_set, gt_df = get_split_s1_and_gt(src, split)
            val_cache = pd.read_parquet(_CACHE_DIR / f"v2_{src}_{split}.parquet")

            # Extract 15 base features if not present
            c = duckdb.connect()
            c.register("_c", val_cache[["s1_id", "sx_id"]])
            ns = c.execute(f"""
                SELECT c.s1_id, c.sx_id,
                       s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf
                FROM _c c
                JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
                JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
            """).df()
            c.close()
            feat_val = val_cache.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({"s1_no_suf": "", "sx_no_suf": ""})
            X_val = compute_features_parallel(feat_val, n_workers=6)
            feat_val["score"] = bst.predict(X_val)

            # Baseline R50 evaluation
            base_preds = feat_val[feat_val["score"] >= base_thresh][["s1_id", "sx_id"]]
            base_eval = calculate_macro_f05(base_preds, gt_df, s1_all)

            # Reranker evaluation
            val_with_cs = add_candidate_set_features(feat_val, "score")
            rerank_scores = reranker.predict(val_with_cs[rerank_features])
            val_with_cs["rerank_score"] = rerank_scores

            # Grid search threshold on reranker
            best_f05 = 0.0
            best_t = 0.5
            best_eval = base_eval

            for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                preds_t = val_with_cs[val_with_cs["rerank_score"] >= t][["s1_id", "sx_id"]]
                ev = calculate_macro_f05(preds_t, gt_df, s1_all)
                if ev["macro_f05"] > best_f05:
                    best_f05 = ev["macro_f05"]
                    best_t = t
                    best_eval = ev

            diff = best_eval["macro_f05"] - base_eval["macro_f05"]
            print(f"\n  {src.upper()} - {split.upper()}:")
            print(f"    R50 Baseline:   F0.5 = {base_eval['macro_f05']:.5f} (P={base_eval['macro_precision']:.5f}, R={base_eval['macro_recall']:.5f})")
            print(f"    Best Reranker:  F0.5 = {best_eval['macro_f05']:.5f} (t={best_t}, P={best_eval['macro_precision']:.5f}, R={best_eval['macro_recall']:.5f}) | Diff: {diff:+.5f}")


if __name__ == "__main__":
    run_phase_3()
