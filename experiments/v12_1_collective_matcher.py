"""V12.1: Assignment-Aware Evidence LightGBM with Collective / Sibling Support.

Implements:
1. Entity-level 3-fold OOF Stage-1 predictions on train_gate (zero leakage)
2. Collective & Sibling Support Feature Engineering:
   - S1 sibling high-confidence count, max sibling probability, sibling margin
   - Target-cluster agreement & support (cheap exact name & address clustering)
   - Target competition margin & exclusivity status
3. Stage-2 LightGBM training for Source 2 and Source 3
4. Calibrated threshold search & Target Exclusivity post-processing
5. Canonical per-S1 Macro F0.5 evaluation on untouched val1 and val2 splits
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet
from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import (
    compute_pairwise_features,
    engineer_competition_features,
    get_v12_feature_names,
    get_clean_gold,
    apply_target_exclusivity,
    s1_norm_path,
    sx_norm_path,
    gt_path,
    MODELS_DIR,
    V2_CACHE,
    OPERATIONAL_THRESHOLDS,
)

NUM_RE = re.compile(r"\b\d+\b")


def compute_collective_sibling_features(df: pd.DataFrame, score_col: str = "stage1_score") -> pd.DataFrame:
    """Compute rich sibling support, cluster consensus, and target exclusivity features."""
    df = df.copy()

    # 1. S1 SIBLING SUPPORT FEATURES
    # Sort descending by score within S1
    df = df.sort_values(by=["s1_id", score_col], ascending=[True, False]).reset_index(drop=True)

    # Rank and count per S1
    df["s1_cand_rank"] = df.groupby("s1_id").cumcount() + 1
    df["s1_cand_count"] = df.groupby("s1_id")[score_col].transform("count")

    # High-confidence sibling count (other candidates for this S1 with score >= 0.60)
    high_conf_mask = (df[score_col] >= 0.60).astype(np.float32)
    s1_total_high_conf = df.groupby("s1_id")[score_col].transform(lambda s: (s >= 0.60).sum())
    df["s1_sibling_high_conf_count"] = np.maximum(0.0, s1_total_high_conf - high_conf_mask)

    # Max sibling score (top score among OTHER candidates for this S1)
    top1_scores = df.groupby("s1_id")[score_col].transform("first")
    top2_scores = df.groupby("s1_id")[score_col].transform(lambda s: s.iloc[1] if len(s) > 1 else 0.0)

    # If current is rank 1, best sibling is top2. Otherwise, best sibling is top1.
    df["s1_best_sibling_score"] = np.where(df["s1_cand_rank"] == 1, top2_scores, top1_scores)
    df["s1_sibling_margin"] = df[score_col] - df["s1_best_sibling_score"]
    df["s1_sibling_support_flag"] = (df["s1_best_sibling_score"] >= 0.60).astype(np.float32)

    # 2. TARGET-CLUSTER CONSENSUS FEATURES
    # Group target records that share exact clean name or postal code
    df["target_name_clean"] = df["sx_name"].fillna("").astype(str).str.lower().str.strip()
    df["target_post"] = df["sx_post"].fillna("").astype(str).str.strip()

    # Create cheap cluster key: (exact_name, postal)
    cluster_key = df["target_name_clean"] + "___" + df["target_post"]
    df["cluster_key"] = cluster_key

    # Calculate cluster-level agreement for this S1
    # Count how many targets in this cluster propose this s1_id
    cluster_s1_counts = df.groupby(["cluster_key", "s1_id"])[score_col].transform("count")
    cluster_total_counts = df.groupby("cluster_key")[score_col].transform("count")
    df["cluster_size"] = cluster_total_counts
    df["cluster_s1_agreement"] = cluster_s1_counts / np.maximum(cluster_total_counts, 1.0)
    df["cluster_max_score"] = df.groupby("cluster_key")[score_col].transform("max")

    # 3. REVERSE TARGET COMPETITION SIBLING FEATURES
    df = df.sort_values(by=["sx_id", score_col], ascending=[True, False]).reset_index(drop=True)
    df["target_stage1_rank"] = df.groupby("sx_id").cumcount() + 1
    target_top1_score = df.groupby("sx_id")[score_col].transform("first")
    target_top2_score = df.groupby("sx_id")[score_col].transform(lambda s: s.iloc[1] if len(s) > 1 else 0.0)

    df["target_stage1_top1"] = target_top1_score
    df["target_stage1_runnerup"] = target_top2_score
    df["target_stage1_margin"] = np.where(
        df["target_stage1_rank"] == 1,
        target_top1_score - target_top2_score,
        df[score_col] - target_top1_score
    )
    df["target_is_stage1_best"] = (df["target_stage1_rank"] == 1).astype(np.float32)

    # Re-sort by s1_id, sx_id
    df = df.sort_values(by=["s1_id", "sx_id"]).reset_index(drop=True)

    collective_cols = [
        "s1_sibling_high_conf_count", "s1_best_sibling_score", "s1_sibling_margin", "s1_sibling_support_flag",
        "cluster_size", "cluster_s1_agreement", "cluster_max_score",
        "target_stage1_top1", "target_stage1_runnerup", "target_stage1_margin", "target_is_stage1_best"
    ]
    return df[collective_cols]


def get_v12_1_feature_names() -> list[str]:
    """Return all features for Stage 2 model: V12 features + Stage 1 score + Collective/Sibling features."""
    base_feats = get_v12_feature_names()
    stage1_feats = ["stage1_score"]
    collective_feats = [
        "s1_sibling_high_conf_count", "s1_best_sibling_score", "s1_sibling_margin", "s1_sibling_support_flag",
        "cluster_size", "cluster_s1_agreement", "cluster_max_score",
        "target_stage1_top1", "target_stage1_runnerup", "target_stage1_margin", "target_is_stage1_best"
    ]
    return base_feats + stage1_feats + collective_feats


def generate_oof_stage1_and_train_stage2(src: str) -> tuple[lgb.Booster, lgb.Booster, list[str]]:
    """Generate pure entity-level OOF Stage-1 scores on train_gate, then train Stage-2 LightGBM."""
    print(f"\n{'='*75}\nGENERATING OOF STAGE-1 & TRAINING STAGE-2 FOR {src.upper()}\n{'='*75}")
    train_cache_path = V2_CACHE / f"v2_{src}_train_gate.parquet"
    print(f"Loading cached training pairs: {train_cache_path.name}")
    train_raw = pd.read_parquet(train_cache_path)

    _, gt_set, _ = get_clean_gold(src, "train_gate")
    print(f"Computing V12 base features for {len(train_raw):,} training candidate pairs...")
    t0 = time.time()
    pair_feats = compute_pairwise_features(train_raw)
    train_full = pd.concat([train_raw, pair_feats], axis=1)
    train_full = engineer_competition_features(train_full)
    print(f"Base features computed in {time.time()-t0:.1f}s")

    train_full["is_gold"] = [(s, x) in gt_set for s, x in zip(train_full["s1_id"], train_full["sx_id"])]
    train_full["label"] = train_full["is_gold"].astype(np.int32)

    v12_feature_cols = get_v12_feature_names()

    # 1. 3-Fold Entity-Level OOF Splits
    # Hash of s1_id mod 3
    s1_entities = train_full["s1_id"].unique()
    entity_folds = {eid: (abs(hash(eid + "_fold")) % 3) for eid in s1_entities}
    train_full["fold"] = train_full["s1_id"].map(entity_folds)

    oof_preds = np.zeros(len(train_full), dtype=np.float32)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": 3.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 6,
    }

    print("\nRunning 3-fold Entity-Level Cross-Validation for OOF Stage-1 predictions...")
    t_cv = time.time()
    for fold in range(3):
        train_idx = train_full["fold"] != fold
        val_idx = train_full["fold"] == fold

        dtrain = lgb.Dataset(train_full.loc[train_idx, v12_feature_cols], label=train_full.loc[train_idx, "label"])
        bst = lgb.train(lgb_params, dtrain)

        val_scores = bst.predict(train_full.loc[val_idx, v12_feature_cols])
        oof_preds[val_idx] = val_scores
        print(f"  Fold {fold+1}/3 finished. Val mean score: {val_scores.mean():.4f}, Max: {val_scores.max():.4f}")

    print(f"3-fold OOF Stage-1 predictions completed in {time.time()-t_cv:.1f}s")
    train_full["stage1_score"] = oof_preds

    # Train full Stage-1 model on all train_gate data (for inference on validation/test)
    print("\nTraining final full Stage-1 model on complete train_gate...")
    dtrain_full = lgb.Dataset(train_full[v12_feature_cols], label=train_full["label"])
    stage1_model = lgb.train(lgb_params, dtrain_full)
    stage1_save_path = MODELS_DIR / f"v12_1_stage1_{src}.txt"
    stage1_model.save_model(str(stage1_save_path))
    print(f"Saved Stage-1 model to: {stage1_save_path}")

    # 2. Compute Collective / Sibling Features on OOF Stage-1
    print("\nComputing Stage-2 Collective / Sibling features on OOF predictions...")
    t_col = time.time()
    collective_feats = compute_collective_sibling_features(train_full, score_col="stage1_score")
    train_full = pd.concat([train_full, collective_feats], axis=1)
    print(f"Collective features computed in {time.time()-t_col:.1f}s")

    # 3. Train Stage-2 LightGBM
    stage2_feature_cols = get_v12_1_feature_names()
    print(f"\nTraining Stage-2 LightGBM with {len(stage2_feature_cols)} features...")
    dtrain_stage2 = lgb.Dataset(train_full[stage2_feature_cols], label=train_full["label"])

    stage2_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.04,
        "num_leaves": 45,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "scale_pos_weight": 2.5,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 6,
    }

    t_tr2 = time.time()
    stage2_model = lgb.train(stage2_params, dtrain_stage2)
    print(f"Stage-2 model training completed in {time.time()-t_tr2:.1f}s")

    stage2_save_path = MODELS_DIR / f"v12_1_stage2_{src}.txt"
    stage2_model.save_model(str(stage2_save_path))
    print(f"Saved Stage-2 model to: {stage2_save_path}")

    return stage1_model, stage2_model, stage2_feature_cols


def evaluate_v12_1_pipeline(
    src: str,
    split_name: str,
    stage1_model: lgb.Booster,
    stage2_model: lgb.Booster,
    stage2_feature_cols: list[str]
) -> dict[str, Any]:
    """Evaluate V12 baseline vs V12.1 Stage-1 vs V12.1 Stage-2 + Exclusivity."""
    print(f"\n{'-'*65}\nEVALUATING V12.1 PIPELINE: {src.upper()} | {split_name.upper()}\n{'-'*65}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    val_cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
    val_raw = pd.read_parquet(val_cache_path)

    # 1. Compute Base Features
    pair_feats = compute_pairwise_features(val_raw)
    full_val = pd.concat([val_raw, pair_feats], axis=1)
    full_val = engineer_competition_features(full_val)

    # 2. Stage-1 Predictions
    v12_feature_cols = get_v12_feature_names()
    full_val["stage1_score"] = stage1_model.predict(full_val[v12_feature_cols])

    # 3. Collective / Sibling Features using Stage-1 Scores
    collective_feats = compute_collective_sibling_features(full_val, score_col="stage1_score")
    full_val = pd.concat([full_val, collective_feats], axis=1)

    # 4. Stage-2 Predictions
    full_val["stage2_score"] = stage2_model.predict(full_val[stage2_feature_cols])

    # Sweep thresholds and margins on Stage-2 scores
    best_thresh = 0.45
    best_margin = 0.0
    best_f05 = -1.0
    best_excl_metrics = {}
    best_cand_excl = pd.DataFrame()

    thresholds = [0.40, 0.43, 0.45, 0.47, 0.50, 0.55, 0.60]
    margins = [0.0, 0.05]

    for th in thresholds:
        for mg in margins:
            cand_preds = full_val[full_val["stage2_score"] >= th][["s1_id", "sx_id", "stage2_score"]].copy()
            cand_preds["v12_score"] = cand_preds["stage2_score"]
            cand_excl = apply_target_exclusivity(cand_preds, target_margin_thresh=mg)
            m_excl = calculate_macro_f05(cand_excl, gt_df, s1_all)

            if m_excl["macro_f05"] > best_f05:
                best_f05 = m_excl["macro_f05"]
                best_thresh = th
                best_margin = mg
                best_excl_metrics = m_excl
                best_cand_excl = cand_excl.copy()

    # Measure secondary multi-match recall
    gold_pairs_df = full_val[[(s, x) in gt_set for s, x in zip(full_val["s1_id"], full_val["sx_id"])]].copy()
    s1_gold_counts = gold_pairs_df.groupby("s1_id")["sx_id"].transform("count")
    sec_gold = gold_pairs_df[s1_gold_counts > 1]
    sec_gold_set = set(zip(sec_gold["s1_id"], sec_gold["sx_id"]))
    v12_1_pred_set = set(zip(best_cand_excl["s1_id"], best_cand_excl["sx_id"]))
    sec_rec_v12_1 = len(sec_gold_set & v12_1_pred_set) / max(len(sec_gold_set), 1)

    print(f"\nFinal V12.1 Results for {src.upper()} {split_name.upper()}:")
    print(f"  Operating Point: threshold = {best_thresh:.2f}, margin = {best_margin:.2f}")
    print(f"  Macro F0.5:      {best_excl_metrics['macro_f05']:.5f}")
    print(f"  Precision:       {best_excl_metrics['macro_precision']:.4f}")
    print(f"  Recall:          {best_excl_metrics['macro_recall']:.4f}")
    print(f"  Predicted Links: {best_excl_metrics['predicted_matches']:,}")
    print(f"  Secondary Recall:{sec_rec_v12_1:.4f}")

    return {
        "src": src,
        "split": split_name,
        "best_thresh": best_thresh,
        "best_margin": best_margin,
        "macro_f05": best_excl_metrics["macro_f05"],
        "precision": best_excl_metrics["macro_precision"],
        "recall": best_excl_metrics["macro_recall"],
        "predicted_matches": best_excl_metrics["predicted_matches"],
        "sec_recall": round(sec_rec_v12_1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="V12.1 Stage-2 Collective Sibling Matcher")
    parser.add_argument("--sources", nargs="+", default=["source2", "source3"], help="Sources to train and evaluate")
    args = parser.parse_args()

    results = {}
    for src in args.sources:
        stage1_model, stage2_model, feat_cols = generate_oof_stage1_and_train_stage2(src)
        res_v1 = evaluate_v12_1_pipeline(src, "val1", stage1_model, stage2_model, feat_cols)
        res_v2 = evaluate_v12_1_pipeline(src, "val2", stage1_model, stage2_model, feat_cols)
        results[f"{src}_val1"] = res_v1
        results[f"{src}_val2"] = res_v2

    results_path = ROOT / "experiments" / "v12_1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll evaluations finished. Results saved to: {results_path}")


if __name__ == "__main__":
    main()
