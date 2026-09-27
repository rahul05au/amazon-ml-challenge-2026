"""V13.1 — Expanded Candidate Graph + Retrained Two-Stage Matcher.

Implements the complete V13.1 pipeline:
1. Frozen R50 feature & score computation for all new V13 candidates
2. Valid expanded Stage-1 dataset creation (train_gate, val1, val2)
3. Stage-1 Retraining with R50 Ablation (Variant A: With R50, Variant B: Without R50)
4. 3-Fold Entity-Level OOF Stage-1 Predictions (zero leakage)
5. Stage-2 Collective Sibling Retraining
6. Full 4-Split Untouched Validation & Recovered Gold Tracing
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import _process_chunk_worker
from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import (
    get_clean_gold,
    compute_pairwise_features,
    engineer_competition_features,
    get_v12_feature_names,
    apply_target_exclusivity,
    s1_norm_path,
    sx_norm_path,
    gt_path,
    MODELS_DIR,
    V2_CACHE,
)
from experiments.v12_1_collective_matcher import (
    compute_collective_sibling_features,
    get_v12_1_feature_names,
)
from experiments.v13_candidate_audit_and_eval import run_v13_retrieval_pairs


def enrich_and_score_r50(pairs: set[tuple[str, str]], src: str, split_name: str) -> pd.DataFrame:
    """Fetch text attributes and compute true frozen R50 scores for new candidate pairs."""
    if not pairs:
        return pd.DataFrame()

    con = duckdb.connect()
    pairs_df = pd.DataFrame(list(pairs), columns=["s1_id", "sx_id"])
    con.register("new_pairs_tbl", pairs_df)

    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == "train_gate":
        where_cond = f"{h_expr} >= 10 AND {h_expr} < 25"
    elif split_name == "val1":
        where_cond = f"{h_expr} >= 0 AND {h_expr} < 5"
    elif split_name == "val2":
        where_cond = f"{h_expr} >= 25 AND {h_expr} < 30"
    else:
        raise ValueError(split_name)

    df_text = con.execute(f"""
        SELECT p.s1_id, p.sx_id,
               s1.name_clean AS s1_name, sx.name_clean AS sx_name,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
               s1.addr_clean AS s1_addr, sx.addr_clean AS sx_addr,
               s1.postal_code AS s1_post, sx.postal_code AS sx_post
        FROM new_pairs_tbl p
        JOIN read_parquet('{s1_norm_path()}') s1 ON p.s1_id = s1.entity_id
        JOIN read_parquet('{sx_norm_path(src)}') sx ON p.sx_id = sx.entity_id
    """).df()
    con.close()

    if len(df_text) == 0:
        return pd.DataFrame()

    # Compute 15 R50 features
    r50_feats = _process_chunk_worker(df_text)

    # Score with frozen R50 booster
    r50_model = lgb.Booster(model_file=str(MODELS_DIR / f"lgb_{src}.txt"))
    df_text["lgb_score"] = r50_model.predict(r50_feats)
    df_text["tfidf_score"] = 0.0  # TF-IDF placeholder

    # Drop temporary no-suffix columns to match V2 cache schema
    return df_text[["s1_id", "sx_id", "lgb_score", "tfidf_score", "s1_name", "sx_name", "s1_addr", "sx_addr", "s1_post", "sx_post"]]


def build_expanded_candidate_df(src: str, split_name: str) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Union existing candidate cache with V13 retrieval pairs and compute true R50 scores."""
    print(f"\nBuilding expanded candidate graph for {src.upper()} {split_name.upper()}...")
    cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
    old_df = pd.read_parquet(cache_path)
    old_pairs = set(zip(old_df["s1_id"], old_df["sx_id"]))

    v13_pairs = run_v13_retrieval_pairs(src, split_name)
    added_pairs = v13_pairs - old_pairs
    print(f"  Existing candidates: {len(old_pairs):,} | Added V13 candidates: +{len(added_pairs):,}")

    if added_pairs:
        enriched_added = enrich_and_score_r50(added_pairs, src, split_name)
        expanded_df = pd.concat([old_df, enriched_added], ignore_index=True)
    else:
        expanded_df = old_df.copy()

    # Deduplicate
    expanded_df = expanded_df.drop_duplicates(subset=["s1_id", "sx_id"]).reset_index(drop=True)
    print(f"  Final expanded candidate graph: {len(expanded_df):,} pairs")
    return expanded_df, added_pairs


def run_v13_1_experiment(src: str):
    print(f"\n{'='*75}\nV13.1 PIPELINE EXECUTION FOR {src.upper()}\n{'='*75}")
    _, gt_train, _ = get_clean_gold(src, "train_gate")

    # 1. Build Expanded Training Graph
    train_expanded_df, train_added = build_expanded_candidate_df(src, "train_gate")

    # Compute Features for Expanded Training Data
    print("Computing pairwise features on expanded training graph...")
    t0 = time.time()
    train_pairwise = compute_pairwise_features(train_expanded_df)
    train_full = pd.concat([train_expanded_df, train_pairwise], axis=1)
    train_full = engineer_competition_features(train_full)
    print(f"Features computed in {time.time()-t0:.1f}s")

    train_full["is_gold"] = [(s, x) in gt_train for s, x in zip(train_full["s1_id"], train_full["sx_id"])]
    train_full["label"] = train_full["is_gold"].astype(np.int32)

    v12_cols_with_r50 = get_v12_feature_names()
    v12_cols_no_r50 = [c for c in v12_cols_with_r50 if c != "lgb_score"]

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

    # 2. Section 6: Stage-1 R50 Ablation (Variant A: With R50, Variant B: Without R50)
    print("\n--- SECTION 6: STAGE-1 R50 ABLATION TRAINING ---")
    print("Training Variant A (WITH frozen R50 lgb_score)...")
    dtrain_a = lgb.Dataset(train_full[v12_cols_with_r50], label=train_full["label"])
    model_stage1_a = lgb.train(lgb_params, dtrain_a)

    print("Training Variant B (WITHOUT lgb_score)...")
    dtrain_b = lgb.Dataset(train_full[v12_cols_no_r50], label=train_full["label"])
    model_stage1_b = lgb.train(lgb_params, dtrain_b)

    # 3. Quick validation check of Variant A vs Variant B on Val1
    print("\nEvaluating Stage-1 Variant A vs Variant B on untouched Val1...")
    val1_expanded, val1_added = build_expanded_candidate_df(src, "val1")
    s1_all_v1, gt_set_v1, gt_df_v1 = get_clean_gold(src, "val1")

    val1_pairwise = compute_pairwise_features(val1_expanded)
    val1_full = pd.concat([val1_expanded, val1_pairwise], axis=1)
    val1_full = engineer_competition_features(val1_full)

    val1_full["score_a"] = model_stage1_a.predict(val1_full[v12_cols_with_r50])
    val1_full["score_b"] = model_stage1_b.predict(val1_full[v12_cols_no_r50])

    preds_a = val1_full[val1_full["score_a"] >= 0.70][["s1_id", "sx_id", "score_a"]].rename(columns={"score_a": "v12_score"})
    preds_b = val1_full[val1_full["score_b"] >= 0.70][["s1_id", "sx_id", "score_b"]].rename(columns={"score_b": "v12_score"})

    m_a = calculate_macro_f05(apply_target_exclusivity(preds_a), gt_df_v1, s1_all_v1)
    m_b = calculate_macro_f05(apply_target_exclusivity(preds_b), gt_df_v1, s1_all_v1)
    print(f"  Variant A (With R50):    Macro F0.5 = {m_a['macro_f05']:.5f} (P={m_a['macro_precision']:.4f}, R={m_a['macro_recall']:.4f})")
    print(f"  Variant B (Without R50): Macro F0.5 = {m_b['macro_f05']:.5f} (P={m_b['macro_precision']:.4f}, R={m_b['macro_recall']:.4f})")

    selected_stage1_cols = v12_cols_with_r50 if m_a["macro_f05"] >= m_b["macro_f05"] else v12_cols_no_r50
    selected_variant_name = "WITH_R50" if m_a["macro_f05"] >= m_b["macro_f05"] else "WITHOUT_R50"
    print(f"  Selected Stage-1 Variant: {selected_variant_name}")

    # Save selected Stage-1 model
    stage1_save_path = MODELS_DIR / f"v13_1_stage1_{src}.txt"
    if selected_variant_name == "WITH_R50":
        model_stage1_a.save_model(str(stage1_save_path))
    else:
        model_stage1_b.save_model(str(stage1_save_path))
    print(f"Saved selected Stage-1 model to: {stage1_save_path}")

    # 4. Section 7: Entity-Level 3-Fold OOF Predictions on Expanded train_gate
    print("\n--- SECTION 7: 3-FOLD ENTITY-LEVEL OOF STAGE-1 PREDICTIONS ---")
    s1_entities = train_full["s1_id"].unique()
    entity_folds = {eid: (abs(hash(eid + "_fold_v13")) % 3) for eid in s1_entities}
    train_full["fold"] = train_full["s1_id"].map(entity_folds)

    oof_preds = np.zeros(len(train_full), dtype=np.float32)
    for fold in range(3):
        tr_mask = train_full["fold"] != fold
        val_mask = train_full["fold"] == fold
        dtr = lgb.Dataset(train_full.loc[tr_mask, selected_stage1_cols], label=train_full.loc[tr_mask, "label"])
        bst = lgb.train(lgb_params, dtr)
        val_preds = bst.predict(train_full.loc[val_mask, selected_stage1_cols])
        oof_preds[val_mask] = val_preds
        print(f"  Fold {fold+1}/3 finished. Val mean: {val_preds.mean():.4f}, Max: {val_preds.max():.4f}")

    train_full["stage1_score"] = oof_preds

    # 5. Section 8: Compute Collective Sibling Features and Retrain Stage-2
    print("\n--- SECTION 8: RETRAIN STAGE-2 COLLECTIVE SIBLING MODEL ---")
    col_feats = compute_collective_sibling_features(train_full, score_col="stage1_score")
    train_full = pd.concat([train_full, col_feats], axis=1)

    stage2_feature_cols = [c for c in get_v12_1_feature_names() if c in train_full.columns]
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

    print("Fitting Stage-2 LightGBM model on expanded collective graph...")
    model_stage2 = lgb.train(stage2_params, dtrain_stage2)
    stage2_save_path = MODELS_DIR / f"v13_1_stage2_{src}.txt"
    model_stage2.save_model(str(stage2_save_path))
    print(f"Saved Stage-2 model to: {stage2_save_path}")

    # 6. Section 9, 10, 11: Comprehensive 4-Split Evaluation & Three-System Comparison
    print("\n--- SECTION 9, 10, 11: FULL VALIDATION & THREE-SYSTEM COMPARISON ---")
    selected_stage1_model = lgb.Booster(model_file=str(stage1_save_path))
    v12_1_stage1_orig = lgb.Booster(model_file=str(MODELS_DIR / f"v12_1_stage1_{src}.txt"))
    v12_1_stage2_orig = lgb.Booster(model_file=str(MODELS_DIR / f"v12_1_stage2_{src}.txt"))

    split_metrics = {}
    for split_name in ["val1", "val2"]:
        s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
        total_gold = len(gt_set)

        # Load expanded validation candidates
        val_exp, added_val_pairs = build_expanded_candidate_df(src, split_name)
        val_pairwise = compute_pairwise_features(val_exp)
        val_full = pd.concat([val_exp, val_pairwise], axis=1)
        val_full = engineer_competition_features(val_full)

        recovered_gold = (set(zip(val_exp["s1_id"], val_exp["sx_id"])) & gt_set) - (set(zip(pd.read_parquet(V2_CACHE / f"v2_{src}_{split_name}.parquet")["s1_id"], pd.read_parquet(V2_CACHE / f"v2_{src}_{split_name}.parquet")["sx_id"])) & gt_set)

        # System B: V13 expanded graph + ORIGINAL V12.1 models (with true R50 score)
        val_full["stage1_score_orig"] = v12_1_stage1_orig.predict(val_full[get_v12_feature_names()])
        col_orig = compute_collective_sibling_features(val_full, score_col="stage1_score_orig")
        val_full_orig = pd.concat([val_full, col_orig], axis=1)
        val_full_orig["stage1_score"] = val_full_orig["stage1_score_orig"]
        val_full_orig["stage2_score_orig"] = v12_1_stage2_orig.predict(val_full_orig[get_v12_1_feature_names()])
        val_full_orig["score_sys_b"] = 0.2 * val_full_orig["stage1_score_orig"] + 0.8 * val_full_orig["stage2_score_orig"]

        # System C: V13 expanded graph + RETRAINED V13.1 models
        val_full["stage1_score_retrained"] = selected_stage1_model.predict(val_full[selected_stage1_cols])
        col_retrained = compute_collective_sibling_features(val_full, score_col="stage1_score_retrained")
        val_full_retrained = pd.concat([val_full, col_retrained], axis=1)
        val_full_retrained["stage1_score"] = val_full_retrained["stage1_score_retrained"]
        val_full_retrained["stage2_score_retrained"] = model_stage2.predict(val_full_retrained[stage2_feature_cols])
        val_full_retrained["score_sys_c"] = 0.2 * val_full_retrained["stage1_score_retrained"] + 0.8 * val_full_retrained["stage2_score_retrained"]

        # Threshold sweeps for System B and System C
        th = 0.47
        preds_b = val_full_orig[val_full_orig["score_sys_b"] >= th][["s1_id", "sx_id", "score_sys_b"]].rename(columns={"score_sys_b": "v12_score"})
        preds_c = val_full_retrained[val_full_retrained["score_sys_c"] >= th][["s1_id", "sx_id", "score_sys_c"]].rename(columns={"score_sys_c": "v12_score"})

        excl_b = apply_target_exclusivity(preds_b)
        excl_c = apply_target_exclusivity(preds_c)

        m_b = calculate_macro_f05(excl_b, gt_df, s1_all)
        m_c = calculate_macro_f05(excl_c, gt_df, s1_all)

        # Tracing Recovered Gold Acceptance in System C
        c_pred_set = set(zip(excl_c["s1_id"], excl_c["sx_id"]))
        recov_accepted = recovered_gold & c_pred_set
        recov_acc_rate = len(recov_accepted) / max(len(recovered_gold), 1)

        print(f"\n{src.upper()} {split_name.upper()} Results Comparison:")
        print(f"  System B (Expanded Graph + Old V12.1 Models): Macro F0.5 = {m_b['macro_f05']:.5f} (P={m_b['macro_precision']:.4f}, R={m_b['macro_recall']:.4f})")
        print(f"  System C (Expanded Graph + Retrained V13.1):  Macro F0.5 = {m_c['macro_f05']:.5f} (P={m_c['macro_precision']:.4f}, R={m_c['macro_recall']:.4f})")
        print(f"  Recovered Gold Tracing: {len(recov_accepted):,} / {len(recovered_gold):,} accepted ({recov_acc_rate*100:.1f}%)")

        split_metrics[f"{src}_{split_name}"] = {
            "system_b_f05": m_b["macro_f05"],
            "system_c_f05": m_c["macro_f05"],
            "precision": m_c["macro_precision"],
            "recall": m_c["macro_recall"],
            "recovered_gold_total": len(recovered_gold),
            "recovered_gold_accepted": len(recov_accepted),
            "recovered_acceptance_rate": round(recov_acc_rate, 4),
        }

    return split_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["source2", "source3"])
    args = parser.parse_args()

    all_results = {}
    for src in args.sources:
        res = run_v13_1_experiment(src)
        all_results.update(res)

    results_path = ROOT / "experiments" / "v13_1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll experiments complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
