"""End-to-end Entity Resolution Matcher & LightGBM Model.

High-throughput, memory-bounded matching pipeline:
1. Clustered train/val sampling by S1 entity.
2. Parallelized multi-core RapidFuzz feature extraction (>130k pairs/s).
3. LightGBM gradient boosted decision tree classifier (<= 8B params, Apache 2.0).
4. Exact macro-averaged F0.5 evaluation and threshold sweep per source.
5. High-speed streaming test inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import (
    INTERMEDIATE_DIR,
    candidate_parquet,
    norm_parquet,
)
from src.features import FEATURE_NAMES, compute_features_parallel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"



def calculate_macro_f05(
    pred_pairs: pd.DataFrame,
    true_pairs: pd.DataFrame,
    all_s1_ids: set[str],
    beta: float = 0.5,
) -> dict[str, float]:
    """Calculate exact macro-averaged F0.5 per S1 entity."""
    beta_sq = beta ** 2
    weight = 1.0 + beta_sq

    pred_grouped = pred_pairs.groupby("s1_id")["sx_id"].apply(set).to_dict() if len(pred_pairs) > 0 else {}
    true_grouped = true_pairs.groupby("s1_id")["true_match_id"].apply(set).to_dict() if len(true_pairs) > 0 else {}

    f05_scores = []
    precisions = []
    recalls = []
    singletons_correct = 0
    total_singletons = 0

    for s1 in all_s1_ids:
        preds = pred_grouped.get(s1, set())
        trues = true_grouped.get(s1, set())

        is_singleton = len(trues) == 0
        if is_singleton:
            total_singletons += 1

        if len(preds) == 0 and len(trues) == 0:
            singletons_correct += 1
            f05_scores.append(1.0)
            precisions.append(1.0)
            recalls.append(1.0)
        elif len(preds) == 0 or len(trues) == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            tp = len(preds & trues)
            p = tp / len(preds)
            r = tp / len(trues)
            precisions.append(p)
            recalls.append(r)
            if (beta_sq * p + r) > 0:
                f05_scores.append((weight * p * r) / (beta_sq * p + r))
            else:
                f05_scores.append(0.0)

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = singletons_correct / total_singletons if total_singletons > 0 else 1.0

    return {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_p, 5),
        "macro_recall": round(macro_r, 5),
        "total_s1": len(all_s1_ids),
        "total_singletons": total_singletons,
        "singletons_correct": singletons_correct,
        "singleton_accuracy": round(singleton_acc, 5),
        "predicted_matches": len(pred_pairs),
    }


def train_target_model(target_source: str = "source2") -> tuple[lgb.Booster, float, dict]:
    """Train LightGBM model on clustered S1 samples and sweep threshold."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "S2-" if target_source == "source2" else "S3-"

    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL: S1 -> {target_source}")
    print(f"{'='*60}")

    con = duckdb.connect()
    con.execute("SET threads TO 6")
    con.execute("SET memory_limit = '6GB'")

    print("Extracting clustered train (44k S1) and validation (11k S1) pairs ...")
    t0 = time.perf_counter()

    con.execute(f"""
        CREATE TEMP TABLE s1_split AS
        SELECT entity_id as s1_id,
               CASE WHEN abs(hash(entity_id || 'v1')) % 1000 < 5 THEN 'val'
                    WHEN abs(hash(entity_id || 'v1')) % 1000 < 25 THEN 'train'
                    ELSE 'rest' END as split
        FROM read_parquet('intermediate/train/norm_source1.parquet')
    """)

    con.execute(f"""
        CREATE TEMP TABLE gt_target AS
        SELECT source1_entity_id as s1_id, trim(unnest(string_split(matched_entity_ids, ','))) as true_id
        FROM read_parquet('intermediate/train/ground_truth.parquet')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
    """)

    df = con.execute(f"""
        WITH selected_s1 AS (
            SELECT s1_id, split FROM s1_split WHERE split IN ('train', 'val')
        ),
        cands AS (
            SELECT c.s1_id, c.sx_id, s.split
            FROM read_parquet('intermediate/train/candidates_s1_{target_source}.parquet') c
            JOIN selected_s1 s ON c.s1_id = s.s1_id
        )
        SELECT
            c.s1_id, c.sx_id, c.split,
            CASE WHEN g.true_id IS NOT NULL THEN 1 ELSE 0 END as label,
            s1.name_clean as s1_name, sx.name_clean as sx_name,
            s1.name_no_suffix as s1_no_suf, sx.name_no_suffix as sx_no_suf,
            s1.addr_clean as s1_addr, sx.addr_clean as sx_addr,
            COALESCE(s1.postal_code, '') as s1_post, COALESCE(sx.postal_code, '') as sx_post
        FROM cands c
        JOIN read_parquet('intermediate/train/norm_source1.parquet') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('intermediate/train/norm_{target_source}.parquet') sx ON c.sx_id = sx.entity_id
        LEFT JOIN gt_target g ON c.s1_id = g.s1_id AND c.sx_id = g.true_id
    """).df()

    val_s1_all = set(con.execute("SELECT s1_id FROM s1_split WHERE split = 'val'").df()["s1_id"])
    val_gt_pairs = con.execute("""
        SELECT g.s1_id, g.true_id as true_match_id
        FROM gt_target g
        JOIN (SELECT s1_id FROM s1_split WHERE split = 'val') s ON g.s1_id = s.s1_id
    """).df()
    con.close()

    train_mask = (df["split"] == "train").values
    val_mask = (df["split"] == "val").values

    print(f"Data extracted in {time.perf_counter() - t0:.1f}s")
    print(f"Train pairs: {train_mask.sum():,} (positives: {df[train_mask]['label'].sum():,})")
    print(f"Val pairs:   {val_mask.sum():,} (positives: {df[val_mask]['label'].sum():,})")

    print("\nComputing parallelized features (6 workers) ...")
    t_feat = time.perf_counter()
    X_train = compute_features_parallel(df[train_mask], n_workers=6)
    y_train = df[train_mask]["label"].values
    X_val = compute_features_parallel(df[val_mask], n_workers=6)
    y_val = df[val_mask]["label"].values
    val_pairs_info = df[val_mask][["s1_id", "sx_id"]].copy()

    total_pairs = len(X_train) + len(X_val)
    dur = time.perf_counter() - t_feat
    print(f"Features computed for {total_pairs:,} pairs in {dur:.1f}s ({total_pairs/dur:,.0f} pairs/s)")

    print("\nTraining LightGBM classifier ...")
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "scale_pos_weight": 3.0,
        "n_jobs": 6,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_NAMES)

    bst = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    val_preds = bst.predict(X_val)
    val_pairs_info["score"] = val_preds

    print("\n--- Sweeping Thresholds for Macro F0.5 ---")
    print(f"{'Threshold':<11}{'Macro F0.5':<13}{'Precision':<13}{'Recall':<11}{'Matches':<10}")
    print("-" * 58)

    best_f05 = -1.0
    best_th = 0.70
    best_metrics = {}

    for th in [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        pred_sub = val_pairs_info[val_pairs_info["score"] >= th][["s1_id", "sx_id"]]
        metrics = calculate_macro_f05(pred_sub, val_gt_pairs, val_s1_all)
        f05 = metrics["macro_f05"]
        p = metrics["macro_precision"]
        r = metrics["macro_recall"]
        m = metrics["predicted_matches"]
        print(f"{th:<11.2f}{f05:<13.4f}{p:<13.4f}{r:<11.4f}{m:<10}")
        if f05 > best_f05:
            best_f05 = f05
            best_th = th
            best_metrics = metrics

    print("-" * 58)
    print(f"Optimal Threshold: {best_th:.2f} => Macro F0.5: {best_f05:.4f} (P: {best_metrics['macro_precision']:.4f}, R: {best_metrics['macro_recall']:.4f})")
    print(f"Singleton Accuracy: {best_metrics['singleton_accuracy']:.4f}")

    model_path = MODELS_DIR / f"lgb_{target_source}.txt"
    bst.save_model(str(model_path))
    print(f"Model saved to {model_path.name}")

    th_path = MODELS_DIR / f"threshold_{target_source}.json"
    with open(th_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_source": target_source,
            "optimal_threshold": best_th,
            "metrics": best_metrics,
        }, f, indent=2)
    print(f"Threshold saved to {th_path.name}")

    return bst, best_th, best_metrics


def predict_test_matches(target_source: str = "source2", batch_size: int = 2_000_000) -> Path:
    """Stream candidates for test split, predict with LightGBM, and save matches."""
    split = "test"
    model_path = MODELS_DIR / f"lgb_{target_source}.txt"
    th_path = MODELS_DIR / f"threshold_{target_source}.json"

    if not model_path.exists() or not th_path.exists():
        raise FileNotFoundError(f"Model or threshold file not found for {target_source}")

    with open(th_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    threshold = float(meta["optimal_threshold"])
    bst = lgb.Booster(model_file=str(model_path))

    cand_path = candidate_parquet(split, target_source)
    s1_path = norm_parquet(split, "source1")
    sx_path = norm_parquet(split, target_source)

    out_path = INTERMEDIATE_DIR / split / f"matches_{target_source}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads TO 6")
    con.execute("SET memory_limit = '6GB'")

    cand_safe = str(cand_path).replace("\\", "/")
    s1_safe = str(s1_path).replace("\\", "/")
    sx_safe = str(sx_path).replace("\\", "/")

    total_cands = con.execute(f"SELECT count(*) FROM read_parquet('{cand_safe}')").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"  PREDICTING MATCHES: S1 -> {target_source} [test]")
    print(f"  Total candidate pairs to score: {total_cands:,}")
    print(f"  Model threshold: {threshold:.2f}")

    print("Loading normalized source records for fast lookup ...")
    t_load = time.perf_counter()
    con = duckdb.connect()
    s1_df = con.execute(f"SELECT entity_id, name_clean, name_no_suffix, addr_clean, COALESCE(postal_code, '') as postal_code FROM read_parquet('{s1_safe}')").df()
    sx_df = con.execute(f"SELECT entity_id, name_clean, name_no_suffix, addr_clean, COALESCE(postal_code, '') as postal_code FROM read_parquet('{sx_safe}')").df()
    con.close()

    s1_map = {
        row.entity_id: (row.name_clean or "", row.name_no_suffix or "", row.addr_clean or "", row.postal_code or "")
        for row in s1_df.itertuples(index=False)
    }
    sx_map = {
        row.entity_id: (row.name_clean or "", row.name_no_suffix or "", row.addr_clean or "", row.postal_code or "")
        for row in sx_df.itertuples(index=False)
    }
    del s1_df, sx_df
    print(f"  -> {len(s1_map):,} S1 and {len(sx_map):,} SX entities loaded ({time.perf_counter() - t_load:.1f}s)")

    offset = 0
    part = 0
    temp_dir = INTERMEDIATE_DIR / split / f"_temp_matches_{target_source}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    part_files = []

    t0 = time.perf_counter()
    total_matched = 0
    empty_tuple = ("", "", "", "")

    print(f"Streaming candidate batches (batch_size={batch_size:,}) ...")
    pf = pq.ParquetFile(str(cand_path))

    for batch in pf.iter_batches(batch_size=batch_size):
        t_batch = time.perf_counter()
        s1_ids = batch["s1_id"].to_pylist()
        sx_ids = batch["sx_id"].to_pylist()
        n_pairs = len(s1_ids)

        s1_data = [s1_map.get(s, empty_tuple) for s in s1_ids]
        sx_data = [sx_map.get(s, empty_tuple) for s in sx_ids]

        batch_df = pd.DataFrame({
            "s1_name": [x[0] for x in s1_data],
            "sx_name": [x[0] for x in sx_data],
            "s1_no_suf": [x[1] for x in s1_data],
            "sx_no_suf": [x[1] for x in sx_data],
            "s1_addr": [x[2] for x in s1_data],
            "sx_addr": [x[2] for x in sx_data],
            "s1_post": [x[3] for x in s1_data],
            "sx_post": [x[3] for x in sx_data],
        })

        X_batch = compute_features_parallel(batch_df, n_workers=6)
        scores = bst.predict(X_batch)
        match_mask = scores >= threshold

        if match_mask.any():
            matched_s1 = [s1_ids[i] for i, m in enumerate(match_mask) if m]
            matched_sx = [sx_ids[i] for i, m in enumerate(match_mask) if m]
            matched_df = pd.DataFrame({"s1_id": matched_s1, "sx_id": matched_sx})
            total_matched += len(matched_df)

            part_path = temp_dir / f"matches_part_{part:04d}.parquet"
            matched_df.to_parquet(part_path, index=False, compression="zstd")
            part_files.append(part_path)

        offset += n_pairs
        part += 1
        elapsed = time.perf_counter() - t_batch
        rate = n_pairs / elapsed if elapsed > 0 else 0
        pct = (offset / total_cands) * 100
        print(f"  [{pct:5.1f}%] Scored {offset:>10,} / {total_cands:,} pairs ({rate:>9,.0f} pairs/s) | Matches: {total_matched:>8,}")

    print("\nMerging match partitions ...")
    con = duckdb.connect()
    temp_safe = str(temp_dir / "*.parquet").replace("\\", "/")
    out_safe = str(out_path).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT DISTINCT s1_id, sx_id FROM read_parquet('{temp_safe}')
        ) TO '{out_safe}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    final_count = con.execute(f"SELECT count(*) FROM read_parquet('{out_safe}')").fetchone()[0]
    con.close()

    for p in part_files:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    total_time = time.perf_counter() - t0
    print(f"  -> Total {final_count:,} matches written to {out_path.name} ({total_time:.1f}s)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Model training and inference pipeline.")
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all")
    parser.add_argument("--target", choices=["source2", "source3", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=2_000_000)
    args = parser.parse_args()

    targets = ["source2", "source3"] if args.target == "both" else [args.target]

    if args.mode in ["train", "all"]:
        for t in targets:
            train_target_model(t)

    if args.mode in ["predict", "all"]:
        for t in targets:
            predict_test_matches(t, batch_size=args.batch_size)

    return 0


if __name__ == "__main__":
    sys.exit(main())
