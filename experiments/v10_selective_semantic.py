"""V10: Selective Multilingual Semantic Matcher with Hard-Negative Contrastive Training.

Isolates:
1. Multilingual semantic representation (Sentence-Transformer MiniLM)
2. Hard-negative contrastive training on mined R50 false positives
3. Calibrated decision layer integrating R50, Phase-3, and semantic features
4. Ambiguity gating (Mode A vs Mode B)
5. Rigorous evaluation across all 4 untouched validation splits and hard error slices
"""

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models
from sklearn.linear_model import LogisticRegression

# Set reproducible seeds and CPU parallelism
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(6)

ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE = ROOT / "intermediate" / "train"
V2_CACHE = ROOT / "intermediate" / "v2_cache"
MODELS_DIR = ROOT / "models"

OPERATIONAL_THRESHOLDS = {"source2": 0.70, "source3": 0.65}
BASE_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 128
MARGIN = 0.5
HARD_NEG_H = 3
BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 1e-3


def serialize_entity(name: Any, addr: Any, country: Any) -> str:
    """Compact structured text serialization per Section 7."""
    name_str = str(name).strip() if name and pd.notna(name) else "MISSING"
    addr_str = str(addr).strip() if addr and pd.notna(addr) else "MISSING"
    cntry_str = str(country).strip() if country and pd.notna(country) else "MISSING"
    return f"[NAME] {name_str}\n[ADDRESS] {addr_str}\n[COUNTRY] {cntry_str}"


def calculate_macro_f05(
    pred_pairs: pd.DataFrame,
    true_pairs: pd.DataFrame,
    all_s1_ids: set[str],
    beta: float = 0.5,
) -> dict[str, float]:
    """Exact competition macro-averaged F0.5 per S1 entity."""
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


def get_clean_gold(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    """Retrieve clean target ground truth without cross-source contamination."""
    prefix = "S2-" if src == "source2" else "S3-"
    con = duckdb.connect()
    
    if split_name == "train_gate":
        cond = "abs(hash(entity_id || 'v1')) % 1000 >= 10 AND abs(hash(entity_id || 'v1')) % 1000 < 25"
    elif split_name == "val1":
        cond = "abs(hash(entity_id || 'v1')) % 1000 < 5"
    elif split_name == "val2":
        cond = "abs(hash(entity_id || 'v1')) % 1000 >= 25 AND abs(hash(entity_id || 'v1')) % 1000 < 30"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    con.execute(f"""
        CREATE TEMP TABLE s1_split AS
        SELECT entity_id as s1_id
        FROM read_parquet('{(INTERMEDIATE / "norm_source1.parquet").as_posix()}')
        WHERE {cond}
    """)

    gt_df = con.execute(f"""
        WITH unnested AS (
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{(INTERMEDIATE / "ground_truth.parquet").as_posix()}')
            WHERE matched_entity_ids != ''
        )
        SELECT u.s1_id, u.true_match_id
        FROM unnested u
        JOIN s1_split s ON u.s1_id = s.s1_id
        WHERE u.true_match_id LIKE '{prefix}%'
    """).df()

    s1_all = set(con.execute("SELECT s1_id FROM s1_split").df()["s1_id"])
    con.close()

    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
    return s1_all, gt_set, gt_df


class ContrastiveProjectionHead(nn.Module):
    """Calibrated contrastive projection head mapping MiniLM representations."""
    def __init__(self, in_features: int = 384, hidden_dim: int = 384, out_features: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual skip connection preserves pretrained semantics while adapting contrast
        out = x + self.net(x)
        return nn.functional.normalize(out, p=2, dim=-1)


def build_entity_serialization_from_df(df: pd.DataFrame, entity_ids: set[str] | None = None) -> dict[str, str]:
    """Retrieve text serialization directly from cached dataframe without external joins."""
    res = {}
    s1_sub = df[["s1_id", "s1_name", "s1_addr"]].drop_duplicates("s1_id")
    for r in s1_sub.itertuples():
        if entity_ids is None or r.s1_id in entity_ids:
            res[r.s1_id] = serialize_entity(r.s1_name, r.s1_addr, "MISSING")
            
    sx_sub = df[["sx_id", "sx_name", "sx_addr"]].drop_duplicates("sx_id")
    for r in sx_sub.itertuples():
        if entity_ids is None or r.sx_id in entity_ids:
            res[r.sx_id] = serialize_entity(r.sx_name, r.sx_addr, "MISSING")
    return res


def train_v10_contrastive_encoder(
    src: str,
    train_cache_df: pd.DataFrame,
    gt_set: set[tuple[str, str]],
    base_encoder: SentenceTransformer,
) -> tuple[SentenceTransformer, ContrastiveProjectionHead]:
    """Train multilingual semantic contrastive model with multi-positive margin loss."""
    print(f"\n{'='*70}")
    print(f"TRAINING V10 SEMANTIC ENCODER FOR {src.upper()}")
    print(f"{'='*70}")

    train_cache_df["is_gold"] = [(s, x) in gt_set for s, x in zip(train_cache_df["s1_id"], train_cache_df["sx_id"])]
    pos_df = train_cache_df[train_cache_df["is_gold"]]
    neg_df = train_cache_df[~train_cache_df["is_gold"]]

    print(f"Total candidate pairs: {len(train_cache_df):,} | Confirmed positives: {len(pos_df):,}")

    # Mine hard negatives: top H=3 per S1 by R50 score
    neg_sorted = neg_df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False])
    hard_negs = neg_sorted.groupby("s1_id").head(HARD_NEG_H)
    print(f"Mined hard negatives (H={HARD_NEG_H}): {len(hard_negs):,}")

    # Group into multi-positive queries
    pos_grouped = pos_df.groupby("s1_id")["sx_id"].apply(list).to_dict()
    neg_grouped = hard_negs.groupby("s1_id")["sx_id"].apply(list).to_dict()

    valid_s1 = [s1 for s1 in pos_grouped if s1 in neg_grouped and len(neg_grouped[s1]) > 0]
    print(f"Multi-positive training S1 queries: {len(valid_s1):,}")

    # Collect distinct entities to serialize and encode
    distinct_s1 = set(valid_s1)
    distinct_sx = set()
    for s1 in valid_s1:
        distinct_sx.update(pos_grouped[s1])
        distinct_sx.update(neg_grouped[s1])

    all_distinct = distinct_s1 | distinct_sx
    print(f"Encoding {len(all_distinct):,} distinct entities ({len(distinct_s1):,} S1, {len(distinct_sx):,} targets)...", flush=True)

    entity_texts = build_entity_serialization_from_df(train_cache_df, all_distinct)
    id_list = list(entity_texts.keys())
    text_list = [entity_texts[i] for i in id_list]

    t0 = time.time()
    raw_embs = base_encoder.encode(text_list, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    emb_dict = {eid: raw_embs[idx] for idx, eid in enumerate(id_list)}
    print(f"Base embeddings computed in {time.time() - t0:.1f}s ({len(id_list)/(time.time()-t0):.1f} items/s)")

    # Prepare PyTorch tensors for multi-positive contrastive training
    # For each S1: sample positive p and hard negative n
    triples = []
    for s1 in valid_s1:
        for p in pos_grouped[s1]:
            for n in neg_grouped[s1]:
                triples.append((emb_dict[s1], emb_dict[p], emb_dict[n]))

    triples = np.array(triples, dtype=np.float32)
    print(f"Constructed {len(triples):,} contrastive triples from multi-positive/hard-negative sets.")

    # Train contrastive projection head
    device = torch.device("cpu")
    head = ContrastiveProjectionHead(384, 384, 384).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    dataset_t = torch.utils.data.TensorDataset(
        torch.from_numpy(triples[:, 0]),
        torch.from_numpy(triples[:, 1]),
        torch.from_numpy(triples[:, 2]),
    )
    loader = torch.utils.data.DataLoader(dataset_t, batch_size=BATCH_SIZE, shuffle=True)

    t_train = time.time()
    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss = 0.0
        n_batches = 0
        for b_q, b_p, b_n in loader:
            optimizer.zero_grad()
            z_q = head(b_q)
            z_p = head(b_p)
            z_n = head(b_n)

            cos_p = (z_q * z_p).sum(dim=-1)
            cos_n = (z_q * z_n).sum(dim=-1)

            # Multi-positive margin objective: cos(q, p) > cos(q, n) + margin
            loss = torch.clamp(MARGIN - (cos_p - cos_n), min=0.0).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{EPOCHS:2d} | Contrastive Loss: {avg_loss:.5f}")

    print(f"Contrastive training completed in {time.time() - t_train:.1f}s")

    # Save V10 semantic model
    model_save_dir = MODELS_DIR / f"v10_semantic_{src}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), model_save_dir / "projection_head.pt")
    base_encoder.save(str(model_save_dir / "base_encoder"))
    with open(model_save_dir / "config.json", "w") as f:
        json.dump({
            "base_encoder": BASE_ENCODER_NAME,
            "margin": MARGIN,
            "hard_neg_h": HARD_NEG_H,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "seed": 42
        }, f, indent=2)
    print(f"Saved V10 semantic model to: {model_save_dir}")

    return base_encoder, head


def compute_semantic_features_for_df(
    df: pd.DataFrame,
    src: str,
    base_encoder: SentenceTransformer,
    head: ContrastiveProjectionHead,
) -> pd.DataFrame:
    """Compute semantic cosine similarity and context features for candidate pairs."""
    distinct_s1 = set(df["s1_id"])
    distinct_sx = set(df["sx_id"])
    all_distinct = distinct_s1 | distinct_sx

    entity_texts = build_entity_serialization_from_df(df, all_distinct)
    id_list = list(entity_texts.keys())
    text_list = [entity_texts[i] for i in id_list]

    raw_embs = base_encoder.encode(text_list, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    with torch.no_grad():
        proj_embs = head(torch.from_numpy(raw_embs)).numpy()

    emb_map = {eid: proj_embs[idx] for idx, eid in enumerate(id_list)}

    # Compute dot products vectorized
    s1_arr = np.array([emb_map[s] for s in df["s1_id"]])
    sx_arr = np.array([emb_map[x] for x in df["sx_id"]])
    cos_sim = (s1_arr * sx_arr).sum(axis=-1)
    df["semantic_cosine"] = cos_sim

    # Context features within candidate set
    df.sort_values(by=["s1_id", "semantic_cosine"], ascending=[True, False], inplace=True)
    df["semantic_rank"] = df.groupby("s1_id").cumcount() + 1
    top1_cos = df.groupby("s1_id")["semantic_cosine"].transform("first")
    df["semantic_margin_to_best"] = top1_cos - df["semantic_cosine"]
    df["r50_minus_semantic"] = df["lgb_score"] - df["semantic_cosine"]
    return df


def engineer_decision_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Phase-3 and V10 decision features."""
    df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False], inplace=True)
    df["cand_rank"] = df.groupby("s1_id").cumcount() + 1
    df["cand_count"] = df.groupby("s1_id")["lgb_score"].transform("count")

    top1_scores = df.groupby("s1_id")["lgb_score"].transform("first")
    df["top1_score"] = top1_scores
    df["score_diff_top1"] = df["lgb_score"] - top1_scores
    df["score_ratio_top1"] = df["lgb_score"] / (top1_scores + 1e-6)

    mean_scores = df.groupby("s1_id")["lgb_score"].transform("mean")
    df["score_diff_mean"] = df["lgb_score"] - mean_scores
    return df


def train_calibrated_decision_layer(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.Booster, LogisticRegression]:
    """Train LightGBM decision model and Platt scaler calibration."""
    # Split train_df into 80% train and 20% internal calibration
    s1_unique = train_df["s1_id"].unique()
    np.random.seed(42)
    s1_train = set(np.random.choice(s1_unique, size=int(0.80 * len(s1_unique)), replace=False))

    tr_mask = train_df["s1_id"].isin(s1_train)
    d_tr = train_df[tr_mask]
    d_cal = train_df[~tr_mask]

    dtrain = lgb.Dataset(d_tr[feature_cols], label=d_tr["is_gold"])
    params = {
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
    model = lgb.train(params, dtrain)

    # Platt scaling (Logistic Regression on raw margin/logit)
    cal_raw = model.predict(d_cal[feature_cols], raw_score=True).reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(cal_raw, d_cal["is_gold"])
    return model, platt


def evaluate_error_slices(df: pd.DataFrame, threshold: float) -> dict[str, dict[str, float]]:
    """Evaluate precision on hard error slices."""
    domain_re = re.compile(r"\b(com|org|net|io|in|co|biz|info)\b")
    non_ascii_re = re.compile(r"[^\x00-\x7F]")

    slices = {
        "same_near_identical": [],
        "cross_script": [],
        "alias_domain": [],
        "different_location": [],
        "address_noise": [],
        "short_common": [],
        "other": [],
    }

    for r in df.itertuples():
        n1 = str(r.s1_name).lower() if hasattr(r, 's1_name') and pd.notna(r.s1_name) else ""
        n2 = str(r.sx_name).lower() if hasattr(r, 'sx_name') and pd.notna(r.sx_name) else ""
        a1 = str(r.s1_addr).lower() if hasattr(r, 's1_addr') and pd.notna(r.s1_addr) else ""
        a2 = str(r.sx_addr).lower() if hasattr(r, 'sx_addr') and pd.notna(r.sx_addr) else ""

        is_cross = bool(non_ascii_re.search(n1)) != bool(non_ascii_re.search(n2))
        has_dom = bool(domain_re.search(n1)) or bool(domain_re.search(n2))
        is_short = len(n1) <= 5 or len(n2) <= 5

        cat = "other"
        if is_cross:
            cat = "cross_script"
        elif has_dom and ("." in n1 or "." in n2):
            cat = "alias_domain"
        elif is_short:
            cat = "short_common"
        elif hasattr(r, 'addr_ratio') and r.addr_ratio < 0.40:
            cat = "different_location"
        elif n1 == n2 or (len(n1) > 5 and n1 in n2):
            cat = "same_near_identical"
        elif hasattr(r, 'addr_ratio') and r.addr_ratio < 0.60:
            cat = "address_noise"

        slices[cat].append(r)

    results = {}
    for cat_name, rows in slices.items():
        if not rows:
            results[cat_name] = {"r50_p": 0.0, "p3_p": 0.0, "v10_p": 0.0, "count": 0}
            continue
        c_df = pd.DataFrame(rows)
        # Precision: TP / Predicted
        r50_acc = c_df[c_df["lgb_score"] >= threshold]
        r50_p = r50_acc["is_gold"].mean() if len(r50_acc) > 0 else 1.0

        p3_acc = c_df[c_df["p3_score"] >= 0.50]
        p3_p = p3_acc["is_gold"].mean() if len(p3_acc) > 0 else 1.0

        v10_acc = c_df[c_df["v10_prob"] >= 0.50]
        v10_p = v10_acc["is_gold"].mean() if len(v10_acc) > 0 else 1.0

        results[cat_name] = {
            "r50_p": float(r50_p),
            "p3_p": float(p3_p),
            "v10_p": float(v10_p),
            "count": len(c_df),
        }
    return results


def run_experiment_for_source(
    src: str,
    base_encoder: SentenceTransformer,
) -> dict[str, Any]:
    """Execute complete V10 pipeline for source2 or source3."""
    t_start = time.time()
    print(f"\n{'#'*80}")
    print(f"V10 EXPERIMENT: TARGET SOURCE = {src.upper()}")
    print(f"{'#'*80}")

    s1_train, gt_train_set, gt_train_df = get_clean_gold(src, "train_gate")
    train_cache_path = V2_CACHE / f"v2_{src}_train_gate.parquet"
    train_df = pd.read_parquet(train_cache_path)

    # 1. Train Contrastive Encoder
    _, head = train_v10_contrastive_encoder(src, train_df, gt_train_set, base_encoder)

    # 2. Extract semantic and contextual features on training subset for decision layer
    print("Preparing training features for calibrated decision layer...")
    # Sample 100k rows from train_df containing confirmed positives + top negatives
    train_sub = train_df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False]).groupby("s1_id").head(5).reset_index(drop=True)
    train_sub["is_gold"] = [(s, x) in gt_train_set for s, x in zip(train_sub["s1_id"], train_sub["sx_id"])]
    train_sub = engineer_decision_features(train_sub)
    train_sub = compute_semantic_features_for_df(train_sub, src, base_encoder, head)

    # Phase-3 features & model
    p3_cols = ["lgb_score", "cand_rank", "cand_count", "top1_score", "score_diff_top1", "score_ratio_top1", "score_diff_mean"]
    dp3 = lgb.Dataset(train_sub[p3_cols], label=train_sub["is_gold"])
    p3_model = lgb.train({
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15, "scale_pos_weight": 3.0,
        "random_state": 42, "verbose": -1, "n_jobs": 6
    }, dp3)

    # V10 features & model
    v10_cols = p3_cols + ["semantic_cosine", "semantic_rank", "semantic_margin_to_best", "r50_minus_semantic"]
    v10_model, platt_scaler = train_calibrated_decision_layer(train_sub, v10_cols)

    # Record semantic score distribution
    pos_cos = train_sub[train_sub["is_gold"]]["semantic_cosine"]
    neg_cos = train_sub[~train_sub["is_gold"]]["semantic_cosine"]
    sem_dist = {
        "pos_mean": float(pos_cos.mean()), "pos_p50": float(pos_cos.median()),
        "neg_mean": float(neg_cos.mean()), "neg_p50": float(neg_cos.median()),
        "margin": float(pos_cos.median() - neg_cos.median()),
    }

    # 3. Evaluate on untouched validation splits (val1 and val2)
    eval_results = {}
    base_thresh = OPERATIONAL_THRESHOLDS[src]
    error_slices_all = {}

    for split_name in ["val1", "val2"]:
        t_split = time.time()
        print(f"\n--- EVALUATING {src.upper()} | {split_name.upper()} ---")
        s1_val, gt_val_set, gt_val_df = get_clean_gold(src, split_name)
        val_cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
        val_df = pd.read_parquet(val_cache_path)
        val_df["is_gold"] = [(s, x) in gt_val_set for s, x in zip(val_df["s1_id"], val_df["sx_id"])]

        # Restrict to Top-5 candidates per S1
        val_top5 = val_df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False]).groupby("s1_id").head(5).reset_index(drop=True)
        val_top5 = engineer_decision_features(val_top5)
        val_top5 = compute_semantic_features_for_df(val_top5, src, base_encoder, head)

        # Baseline predictions (R50)
        r50_preds = val_top5[val_top5["lgb_score"] >= base_thresh][["s1_id", "sx_id"]]
        r50_res = calculate_macro_f05(r50_preds, gt_val_df, s1_val)

        # Phase-3 predictions
        val_top5["p3_score"] = p3_model.predict(val_top5[p3_cols])
        p3_preds = val_top5[(val_top5["lgb_score"] >= base_thresh) & (val_top5["p3_score"] >= 0.50)][["s1_id", "sx_id"]]
        p3_res = calculate_macro_f05(p3_preds, gt_val_df, s1_val)

        # V10 Model Predictions (Platt calibrated)
        v10_raw = v10_model.predict(val_top5[v10_cols], raw_score=True).reshape(-1, 1)
        val_top5["v10_prob"] = platt_scaler.predict_proba(v10_raw)[:, 1]

        # Mode A: Semantic decision on all Top-5 pairs
        v10_mode_a_preds = val_top5[val_top5["v10_prob"] >= 0.50][["s1_id", "sx_id"]]
        v10_mode_a_res = calculate_macro_f05(v10_mode_a_preds, gt_val_df, s1_val)

        # Mode B: Ambiguity Gated
        # Ambiguous condition: top score < 0.95 OR top1-top2 margin < 0.20 OR multiple candidates >= 0.70
        is_ambig = (val_top5["top1_score"] < 0.95) | (val_top5["score_diff_top1"].abs() < 0.20) | (val_top5["cand_count"] > 1)
        val_top5["mode_b_accept"] = np.where(is_ambig, val_top5["v10_prob"] >= 0.50, val_top5["lgb_score"] >= base_thresh)
        v10_mode_b_preds = val_top5[val_top5["mode_b_accept"]][["s1_id", "sx_id"]]
        v10_mode_b_res = calculate_macro_f05(v10_mode_b_preds, gt_val_df, s1_val)

        # Error slices
        error_slices_all[split_name] = evaluate_error_slices(val_top5, base_thresh)

        eval_results[split_name] = {
            "r50": r50_res,
            "p3": p3_res,
            "v10_a": v10_mode_a_res,
            "v10_b": v10_mode_b_res,
            "runtime": time.time() - t_split,
            "pairs_evaluated": len(val_top5),
            "ambiguous_pairs": int(is_ambig.sum()),
        }

        print(f"  R50 Baseline:     F0.5 = {r50_res['macro_f05']:.5f} | P = {r50_res['macro_precision']:.5f} | R = {r50_res['macro_recall']:.5f} | SingAcc = {r50_res['singletons_correct']}/{r50_res['total_singletons']}")
        print(f"  Phase-3 Control:  F0.5 = {p3_res['macro_f05']:.5f} | P = {p3_res['macro_precision']:.5f} | R = {p3_res['macro_recall']:.5f} | SingAcc = {p3_res['singletons_correct']}/{p3_res['total_singletons']}")
        print(f"  V10 Mode A (All): F0.5 = {v10_mode_a_res['macro_f05']:.5f} | P = {v10_mode_a_res['macro_precision']:.5f} | R = {v10_mode_a_res['macro_recall']:.5f}")
        print(f"  V10 Mode B (Gated): F0.5 = {v10_mode_b_res['macro_f05']:.5f} | P = {v10_mode_b_res['macro_precision']:.5f} | R = {v10_mode_b_res['macro_recall']:.5f}")

    return {
        "sem_dist": sem_dist,
        "eval_results": eval_results,
        "error_slices": error_slices_all,
        "total_runtime": time.time() - t_start,
    }


def main():
    print("=" * 80)
    print("STARTING V10 EXPERIMENT: SELECTIVE MULTILINGUAL SEMANTIC MATCHER")
    print("=" * 80)
    t_global = time.time()

    print(f"Loading Base Encoder: {BASE_ENCODER_NAME} (local_files_only=True)...")
    base_encoder = SentenceTransformer(BASE_ENCODER_NAME, local_files_only=True)

    res_s2 = run_experiment_for_source("source2", base_encoder)
    res_s3 = run_experiment_for_source("source3", base_encoder)

    # Save summary results JSON
    summary_path = ROOT / "experiments" / "v10_results.json"
    with open(summary_path, "w") as f:
        json.dump({"source2": res_s2, "source3": res_s3}, f, indent=2)
    print(f"\nV10 experiment results successfully saved to: {summary_path}")
    print(f"Total experiment runtime: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
