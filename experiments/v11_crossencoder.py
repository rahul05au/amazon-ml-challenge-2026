"""V11: GPU Multilingual Cross-Encoder Matcher & Calibrated Fusion Layer.

Architecture:
1. Frozen Candidate Generation: 5-pass DuckDB + TF-IDF (k=20, tau_2=0.53, tau_3=0.51)
2. Sequence-pair Joint Cross-Encoder: FacebookAI/xlm-roberta-base with BCEWithLogitsLoss
3. Multi-Positive + Hard-Negative Mining (up to 5 hard negatives per S1 from R50 pool)
4. Calibrated Decision Fusion Layer (LightGBM): R50 + Phase-3 Context + CrossEncoder Probability
5. Exact Canonical Evaluator: Macro F0.5 per S1 entity with strict source filtering
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parent.parent
INTERMEDIATE = ROOT / "intermediate" / "train"
V2_CACHE = ROOT / "intermediate" / "v2_cache"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

DEFAULT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
OPERATIONAL_THRESHOLDS = {"source2": 0.70, "source3": 0.65}


def serialize_entity(name: Any, addr: Any, country: Any) -> str:
    """Format single entity metadata into clean text representation."""
    n = str(name).strip() if pd.notna(name) and str(name).strip() else "MISSING"
    a = str(addr).strip() if pd.notna(addr) and str(addr).strip() else "MISSING"
    c = str(country).strip() if pd.notna(country) and str(country).strip() else "MISSING"
    return f"NAME: {n} ADDRESS: {a} COUNTRY: {c}"


def format_cross_pair(s1_name: Any, s1_addr: Any, s1_country: Any,
                      sx_name: Any, sx_addr: Any, sx_country: Any) -> str:
    """Structured sequence-pair serialization per Section 7."""
    ent_a = serialize_entity(s1_name, s1_addr, s1_country)
    ent_b = serialize_entity(sx_name, sx_addr, sx_country)
    return f"[ENTITY_A] {ent_a} [ENTITY_B] {ent_b}"


class CrossEncoderPairDataset(Dataset):
    """PyTorch Dataset for sequence-pair classification."""
    def __init__(self, texts: list[str], labels: list[float] | None = None):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = float(self.labels[idx])
        return item


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int = 192):
    """Dynamic padding batch collator."""
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            encoded["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        return encoded
    return collate_fn


def calculate_macro_f05(
    pred_pairs: pd.DataFrame,
    true_pairs: pd.DataFrame,
    all_s1_ids: set[str],
    beta: float = 0.5,
) -> dict[str, float]:
    """Exact canonical macro-averaged F0.5 per S1 entity."""
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


def mine_crossencoder_training_data(
    train_cache_df: pd.DataFrame,
    gt_set: set[tuple[str, str]],
    h_neg: int = 5,
) -> pd.DataFrame:
    """Mine confirmed positives and up to H=5 hard negatives per S1 entity."""
    train_cache_df["is_gold"] = [(s, x) in gt_set for s, x in zip(train_cache_df["s1_id"], train_cache_df["sx_id"])]
    pos_df = train_cache_df[train_cache_df["is_gold"]].copy()
    neg_df = train_cache_df[~train_cache_df["is_gold"]].copy()

    # Mine top H hard negatives ranked by R50 score per S1
    neg_sorted = neg_df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False])
    hard_negs = neg_sorted.groupby("s1_id").head(h_neg).copy()

    combined = pd.concat([pos_df, hard_negs], ignore_index=True)
    combined["label"] = combined["is_gold"].astype(np.float32)

    # Format sequence-pair texts
    combined["pair_text"] = [
        format_cross_pair(n1, a1, "MISSING", n2, a2, "MISSING")
        for n1, a1, n2, a2 in zip(combined["s1_name"], combined["s1_addr"], combined["sx_name"], combined["sx_addr"])
    ]

    # Shuffle deterministically
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return combined


def train_cross_encoder(
    train_data: pd.DataFrame,
    model_name: str,
    device: torch.device,
    args: argparse.Namespace,
    save_dir: Path,
) -> tuple[nn.Module, AutoTokenizer]:
    """Fine-tune XLM-RoBERTa cross-encoder with BCE loss and mixed precision."""
    print(f"\nInitializing Cross-Encoder: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # Internal holdout (10% of training S1s for early validation)
    unique_s1 = train_data["s1_id"].unique()
    np.random.seed(42)
    val_s1 = set(np.random.choice(unique_s1, size=int(0.10 * len(unique_s1)), replace=False))

    tr_df = train_data[~train_data["s1_id"].isin(val_s1)].reset_index(drop=True)
    val_df = train_data[train_data["s1_id"].isin(val_s1)].reset_index(drop=True)

    print(f"Training pairs: {len(tr_df):,} (Pos: {int(tr_df['label'].sum()):,}, Neg: {len(tr_df)-int(tr_df['label'].sum()):,})")
    print(f"Internal val pairs: {len(val_df):,} (Pos: {int(val_df['label'].sum()):,}, Neg: {len(val_df)-int(val_df['label'].sum()):,})")

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        CrossEncoderPairDataset(tr_df["pair_text"].tolist(), tr_df["label"].tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        CrossEncoderPairDataset(val_df["pair_text"].tolist(), val_df["label"].tolist()),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion = nn.BCEWithLogitsLoss()
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not args.bf16)
    amp_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\nStarting Cross-Encoder Training ({args.epochs} epochs, {total_steps:,} steps, amp={use_amp}, dtype={amp_dtype})...")
    t0 = time.time()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss = criterion(logits, labels)

            if use_amp and not args.bf16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()

            if (step + 1) % 200 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - t0
                print(f"  Epoch {epoch}/{args.epochs} | Step {step+1:5d}/{len(train_loader)} | Loss: {train_loss/(step+1):.5f} | Time: {elapsed:.1f}s", flush=True)

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                    v_loss = criterion(logits, labels)

                val_loss += v_loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        auc = roc_auc_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.5
        pr_auc = average_precision_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.5
        print(f"--> Epoch {epoch} Validation | Loss: {avg_val_loss:.5f} | AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}", flush=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved best checkpoint to {save_dir}")

    print(f"Cross-encoder training completed in {time.time() - t0:.1f}s")
    return model, tokenizer


def predict_cross_encoder_probabilities(
    df: pd.DataFrame,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    """Run batched GPU inference to predict P(match | entity_a, entity_b)."""
    model.eval()
    pair_texts = [
        format_cross_pair(n1, a1, "MISSING", n2, a2, "MISSING")
        for n1, a1, n2, a2 in zip(df["s1_name"], df["s1_addr"], df["sx_name"], df["sx_addr"])
    ]

    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)
    loader = DataLoader(
        CrossEncoderPairDataset(pair_texts),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_bf16_supported()) else torch.float16
    probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy().tolist())

    return np.array(probs, dtype=np.float32)


def engineer_fusion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute combined R50, Phase-3, and Cross-Encoder contextual features."""
    df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False], inplace=True)
    df["cand_rank"] = df.groupby("s1_id").cumcount() + 1
    df["cand_count"] = df.groupby("s1_id")["lgb_score"].transform("count")

    top1_scores = df.groupby("s1_id")["lgb_score"].transform("first")
    df["top1_score"] = top1_scores
    df["score_diff_top1"] = df["lgb_score"] - top1_scores
    df["score_ratio_top1"] = df["lgb_score"] / (top1_scores + 1e-6)

    mean_scores = df.groupby("s1_id")["lgb_score"].transform("mean")
    df["score_diff_mean"] = df["lgb_score"] - mean_scores

    if "cross_prob" in df.columns:
        # CrossEncoder candidate-set context
        top1_cross = df.groupby("s1_id")["cross_prob"].transform("max")
        df["cross_margin_top1"] = top1_cross - df["cross_prob"]
        df["cross_rank_within_candidates"] = df.groupby("s1_id")["cross_prob"].rank(ascending=False, method="min")

    return df


def train_fusion_model(
    train_df: pd.DataFrame,
    fusion_features: list[str],
) -> lgb.Booster:
    """Train calibrated LightGBM decision layer on joint features."""
    dtrain = lgb.Dataset(train_df[fusion_features], label=train_df["is_gold"])
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
    return model


def evaluate_error_slices(df: pd.DataFrame, threshold: float) -> dict[str, dict[str, float]]:
    """Evaluate precision on hard error categories."""
    domain_re = re.compile(r"\b(com|org|net|io|in|co|biz|info)\b")
    non_ascii_re = re.compile(r"[^\x00-\x7F]")

    slices = {
        "same_near_identical": [],
        "cross_script": [],
        "short_common": [],
        "other": [],
    }

    for r in df.itertuples():
        n1 = str(r.s1_name).lower() if hasattr(r, 's1_name') and pd.notna(r.s1_name) else ""
        n2 = str(r.sx_name).lower() if hasattr(r, 'sx_name') and pd.notna(r.sx_name) else ""

        is_cross = bool(non_ascii_re.search(n1)) != bool(non_ascii_re.search(n2))
        has_dom = bool(domain_re.search(n1)) or bool(domain_re.search(n2))
        is_short = len(n1) <= 5 or len(n2) <= 5

        cat = "other"
        if is_cross:
            cat = "cross_script"
        elif is_short:
            cat = "short_common"
        elif n1 == n2 or (len(n1) > 5 and n1 in n2):
            cat = "same_near_identical"

        slices[cat].append(r)

    results = {}
    for cat_name, rows in slices.items():
        if not rows:
            results[cat_name] = {"r50_p": 0.0, "p3_p": 0.0, "cross_p": 0.0, "fusion_p": 0.0, "count": 0}
            continue
        c_df = pd.DataFrame(rows)
        r50_acc = c_df[c_df["lgb_score"] >= threshold]
        r50_p = r50_acc["is_gold"].mean() if len(r50_acc) > 0 else 1.0

        p3_acc = c_df[c_df["p3_score"] >= 0.50]
        p3_p = p3_acc["is_gold"].mean() if len(p3_acc) > 0 else 1.0

        cross_acc = c_df[c_df["cross_prob"] >= 0.50]
        cross_p = cross_acc["is_gold"].mean() if len(cross_acc) > 0 else 1.0

        fusion_acc = c_df[c_df["fusion_prob"] >= 0.50]
        fusion_p = fusion_acc["is_gold"].mean() if len(fusion_acc) > 0 else 1.0

        results[cat_name] = {
            "r50_p": float(r50_p),
            "p3_p": float(p3_p),
            "cross_p": float(cross_p),
            "fusion_p": float(fusion_p),
            "count": len(c_df),
        }
    return results


def run_v11_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute complete end-to-end V11 experiment."""
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("=" * 80)
    print(f"V11: GPU MULTILINGUAL CROSS-ENCODER EXPERIMENT ({device.type.upper()})")
    if device.type == "cuda":
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
    print("=" * 80)

    targets = ["source2", "source3"] if args.target == "both" else [args.target]
    all_results = {}

    for src in targets:
        print(f"\n{'#'*80}")
        print(f"PROCESSING TARGET: {src.upper()}")
        print(f"{'#'*80}")

        # 1. Load train data and mine hard negatives
        s1_tr, gt_tr_set, gt_tr_df = get_clean_gold(src, "train_gate")
        tr_cache = pd.read_parquet(V2_CACHE / f"v2_{src}_train_gate.parquet")
        train_pairs = mine_crossencoder_training_data(tr_cache, gt_tr_set, h_neg=5)

        # 2. Train Cross-Encoder
        save_model_dir = MODELS_DIR / f"v11_crossencoder_{src}"
        model, tokenizer = train_cross_encoder(train_pairs, args.model_name, device, args, save_model_dir)

        # 3. Small Slice Diagnostic on S2 val1 (Section 14)
        if src == "source2":
            print(f"\n--- RUNNING SMALL-SLICE DIAGNOSTIC ON S2 VAL1 ---")
            s1_v1, gt_v1_set, _ = get_clean_gold(src, "val1")
            val1_raw = pd.read_parquet(V2_CACHE / f"v2_{src}_val1.parquet")
            val1_raw["is_gold"] = [(s, x) in gt_v1_set for s, x in zip(val1_raw["s1_id"], val1_raw["sx_id"])]
            slice_df = val1_raw.head(1000).copy()
            slice_probs = predict_cross_encoder_probabilities(slice_df, model, tokenizer, device, args)
            slice_df["cross_prob"] = slice_probs

            slice_auc = roc_auc_score(slice_df["is_gold"], slice_probs) if len(set(slice_df["is_gold"])) > 1 else 0.5
            slice_pr_auc = average_precision_score(slice_df["is_gold"], slice_probs) if len(set(slice_df["is_gold"])) > 1 else 0.5
            pos_med = float(np.median(slice_probs[slice_df["is_gold"]])) if slice_df["is_gold"].sum() > 0 else 0.0
            neg_med = float(np.median(slice_probs[~slice_df["is_gold"]])) if (~slice_df["is_gold"]).sum() > 0 else 0.0

            print(f"  Small Slice AUC: {slice_auc:.4f} | PR-AUC: {slice_pr_auc:.4f}")
            print(f"  Pos Median: {pos_med:.4f} | Neg Median: {neg_med:.4f} | Margin: {pos_med - neg_med:.4f}")

            if args.small_slice_only:
                print("Small slice diagnostic completed. Stopping as requested.")
                return {"slice_auc": slice_auc, "slice_pr_auc": slice_pr_auc}

        # 4. Train Calibrated Decision Layer on training gate Top-5
        print("\nTraining Calibrated Fusion Model...")
        tr_top5 = tr_cache.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False]).groupby("s1_id").head(5).copy()
        tr_top5["is_gold"] = [(s, x) in gt_tr_set for s, x in zip(tr_top5["s1_id"], tr_top5["sx_id"])]
        tr_top5["cross_prob"] = predict_cross_encoder_probabilities(tr_top5, model, tokenizer, device, args)
        tr_top5 = engineer_fusion_features(tr_top5)

        # Baseline Phase-3 model
        p3_cols = ["lgb_score", "cand_rank", "cand_count", "top1_score", "score_diff_top1", "score_ratio_top1", "score_diff_mean"]
        dp3 = lgb.Dataset(tr_top5[p3_cols], label=tr_top5["is_gold"])
        p3_model = lgb.train({
            "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
            "n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15, "scale_pos_weight": 3.0,
            "random_state": 42, "verbose": -1, "n_jobs": 6
        }, dp3)

        # Fusion model
        fusion_cols = p3_cols + ["cross_prob", "cross_margin_top1", "cross_rank_within_candidates"]
        fusion_model = train_fusion_model(tr_top5, fusion_cols)
        fusion_model_path = MODELS_DIR / f"v11_fusion_{src}.txt"
        fusion_model.save_model(str(fusion_model_path))
        print(f"Saved Fusion Model to: {fusion_model_path}")

        # 5. Validation across val1 and val2
        src_val_results = {}
        base_thresh = OPERATIONAL_THRESHOLDS[src]
        for split_name in ["val1", "val2"]:
            print(f"\n--- EVALUATING: {src.upper()} | {split_name.upper()} ---")
            s1_val, gt_val_set, gt_val_df = get_clean_gold(src, split_name)
            val_df = pd.read_parquet(V2_CACHE / f"v2_{src}_{split_name}.parquet")
            val_df["is_gold"] = [(s, x) in gt_val_set for s, x in zip(val_df["s1_id"], val_df["sx_id"])]

            val_top5 = val_df.sort_values(by=["s1_id", "lgb_score"], ascending=[True, False]).groupby("s1_id").head(5).copy()
            val_top5["p3_score"] = p3_model.predict(val_top5[p3_cols])
            val_top5["cross_prob"] = predict_cross_encoder_probabilities(val_top5, model, tokenizer, device, args)
            val_top5 = engineer_fusion_features(val_top5)
            val_top5["fusion_prob"] = fusion_model.predict(val_top5[fusion_cols])

            # R50 evaluation
            r50_preds = val_top5[val_top5["lgb_score"] >= base_thresh][["s1_id", "sx_id"]]
            r50_res = calculate_macro_f05(r50_preds, gt_val_df, s1_val)

            # Phase-3 evaluation
            p3_preds = val_top5[(val_top5["lgb_score"] >= base_thresh) & (val_top5["p3_score"] >= 0.50)][["s1_id", "sx_id"]]
            p3_res = calculate_macro_f05(p3_preds, gt_val_df, s1_val)

            # Cross-Encoder only evaluation (raw probability threshold)
            cross_preds = val_top5[val_top5["cross_prob"] >= 0.50][["s1_id", "sx_id"]]
            cross_res = calculate_macro_f05(cross_preds, gt_val_df, s1_val)

            # V11 Fusion evaluation (threshold 0.50)
            fusion_preds = val_top5[val_top5["fusion_prob"] >= 0.50][["s1_id", "sx_id"]]
            fusion_res = calculate_macro_f05(fusion_preds, gt_val_df, s1_val)

            # Secondary true-match recall
            multi_s1 = set(gt_val_df.groupby("s1_id").filter(lambda g: len(g) >= 2)["s1_id"])
            sec_gold = val_top5[val_top5["s1_id"].isin(multi_s1) & val_top5["is_gold"] & (val_top5["cand_rank"] > 1)]
            tot_sec = len(sec_gold)

            r50_sec_rec = len(sec_gold[sec_gold["lgb_score"] >= base_thresh]) / tot_sec if tot_sec else 0.0
            p3_sec_rec = len(sec_gold[(sec_gold["lgb_score"] >= base_thresh) & (sec_gold["p3_score"] >= 0.50)]) / tot_sec if tot_sec else 0.0
            fusion_sec_rec = len(sec_gold[sec_gold["fusion_prob"] >= 0.50]) / tot_sec if tot_sec else 0.0

            slices = evaluate_error_slices(val_top5, base_thresh)

            src_val_results[split_name] = {
                "r50": r50_res,
                "p3": p3_res,
                "cross_only": cross_res,
                "fusion": fusion_res,
                "secondary_recall": {
                    "r50": r50_sec_rec,
                    "p3": p3_sec_rec,
                    "fusion": fusion_sec_rec,
                    "total_secondary": tot_sec,
                },
                "slices": slices,
            }

            print(f"  R50 Baseline:     F0.5 = {r50_res['macro_f05']:.5f} | P = {r50_res['macro_precision']:.5f} | R = {r50_res['macro_recall']:.5f}")
            print(f"  Phase-3 Control:  F0.5 = {p3_res['macro_f05']:.5f} | P = {p3_res['macro_precision']:.5f} | R = {p3_res['macro_recall']:.5f}")
            print(f"  CrossEncoder-only:F0.5 = {cross_res['macro_f05']:.5f} | P = {cross_res['macro_precision']:.5f} | R = {cross_res['macro_recall']:.5f}")
            print(f"  V11 Fusion:       F0.5 = {fusion_res['macro_f05']:.5f} | P = {fusion_res['macro_precision']:.5f} | R = {fusion_res['macro_recall']:.5f}")
            print(f"  Secondary Recall: R50={r50_sec_rec:.4f} | Phase3={p3_sec_rec:.4f} | Fusion={fusion_sec_rec:.4f}")

        all_results[src] = src_val_results

    # Save summary json
    res_path = ROOT / "experiments" / "v11_results.json"
    with open(res_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nV11 Experiment results saved to: {res_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="V11: GPU Multilingual Cross-Encoder Matcher")
    parser.add_argument("--target", type=str, default="both", choices=["source2", "source3", "both"])
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--small-slice-only", action="store_true", default=False)
    args = parser.parse_args()

    run_v11_pipeline(args)


if __name__ == "__main__":
    main()
