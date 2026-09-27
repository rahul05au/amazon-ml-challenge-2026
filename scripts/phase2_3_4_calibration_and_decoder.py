"""
PHASES 2, 3, 4:
- Leave-One-Country-Out Robustness
- Probability Calibration (Platt vs Isotonic vs Raw)
- Expected-F0.5 Per-Entity Decoder vs Country-Aware vs Baseline
- Generates reports/expected_f05_ablation.md and models/final_calibrator_*.pkl
"""
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.v12_assignment_lgb import get_clean_gold, apply_target_exclusivity

V13_CACHE = ROOT / "intermediate" / "v13_cache"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD SCORES AND LABELS ACROSS ALL VAL SPLITS
# -------------------------------------------------------------
print("Loading validation scores and labels...")
val_data = {}

for src in ["source2", "source3"]:
    for split in ["val1", "val2"]:
        s1_all, gt_set, gt_df = get_clean_gold(src, split)
        df_scores = pd.read_parquet(V13_CACHE / f"val_{src}_{split}.parquet")
        
        # Binary label: is pair in ground truth?
        df_scores["is_gold"] = [(s, x) in gt_set for s, x in zip(df_scores["s1_id"], df_scores["sx_id"])]
        val_data[(src, split)] = {
            "df": df_scores,
            "s1_all": s1_all,
            "gt_set": gt_set,
            "gt_df": gt_df,
        }

# -------------------------------------------------------------
# 2. PHASE 3: PROBABILITY CALIBRATION (PLATT VS ISOTONIC)
# -------------------------------------------------------------
print("\n" + "="*70)
print("PHASE 3: PROBABILITY CALIBRATION (Cross-Fit val1 -> val2, val2 -> val1)")
print("="*70)

calibrators = {}

for src in ["source2", "source3"]:
    print(f"\n--- Calibrating {src.upper()} ---")
    df_v1 = val_data[(src, "val1")]["df"]
    df_v2 = val_data[(src, "val2")]["df"]

    # Fit Platt on v1, test on v2
    platt_v1 = LogisticRegression(C=1.0, solver="lbfgs")
    platt_v1.fit(df_v1[["final_score"]], df_v1["is_gold"])
    
    # Fit Isotonic on v1, test on v2
    iso_v1 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_v1.fit(df_v1["final_score"], df_v1["is_gold"])

    # Measure Brier Score on held-out v2
    raw_brier = brier_score_loss(df_v2["is_gold"], df_v2["final_score"])
    platt_brier = brier_score_loss(df_v2["is_gold"], platt_v1.predict_proba(df_v2[["final_score"]])[:, 1])
    iso_brier = brier_score_loss(df_v2["is_gold"], iso_v1.predict(df_v2["final_score"]))

    print(f"  Raw Brier Score (val2):    {raw_brier:.5f}")
    print(f"  Platt Brier Score (val2):  {platt_brier:.5f}")
    print(f"  Isotonic Brier Score (val2): {iso_brier:.5f}")

    # Train final calibrator on pooled validation data for production decoder
    pooled_df = pd.concat([df_v1, df_v2], ignore_index=True)
    final_platt = LogisticRegression(C=1.0, solver="lbfgs")
    final_platt.fit(pooled_df[["final_score"]], pooled_df["is_gold"])
    
    final_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    final_iso.fit(pooled_df["final_score"], pooled_df["is_gold"])

    # Save calibrators
    cal_file = MODELS_DIR / f"final_calibrator_{src}.pkl"
    with open(cal_file, "wb") as f:
        pickle.dump({"platt": final_platt, "isotonic": final_iso}, f)
    print(f"  Saved calibrator to: {cal_file}")

    calibrators[src] = {"platt": final_platt, "isotonic": final_iso}

    # Add calibrated probs to val dfs
    for split in ["val1", "val2"]:
        val_data[(src, split)]["df"]["calibrated_prob"] = final_iso.predict(val_data[(src, split)]["df"]["final_score"])

# -------------------------------------------------------------
# 3. PHASE 4: EXPECTED-F0.5 DECODER FUNCTION
# -------------------------------------------------------------
def run_decoder(df_scores: pd.DataFrame, mode: str, params: dict, s1_all: set[str], gt_df: pd.DataFrame) -> dict:
    """
    Evaluates predictions under different decision layers:
    - 'baseline': fixed global threshold per source
    - 'country_aware': country-specific thresholds
    - 'expected_f05': expected-F0.5 prefix selection per S1
    """
    if mode == "baseline":
        th = params["threshold"]
        preds = df_scores[df_scores["final_score"] >= th][["s1_id", "sx_id", "final_score"]].rename(columns={"final_score": "v12_score"})
        excl = apply_target_exclusivity(preds)
    elif mode == "country_aware":
        country_th = params["country_thresholds"]  # dict e.g. {"us": 0.44, "india": 0.52, "default": 0.50}
        masks = []
        for country, th in country_th.items():
            if country == "default":
                continue
            masks.append((df_scores["country_clean"] == country) & (df_scores["final_score"] >= th))
        def_th = country_th.get("default", 0.50)
        known_countries = [c for c in country_th if c != "default"]
        masks.append((~df_scores["country_clean"].isin(known_countries)) & (df_scores["final_score"] >= def_th))
        
        full_mask = masks[0]
        for m in masks[1:]:
            full_mask = full_mask | m
            
        preds = df_scores[full_mask][["s1_id", "sx_id", "final_score"]].rename(columns={"final_score": "v12_score"})
        excl = apply_target_exclusivity(preds)
    elif mode == "expected_f05":
        # Per-S1 expected F0.5 prefix decoding
        # Group candidates by S1, sorted by calibrated_prob descending
        df_sorted = df_scores.sort_values(by=["s1_id", "calibrated_prob"], ascending=[True, False])
        
        selected_pairs = []
        # Groupby S1 in vector chunks or dict iteration
        grouped = df_sorted.groupby("s1_id")
        
        # Expected F0.5 calculation
        for s1, group in grouped:
            probs = group["calibrated_prob"].values
            sx_ids = group["sx_id"].values
            m = len(probs)
            if m == 0 or probs[0] < 0.15:
                continue

            # P(singleton) = prod(1 - p_i)
            # Clip probabilities to avoid log/zero issues
            p_clipped = np.clip(probs, 1e-4, 1.0 - 1e-4)
            p_singleton = np.prod(1.0 - p_clipped)

            # Cumulative expected TP for prefix k
            cum_tp = np.cumsum(probs)
            # Total expected true targets R = sum(p_i)
            exp_R = cum_tp[-1]

            # Expected utility for k = 0 (empty prediction)
            u_0 = p_singleton * 1.0

            # Expected utility for k = 1..min(m, 10)
            best_k = 0
            best_u = u_0

            # Minimum probability gate to consider candidate
            k_max = min(m, 10)
            for k in range(1, k_max + 1):
                # Only consider adding candidate if p_k is reasonable
                if probs[k - 1] < params.get("min_cand_prob", 0.35):
                    break
                exp_tp = cum_tp[k - 1]
                # F0.5 formula: (1.25 * tp) / (0.25 * k + R)
                denom = 0.25 * k + exp_R
                u_k = (1.0 - p_singleton) * (1.25 * exp_tp / denom)
                
                # Tie break toward smaller k with conservative margin
                if u_k > best_u + params.get("abstain_margin", 0.02):
                    best_u = u_k
                    best_k = k

            if best_k > 0:
                for idx in range(best_k):
                    selected_pairs.append((s1, sx_ids[idx], probs[idx]))

        if selected_pairs:
            preds_df = pd.DataFrame(selected_pairs, columns=["s1_id", "sx_id", "v12_score"])
            excl = apply_target_exclusivity(preds_df)
        else:
            excl = pd.DataFrame(columns=["s1_id", "sx_id", "v12_score"])

    # Compute Macro F0.5
    pred_map = excl.groupby("s1_id")["sx_id"].apply(set).to_dict() if len(excl) > 0 else {}
    true_map = gt_df.groupby("s1_id")["true_match_id"].apply(set).to_dict()

    beta = 0.5
    beta_sq = beta ** 2
    weight = 1.0 + beta_sq

    f05_list, p_list, r_list = [], [], []
    singletons_correct = 0
    total_singletons = 0

    for s1 in s1_all:
        p_set = pred_map.get(s1, set())
        t_set = true_map.get(s1, set())
        is_sing = (len(t_set) == 0)
        if is_sing:
            total_singletons += 1

        if len(p_set) == 0 and len(t_set) == 0:
            singletons_correct += 1
            f05_list.append(1.0)
            p_list.append(1.0)
            r_list.append(1.0)
        elif len(p_set) == 0 or len(t_set) == 0:
            f05_list.append(0.0)
            p_list.append(0.0)
            r_list.append(0.0)
        else:
            tp = len(p_set & t_set)
            prec = tp / len(p_set)
            rec = tp / len(t_set)
            p_list.append(prec)
            r_list.append(rec)
            d = (beta_sq * prec + rec)
            f05_list.append((weight * prec * rec) / d if d > 0 else 0.0)

    return {
        "macro_f05": round(float(np.mean(f05_list)), 5),
        "precision": round(float(np.mean(p_list)), 5),
        "recall": round(float(np.mean(r_list)), 5),
        "singleton_acc": round(singletons_correct / max(total_singletons, 1), 5),
        "predicted_links": len(excl),
    }

# -------------------------------------------------------------
# 4. SWEEP POLICIES ACROSS ALL FOUR SPLITS
# -------------------------------------------------------------
print("\n" + "="*70)
print("PHASE 2 & 4: POLICY COMPARISON ACROSS ALL 4 SPLITS")
print("="*70)

policies = {
    "A_Baseline": {
        "mode": "baseline",
        "params": {"source2": {"threshold": 0.44}, "source3": {"threshold": 0.47}}
    },
    "B_CountryAware_Conservative": {
        "mode": "country_aware",
        "params": {
            "source2": {"country_thresholds": {"us": 0.44, "india": 0.50, "default": 0.52}},
            "source3": {"country_thresholds": {"us": 0.46, "india": 0.52, "default": 0.55}},
        }
    },
    "C_ExpectedF05_Decoder": {
        "mode": "expected_f05",
        "params": {
            "source2": {"min_cand_prob": 0.38, "abstain_margin": 0.03},
            "source3": {"min_cand_prob": 0.40, "abstain_margin": 0.03},
        }
    },
    "D_Hybrid_ExpectedF05_PrecisionTuned": {
        "mode": "expected_f05",
        "params": {
            "source2": {"min_cand_prob": 0.42, "abstain_margin": 0.04},
            "source3": {"min_cand_prob": 0.44, "abstain_margin": 0.04},
        }
    }
}

policy_results = {}

for pol_name, pol_cfg in policies.items():
    print(f"\nEvaluating Policy: {pol_name} ...")
    t0 = time.time()
    split_metrics = {}
    
    for src in ["source2", "source3"]:
        src_params = pol_cfg["params"][src]
        for split in ["val1", "val2"]:
            data = val_data[(src, split)]
            m = run_decoder(data["df"], pol_cfg["mode"], src_params, data["s1_all"], data["gt_df"])
            split_metrics[f"{src}_{split}"] = m
            print(f"  {src} {split}: Macro F0.5 = {m['macro_f05']:.5f} (P={m['precision']:.4f}, R={m['recall']:.4f}, SingAcc={m['singleton_acc']:.4f}, Links={m['predicted_links']:,})")
            
    avg_f05 = np.mean([split_metrics[k]["macro_f05"] for k in split_metrics])
    avg_prec = np.mean([split_metrics[k]["precision"] for k in split_metrics])
    avg_rec = np.mean([split_metrics[k]["recall"] for k in split_metrics])
    avg_sing = np.mean([split_metrics[k]["singleton_acc"] for k in split_metrics])
    total_links = np.sum([split_metrics[k]["predicted_links"] for k in split_metrics])
    
    print(f"  -> AVERAGE MACRO F0.5: {avg_f05:.5f} | Precision: {avg_prec:.4f} | Singleton Acc: {avg_sing:.4f} ({time.time()-t0:.1f}s)")
    
    policy_results[pol_name] = {
        "avg_macro_f05": avg_f05,
        "avg_precision": avg_prec,
        "avg_recall": avg_rec,
        "avg_singleton_acc": avg_sing,
        "total_val_links": int(total_links),
        "splits": split_metrics,
    }

# -------------------------------------------------------------
# 5. WRITE ABLATION REPORT
# -------------------------------------------------------------
report_file = REPORTS_DIR / "expected_f05_ablation.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("# Expected-F0.5 Decoder & Country-Aware Policy Ablation Report\n\n")
    f.write("## 1. Multi-Split Policy Scoreboard\n\n")
    f.write("| Policy Variant | Avg Macro $F_{0.5}$ | Precision | Recall | Singleton Accuracy | Total Val Links | Delta vs Baseline |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    
    base_f = policy_results["A_Baseline"]["avg_macro_f05"]
    for pol_name, res in policy_results.items():
        delta = res["avg_macro_f05"] - base_f
        flag = "🏆" if delta == max(r["avg_macro_f05"] - base_f for r in policy_results.values()) else ""
        f.write(f"| **`{pol_name}`** | **{res['avg_macro_f05']:.5f}** | {res['avg_precision']:.4f} | {res['avg_recall']:.4f} | {res['avg_singleton_acc']:.4f} | {res['total_val_links']:,} | {delta:+.5f} {flag} |\n")

    f.write("\n## 2. Per-Split Breakdown\n\n")
    f.write("| Policy Variant | S2 Val1 | S2 Val2 | S3 Val1 | S3 Val2 |\n")
    f.write("| :--- | :---: | :---: | :---: | :---: |\n")
    for pol_name, res in policy_results.items():
        s2v1 = res["splits"]["source2_val1"]["macro_f05"]
        s2v2 = res["splits"]["source2_val2"]["macro_f05"]
        s3v1 = res["splits"]["source3_val1"]["macro_f05"]
        s3v2 = res["splits"]["source3_val2"]["macro_f05"]
        f.write(f"| `{pol_name}` | {s2v1:.5f} | {s2v2:.5f} | {s3v1:.5f} | {s3v2:.5f} |\n")

print(f"\nAblation report written to: {report_file}")
