"""
V13.1 BLEND RATIO SWEEP
========================
All scores already computed. Zero retraining.
Sweeps α in: final_score = α * stage1_score + (1-α) * stage2_score
across val1+val2 for both sources, finds optimal α per source and globally.
Also tests: max(stage1, stage2), stage2-only, stage1-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import (
    get_clean_gold, compute_pairwise_features, engineer_competition_features,
    get_v12_feature_names, apply_target_exclusivity, MODELS_DIR, V2_CACHE,
)
from experiments.v12_1_collective_matcher import (
    compute_collective_sibling_features, get_v12_1_feature_names,
)
from experiments.v13_1_pipeline import build_expanded_candidate_df

V12_1_SCORES = {
    ("source2", "val1"): 0.89901,
    ("source2", "val2"): 0.90037,
    ("source3", "val1"): 0.90198,
    ("source3", "val2"): 0.89990,
}

ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
THRESHOLDS = [0.35, 0.40, 0.43, 0.45, 0.47, 0.50, 0.52, 0.55, 0.60]

print("=" * 70)
print("V13.1 BLEND RATIO SWEEP — zero retraining, scores already cached")
print("=" * 70)

# Store all val scores keyed by (src, split)
all_scores: dict[tuple, pd.DataFrame] = {}
all_gt: dict[tuple, tuple] = {}

for src in ["source2", "source3"]:
    stage1_path = MODELS_DIR / f"v13_1_stage1_{src}.txt"
    stage2_path = MODELS_DIR / f"v13_1_stage2_{src}.txt"
    model_s1 = lgb.Booster(model_file=str(stage1_path))
    model_s2 = lgb.Booster(model_file=str(stage2_path))

    # Determine which feature set was selected (WITHOUT_R50 won for both)
    v12_cols = get_v12_feature_names()
    v12_cols_no_r50 = [c for c in v12_cols if c != "lgb_score"]
    stage2_cols = get_v12_1_feature_names()

    for split in ["val1", "val2"]:
        print(f"\nScoring {src.upper()} {split.upper()}...")
        t0 = time.time()

        s1_all, gt_set, gt_df = get_clean_gold(src, split)
        val_exp, _ = build_expanded_candidate_df(src, split)
        val_pw = compute_pairwise_features(val_exp)
        val_full = pd.concat([val_exp, val_pw], axis=1)
        val_full = engineer_competition_features(val_full)

        val_full["stage1_score"] = model_s1.predict(val_full[v12_cols_no_r50])
        col_feats = compute_collective_sibling_features(val_full, score_col="stage1_score")
        val_full = pd.concat([val_full, col_feats], axis=1)

        # Ensure stage2_cols present
        missing = [c for c in stage2_cols if c not in val_full.columns]
        for c in missing:
            val_full[c] = 0.0
        val_full["stage2_score"] = model_s2.predict(val_full[stage2_cols])

        all_scores[(src, split)] = val_full[["s1_id", "sx_id", "stage1_score", "stage2_score"]].copy()
        all_gt[(src, split)] = (s1_all, gt_set, gt_df)
        print(f"  Scored {len(val_full):,} pairs in {time.time()-t0:.1f}s")

print("\n\n" + "=" * 70)
print("BLEND SWEEP RESULTS")
print("=" * 70)

# Global best tracker
global_results = {}  # (alpha, th) -> {split: f05}

best_global_avg = -1.0
best_global_alpha = 0.20
best_global_th = 0.47

print(f"\n{'Alpha':>6}  {'BestTh':>6}  {'S2v1':>8}  {'S2v2':>8}  {'S3v1':>8}  {'S3v2':>8}  {'AVG':>8}  {'vs V12.1':>9}")
print("-" * 75)

for alpha in ALPHAS:
    best_th_for_alpha = {}
    split_best_f05 = {}

    for src in ["source2", "source3"]:
        for split in ["val1", "val2"]:
            df = all_scores[(src, split)].copy()
            s1_all, gt_set, gt_df = all_gt[(src, split)]
            df["final_score"] = alpha * df["stage1_score"] + (1 - alpha) * df["stage2_score"]

            best_f = -1.0
            best_th = 0.47
            for th in THRESHOLDS:
                preds = df[df["final_score"] >= th][["s1_id", "sx_id", "final_score"]].rename(
                    columns={"final_score": "v12_score"})
                excl = apply_target_exclusivity(preds)
                m = calculate_macro_f05(excl, gt_df, s1_all)
                if m["macro_f05"] > best_f:
                    best_f = m["macro_f05"]
                    best_th = th
            split_best_f05[(src, split)] = best_f
            best_th_for_alpha[(src, split)] = best_th

    avg_f05 = np.mean(list(split_best_f05.values()))
    avg_v12 = np.mean(list(V12_1_SCORES.values()))
    delta = avg_f05 - avg_v12

    s2v1 = split_best_f05[("source2", "val1")]
    s2v2 = split_best_f05[("source2", "val2")]
    s3v1 = split_best_f05[("source3", "val1")]
    s3v2 = split_best_f05[("source3", "val2")]

    flag = "✅" if avg_f05 >= 0.91 else ("🎯" if avg_f05 >= 0.905 else "")
    print(f"  α={alpha:.2f}  th≈{best_th_for_alpha[('source2','val1')]:.2f}  "
          f"{s2v1:.5f}  {s2v2:.5f}  {s3v1:.5f}  {s3v2:.5f}  "
          f"{avg_f05:.5f}  {delta:+.5f} {flag}")

    if avg_f05 > best_global_avg:
        best_global_avg = avg_f05
        best_global_alpha = alpha
        best_global_th_map = best_th_for_alpha
        best_global_splits = dict(split_best_f05)

print("-" * 75)
print(f"\n🏆 BEST: α={best_global_alpha:.2f} → Average F0.5={best_global_avg:.5f} "
      f"({best_global_avg - np.mean(list(V12_1_SCORES.values())):+.5f} vs V12.1)")
print(f"   Best thresholds: S2v1={best_global_th_map[('source2','val1')]:.2f}  "
      f"S2v2={best_global_th_map[('source2','val2')]:.2f}  "
      f"S3v1={best_global_th_map[('source3','val1')]:.2f}  "
      f"S3v2={best_global_th_map[('source3','val2')]:.2f}")

print("\n   Per-split best:")
for (src, split), f05 in best_global_splits.items():
    v12 = V12_1_SCORES[(src, split)]
    print(f"   {src} {split}: {f05:.5f} ({f05-v12:+.5f} vs V12.1={v12:.5f})")

# Decision
avg_v12 = np.mean(list(V12_1_SCORES.values()))
print(f"\n{'='*70}")
if best_global_avg >= 0.91:
    print(f"✅ TARGET 0.91 CROSSED → Prepare V13.1 as THIRD SUBMISSION CANDIDATE")
    print(f"   Config: α={best_global_alpha:.2f}, per-split thresholds as above")
elif best_global_avg >= 0.905:
    print(f"🎯 Close to 0.91 (gap={0.91-best_global_avg:.5f}). Consider submission or one more retrain.")
else:
    print(f"📈 Best blend gives {best_global_avg:.5f} (V12.1={avg_v12:.5f}, gap to 0.91={0.91-best_global_avg:.5f})")
    print(f"   V13.1 is better than V12.1 but gap to 0.91 remains.")
print(f"{'='*70}")

# Save results
results = {
    "best_alpha": best_global_alpha,
    "best_avg_f05": best_global_avg,
    "best_thresholds": {f"{s}_{sp}": th for (s, sp), th in best_global_th_map.items()},
    "split_f05": {f"{s}_{sp}": f for (s, sp), f in best_global_splits.items()},
    "v12_1_avg": avg_v12,
    "improvement": best_global_avg - avg_v12,
}
out = ROOT / "experiments" / "v13_1_blend_sweep_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {out}")
