import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import (
    get_clean_gold,
    apply_target_exclusivity,
    compute_pairwise_features,
    engineer_competition_features,
    get_v12_feature_names,
    MODELS_DIR,
)
from experiments.v12_1_collective_matcher import (
    compute_collective_sibling_features,
    get_v12_1_feature_names,
)

def test_blend_and_calibration(src: str):
    print(f"\n{'='*60}\nTESTING BLEND & CALIBRATION FOR {src.upper()}\n{'='*60}")
    stage1_model = lgb.Booster(model_file=str(MODELS_DIR / f"v12_1_stage1_{src}.txt"))
    stage2_model = lgb.Booster(model_file=str(MODELS_DIR / f"v12_1_stage2_{src}.txt"))
    base_cols = get_v12_feature_names()
    stage2_cols = get_v12_1_feature_names()

    for split in ["val1", "val2"]:
        s1_all, gt_set, gt_df = get_clean_gold(src, split)
        val_raw = pd.read_parquet(ROOT / f"intermediate/v2_cache/v2_{src}_{split}.parquet")
        
        full_val = pd.concat([val_raw, compute_pairwise_features(val_raw)], axis=1)
        full_val = engineer_competition_features(full_val)
        full_val["stage1_score"] = stage1_model.predict(full_val[base_cols])
        full_val = pd.concat([full_val, compute_collective_sibling_features(full_val, "stage1_score")], axis=1)
        full_val["stage2_score"] = stage2_model.predict(full_val[stage2_cols])

        # Test blend weights
        for w2 in [1.0, 0.8, 0.7, 0.5]:
            w1 = 1.0 - w2
            full_val["blended_score"] = w1 * full_val["stage1_score"] + w2 * full_val["stage2_score"]
            for th in [0.42, 0.45, 0.47, 0.50]:
                cand = full_val[full_val["blended_score"] >= th][["s1_id", "sx_id", "blended_score"]].copy()
                cand["v12_score"] = cand["blended_score"]
                cand_excl = apply_target_exclusivity(cand, 0.0)
                m = calculate_macro_f05(cand_excl, gt_df, s1_all)
                if m["macro_f05"] >= 0.898:
                    print(f"[{split}] w2={w2:.1f}, th={th:.2f} -> Macro F0.5 = {m['macro_f05']:.5f} (P={m['macro_precision']:.4f}, R={m['macro_recall']:.4f})")

if __name__ == "__main__":
    test_blend_and_calibration("source2")
    test_blend_and_calibration("source3")
