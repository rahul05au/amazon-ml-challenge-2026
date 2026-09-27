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

stage1_model = lgb.Booster(model_file=str(MODELS_DIR / "v12_1_stage1_source2.txt"))
stage2_model = lgb.Booster(model_file=str(MODELS_DIR / "v12_1_stage2_source2.txt"))
feat_cols = get_v12_1_feature_names()
base_cols = get_v12_feature_names()

for split in ["val1", "val2"]:
    s1_all, gt_set, gt_df = get_clean_gold("source2", split)
    val_raw = pd.read_parquet(ROOT / f"intermediate/v2_cache/v2_source2_{split}.parquet")
    
    full_val = pd.concat([val_raw, compute_pairwise_features(val_raw)], axis=1)
    full_val = engineer_competition_features(full_val)
    full_val["stage1_score"] = stage1_model.predict(full_val[base_cols])
    full_val = pd.concat([full_val, compute_collective_sibling_features(full_val, "stage1_score")], axis=1)
    full_val["stage2_score"] = stage2_model.predict(full_val[feat_cols])
    
    print(f"\n{'='*50}\nThreshold Sweep for SOURCE2 {split.upper()}\n{'='*50}")
    for th in [0.35, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60]:
        for mg in [0.0, 0.05]:
            cand = full_val[full_val["stage2_score"] >= th][["s1_id", "sx_id", "stage2_score"]].copy()
            cand["v12_score"] = cand["stage2_score"]
            cand_excl = apply_target_exclusivity(cand, mg)
            m = calculate_macro_f05(cand_excl, gt_df, s1_all)
            print(f"th={th:.2f}, mg={mg:.2f} -> F0.5={m['macro_f05']:.5f} (P={m['macro_precision']:.4f}, R={m['macro_recall']:.4f}, Preds={m['predicted_matches']:,})")
