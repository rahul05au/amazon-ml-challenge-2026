import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import _process_chunk_worker, FEATURE_NAMES
from experiments.v12_assignment_lgb import get_clean_gold, s1_norm_path, sx_norm_path, V2_CACHE
from experiments.v13_candidate_audit_and_eval import run_v13_retrieval_pairs

def r50_rescore_diagnostic(src: str, split_name: str):
    print(f"\n{'='*70}\nV13.1 STEP 3: FROZEN R50 RESCORE DIAGNOSTIC FOR {src.upper()} | {split_name.upper()}\n{'='*70}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    val_cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
    v2_df = pd.read_parquet(val_cache_path, columns=["s1_id", "sx_id"])
    v2_set = set(zip(v2_df["s1_id"], v2_df["sx_id"]))

    new_retrieved = run_v13_retrieval_pairs(src, split_name)
    recovered_gold = (new_retrieved - v2_set) & gt_set
    print(f"Total recovered gold pairs: {len(recovered_gold):,}")

    # Fetch text columns from normalized files
    con = duckdb.connect()
    recov_df_input = pd.DataFrame(list(recovered_gold), columns=["s1_id", "sx_id"])
    con.register("recov_tbl", recov_df_input)

    df_text = con.execute(f"""
        SELECT r.s1_id, r.sx_id,
               s1.name_clean AS s1_name, sx.name_clean AS sx_name,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
               s1.addr_clean AS s1_addr, sx.addr_clean AS sx_addr,
               s1.postal_code AS s1_post, sx.postal_code AS sx_post
        FROM recov_tbl r
        JOIN read_parquet('{s1_norm_path()}') s1 ON r.s1_id = s1.entity_id
        JOIN read_parquet('{sx_norm_path(src)}') sx ON r.sx_id = sx.entity_id
    """).df()
    con.close()

    # Compute exact 15 R50 features
    r50_feats = _process_chunk_worker(df_text)

    # Run frozen R50 model
    r50_model = lgb.Booster(model_file=str(ROOT / f"models/lgb_{src}.txt"))
    scores = r50_model.predict(r50_feats)
    df_text["r50_score"] = scores

    print(f"\nFrozen R50 Score Statistics on {len(scores)} Recovered Gold Pairs:")
    print(f"  Mean Score:   {np.mean(scores):.4f}")
    print(f"  Median Score: {np.median(scores):.4f}")
    print(f"  P90 Score:    {np.percentile(scores, 90):.4f}")
    print(f"  P95 Score:    {np.percentile(scores, 95):.4f}")
    print(f"  Min Score:    {np.min(scores):.4f}")
    print(f"  Max Score:    {np.max(scores):.4f}")
    print(f"  Fraction > 0.1: {(scores > 0.1).mean()*100:.1f}% ({np.sum(scores > 0.1)}/{len(scores)})")
    print(f"  Fraction > 0.3: {(scores > 0.3).mean()*100:.1f}% ({np.sum(scores > 0.3)}/{len(scores)})")
    print(f"  Fraction > 0.5: {(scores > 0.5).mean()*100:.1f}% ({np.sum(scores > 0.5)}/{len(scores)})")
    print(f"  Fraction > 0.7: {(scores > 0.7).mean()*100:.1f}% ({np.sum(scores > 0.7)}/{len(scores)})")

    # Sample of scores
    print("\nSample of Recovered Gold Pairs with True Frozen R50 Scores:")
    for idx, (_, r) in enumerate(df_text.head(10).iterrows()):
        print(f"  [{idx+1}] S1: {r['s1_name'][:30]:<30} | SX: {r['sx_name'][:30]:<30} | R50 Score: {r['r50_score']:.4f}")

    return df_text

if __name__ == "__main__":
    r50_rescore_diagnostic("source2", "val1")
