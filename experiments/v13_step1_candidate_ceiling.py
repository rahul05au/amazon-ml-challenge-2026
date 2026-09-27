"""Step 3: Measure exact gold coverage, missing gold, candidate recall, and oracle Macro F0.5.

Evaluates current V12.1 candidate pool across all four untouched validation splits:
- Source 2 Val1
- Source 2 Val2
- Source 3 Val1
- Source 3 Val2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import get_clean_gold, V2_CACHE


def evaluate_candidate_ceiling(src: str, split_name: str) -> dict[str, float | int]:
    print(f"\nEvaluating Candidate Ceiling for {src.upper()} {split_name.upper()}...")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    total_gold = len(gt_set)

    val_cache_path = V2_CACHE / f"v2_{src}_{split_name}.parquet"
    cand_df = pd.read_parquet(val_cache_path, columns=["s1_id", "sx_id"])
    cand_set = set(zip(cand_df["s1_id"], cand_df["sx_id"]))
    total_cands = len(cand_set)

    covered_gold = cand_set & gt_set
    missing_gold = gt_set - cand_set
    cand_recall = len(covered_gold) / max(total_gold, 1)

    # Oracle candidate pool: take all candidate pairs that are in gt_set
    oracle_preds = pd.DataFrame(list(covered_gold), columns=["s1_id", "sx_id"])
    oracle_metrics = calculate_macro_f05(oracle_preds, gt_df, s1_all)

    print(f"  Total Candidates:     {total_cands:,}")
    print(f"  Total Gold Pairs:     {total_gold:,}")
    print(f"  Gold Pairs Present:   {len(covered_gold):,} ({cand_recall*100:.2f}%)")
    print(f"  Gold Pairs Missing:   {len(missing_gold):,} ({(1-cand_recall)*100:.2f}%)")
    print(f"  Oracle Precision:     {oracle_metrics['macro_precision']:.4f}")
    print(f"  Oracle Recall:        {oracle_metrics['macro_recall']:.4f}")
    print(f"  Oracle Macro F0.5:    {oracle_metrics['macro_f05']:.5f}")

    return {
        "src": src,
        "split": split_name,
        "total_cands": total_cands,
        "total_gold": total_gold,
        "gold_present": len(covered_gold),
        "gold_missing": len(missing_gold),
        "cand_recall": round(cand_recall, 5),
        "oracle_f05": round(oracle_metrics["macro_f05"], 5),
        "oracle_precision": round(oracle_metrics["macro_precision"], 4),
        "oracle_recall": round(oracle_metrics["macro_recall"], 4),
    }


def main():
    results = {}
    for src in ["source2", "source3"]:
        for split in ["val1", "val2"]:
            res = evaluate_candidate_ceiling(src, split)
            results[f"{src}_{split}"] = res

    out_file = ROOT / "experiments" / "v13_step1_ceiling_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll ceiling evaluations saved to {out_file}")


if __name__ == "__main__":
    main()
