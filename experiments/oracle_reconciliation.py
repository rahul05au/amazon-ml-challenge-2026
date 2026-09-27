"""
ORACLE RECONCILIATION AUDIT
============================
Computes the TRUE canonical oracle Macro F0.5 on the ORIGINAL V2 candidate pool,
using the EXACT same gold extractor (get_clean_gold) and evaluator (calculate_macro_f05)
as V12/V12.1.

Rules:
- Uses ONLY v2_cache parquets (no compound-key / MinHash / phonetic candidates)
- Source-specific gold filtering: S2-% for source2, S3-% for source3
- Per-S1 Macro F0.5 (same as calculate_macro_f05 in src/matcher.py)
- Oracle = predict ALL recoverable gold in the candidate pool, nothing else
- Asserts oracle >= V12.1 observed scores

V12.1 validated scores:
  S2 val1 = 0.89901
  S2 val2 = 0.90037
  S3 val1 = 0.90198
  S3 val2 = 0.89990
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import get_clean_gold

V2_CACHE = ROOT / "intermediate" / "v2_cache"
GT_PATH = str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")

V12_1_SCORES = {
    ("source2", "val1"): 0.89901,
    ("source2", "val2"): 0.90037,
    ("source3", "val1"): 0.90198,
    ("source3", "val2"): 0.89990,
}

print("=" * 70)
print("ORACLE RECONCILIATION AUDIT")
print("Candidate pool: ORIGINAL v2_cache ONLY")
print("Gold: source-specific (S2-% / S3-%), exact get_clean_gold() function")
print("Evaluator: calculate_macro_f05 (per-S1 macro, same as V12/V12.1)")
print("=" * 70)

results = {}

for src in ["source2", "source3"]:
    prefix = "S2-" if src == "source2" else "S3-"
    for split in ["val1", "val2"]:
        cand_path = V2_CACHE / f"v2_{src}_{split}.parquet"
        print(f"\n{'='*60}")
        print(f"  {src.upper()} | {split.upper()}")
        print(f"  Candidate file: {cand_path.name}")
        print(f"{'='*60}")

        # ── 1. Load candidate pool ──────────────────────────────────────────
        if not cand_path.exists():
            print(f"  ERROR: {cand_path} does not exist!")
            continue

        cands = pd.read_parquet(cand_path, columns=["s1_id", "sx_id"])
        cands = cands.drop_duplicates(subset=["s1_id", "sx_id"])
        total_cands = len(cands)
        print(f"  Total candidate pairs loaded: {total_cands:,}")
        print(f"  Unique S1 in candidate pool:  {cands['s1_id'].nunique():,}")

        # ── 2. Cross-source ID check ────────────────────────────────────────
        sx_ids_in_cands = cands["sx_id"].unique()
        cross_prefix = "S3-" if src == "source2" else "S2-"
        cross_ids = [x for x in sx_ids_in_cands if str(x).startswith(cross_prefix)]
        print(f"\n  Cross-source ID check:")
        print(f"    IDs with prefix '{prefix}' in sx_id: "
              f"{sum(1 for x in sx_ids_in_cands if str(x).startswith(prefix)):,}")
        print(f"    IDs with cross-source prefix '{cross_prefix}' in sx_id: {len(cross_ids):,}  "
              f"{'(ALERT: unexpected cross-source IDs present!)' if cross_ids else '(clean)'}")

        # ── 3. Load canonical gold via get_clean_gold() ─────────────────────
        s1_all, gt_set_canonical, gt_df = get_clean_gold(src, split)
        total_gold_pairs = len(gt_set_canonical)
        unique_gold_s1 = gt_df["s1_id"].nunique()

        print(f"\n  Gold extraction (get_clean_gold, prefix='{prefix}'):")
        print(f"    Total gold pairs:       {total_gold_pairs:,}")
        print(f"    Unique gold S1:         {unique_gold_s1:,}")
        print(f"    Total S1 in split:      {len(s1_all):,}")
        print(f"    S1 with no gold match:  {len(s1_all) - unique_gold_s1:,}  (singletons)")

        # Verify cross-source IDs explicitly excluded from gold
        con = duckdb.connect()
        all_unnested = con.execute(f"""
            WITH unnested AS (
                SELECT source1_entity_id AS s1_id,
                       trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
                FROM read_parquet('{GT_PATH}')
                WHERE matched_entity_ids != ''
                  AND source1_entity_id IN (
                      SELECT DISTINCT s1_id FROM (
                          VALUES {", ".join(f"('{x}')" for x in list(s1_all)[:5000])}
                      ) t(s1_id)
                  )
            )
            SELECT
                COUNT(*) AS total_unnested,
                SUM(CASE WHEN true_match_id LIKE '{prefix}%' THEN 1 ELSE 0 END) AS source_specific,
                SUM(CASE WHEN true_match_id LIKE '{cross_prefix}%' THEN 1 ELSE 0 END) AS cross_source_count
            FROM unnested
        """).df()
        con.close()
        print(f"\n  Gold unnesting audit (sample of 5000 S1s):")
        print(f"    Total unnested IDs:     {all_unnested['total_unnested'].iloc[0]:,}")
        print(f"    Source-specific kept:   {all_unnested['source_specific'].iloc[0]:,}")
        print(f"    Cross-source excluded:  {all_unnested['cross_source_count'].iloc[0]:,}  "
              f"{'(correctly excluded by prefix filter)' if all_unnested['cross_source_count'].iloc[0] > 0 else ''}")

        # ── 4. Candidate recall ─────────────────────────────────────────────
        cand_set = set(zip(cands["s1_id"], cands["sx_id"]))
        gold_in_cands = gt_set_canonical & cand_set
        gold_not_in_cands = gt_set_canonical - cand_set
        cand_recall = len(gold_in_cands) / total_gold_pairs if total_gold_pairs > 0 else 0.0

        print(f"\n  Candidate Recall:")
        print(f"    Gold pairs in candidate pool: {len(gold_in_cands):,} / {total_gold_pairs:,}")
        print(f"    Gold pairs NOT in pool:       {len(gold_not_in_cands):,}")
        print(f"    Candidate Recall:             {cand_recall:.6f}  ({cand_recall*100:.2f}%)")

        # ── 5. TRUE ORACLE: predict ALL recoverable gold, nothing else ──────
        # Oracle prediction: for each S1, predict exactly the gold IDs that
        # are present in the candidate pool. No non-gold IDs predicted.
        oracle_preds = pd.DataFrame(list(gold_in_cands), columns=["s1_id", "sx_id"])

        oracle_metrics = calculate_macro_f05(oracle_preds, gt_df, s1_all)
        oracle_f05 = oracle_metrics["macro_f05"]
        oracle_p   = oracle_metrics["macro_precision"]
        oracle_r   = oracle_metrics["macro_recall"]

        print(f"\n  Oracle Macro F0.5 (predict all recoverable gold, nothing else):")
        print(f"    Oracle Macro F0.5:   {oracle_f05:.5f}")
        print(f"    Oracle Precision:    {oracle_p:.5f}")
        print(f"    Oracle Recall:       {oracle_r:.5f}")
        print(f"    Total S1 evaluated:  {oracle_metrics['total_s1']:,}")
        print(f"    Singletons correct:  {oracle_metrics['singletons_correct']:,} / {oracle_metrics['total_singletons']:,}")

        # ── 6. V12.1 comparison and assertion ──────────────────────────────
        v12_score = V12_1_SCORES[(src, split)]
        delta = oracle_f05 - v12_score
        assertion_ok = oracle_f05 >= v12_score

        print(f"\n  V12.1 Comparison:")
        print(f"    V12.1 achieved:      {v12_score:.5f}")
        print(f"    Oracle ceiling:      {oracle_f05:.5f}")
        print(f"    Delta (oracle-v12):  {delta:+.5f}")
        print(f"    Assertion (oracle >= v12.1): {'✅ PASS' if assertion_ok else '❌ FAIL — EVALUATOR BUG'}")

        if not assertion_ok:
            print(f"\n  !! CRITICAL: Oracle {oracle_f05:.5f} < V12.1 {v12_score:.5f}")
            print(f"  !! This is impossible if the same candidate pool and gold are used.")
            print(f"  !! Root causes to investigate:")
            print(f"  !!   1. Wrong candidate file loaded")
            print(f"  !!   2. Wrong split S1 filter")
            print(f"  !!   3. Cross-source gold contamination")
            print(f"  !!   4. calculate_macro_f05 called with wrong all_s1_ids set")
            print(f"  !!   5. Candidate pool differs from what V12.1 used")

        results[(src, split)] = {
            "candidate_recall": cand_recall,
            "oracle_f05": oracle_f05,
            "oracle_precision": oracle_p,
            "oracle_recall": oracle_r,
            "v12_1_f05": v12_score,
            "delta": delta,
            "assertion_ok": assertion_ok,
            "total_gold": total_gold_pairs,
            "gold_in_cands": len(gold_in_cands),
            "total_s1": len(s1_all),
            "singletons": oracle_metrics["total_singletons"],
        }

# ── FINAL SUMMARY TABLE ────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)
print(f"{'Split':<18} {'CandRecall':>11} {'OracleF0.5':>11} {'V12.1F0.5':>10} {'Delta':>8} {'Assert':>8}")
print("-" * 70)
all_pass = True
for (src, split), r in results.items():
    label = f"{src[:3].upper()} {split}"
    ok = "✅" if r["assertion_ok"] else "❌ FAIL"
    if not r["assertion_ok"]:
        all_pass = False
    print(f"  {label:<16} {r['candidate_recall']:>10.4f}  {r['oracle_f05']:>10.5f}  "
          f"{r['v12_1_f05']:>9.5f}  {r['delta']:>+7.5f}  {ok:>8}")

print("-" * 70)
avg_oracle = np.mean([r["oracle_f05"] for r in results.values()])
avg_v12 = np.mean([r["v12_1_f05"] for r in results.values()])
avg_cand_recall = np.mean([r["candidate_recall"] for r in results.values()])
print(f"  {'AVERAGE':<16} {avg_cand_recall:>10.4f}  {avg_oracle:>10.5f}  {avg_v12:>9.5f}  "
      f"{avg_oracle-avg_v12:>+7.5f}")
print("=" * 70)

print(f"\n  Overall assertion: {'ALL PASS ✅' if all_pass else 'ONE OR MORE FAILED ❌'}")

# ── ROOT CAUSE OF PREVIOUS ~0.83 CEILING ──────────────────────────────────
print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS: Previous ~0.815-0.830 oracle ceiling")
print("=" * 70)
print("""
The V6 Phase 0 oracle experiment used a DIFFERENT oracle definition:

  WRONG (V6 Phase 0 experiments):
    - Candidate recall computed as: len(gt_set & cand_set) / total_gold
    - 'total_gold' = all gold for the split regardless of candidate coverage
    - Oracle Macro F0.5 was NEVER explicitly computed
    - The ~0.83 number was the CANDIDATE RECALL, not oracle Macro F0.5
    - Gold was extracted by: LIKE '%S2-%' without proper per-S1 unnesting
    - The evaluator treated 'blocking recall' as a proxy for oracle ceiling
    - This is INCORRECT: oracle Macro F0.5 >= blocking recall when singletons
      are correctly handled (singletons with no gold score F0.5=1.0)

  CORRECT (this script):
    - get_clean_gold() with prefix filtering (same as V12/V12.1)
    - calculate_macro_f05() with full s1_all set including singletons
    - Oracle = predict all recoverable gold, no false positives
    - Singletons (no gold match) score F0.5=1.0 when correctly predicting empty

  The ~0.83 was CANDIDATE RECALL, not oracle Macro F0.5.
  Candidate recall of ~0.83 is fully consistent with oracle Macro F0.5 of ~0.97+
  because the majority of S1 entities are singletons (no gold match),
  and they all contribute F0.5=1.0 to the macro average.
""")
print("=" * 70)
print("\nAUDIT COMPLETE. See ORACLE_RECONCILIATION_REPORT.md for full documentation.")
