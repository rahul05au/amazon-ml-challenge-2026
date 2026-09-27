# ORACLE RECONCILIATION REPORT

> **Status: CASE A — Previous stop report INVALID. Oracle ceiling is ~0.972. Continue matcher optimization.**

---

## 1. Summary Table

| Split | Cand Recall | Oracle F0.5 | V12.1 F0.5 | Delta | Assertion |
|---|---|---|---|---|---|
| S2 val1 | 95.39% | **0.97151** | 0.89901 | +0.07250 | ✅ PASS |
| S2 val2 | 95.30% | **0.96973** | 0.90037 | +0.06936 | ✅ PASS |
| S3 val1 | 95.48% | **0.97476** | 0.90198 | +0.07278 | ✅ PASS |
| S3 val2 | 95.06% | **0.97181** | 0.89990 | +0.07191 | ✅ PASS |
| **Average** | **95.31%** | **0.97195** | 0.90032 | +0.07164 | ✅ ALL PASS |

> All four assertions `oracle_macro_f05 >= V12.1_macro_f05` passed.

---

## 2. Canonical Evaluator Definition

- **Candidate pool**: `intermediate/v2_cache/v2_{source}_{split}.parquet` (original V2 pool only)
- **Gold extractor**: `get_clean_gold(src, split)` from `experiments/v12_assignment_lgb.py`
  - S1 filter: `abs(hash(entity_id || 'v1')) % 1000` in `[0,5)` (val1) or `[25,30)` (val2)
  - Gold filter: `true_match_id LIKE 'S2-%'` (source2) or `true_match_id LIKE 'S3-%'` (source3)
  - Cross-source IDs explicitly excluded by prefix filter
- **Evaluator**: `calculate_macro_f05(pred_pairs, gt_df, s1_all)` from `src/matcher.py`
  - Per-S1 macro average including singletons
  - Singletons (no gold) that predict empty → F0.5 = 1.0
- **Oracle definition**: For each S1, predict ALL gold targets present in its candidate pool; predict NOTHING else

---

## 3. Gold Filtering Audit

| Split | Source-specific kept | Cross-source excluded |
|---|---|---|
| S2 val1 | 8,462 | **8,943** |
| S2 val2 | 8,532 | **8,974** |
| S3 val1 | 8,943 | **8,462** |
| S3 val2 | 8,974 | **8,532** |

All candidate pools contain **zero cross-source sx_ids** (confirmed clean).

---

## 4. Full Per-Split Data

### S2 val1
- Candidate pairs: 558,054 | Unique S1 in pool: 10,940
- Gold pairs: 18,413 | In pool: 17,564 | Missing: 849
- Candidate Recall: **95.39%** | Oracle F0.5: **0.97151** | P: 0.97774 | R: 0.95831
- Singletons: 1,429/1,429 ✅

### S2 val2
- Candidate pairs: 586,825 | Unique S1 in pool: 11,005
- Gold pairs: 18,683 | In pool: 17,805 | Missing: 878
- Candidate Recall: **95.30%** | Oracle F0.5: **0.96973** | P: 0.97592 | R: 0.95676
- Singletons: 1,400/1,400 ✅

### S3 val1
- Candidate pairs: 604,859 | Unique S1 in pool: 10,934
- Gold pairs: 19,745 | In pool: 18,853 | Missing: 892
- Candidate Recall: **95.48%** | Oracle F0.5: **0.97476** | P: 0.98192 | R: 0.95945
- Singletons: 1,365/1,365 ✅

### S3 val2
- Candidate pairs: 633,842 | Unique S1 in pool: 11,024
- Gold pairs: 19,768 | In pool: 18,791 | Missing: 977
- Candidate Recall: **95.06%** | Oracle F0.5: **0.97181** | P: 0.97916 | R: 0.95640
- Singletons: 1,328/1,328 ✅

---

## 5. Root Cause of Previous ~0.815–0.830 "Oracle Ceiling"

| Dimension | V6 Phase 0 (WRONG) | This Audit (CORRECT) |
|---|---|---|
| What was computed | `len(gold ∩ cands) / total_gold` | `calculate_macro_f05(oracle_preds, gt_df, s1_all)` |
| Metric name | **Candidate recall** | **Oracle Macro F0.5** |
| Singleton handling | Not included | Singletons score F0.5=1.0 |
| Gold filtering | `LIKE '%S2-%'` (fragile substring) | `get_clean_gold()` exact prefix on unnested IDs |
| Result | ~0.48–0.51 blocking recall | **~0.97 oracle F0.5** |

**The critical insight**: ~13% of S1 entities per split are singletons (no gold match). A perfect oracle predicts empty for them → F0.5=1.0. These 1,300–1,400 entities each contributing 1.0 significantly lift the macro average above the raw candidate recall rate. Candidate recall of ~95% maps to oracle F0.5 of ~0.97, **not ~0.83**.

> [!IMPORTANT]
> The ~0.83 number reported as the "oracle ceiling" was **candidate recall** (blocking recall), not oracle Macro F0.5. These are completely different metrics. The stop condition was triggered on a false premise.

---

## 6. Comparison Against Previous Oracle Reports

| Split | Previous (WRONG — was cand recall) | Correct Oracle F0.5 | Gap |
|---|---|---|---|
| S2 val1 | ~0.815–0.830 | **0.97151** | +0.14 |
| S2 val2 | ~0.815–0.830 | **0.96973** | +0.14 |
| S3 val1 | ~0.815–0.830 | **0.97476** | +0.14 |
| S3 val2 | ~0.815–0.830 | **0.97181** | +0.14 |

---

## 7. Decision: CASE A

> **Oracle ~0.972 >> Target 0.91 > V12.1 ~0.900**

The previous stop-condition report is **INVALID**. The oracle ceiling is **~0.972**, not ~0.83.

- V12.1 achieved: **0.90032** average
- Oracle ceiling: **0.97195** average  
- Remaining headroom: **+0.072**
- Target (0.91): only **+0.010** above V12.1 — well within reach

### ✅ Continue with V13.1

1. Expanded V13 candidates (valid candidate pool)
2. Valid R50 rescoring
3. Retrained Stage-1 on expanded pool
4. OOF Stage-2
5. Final decoding with target exclusivity

> [!NOTE]
> Do NOT expand the candidate pool. Blocking is NOT the bottleneck (95.3% recall already). The bottleneck is matcher discrimination within the existing pool (gap between oracle 0.972 and V12.1 0.900 = 0.072).

---

*Script*: [`experiments/oracle_reconciliation.py`](file:///c:/Users/rahul/Downloads/6ab10eb3b23ba_student_resource/student_resource/experiments/oracle_reconciliation.py)  
*Run*: 2026-09-27 10:17 IST | Exit code: 0 | All assertions: PASS
