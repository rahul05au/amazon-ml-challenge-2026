# V1 Frozen Baseline — Official Results & Methodology

**Frozen Baseline Version:** V1.0  
**Snapshot Date:** 2026-09-25  
**Evaluation Metric:** Macro-averaged $F_{0.5}$ per $S1$ entity  

---

## 1. Executive Summary
The V1 baseline is an industrial-grade, memory-bounded entity resolution pipeline capable of resolving 1.73M `S1-*` entities against 4.89M `S2-*` and 5.08M `S3-*` records (11.7M test entities total) within strict competition limits (<= 8B model parameters, 8GB memory limit, no external APIs or geocoding).

Key components:
1. **Multi-Lingual Normalization:** Unicode NFC normalization preserving Indic combining marks (`\p{M}`), accent stripping for Latin diacritics, and legal suffix normalization.
2. **5-Pass Inverted Index Blocking (DuckDB):** Inverted index over rare tokens and compact keys yielding 86.9% recall at ~50 candidates/entity without full Cartesian joins.
3. **High-Throughput Vectorized Features:** 15 RapidFuzz string, token, length, and address similarity features computed in parallel (>110k pairs/s).
4. **LightGBM Matching Models:** Precision-tuned gradient boosted decision trees optimizing Macro $F_{0.5}$ and preserving singleton accuracy (>90%).

---

## 2. Verified Measured Results

### 2.1 Validation Performance (Clustered Entity Split)
Evaluated on 11,000 clustered S1 entities with full candidate sets (531k S2 pairs, 576k S3 pairs):

| Source | Optimal Threshold | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1 -> S2** | $\theta_2^* = 0.70$ | **0.7016** | 85.52% | 41.31% | 90.27% |
| **S1 -> S3** | $\theta_3^* = 0.65$ | **0.7066** | 84.99% | 42.44% | 87.40% |
| **Combined** | — | **~0.704** | **~85.3%** | **~41.8%** | **~88.8%** |

### 2.2 Blocking Recall (Ground Truth Split)
Evaluated on deterministic 10% S1 validation split (221,253 entities):
- **S1 -> S2:** **86.88% recall** (321,998 / 370,629 true matches captured), 50.5 avg candidates/S1, 96.8% coverage.
- **S1 -> S3:** **86.62% recall** (342,462 / 395,384 true matches captured), 54.8 avg candidates/S1, 97.0% coverage.

### 2.3 Full Test Inference Metrics
- **Test Entities:** 1,732,544 ($S1$), 4,891,956 ($S2$), 5,077,633 ($S3$)
- **Candidates Scored:**
  - $S1 \to S2$: 75,111,374 pairs
  - $S1 \to S3$: 83,777,885 pairs
  - Total: 158,889,259 candidate pairs
- **Inference Runtime:**
  - $S1 \to S2$: 1,732.2s (~28.9 min)
  - $S1 \to S3$: 1,992.3s (~33.2 min)
  - Total inference: 3,724.5s (~1h 02m)
  - TSV export: 52.1s
- **Test Match Counts:**
  - S2 predicted matches: 3,104,287
  - S3 predicted matches: 3,481,820
  - Total predicted matches: 6,586,107
- **Test Singleton Count:** 121,033 S1 entities (7.0%) predicted with 0 matches (clean empty tab entry).
- **Test Non-Empty S1 Count:** 1,611,511 S1 entities (93.0%) with $\ge 1$ match.

---

## 3. Strict Submission Validator Results
Executed with `--check-ids` against `dataset/test`:
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (121033 empty, 1611511 non-empty).
  candidate_pairs.tsv: 1732544 rows (14800 empty, 1717744 non-empty).

PASS — no blocking issues found. Safe to submit.
```
- **Exit Code:** 0
- **Invalid / Missing IDs:** 0
- **Candidate Containment:** 100% Pass (0 offenders, all matches are subsets of candidates)
- **Formatting:** Exactly 1 header + 1,732,544 rows in both TSVs. Genuine tab-delimited empty strings for singletons (`QUOTE ''`).

---

## 4. Pipeline Parameters & Configuration

### Blocking (5 Orthogonal Passes)
- **Pass A:** Compact unaccented name (without legal suffixes/domains) + country
- **Pass B:** Rare name tokens (length $\ge 3$, freq $\le 150$) + country
- **Pass C1:** Exact normalized address + country
- **Pass C2:** Street number + primary street token + country
- **Pass C3:** Rare address tokens (length $\ge 5$, freq $\le 40$) + country
- **Max block size cap:** 5,000

### Feature Vector (15 Features)
1. `name_exact`: Exact match binary indicator
2. `name_no_suf_exact`: Suffix-stripped name equality
3. `name_ratio`: Levenshtein similarity ratio
4. `name_wratio`: Weighted ratio (token alignment & prefix)
5. `name_token_set`: Token set similarity
6. `name_len_diff`: Absolute character length difference
7. `name_len_ratio`: Ratio of shorter to longer name length
8. `name_jaccard`: Word-level token Jaccard similarity
9. `addr_exact`: Exact address equality
10. `addr_ratio`: Address Levenshtein ratio
11. `addr_token_set`: Address token set similarity
12. `addr_jaccard`: Address token Jaccard similarity
13. `number_exact`: Street/building number equality (+1 match, 0 mismatch, -1 missing)
14. `postal_exact`: Postal code equality (+1 match, 0 mismatch, -1 missing)
15. `name_addr_mean`: Composite mean of name WRatio and address ratio

### Matching Models
- **Algorithm:** LightGBM GBDT (150 trees, max depth 6, 31 leaves, scale_pos_weight 3.0)
- **Parameters:** < 100,000 parameters (well within <= 8B parameter constraint)
- **Thresholds:** $\theta_2^* = 0.70$, $\theta_3^* = 0.65$

---

## 5. Known V1 Limitations & Targets for V2
1. **Blocking Recall Ceiling (~86.9%):** ~13.1% of true matches missed during initial blocking. V2 will introduce rare phonetic / n-gram blocking keys.
2. **Candidate Density (~50/entity):** Candidate set is 158M pairs. Candidate pruning / meta-blocking could discard low-probability candidates before scoring.
3. **Threshold vs Precision Trade-off:** High thresholds preserve precision (85%) but suppress recall (42%). Hard-negative training with semi-supervised negatives can improve separation.
