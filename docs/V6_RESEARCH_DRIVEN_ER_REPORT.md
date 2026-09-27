# V6: Research-Driven High-Upside Entity Resolution Pipeline Report

**Date:** September 26, 2026  
**Status:** Completed  
**Baseline Model:** R50 (V5b Hard-Negative Replacement Retraining)  
**Evaluated On:** Untouched Validation Splits `val1` (`hash % 1000 in [0, 5)`) and `val2` (`hash % 1000 in [25, 30)`)

---

## 1. Executive Summary & Core Verdict

The objective of V6 was to evaluate whether research-driven methods—namely **Additive Character N-gram Retrieval**, **Multilingual Dense Embeddings**, and **Candidate-Set Reranking**—can provide a realistic path toward Macro $F_{0.5} \ge 0.91$ (from the public leaderboard baseline of 0.815).

### Key Findings:
1. **The Blocking Ceiling (Phase 0):** Current blocking recall is constrained to **~48.2% on Source 2** and **~51.0% on Source 3**. An absolute 100% precision/recall oracle matcher on the existing candidate pool achieves an $F_{0.5}$ ceiling of only **~0.817 (S2)** and **~0.830 (S3)**. Reaching $\ge 0.91$ strictly requires expanding candidate recall.
2. **Character N-Gram Retrieval (Phase 1):** Conclusively **DISCARDED**. While 3–5 char n-gram TF-IDF retrieved some matches, it added >1,000 noise candidates per recovered match, incurred massive memory overhead (~20GB RAM), and **regressed downstream $F_{0.5}$ in 7 out of 8 validation settings**.
3. **Dense Multilingual Retrieval (Phase 2):** Conclusively **INFEASIBLE**. On CPU hardware without CUDA, encoding the 2.0M+ entity target corpus is extrapolated to require **>1.3 hours per source** (~80 minutes per split), directly exceeding competition timeout limits (>60 min end-to-end limit).
4. **Candidate-Set Reranking (Phase 3):** Conclusively **VALIDATED**. Modeling intra-candidate competition (score difference to top-1, candidate rank, candidate set size) pruned false positives and achieved consistent Macro $F_{0.5}$ improvements across **ALL FOUR validation splits** (+0.00224 to +0.00447).

---

## 2. Phase 0: Oracle Ceiling & Error Decomposition

| Source | Split | Total Gold | Gold in Candidates | Missed by Blocking | Blocking Recall | Active R50 $F_{0.5}$ | Oracle Matcher Ceiling | Gap to Oracle |
|---|---|---|---|---|---|---|---|---|
| **Source 2** | `val1` | 36,494 | 17,564 | 18,930 | **48.13%** | 0.72994 | **0.81760** | +0.08766 |
| **Source 2** | `val2` | 36,864 | 17,805 | 19,059 | **48.30%** | 0.72984 | **0.81647** | +0.08663 |
| **Source 3** | `val1` | 36,747 | 18,853 | 17,894 | **51.30%** | 0.73867 | **0.83293** | +0.09426 |
| **Source 3** | `val2` | 37,067 | 18,791 | 18,276 | **50.69%** | 0.73507 | **0.82810** | +0.09303 |

**Bottleneck Diagnosis:** **Blocking-Dominant for the $\ge 0.91$ Target**. Even if the matcher were theoretically perfect, Macro $F_{0.5}$ cannot exceed ~0.82–0.83 without recovering the ~50% of true matches missed by candidate generation.

---

## 3. Phase 1: Character N-Gram Additive Retrieval

Tested 3–5 char n-gram TF-IDF on `business_name`, partitioned by country, unioned with the baseline candidate pool.

| Source | Split | $k$ | Base Recall | Union Recall | Added Candidates | Recovered Gold | Added / Gold Ratio | Base $F_{0.5}$ | Union $F_{0.5}$ | Diff | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Source 2** | `val1` | 20 | 0.4813 | 0.4846 | 127,518 | 122 | 1,045.2 | 0.72994 | 0.73000 | +0.00006 | Discard |
| **Source 2** | `val1` | 40 | 0.4813 | 0.4853 | 298,887 | 145 | 2,061.3 | 0.72994 | 0.72862 | -0.00132 | Discard |
| **Source 2** | `val2` | 20 | 0.4830 | 0.4863 | 126,710 | 121 | 1,047.2 | 0.72984 | 0.72981 | -0.00003 | Discard |
| **Source 2** | `val2` | 40 | 0.4830 | 0.4869 | 298,177 | 143 | 2,085.2 | 0.72984 | 0.72877 | -0.00107 | Discard |
| **Source 3** | `val1` | 20 | 0.5130 | 0.5174 | 123,546 | 159 | 777.0 | 0.73867 | 0.73801 | -0.00066 | Discard |
| **Source 3** | `val1` | 40 | 0.5130 | 0.5183 | 286,529 | 194 | 1,477.0 | 0.73867 | 0.73651 | -0.00216 | Discard |
| **Source 3** | `val2` | 20 | 0.5069 | 0.5115 | 123,084 | 167 | 737.0 | 0.73507 | 0.73446 | -0.00061 | Discard |
| **Source 3** | `val2` | 40 | 0.5069 | 0.5126 | 286,181 | 210 | 1,362.8 | 0.73507 | 0.73282 | -0.00225 | Discard |

**Phase 1 Gate Decision:** **DISCARD**. In 7 out of 8 cases, downstream $F_{0.5}$ dropped due to the extreme noise-to-signal ratio (>1,000 false candidates per recovered gold match).

---

## 4. Phase 2: Multilingual Dense Retrieval Feasibility

Evaluated local `all-MiniLM-L6-v2` dense embedding retrieval on CPU:

* **Hardware Available:** CPU only (No CUDA GPU).
* **Target Encoding Throughput:** 428.2 records / sec.
* **Extrapolated Full Target Corpus Encoding Time (2.0M records):** ~1.30 hours (~78 minutes).
* **Full Query Encoding Time (37k queries):** ~1.2 minutes.
* **Total Projected Inference Time per Split:** **~79 minutes** (>60 min end-to-end competition timeout).
* **Vector Memory Footprint:** ~2.86 GB.
* **Phase 2 Decision:** **DISCARD/SKIP**. Dense retrieval is computationally prohibitive within local competition hardware and runtime constraints.

---

## 5. Phase 3: Candidate-Set Reranker

Trained a candidate-set reranker using intra-candidate competition features (`score`, `cand_rank`, `cand_count`, `top1_score`, `score_diff_top1`, `score_ratio_top1`, `score_diff_mean`) strictly on training data (`train_gate`) and evaluated on untouched validation splits:

| Source | Split | Baseline R50 $F_{0.5}$ | Baseline Precision | Baseline Recall | Reranker $F_{0.5}$ | Reranker Precision | Reranker Recall | $F_{0.5}$ Gain |
|---|---|---|---|---|---|---|---|---|
| **Source 2** | `val1` | 0.72994 | 0.88384 | 0.50191 | **0.73218** | **0.89059** | 0.49844 | **+0.00224** |
| **Source 2** | `val2` | 0.72984 | 0.88319 | 0.50144 | **0.73431** | **0.89430** | 0.49778 | **+0.00447** |
| **Source 3** | `val1` | 0.73867 | 0.88004 | 0.51849 | **0.74189** | **0.89118** | 0.51167 | **+0.00322** |
| **Source 3** | `val2` | 0.73507 | 0.87842 | 0.51407 | **0.73888** | **0.88982** | 0.50770 | **+0.00381** |

**Phase 3 Decision:** **VALIDATED & PROMOTED**.
Candidate-set reranking produces positive gains across **all 4 evaluation splits** with zero additional external dependencies and instant millisecond execution time.

---

## 6. Structural Reality vs 0.91 Target

The empirical findings from V6 provide a mathematically definitive picture:
1. **The 0.815 Leaderboard Score:** Achieved with the existing 5-blocker pipeline and R50 thresholding.
2. **The 0.830 Ceiling:** The theoretical maximum possible performance without new high-precision candidate blocking generators.
3. **The 0.910 Target:** Cannot be reached purely by modifying matchers, rerankers, or feature weights. Reaching $\ge 0.91$ requires recovering matches that currently share zero tokens, zero address strings, and zero postal codes without flooding the system with millions of false positive candidates.
