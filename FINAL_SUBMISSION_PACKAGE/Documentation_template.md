# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution
## Submission #5 Technical Documentation

**Team Name:** Antigravity ER  
**Submission Identifier:** `Rahul_AmazonML_Submission5`  
**Public Leaderboard Score:** `0.848812`  
**Pipeline Version:** `V13.1 Two-Stage Collective Matcher + conservative singleton recovery`  
**Validation:** V13.1 core: Macro $F_{0.5} = \mathbf{0.90143}$ across four held-out splits; #5 recovery ablation: `0.89516` baseline to `0.89617` for the selected policy. These are offline results, not the public score.  
**Submission Validator Status:** `PASS` for required matching-file format and row checks; optional candidate containment was not part of this validation run.  

---

## 1. Executive Summary

Submission #5 keeps Submission #3 as its frozen prediction core and adds only 7,847 high-confidence empty-to-singleton recoveries. The rescue gate is rank 1, score at least `0.85`, and top-1/top-2 margin at least `0.15`; it added no links to already non-empty S1 rows and removed no existing links. The public leaderboard score recorded for #5 is **`0.848812`**. The score is not 0.91+, and the offline figures below must not be represented as leaderboard performance.

The underlying Submission #3 matcher is the **V13.1 Two-Stage Collective LightGBM Matcher** with $\alpha = 0.50$, bounded multi-view retrieval, R50 rescoring, collective sibling evidence, and target exclusivity. Its four-split offline result was reported as `0.90143`. The separate #5 recovery ablation reports `0.89516` for its baseline and `0.89617` for the selected recovery policy; use these only with their respective evaluation context.

---

## 2. Methodology

### 2.1 Entity Resolution Formulation
The challenge requires matching records from Source 1 (`S1-*`) to corresponding real-world entities in Source 2 (`S2-*`) and Source 3 (`S3-*`):
- **1-to-many S1 mapping:** A single S1 entity can match multiple target records (branch locations, multi-lingual aliases, legal entity siblings).
- **Target Exclusivity:** A single target record in S2 or S3 can belong to at most one true business entity in S1.
- **Evaluation Metric:** Macro-averaged $F_{0.5}$, placing $4\times$ more weight on precision than recall ($\beta = 0.5$), making false positive avoidance and singleton discrimination critical.

### 2.2 Preprocessing & Unicode Normalization
- **Unicode Combining Mark Preservation:** Used `[^\p{L}\p{M}\p{N}\s]` regular expressions to prevent separating Indian vowel matras from base consonants across Hindi, Telugu, Tamil, and Kannada scripts.
- **Diacritical Stripping:** Stripped Latin accents (`cóálition` $\to$ `coalition`) while keeping Indic scripts intact.
- **Legal Entity & Web Artifacts:** Normalization of company suffixes (`pvt ltd`, `inc`, `llc`, `gmbh`) and stripping of domain suffixes (`.com`, `.net`, `.org`).

### 2.3 Candidate Generation (Blocking) & V13 Bounded Sparse Retrieval
To balance recall and candidate volume:
1. **Multi-Pass Core Blocking:** Compact unaccented name, rare name tokens (DF $\le 150$), normalized address, street number + street token, and rare address tokens (DF $\le 40$).
2. **V13 Bounded Sparse Retrieval:**
   - **View A:** Mark-safe Unicode normalization retaining dual native-script and Latin representations.
   - **View B:** Learned Indic-to-Latin transliteration dictionary mined strictly on `train_gate` (1,322 Source 2 tokens, 1,135 Source 3 tokens; zero test leakage).
   - **View C:** Fast OCR-tolerant prefix retrieval mapping common character confusions ('1' $\leftrightarrow$ 'l', '0' $\leftrightarrow$ 'o').
   - **View D:** Address token joins within country partitions.

### 2.4 True Frozen R50 Rescoring
All candidate pairs are enriched with the frozen R50 feature vector and evaluated directly through `models/lgb_source2.txt` and `models/lgb_source3.txt`. This eliminates missing-feature degradation and recovers high base scores (mean 0.538, 83.0% > 0.1) on newly retrieved candidates.

### 2.5 Two-Stage Collective Matcher & Sibling Evidence
1. **Stage-1 LightGBM:** Trained with 41 deterministic pairwise string, token, edit-distance, address, and competition features.
2. **Entity-Level 3-Fold Out-Of-Fold (OOF) Prediction:** Generated unbiased Stage-1 probabilities across training entities with zero entity leakage.
3. **Collective Sibling Features:** Extracted 12 collective signals capturing sibling consensus:
   - High-confidence sibling count (`s1_sibling_high_conf_count`)
   - Maximum sibling score and margin (`s1_best_sibling_score`, `s1_sibling_margin`)
   - Target cluster consensus agreement (`cluster_s1_agreement`, `cluster_max_score`)
   - Reverse target competition (`target_stage1_margin`, `target_is_stage1_best`)
4. **Stage-2 LightGBM:** Trained on 69 combined features to model multi-entity sibling consensus.

### 2.6 Alpha Blend Ensemble & Target Exclusivity
- **Ensemble Blend:** $\text{Final Score} = 0.50 \cdot \text{Stage-1} + 0.50 \cdot \text{Stage-2}$.
- **Selected Thresholds:**
  - Source 2: $\tau = 0.44$
  - Source 3: $\tau = 0.47$
- **Global Target Exclusivity:** Enforces that each target record $T$ is assigned to at most one S1:
  $$\text{S1}^*(T) = \arg\max_{S1} \left(\text{Final Score}(S1, T)\right)$$
  Pruned 318,915 ambiguous collision links, completely eliminating target conflicts (0 violations).

---

## 3. Validation Results

The table below reports the underlying V13.1 core before Submission #5's test-only singleton recovery. It is offline validation on `val1` and `val2`, not a leaderboard result. The separate #5 recovery ablation reports `0.89516` for its baseline and `0.89617` for the selected `score >= 0.85`, margin `>= 0.15`, rank-1 policy.

| Model Configuration | Source 2 Val1 | Source 2 Val2 | Source 3 Val1 | Source 3 Val2 | **Macro $F_{0.5}$ Average** | Delta vs V12.1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R50 Baseline** | 0.85698 | 0.85873 | 0.86017 | 0.85810 | **0.85850** | -0.04182 |
| **V12 Assignment-Aware** | 0.89332 | 0.89244 | 0.89734 | 0.89507 | **0.89454** | -0.00578 |
| **V12.1 Collective Matcher** | 0.89901 | 0.90037 | 0.90198 | 0.89990 | **0.90032** | Baseline |
| **V13.1 Pure Models** | 0.89935 | 0.89982 | 0.90412 | 0.90055 | **0.90096** | +0.00064 |
| **V13.1 Blended ($\alpha=0.50$) (Ours)** | **0.90055** | **0.89954** | **0.90495** | **0.90067** | **0.90143** | **+0.00111** |

---

## 4. Final Test Statistics

Complete test inference was executed across the full test candidate graph:

* **S1 rows (matching_results.tsv):** `1,732,544`
* **S1 rows (candidate_pairs.tsv):** `1,732,544`
* **S2 links:** `2,231,034`
* **S3 links:** `2,400,847`
* **Total predicted links:** `4,631,881`
* **Empty S1 entities:** `161,929`
* **Non-empty S1 entities:** `1,570,615`
* **#3-to-#5 changes:** `7,847` entities, all empty-to-non-empty; 7,847 links added, zero removed.
* **Singleton recoveries preserved:** `7,847 / 7,847` in the frozen #5 matching artifact.
* **Candidate containment:** The #5 candidate file is byte-identical to #3's. The official validator's optional large candidate cross-check was not run in the final matching-only check.
* **Validator:** Matching-file required row and format checks passed; candidate containment and optional ID-existence checks remain unverified here.

---

## 5. Artifact Hashes & Integrity

* **matching_results.tsv SHA-256:** `b3df36d5ff924d7cf7e0f0040a69dc484257ce69196545308c2288bb53d9e87c`
* **candidate_pairs.tsv SHA-256:** `8452fd1f0b586ff1d1374fef108273a24ab30f6782043a297f2f65a4ee7e39f6`
* **models/lgb_source2.txt SHA-256:** `ecabacfbe6acb9b567d04956928b5b3011e9951a0bb837441bede0e512b2aee9` (**VERIFIED UNTOUCHED**)
* **models/lgb_source3.txt SHA-256:** `304aa86a6cd1bdccb025ffefc852958fa7c3b5d165ced2bdc42d3429179f2f40` (**VERIFIED UNTOUCHED**)
* **models/v13_1_stage1_source2.txt SHA-256:** `1998672fdae6f094f2ccbf965e8f23517344638ce6598ed29ff9adc5169af67d`
* **models/v13_1_stage2_source2.txt SHA-256:** `354d66b4ca47636b8b70200321b158f75ae5f23caaaa87170f623a71f2601018`
* **models/v13_1_stage1_source3.txt SHA-256:** `7436fdd685e6ec143e3d90cb99908ba2aa8be8580618396470831009016bc98f`
* **models/v13_1_stage2_source3.txt SHA-256:** `8f4c5ad5ed4480654b6622b93cedd7aa5c18eef6b61ca692382a40f0d177dceb`
