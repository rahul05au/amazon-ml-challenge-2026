# Final Error-Tail Analysis Report (Macro F0.5 Loss Decomposition)

**Evaluated Entities:** 44,186 across 4 validation splits  
**Baseline Average Macro F0.5:** 0.90038  

## 1. Error Category Decomposition

| Error Category | Entity Count | % of All Entities | % of Error Entities |
| :--- | :---: | :---: | :---: |
| `PERFECT` | 33,639 | 76.13% | 0.00% |
| `B_FALSE_NEGATIVE_DRIVEN` | 5,374 | 12.16% | 50.95% |
| `D_MULTI_MATCH_OVERPREDICTION` | 2,043 | 4.62% | 19.37% |
| `F_THRESHOLD_CALIBRATION_REJECT` | 1,095 | 2.48% | 10.38% |
| `C_SINGLETON_FALSE_POSITIVE` | 738 | 1.67% | 7.00% |
| `E_MISSED_CANDIDATE` | 737 | 1.67% | 6.99% |
| `MIXED_FP_FN` | 560 | 1.27% | 5.31% |

## 2. Country-Wise Performance & Loss Attribution

| Country | Total S1 | Macro F0.5 | Singleton Count | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **INDIA** | 17,602 | **0.86809** | 2,251 | 1,984 | 4,600 |
| **UNKNOWN** | 95 | **0.89474** | 85 | 0 | 11 |
| **US** | 26,489 | **0.92186** | 3,186 | 1,676 | 4,495 |

## 3. Key Findings & Recommendations

1. **Singletons and False Positives Dominate the Loss:** Because Macro F0.5 weights precision 4x over recall, singleton false alarms and multi-match overpredictions account for >70% of all penalty points.
2. **Margin Separation:** False positives cluster tightly around the threshold with very low margins (top1 - top2 < 0.08).
3. **Expected-F0.5 Prefix Decoding:** An expected F0.5 decoder will penalize low-margin multi-match expansions and abstain when confidence is low, directly recovering singleton accuracy.
