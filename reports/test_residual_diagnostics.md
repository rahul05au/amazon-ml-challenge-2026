# Test-Side Residual Diagnostics & Risky Tail Report

**Total Test S1 Entities:** 1,732,544  
**Baseline Predicted Links:** 4,624,034 across 1,562,768 non-empty entities  

## 1. Confidence Bucket Distribution

| Confidence Bucket | S1 Count | % of All S1 | Total Predicted Links | Empty S1 (Singletons) | Avg Links / S1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`high`** | 620,391 | 35.81% | 1,949,571 | 27,964 | 3.14 |
| **`medium`** | 591,471 | 34.14% | 1,445,265 | 96,058 | 2.44 |
| **`ultra-high`** | 516,572 | 29.82% | 1,211,274 | 45,754 | 2.34 |
| **`uncertain`** | 4,110 | 0.24% | 17,924 | 0 | 4.36 |

## 2. Risky Tail Quantile Analysis (Phases 3 & 4 Target Entities)

| Tail Cutoff | Entities ($K$) | Total Links | Empty S1 | Singletons | Multi-Matches | Avg Cands | India | US | France |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bottom 0.10%** | 1,733 | 8,035 | 0 | 0 | 1,733 | 288.2 | 146 | 114 | 1,473 |
| **Bottom 0.25%** | 4,331 | 19,394 | 0 | 0 | 4,331 | 211.8 | 166 | 114 | 4,051 |
| **Bottom 0.50%** | 8,663 | 45,307 | 0 | 0 | 8,663 | 291.4 | 1,288 | 114 | 7,261 |
| **Bottom 1.00%** | 17,325 | 91,353 | 0 | 0 | 17,325 | 205.0 | 5,732 | 114 | 11,479 |
| **Bottom 2.00%** | 34,651 | 161,559 | 4,624 | 0 | 30,027 | 167.8 | 10,744 | 114 | 23,793 |

## 3. Structural Findings on Residual Error Tail

1. **The 1% Risky Tail (17,325 Entities):** The bottom 1% uncertainty entities contain a severe concentration of cross-source asymmetric links (where S1 matched S2 with 1-2 targets but 0 from S3 despite dozens of S3 candidates) and high-density multi-match overpredictions.
2. **France Disproportion:** In the bottom 0.5% tail, France entities are represented at 2.4x their population frequency due to uncalibrated name token collisions.
3. **Precision-Safe Scope:** Micro-corrections targeting only the bottom 0.5% – 1% (8,662 to 17,325 entities) leave 99% of the 0.9918 champion completely untouched.
