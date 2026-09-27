# Expected-F0.5 Decoder & Country-Aware Policy Ablation Report

## 1. Multi-Split Policy Scoreboard

| Policy Variant | Avg Macro $F_{0.5}$ | Precision | Recall | Singleton Accuracy | Total Val Links | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A_Baseline`** | **0.90039** | 0.9148 | 0.8825 | 0.8664 | 71,163 | +0.00000 🏆 |
| **`B_CountryAware_Conservative`** | **0.90034** | 0.9156 | 0.8801 | 0.8768 | 70,531 | -0.00005  |
| **`C_ExpectedF05_Decoder`** | **0.89635** | 0.9066 | 0.8918 | 0.8629 | 73,490 | -0.00404  |
| **`D_Hybrid_ExpectedF05_PrecisionTuned`** | **0.89792** | 0.9092 | 0.8898 | 0.8653 | 72,870 | -0.00247  |

## 2. Per-Split Breakdown

| Policy Variant | S2 Val1 | S2 Val2 | S3 Val1 | S3 Val2 |
| :--- | :---: | :---: | :---: | :---: |
| `A_Baseline` | 0.89947 | 0.89877 | 0.90415 | 0.89916 |
| `B_CountryAware_Conservative` | 0.90005 | 0.89878 | 0.90328 | 0.89925 |
| `C_ExpectedF05_Decoder` | 0.89604 | 0.89559 | 0.89877 | 0.89500 |
| `D_Hybrid_ExpectedF05_PrecisionTuned` | 0.89678 | 0.89632 | 0.90124 | 0.89732 |
