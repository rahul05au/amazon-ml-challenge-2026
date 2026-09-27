"""Compile full comprehensive metric comparison across all 4 splits."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

comparison = {
    "R50_Baseline": {
        "S2_val1": {"macro_f05": 0.85698, "precision": 0.8659, "recall": 0.8229, "predicted": 16568, "sec_recall": 0.8957},
        "S2_val2": {"macro_f05": 0.85873, "precision": 0.8671, "recall": 0.8269, "predicted": 16922, "sec_recall": 0.8959},
        "S3_val1": {"macro_f05": 0.86017, "precision": 0.8711, "recall": 0.8193, "predicted": 17462, "sec_recall": 0.9056},
        "S3_val2": {"macro_f05": 0.85810, "precision": 0.8690, "recall": 0.8172, "predicted": 17478, "sec_recall": 0.9051},
        "Average_F05": 0.85850,
    },
    "Phase3_Control": {
        "S2_val1": {"macro_f05": 0.85966, "precision": 0.8679, "recall": 0.8284, "predicted": 16752},
        "S2_val2": {"macro_f05": 0.86154, "precision": 0.8694, "recall": 0.8317, "predicted": 17094},
        "S3_val1": {"macro_f05": 0.86372, "precision": 0.8739, "recall": 0.8252, "predicted": 17621},
        "S3_val2": {"macro_f05": 0.86171, "precision": 0.8719, "recall": 0.8231, "predicted": 17641},
        "Average_F05": 0.86166,
    },
    "V12_Assignment_LGB": {
        "S2_val1": {"macro_f05": 0.89332, "precision": 0.9104, "recall": 0.8672, "predicted": 16421, "sec_recall": 0.8973},
        "S2_val2": {"macro_f05": 0.89244, "precision": 0.9094, "recall": 0.8674, "predicted": 16688, "sec_recall": 0.8967},
        "S3_val1": {"macro_f05": 0.89734, "precision": 0.9149, "recall": 0.8710, "predicted": 17674, "sec_recall": 0.9028},
        "S3_val2": {"macro_f05": 0.89507, "precision": 0.9131, "recall": 0.8684, "predicted": 17672, "sec_recall": 0.8998},
        "Average_F05": 0.89454,
    },
    "V12_1_Collective_Sibling": {
        "S2_val1": {"macro_f05": 0.89901, "precision": 0.9149, "recall": 0.8756, "predicted": 16705, "sec_recall": 0.9012, "threshold": 0.47, "w2": 0.8},
        "S2_val2": {"macro_f05": 0.90037, "precision": 0.9145, "recall": 0.8824, "predicted": 17142, "sec_recall": 0.9025, "threshold": 0.45, "w2": 0.8},
        "S3_val1": {"macro_f05": 0.90198, "precision": 0.9180, "recall": 0.8799, "predicted": 18012, "sec_recall": 0.9142, "threshold": 0.47, "w2": 0.8},
        "S3_val2": {"macro_f05": 0.89990, "precision": 0.9161, "recall": 0.8780, "predicted": 17954, "sec_recall": 0.9018, "threshold": 0.47, "w2": 0.8},
        "Average_F05": 0.90032,
    }
}

with open(ROOT / "experiments/v12_1_full_comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2)

print("Saved full comparison to experiments/v12_1_full_comparison.json")
