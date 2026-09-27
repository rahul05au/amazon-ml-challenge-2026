# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

## Problem
Given three business record sources:
- **Source 1 ($S1$):** Query entities (~1.73M test records)
- **Source 2 ($S2$) and Source 3 ($S3$):** Target database entities (~4.89M and ~5.08M test records)

For every $S1$ entity, find zero, one, or multiple matching entities from $S2$ and $S3$.
The evaluation metric is **Macro-averaged $F_{0.5}$** across all $S1$ entities (precision prioritized $4\times$ over recall; singletons with no match must be empty strings).

---

## Solution Overview
Our solution is an industrial-grade, memory-bounded, multi-source entity resolution system:
- **Multi-lingual Unicode Normalization:** Unicode NFC normalization preserving Indic combining marks (`\p{M}`), stripping Latin diacritics, and expanding/stripping legal entity suffixes.
- **5-Pass Inverted Index Blocking (DuckDB):** Partitions the candidate space without Cartesian joins, capturing 86.9% recall at ~50 candidates/entity.
- **Vectorized Parallel Feature Extraction:** 15 RapidFuzz string, token, length, and address similarity features computed across multiprocessing workers (>110,000 pairs/s).
- **LightGBM Matching Models:** Precision-tuned gradient boosted decision trees optimizing Macro $F_{0.5}$ and preserving singleton accuracy (>90%).

---

## Pipeline
The production workflow operates strictly in order:
```text
Raw TSVs (dataset/)
       │
       ▼
1. Ingestion (`src/ingestion.py`)
       │
       ▼
2. Normalization (`src/normalization.py`)
       │
       ▼
3. Multi-Pass Blocking (`src/blocking.py`)
       │
       ▼
4. Feature Extraction (`src/features.py`)
       │
       ▼
5. Matching & Thresholding (`src/matcher.py`)
       │
       ▼
6. Submission Export (`src/submission.py`)
       │
       ▼
7. Validation (`utils/validate_submission.py`)
```

---

## Repository Structure
```text
student_resource/
├── models/                    # Trained LightGBM models & thresholds
│   ├── lgb_source2.txt        # LightGBM booster for S1 -> S2
│   ├── lgb_source3.txt        # LightGBM booster for S1 -> S3
│   ├── threshold_source2.json # Source 2 threshold
│   ├── threshold_source3.json # Source 3 threshold
│   ├── v13_1_stage1_source2.txt
│   ├── v13_1_stage1_source3.txt
│   ├── v13_1_stage2_source2.txt
│   └── v13_1_stage2_source3.txt
├── src/                       # Core ML pipeline modules
│   ├── __init__.py
│   ├── config.py              # Centralized paths and configuration
│   ├── normalization.py       # Multi-lingual text & Unicode normalizer
│   ├── ingestion.py           # DuckDB streaming ingestion
│   ├── blocking.py            # Multi-pass candidate indexing
│   ├── lexical_blocking.py    # Lexical blocking rules
│   ├── features.py            # Vectorized RapidFuzz similarity features
│   ├── matcher.py             # Matching engine & F0.5 metrics
│   └── submission.py          # TSV submission export
├── experiments/               # Core pipeline stages and collective matchers
│   ├── v12_assignment_lgb.py
│   ├── v12_1_collective_matcher.py
│   ├── v12_1_candidate_expansion.py
│   ├── v13_1_pipeline.py
│   ├── v13_1_execute.py
│   └── v13_candidate_audit_and_eval.py
├── scripts/                   # Production inference & recovery scripts
│   ├── generate_final_third_submission.py
│   ├── apply_submission5_recoveries.py
│   └── import_submission_candidates.py
├── utils/                     # Official competition submission validator
│   └── validate_submission.py
├── Documentation_template.md  # Official submission technical documentation
├── requirements.txt
└── README.md
```

---

## Requirements
- Python 3.8+ (tested on Python 3.12)
- Dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Core libraries:
- `duckdb` (In-process SQL OLAP engine for multi-million row joins)
- `lightgbm` (Gradient boosted decision trees, Apache-2.0, <= 8B parameters)
- `rapidfuzz` (C++ vectorized string similarity calculations)
- `pyarrow` (Parquet streaming and memory management)
- `pandas`, `numpy`

---

## Running the Pipeline

### 1. Ingest & Normalize Data
```bash
python -m src.ingestion --split train
python -m src.ingestion --split test
```

### 2. Generate Candidate Pairs
```bash
python -m src.blocking --split test
```

### 3. Generate Submission Predictions
```bash
python scripts/generate_final_third_submission.py
python scripts/apply_submission5_recoveries.py
```

---

## Validation
Run the submission validator against output files:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Technical Documentation
For full details on methodology, candidate generation, two-stage collective LightGBM matching, target exclusivity, singleton recovery, and test set metrics, please refer to **[Documentation_template.md](Documentation_template.md)**.
