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
├── dataset/                   # Raw train and test TSVs
│   ├── test/
│   └── train/
├── docs/                      # Technical documentation and baseline
│   ├── README.md              # Documentation index
│   └── V1_BASELINE.md         # Full V1 methodology & measured metrics
├── models/                    # Trained LightGBM models & thresholds
│   ├── lgb_source2.txt        # LightGBM booster for S1 -> S2
│   ├── lgb_source3.txt        # LightGBM booster for S1 -> S3
│   ├── threshold_source2.json # Optimal threshold (0.70)
│   └── threshold_source3.json # Optimal threshold (0.65)
├── src/                       # Production ML pipeline
│   ├── __init__.py
│   ├── config.py              # Centralized paths and parameters
│   ├── normalization.py       # Multi-lingual text & Unicode normalizer
│   ├── ingestion.py           # DuckDB TSV streaming ingestion
│   ├── blocking.py            # 5-pass candidate pair indexing
│   ├── blocking_eval.py       # Candidate recall evaluation on ground truth
│   ├── features.py            # 15 vectorized RapidFuzz similarity features
│   ├── matcher.py             # Model training, thresholding & inference
│   └── submission.py          # TSV export and formatting
├── tests/                     # Unit and regression test suite
│   ├── __init__.py
│   ├── test_features.py       # Feature generation tests
│   └── test_normalization.py  # Text normalization and regex tests
├── utils/                     # Competition submission validator
│   └── validate_submission.py
├── .gitignore
├── README.md
└── requirements.txt
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
- `pandas`, `numpy`, `pytest`

---

## Running the Pipeline

### 1. Run Unit Tests
```bash
python -m pytest tests/ -v
```

### 2. Ingest & Normalize Data
```bash
python -m src.ingestion --split train
python -m src.ingestion --split test
```

### 3. Generate Candidates via Multi-Pass Blocking
```bash
python -m src.blocking --split train
python -m src.blocking --split test
```

### 4. Evaluate Candidate Recall
```bash
python -m src.blocking_eval --target source2
python -m src.blocking_eval --target source3
```

### 5. Train Matching Models
```bash
python -m src.matcher --mode train --target both
```

### 6. Predict Test Matches
```bash
python -m src.matcher --mode predict --target both
```

### 7. Export Submission TSVs
```bash
python -c "from src.submission import export_candidate_pairs, export_matching_results; export_candidate_pairs('test'); export_matching_results('intermediate/test/matches_source2.parquet', 'intermediate/test/matches_source3.parquet', 'test')"
```

---

## Validation
Run the submission validator against output files:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

---

## Current V1 Baseline
Detailed technical architecture, metric breakdown, and error analysis are documented in **[docs/V1_BASELINE.md](docs/V1_BASELINE.md)**:
- **Validation Macro $F_{0.5}$:** `0.7016` ($S2$), `0.7066` ($S3$), combined `~0.704`
- **Validation Precision:** ~85.3%
- **Blocking Recall:** 86.88% ($S2$), 86.62% ($S3$)
- **Candidates Scored:** 158,889,259 pairs across 1,732,544 test $S1$ entities
- **Total Test Matches:** 6,586,107 matches
- **Singleton Accuracy:** >90% (121,033 singletons identified)
- **Submission Validation:** `PASS` (0 invalid IDs, 100% candidate containment)

---

## Future Experiments
Future research directions targeting V2/V3 enhancements:
1. **Rare Token & Phonetic Blocking:** Expanding blocking passes to capture phonetic and n-gram variations.
2. **Candidate Pruning & Meta-Blocking:** Discarding low-probability candidate pairs prior to feature extraction to reduce candidate density.
3. **Hard-Negative Training:** Sampling harder negatives to improve model discriminability.
4. **Post-Processing Vetoes:** Applying precise cross-source location consistency vetoes to further minimize false positive merges.
