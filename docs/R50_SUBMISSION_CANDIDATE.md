# Active Submission Candidate: R50 Hard-Negative Replacement

## Active Submission Model
**Architecture:** R50 Hard-Negative Replacement  
**Base:** LightGBM GBDT (15 rapidfuzz string & token features, scale_pos_weight = 3.0)  
**Negative Strategy:** Replaced 50,000 easy negatives with confirmed high-scoring hard negatives mined from TF-IDF candidates above threshold and ranked by V2 model score with per-S1 cap = 20.  

## Untouched Validation Metrics (Dual-Split)
### Source 2
- **val1 Macro F0.5:** 0.72331 → 0.72994 (+0.00663)
- **val2 Macro F0.5:** 0.72461 → 0.72984 (+0.00523)

### Source 3
- **val1 Macro F0.5:** 0.73625 → 0.73867 (+0.00242)
- **val2 Macro F0.5:** 0.72912 → 0.73507 (+0.00595)

## Backups & Invariants
- **V2 Preserved as Backup:**
  - `models/v2_frozen_backup_lgb_source2.txt` (SHA-256: `EE8CDA0507240C6B18FE2353FAB82E40AC81D943D53097237423C08B36A539B3`)
  - `models/v2_frozen_backup_lgb_source3.txt` (SHA-256: `327B6B1E482738A342533CFED9073819AE239602B2D00E92CF3584A2F4E2F33B`)
- **Active Model Files:**
  - `models/lgb_source2.txt` (SHA-256: `ECABACFBE6ACB9B567D04956928B5B3011E9951A0BB837441BEDE0E512B2AEE9`)
  - `models/lgb_source3.txt` (SHA-256: `304AA86A6CD1BDCCB025FFEFC852958FA7C3B5D165CED2BDC42D3429179F2F40`)
- **Candidate Generation & Blocking:** UNCHANGED
- **Model Thresholds:** UNCHANGED (S2=0.70 / 0.65, S3=0.65)
- **Feature Set:** UNCHANGED (15 features)

## Full Test Inference & Validation
- **Validator Status:** PASS (Checked with `utils/validate_submission.py --check-ids`)
- **Required S1 count:** 1,732,544
- **Valid S2/S3 match IDs:** 9,969,589
- **matching_results.tsv:** 1,732,544 rows (128,280 empty singletons, 1,604,264 non-empty)
- **candidate_pairs.tsv:** 1,732,544 rows (14,800 empty, 1,717,744 non-empty)
- **Candidate containment:** 100.0%
