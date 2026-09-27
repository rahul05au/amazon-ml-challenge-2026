# Antigravity ER: Submission #5

## Submission Status

This package contains the frozen Submission #5 artifacts. The recorded public leaderboard score is **0.848812**. The V13.1 core's reported four-split offline score of 0.90143 is not an online score. The follow-up borderline-only Model B experiment produced no accepted validation changes, so this package remains #5 and no Submission #8 was created.

## Package Layout

- `../../output/matching_results.tsv`: canonical #5 predictions, one row per test S1.
- `../../output/candidate_pairs.tsv`: frozen candidate graph supplied to inference.
- `../../FINAL_REPORT.md`: concise submission status, validation scope, and artifact hashes.
- `src/`: normalization, ingestion, features, blocking, matcher and output modules.
- `experiments/`: V12/V13.1 feature and collective scoring modules used by test inference.
- `scripts/`: candidate import, V13.1 inference and the audited #5 recovery-delta application.
- `models/`: frozen V13.1 stage models and baseline LightGBM weights.
- `artifacts/submission5_recovery_delta.tsv`: exact 7,847-row #3-to-#5 empty-to-singleton delta.

The raw challenge dataset is not redistributed. Place the provided `dataset/train/` and `dataset/test/` directories under this project directory before running preprocessing.

## Environment

Use Python 3.12 and install pinned dependencies:

```powershell
py -3 -m pip install -r requirements.txt
```

## Reproduction

Run commands from this directory (`code/business_entity_resolution`):

```powershell
py -3 -m src.ingestion --split test
py -3 scripts/import_submission_candidates.py
py -3 scripts/generate_final_third_submission.py
py -3 scripts/apply_submission5_recoveries.py
```

The import step materializes the shipped final candidate graph as the per-source Parquet inputs consumed by inference. The generator writes the V13.1 core prediction file to `FINAL_THIRD_SUBMISSION/`; the recovery step applies only the frozen, audited #5 additions to create the final file at `../../output/matching_results.tsv`. The recovery utility refuses to run unless its delta has exactly 7,847 rows and every affected S1 row is empty in the baseline.

The candidate graph is included as the exact inference input. Regenerating the original multi-view retrieval graph from raw data is a separate blocking run and is not needed to inspect or rerun scoring against this frozen candidate set.

## Validation

From this directory, run the official validator:

```powershell
py -3 utils/validate_submission.py `
  --matching ../../output/matching_results.tsv `
  --candidate ../../output/candidate_pairs.tsv `
  --test-dir dataset/test
```

The check that matched IDs exist can optionally be enabled with `--check-ids`; it is memory intensive on the complete data.

## Recorded Results

- Public leaderboard: `0.848812` (Submission #5).
- V13.1 core offline Macro F0.5: `0.90143` reported across four held-out splits; not comparable to or a substitute for the public score.
- #5 recovery ablation: `0.89516` baseline to `0.89617` for the selected strict recovery policy.
- Test rows: `1,732,544`; empty S1 rows: `161,929`.
- Test links: `2,231,034` S2 and `2,400,847` S3.
- #5 rescue: `7,847` rows added, zero links removed.
- Matching SHA-256: `b3df36d5ff924d7cf7e0f0040a69dc484257ce69196545308c2288bb53d9e87c`.
- Candidate SHA-256: `8452fd1f0b586ff1d1374fef108273a24ab30f6782043a297f2f65a4ee7e39f6`.
- Candidate containment audit: all 4,631,881 predicted links found in the packaged candidates; zero missing links or S1 rows.
