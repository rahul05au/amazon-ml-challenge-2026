# Final Submission Report: Submission #5

## Status

**Final status: KEEP SUBMISSION #5.** The public leaderboard score is **0.848812**. No 0.91+ online score has been achieved. Submission #8 was not generated or uploaded.

The package preserves Submission #5 exactly. Its matching file has SHA-256 `b3df36d5ff924d7cf7e0f0040a69dc484257ce69196545308c2288bb53d9e87c`, and its candidate file has SHA-256 `8452fd1f0b586ff1d1374fef108273a24ab30f6782043a297f2f65a4ee7e39f6`.

## Prediction Lineage

Submission #5 is the frozen Submission #3 prediction file plus 7,847 strict, high-confidence empty-to-singleton recoveries. Compared with #3, it changed 7,847 of 1,732,544 S1 entities (0.453%), added 7,847 links, removed zero links, and made zero non-empty-to-empty transitions. The packaged recovery delta can be applied to the #3 result and is guarded to reject non-empty baseline rows or a delta count other than 7,847.

The frozen #5 matching file contains 1,732,544 rows, of which 161,929 are empty and 1,570,615 are non-empty. It contains 2,231,034 S2 links and 2,400,847 S3 links.

## Validation and Further Experiment

The underlying V13.1 core has a reported four-split offline Macro F0.5 of 0.90143. Separately, the #5 recovery ablation reports 0.89516 for its baseline and 0.89617 for the selected recovery policy. These are offline training-data validation metrics and do not predict or claim a 0.91+ online result.

A borderline-only Model B ablation was evaluated at five bands (±0.005, ±0.010, ±0.020, ±0.030, ±0.050). The strict evidence gates accepted zero validation additions. Four-fold Macro F0.5 and LOCO were unchanged, so the acceptance gate failed and no #8 test candidate was produced. Test-side eligible pairs were not rescored, preserving the explicit restriction against global test rescoring.

The official validator passed required matching-file format and S1 row checks in matching-only mode. Separately, a DuckDB audit checked all **4,631,881** predicted links against the packaged candidate file: zero missing S1 rows and zero links outside the candidate set. The optional ID-existence check was not run.

## Upload Note

This workspace has no authenticated challenge portal session or submission receipt. No online upload was performed. The archive contains the frozen #5 artifacts; uploading it will not change the public score unless a new prediction file is accepted and scored by the challenge portal.
