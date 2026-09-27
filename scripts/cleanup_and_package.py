"""
FINAL CLEANUP + SUBMISSION PACKAGER
Strips comments from all production Python files,
deletes all experimental/dead files, creates clean final zip.
"""
import ast
import os
import shutil
import sys
import tokenize
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Files to DELETE entirely ──────────────────────────────────────────────────
DELETE_FILES = [
    # Old experiments
    "experiments/v4_precision_veto.py",
    "experiments/v5_hard_negative_retrain.py",
    "experiments/v5b_hard_negative_replacement.py",
    "experiments/v6_character_retrieval.py",
    "experiments/v6_dense_slice.py",
    "experiments/v6_oracle_ceiling.py",
    "experiments/v6_reranker_experiment.py",
    "experiments/v7_compound_key.py",
    "experiments/v7_minhash_lsh.py",
    "experiments/v7_phonetic_address_key.py",
    "experiments/v8_ranking_diagnostics.py",
    "experiments/v9_lambdarank.py",
    "experiments/v10_selective_semantic.py",
    "experiments/v10_results.json",
    "experiments/v11_crossencoder.py",
    "experiments/v12_1_candidate_audit.py",
    "experiments/v12_1_candidate_expansion.py",
    "experiments/v12_1_fine_sweep.py",
    "experiments/v12_1_full_comparison.json",
    "experiments/v12_1_results.json",
    "experiments/v12_results.json",
    "experiments/v13_1_blend_sweep.py",
    "experiments/v13_1_blend_sweep_results.json",
    "experiments/v13_1_results.json",
    "experiments/v13_1_step3_r50_rescore.py",
    "experiments/v13_audit_results.json",
    "experiments/v13_candidate_audit_and_eval.py",
    "experiments/v13_evaluate_all_splits.py",
    "experiments/v13_fast_retrieval.py",
    "experiments/v13_results.json",
    "experiments/v13_sparse_retrieval.py",
    "experiments/v13_step1_candidate_ceiling.py",
    "experiments/v13_step1_ceiling_results.json",
    "experiments/v14_ood_robust_model.py",
    "experiments/compile_comparison.py",
    "experiments/field_inventory.py",
    "experiments/inspect_misses.py",
    "experiments/missed_pairs_sample.txt",
    "experiments/oracle_reconciliation.py",
    "experiments/test_blend.py",
    "experiments/test_targeted_channels.py",
    "experiments/v12_audit_results.json",
    # Old scripts
    "scripts/build_submission_4.py",
    "scripts/build_submission_6.py",
    "scripts/build_submission_7.py",
    "scripts/build_submission_8.py",
    "scripts/package_submission_4.py",
    "scripts/package_submission_5.py",
    "scripts/package_submission_6.py",
    "scripts/package_third_submission.py",
    "scripts/phase10_test_safe_apply.py",
    "scripts/phase1_error_tail_analysis.py",
    "scripts/phase2_3_4_calibration_and_decoder.py",
    "scripts/phase2_3_test_residual_diagnostics.py",
    "scripts/phase4_9_micro_ablations.py",
    "scripts/run_v11_aws.sh",
    # Dead src files
    "src/eval_v2_experiment1b.py",
    "src/eval_v2_experiment1c.py",
    "src/eval_v2_stability.py",
    "src/blocking_eval.py",
    "src/features.py",
    "src/ingestion.py",
    "src/lexical_blocking.py",
    "src/submission.py",
    # Old reports / docs
    "V12_FINAL_REPORT.md",
    "V12_1_THIRD_SUBMISSION_REPORT.md",
    "V13_THIRD_SUBMISSION_REPORT.md",
    "V13_1_THIRD_SUBMISSION_REPORT.md",
    "ORACLE_RECONCILIATION_REPORT.md",
    # Scratch
    "experiments/v12_results.json",
]

DELETE_DIRS = [
    "reports",
    "scratch",
    "docs",
    "tests",
    "SUBMISSION_V1",
    "SUBMISSION_4",
    "SUBMISSION_5",
    "SUBMISSION_6",
    "SUBMISSION_7",
    "SUBMISSION_8",
    "SUBMISSION_HISTORY",
    "FINAL_9918_BASELINE",
    "FINAL_9918_PLUS",
    "FINAL_PUBLIC_BASELINE_084814",
    "FINAL_PUBLIC_BASELINE_0849",
    "FINAL_THIRD_SUBMISSION",
]

# ── Production Python files to strip comments from ────────────────────────────
PRODUCTION_FILES = [
    "src/__init__.py",
    "src/blocking.py",
    "src/config.py",
    "src/matcher.py",
    "src/normalization.py",
    "experiments/v12_assignment_lgb.py",
    "experiments/v12_1_collective_matcher.py",
    "experiments/v13_1_execute.py",
    "experiments/v13_1_pipeline.py",
    "scripts/generate_final_third_submission.py",
    "utils/validate_submission.py",
]


def strip_comments(source: str) -> str:
    """Remove all comments and docstrings from Python source."""
    try:
        result = []
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        prev_toktype = tokenize.ENCODING
        for tok_type, tok_string, tok_start, tok_end, tok_line in tokens:
            if tok_type == tokenize.COMMENT:
                continue
            if tok_type == tokenize.STRING:
                # Check if it's a docstring (expression statement)
                if prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                                    tokenize.ENCODING, 54):  # 54 = OP for class/def
                    continue
            if tok_type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                                 tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                result.append((tok_type, tok_string))
            else:
                result.append((tok_type, tok_string))
            prev_toktype = tok_type

        # Reconstruct via tokenize.untokenize
        cleaned = tokenize.untokenize(result)

        # Remove blank lines (3+ in a row -> 1)
        lines = cleaned.splitlines()
        out = []
        blanks = 0
        for line in lines:
            if line.strip() == "":
                blanks += 1
                if blanks <= 1:
                    out.append(line)
            else:
                blanks = 0
                out.append(line)
        return "\n".join(out).strip() + "\n"
    except Exception:
        return source  # return original if stripping fails


def main():
    print("=" * 70)
    print("FINAL CODEBASE CLEANUP + SUBMISSION PACKAGER")
    print("=" * 70)

    # 1. Delete unwanted files
    print("\n[1] Deleting unwanted files...")
    deleted = 0
    for rel in DELETE_FILES:
        p = ROOT / rel
        if p.exists():
            p.unlink()
            print(f"  ✗ {rel}")
            deleted += 1
    print(f"  Deleted {deleted} files.")

    # 2. Delete unwanted directories
    print("\n[2] Deleting unwanted directories...")
    for rel in DELETE_DIRS:
        p = ROOT / rel
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            print(f"  ✗ {rel}/")
    
    # 3. Remove __pycache__ everywhere
    print("\n[3] Removing __pycache__ directories...")
    for pycache in ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)
    for pyc in ROOT.rglob("*.pyc"):
        pyc.unlink()
    print("  Done.")

    # 4. Strip comments from production files
    print("\n[4] Stripping comments from production files...")
    for rel in PRODUCTION_FILES:
        p = ROOT / rel
        if not p.exists():
            print(f"  SKIP (not found): {rel}")
            continue
        original = p.read_text(encoding="utf-8", errors="ignore")
        cleaned = strip_comments(original)
        p.write_text(cleaned, encoding="utf-8")
        orig_lines = len(original.splitlines())
        new_lines = len(cleaned.splitlines())
        print(f"  ✓ {rel}: {orig_lines} -> {new_lines} lines")

    # 5. Create final submission zip
    print("\n[5] Creating final submission package...")
    zip_path = ROOT / "FINAL_SUBMISSION_PACKAGE.zip"
    if zip_path.exists():
        zip_path.unlink()

    champion_dir = ROOT / "FINAL_PUBLIC_CHAMPION_0849"
    matching_tsv = champion_dir / "matching_results.tsv"
    candidate_tsv = champion_dir / "candidate_pairs.tsv"

    code_files = [
        # src
        ("src/__init__.py", "code/src/__init__.py"),
        ("src/blocking.py", "code/src/blocking.py"),
        ("src/config.py", "code/src/config.py"),
        ("src/matcher.py", "code/src/matcher.py"),
        ("src/normalization.py", "code/src/normalization.py"),
        # experiments (production only)
        ("experiments/v12_assignment_lgb.py", "code/experiments/v12_assignment_lgb.py"),
        ("experiments/v12_1_collective_matcher.py", "code/experiments/v12_1_collective_matcher.py"),
        ("experiments/v13_1_execute.py", "code/experiments/v13_1_execute.py"),
        ("experiments/v13_1_pipeline.py", "code/experiments/v13_1_pipeline.py"),
        # scripts
        ("scripts/generate_final_third_submission.py", "code/scripts/generate_final_third_submission.py"),
        # utils
        ("utils/validate_submission.py", "code/utils/validate_submission.py"),
        # root
        ("requirements.txt", "code/requirements.txt"),
        ("README.md", "code/README.md"),
        ("Documentation_template.md", "code/Documentation_template.md"),
    ]

    # Models (only the 4 production models)
    model_files = [
        ("models/v13_1_stage1_source2.txt", "code/models/v13_1_stage1_source2.txt"),
        ("models/v13_1_stage1_source3.txt", "code/models/v13_1_stage1_source3.txt"),
        ("models/v13_1_stage2_source2.txt", "code/models/v13_1_stage2_source2.txt"),
        ("models/v13_1_stage2_source3.txt", "code/models/v13_1_stage2_source3.txt"),
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Output files
        print(f"  Adding output/matching_results.tsv ({matching_tsv.stat().st_size/1e6:.0f} MB)...")
        zf.write(matching_tsv, arcname="output/matching_results.tsv")
        print(f"  Adding output/candidate_pairs.tsv ({candidate_tsv.stat().st_size/1e9:.2f} GB)...")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # Code files
        for src_rel, arc_name in code_files:
            p = ROOT / src_rel
            if p.exists():
                zf.write(p, arcname=arc_name)
                print(f"  Adding {arc_name}")
            else:
                print(f"  SKIP: {src_rel} not found")

        # Model files
        for src_rel, arc_name in model_files:
            p = ROOT / src_rel
            if p.exists():
                print(f"  Adding {arc_name} ({p.stat().st_size/1e6:.1f} MB)...")
                zf.write(p, arcname=arc_name)

    size_gb = zip_path.stat().st_size / 1e9
    print(f"\n  ✅ Package created: {zip_path.name} ({size_gb:.2f} GB)")

    print("\n[6] Git cleanup commit...")
    print("  Run: git add -A && git commit -m 'cleanup: remove experimental files, strip comments, final submission'")
    print("  Run: git push origin main")

    print("\n" + "=" * 70)
    print("DONE. Final package ready: FINAL_SUBMISSION_PACKAGE.zip")
    print("Public Champion: Submission #5 = 0.848812 (FINAL_PUBLIC_CHAMPION_0849/)")
    print("=" * 70)


if __name__ == "__main__":
    main()
