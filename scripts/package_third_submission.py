"""
PACKAGE FINAL THIRD SUBMISSION ZIP
==================================
Builds: FINAL_THIRD_SUBMISSION/Rahul_AmazonML_ThirdSubmission.zip
Exact required structure:
Rahul_AmazonML_ThirdSubmission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── experiments/
│       ├── scripts/
│       ├── utils/
│       ├── models/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = ROOT / "FINAL_THIRD_SUBMISSION"
ZIP_PATH = SUBMISSION_DIR / "Rahul_AmazonML_ThirdSubmission.zip"

EXCLUDED_DIRS = {
    "__pycache__", ".pytest_cache", ".git", "dataset", "intermediate",
    "scratch", ".system_generated", "FINAL_THIRD_SUBMISSION", "SUBMISSION_V1",
    ".gemini", ".vscode", ".idea"
}

EXCLUDED_EXTS = {
    ".pyc", ".pyo", ".pyd", ".log", ".tmp", ".parquet", ".zip", ".tar", ".gz"
}

# The required model files for reproduction
ALLOWED_MODELS = {
    "lgb_source2.txt",
    "lgb_source3.txt",
    "v13_1_stage1_source2.txt",
    "v13_1_stage2_source2.txt",
    "v13_1_stage1_source3.txt",
    "v13_1_stage2_source3.txt",
    "threshold_source2.json",
    "threshold_source3.json",
}


def should_include_file(path: Path) -> bool:
    name_lower = path.name.lower()
    if "aws" in name_lower or "cred" in name_lower or "secret" in name_lower:
        return False
    for part in path.parts:
        if part in EXCLUDED_DIRS:
            return False
    if path.suffix.lower() in EXCLUDED_EXTS:
        return False
    if "models" in path.parts:
        return path.name in ALLOWED_MODELS
    return True


def main():
    t0 = time.time()
    print("=" * 70)
    print("PACKAGING FINAL THIRD SUBMISSION ZIP")
    print("=" * 70)

    if ZIP_PATH.exists():
        print(f"Removing existing zip at {ZIP_PATH}...")
        ZIP_PATH.unlink()

    print(f"Creating zip at: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        # 1. output/ files
        match_src = SUBMISSION_DIR / "matching_results.tsv"
        cand_src = SUBMISSION_DIR / "candidate_pairs.tsv"

        print(f"  Adding output/matching_results.tsv ({match_src.stat().st_size / (1024*1024):.1f} MB) ...")
        zf.write(match_src, arcname="output/matching_results.tsv")

        print(f"  Adding output/candidate_pairs.tsv ({cand_src.stat().st_size / (1024*1024):.1f} MB) ...")
        zf.write(cand_src, arcname="output/candidate_pairs.tsv")

        # 2. Documentation_template.md at root
        doc_src = ROOT / "Documentation_template.md"
        print(f"  Adding Documentation_template.md ...")
        zf.write(doc_src, arcname="Documentation_template.md")

        # 3. code/business_entity_resolution/
        code_root = "code/business_entity_resolution"
        
        # README.md
        readme_src = SUBMISSION_DIR / "PACKAGED_README.md"
        zf.write(readme_src, arcname=f"{code_root}/README.md")
        
        # requirements.txt
        req_src = ROOT / "requirements.txt"
        zf.write(req_src, arcname=f"{code_root}/requirements.txt")

        # src, experiments, scripts, utils, models
        subdirs_to_pack = ["src", "experiments", "scripts", "utils", "models"]
        for subdir in subdirs_to_pack:
            dir_path = ROOT / subdir
            if not dir_path.exists():
                continue
            for root, dirs, files in os.walk(dir_path):
                # Filter dirs in-place
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
                for file in files:
                    file_path = Path(root) / file
                    if should_include_file(file_path):
                        rel_path = file_path.relative_to(ROOT)
                        arcname = f"{code_root}/{rel_path.as_posix()}"
                        zf.write(file_path, arcname=arcname)

    zip_size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"\nZIP file successfully created! Size: {zip_size_mb:.1f} MB ({time.time()-t0:.1f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
