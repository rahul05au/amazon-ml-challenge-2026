"""Field Inventory for Source1, Source2, and Source3 (Raw and Normalized)."""
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_SOURCES, norm_parquet

c = duckdb.connect()

print("================================================================================")
print("STEP 1: FIELD INVENTORY & SCHEMA DUMP")
print("================================================================================\n")

for src_name, raw_path in TRAIN_SOURCES.items():
    raw_p = str(raw_path).replace("\\", "/")
    print(f"\n========================================================")
    print(f"--- RAW TSV: {src_name.upper()} ({raw_p}) ---")
    print(f"========================================================")
    df_desc = c.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{raw_p}', delim='\t')").df()
    print("Columns & Types:")
    print(df_desc[["column_name", "column_type"]].to_string(index=False))
    
    total_rows = c.execute(f"SELECT count(*) FROM read_csv_auto('{raw_p}', delim='\t')").fetchone()[0]
    print(f"\nTotal Records: {total_rows:,}")
    
    null_cols = []
    for col in df_desc["column_name"]:
        non_null = c.execute(f"SELECT count(\"{col}\") FROM read_csv_auto('{raw_p}', delim='\t') WHERE \"{col}\" IS NOT NULL AND trim(cast(\"{col}\" as varchar)) != ''").fetchone()[0]
        sample = c.execute(f"SELECT \"{col}\" FROM read_csv_auto('{raw_p}', delim='\t') WHERE \"{col}\" IS NOT NULL AND trim(cast(\"{col}\" as varchar)) != '' LIMIT 1").fetchone()
        sample_val = str(sample[0]) if sample else "None"
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."
        pct = (non_null / total_rows) * 100
        null_cols.append({"Column": col, "Non-Null Count": f"{non_null:,}", "Fill Rate": f"{pct:.1f}%", "Sample Value": sample_val})
    
    print("\nField Fill Rates & Samples:")
    print(pd.DataFrame(null_cols).to_string(index=False))

    # Also inspect normalized parquet
    norm_p = str(norm_parquet("train", src_name)).replace("\\", "/")
    if Path(norm_p).exists():
        print(f"\n--- NORMALIZED PARQUET: {src_name.upper()} ---")
        df_norm_desc = c.execute(f"DESCRIBE SELECT * FROM read_parquet('{norm_p}')").df()
        print(df_norm_desc[["column_name", "column_type"]].to_string(index=False))

print("\n================================================================================")
print("STRUCTURED FIELD AVAILABILITY SUMMARY")
print("================================================================================")
