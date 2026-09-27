"""Inspect missed gold pairs attributes in depth."""
import duckdb
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

con = duckdb.connect()

h_expr = "abs(hash(s1.entity_id || 'v1')) % 1000"
val1_cond = f"{h_expr} >= 0 AND {h_expr} < 5"

gt_df = con.execute(f"""
    WITH unnested AS (
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS sx_id
        FROM read_parquet('{str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")}')
        WHERE matched_entity_ids != ''
    )
    SELECT u.s1_id, u.sx_id,
           s1.raw_name AS s1_name, s1.addr_clean AS s1_addr, s1.country_clean AS s1_country, s1.postal_code AS s1_postal,
           sx.raw_name AS sx_name, sx.addr_clean AS sx_addr, sx.country_clean AS sx_country, sx.postal_code AS sx_postal
    FROM unnested u
    JOIN read_parquet('{str(ROOT / "intermediate/train/norm_source1.parquet").replace("\\", "/")}') s1 ON u.s1_id = s1.entity_id
    JOIN read_parquet('{str(ROOT / "intermediate/train/norm_source2.parquet").replace("\\", "/")}') sx ON u.sx_id = sx.entity_id
    WHERE {val1_cond}
      AND u.sx_id LIKE 'S2-%'
""").df()

v2_df = pd.read_parquet(ROOT / "intermediate/v2_cache/v2_source2_val1.parquet", columns=["s1_id", "sx_id"])
v2_set = set(zip(v2_df["s1_id"], v2_df["sx_id"]))

missed = gt_df[~gt_df.apply(lambda r: (r["s1_id"], r["sx_id"]) in v2_set, axis=1)]

out_lines = [f"Total gold in val1: {len(gt_df):,}, Missed: {len(missed):,}"]
for idx, (_, r) in enumerate(missed.head(50).iterrows()):
    out_lines.append(f"--- Pair {idx+1} ---")
    out_lines.append(f"S1: {r['s1_name']} | Country: {r['s1_country']} | Post: {r['s1_postal']} | Addr: {r['s1_addr']}")
    out_lines.append(f"SX: {r['sx_name']} | Country: {r['sx_country']} | Post: {r['sx_postal']} | Addr: {r['sx_addr']}")

(ROOT / "experiments/missed_pairs_sample.txt").write_text("\n".join(out_lines), encoding="utf-8")
print(f"Wrote 50 sample missed pairs to experiments/missed_pairs_sample.txt successfully.")
con.close()
