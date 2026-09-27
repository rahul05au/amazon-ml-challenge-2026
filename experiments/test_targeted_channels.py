import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import re
import time
import duckdb
import pandas as pd
from experiments.v12_1_candidate_expansion import build_learned_transliteration_dict

con = duckdb.connect()

h_expr_s1 = "abs(hash(s1.entity_id || 'v1')) % 1000"
h_expr = "abs(hash(entity_id || 'v1')) % 1000"
val1_cond_s1 = f"{h_expr_s1} >= 0 AND {h_expr_s1} < 5"
val1_cond = f"{h_expr} >= 0 AND {h_expr} < 5"

print("Loading ground truth and baseline V2 candidates...")
gt_df = con.execute(f"""
    WITH unnested AS (
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS sx_id
        FROM read_parquet('{str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")}')
        WHERE matched_entity_ids != ''
    )
    SELECT u.s1_id, u.sx_id
    FROM unnested u
    JOIN read_parquet('{str(ROOT / "intermediate/train/norm_source1.parquet").replace("\\", "/")}') s1 ON u.s1_id = s1.entity_id
    WHERE {val1_cond_s1}
      AND u.sx_id LIKE 'S2-%'
""").df()
gt_set = set(zip(gt_df["s1_id"], gt_df["sx_id"]))
total_gold = len(gt_set)

v2_df = pd.read_parquet(ROOT / "intermediate/v2_cache/v2_source2_val1.parquet", columns=["s1_id", "sx_id"])
v2_set = set(zip(v2_df["s1_id"], v2_df["sx_id"]))
missed_set = gt_set - v2_set
print(f"Total gold: {total_gold:,} | Existing V2 recall: {len(v2_set & gt_set)/total_gold*100:.2f}% ({len(v2_set & gt_set):,}/{total_gold:,}) | Missed: {len(missed_set):,}")

# Build transliteration dictionary
tmap = build_learned_transliteration_dict("source2")
tmap['प्रा'] = 'private'
tmap['लि'] = 'limited'
tmap['प्रा.'] = 'private'
tmap['लि.'] = 'limited'

# Register transliteration function in DuckDB
def py_translit(name):
    if not name:
        return ""
    words = re.sub(r'[^\w\s]', ' ', str(name).lower()).split()
    return " ".join([tmap.get(w, w) for w in words])

con.create_function("translit", py_translit, return_type="VARCHAR")

t0 = time.time()
print("Registering S1 and SX tables in DuckDB...")
con.execute(f"""
    CREATE TEMP TABLE s1_val AS
    SELECT entity_id AS s1_id, name_clean AS s1_name, country_clean AS country, addr_clean AS s1_addr,
           regexp_extract(lower(name_clean), '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS s1_t1,
           regexp_extract(lower(name_clean), '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS s1_t2,
           regexp_extract(lower(addr_clean), '(\\b\\d+[a-z]?\\b)', 1) AS s1_num
    FROM read_parquet('{str(ROOT / "intermediate/train/norm_source1.parquet").replace("\\", "/")}')
    WHERE {val1_cond}
""")

con.execute(f"""
    CREATE TEMP TABLE sx_val AS
    SELECT entity_id AS sx_id, raw_name AS sx_raw, country_clean AS country, addr_clean AS sx_addr,
           translit(raw_name) AS sx_translit,
           regexp_extract(lower(addr_clean), '(\\b\\d+[a-z]?\\b)', 1) AS sx_num
    FROM read_parquet('{str(ROOT / "intermediate/train/norm_source2.parquet").replace("\\", "/")}')
    WHERE country_clean IN (SELECT DISTINCT country FROM s1_val)
""")

print(f"Tables prepared in {time.time()-t0:.2f}s. Running fast join on transliterated 2-token prefix...")
t1 = time.time()
translit_matches = con.execute("""
    WITH sx_toks AS (
        SELECT sx_id, country,
               regexp_extract(sx_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS sx_t1,
               regexp_extract(sx_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS sx_t2
        FROM sx_val
        WHERE sx_translit != ''
    )
    SELECT s1.s1_id, sx.sx_id
    FROM s1_val s1
    JOIN sx_toks sx
      ON s1.country = sx.country
     AND s1.s1_t1 = sx.sx_t1
     AND s1.s1_t2 = sx.sx_t2
    WHERE s1.s1_t1 != '' AND s1.s1_t1 NOT IN ('the', 'and', 'company', 'enterprises', 'group')
""").df()
trans_set = set(zip(translit_matches["s1_id"], translit_matches["sx_id"]))
print(f"Transliterated 2-token join generated {len(trans_set):,} pairs in {time.time()-t1:.2f}s")

comb = v2_set | trans_set
rec = len(comb & gt_set) / total_gold
recov = len(comb & gt_set) - len(v2_set & gt_set)
added = len(comb) - len(v2_set)
print(f"V2 + Transliterated 2-token join:")
print(f"  Recall:             {rec*100:.2f}% (+{(rec - len(v2_set & gt_set)/total_gold)*100:.2f}%)")
print(f"  Gold pairs added:   +{recov:,} ({len(comb & gt_set):,}/{total_gold:,})")
print(f"  Added candidates:   +{added:,}")
if recov > 0:
    print(f"  Efficiency:         {added/recov:.1f} candidates per recovered gold")

con.close()
