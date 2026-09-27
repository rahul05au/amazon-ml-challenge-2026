from __future__ import annotations
from pathlib import Path
import duckdb
ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT.parents[1]
CANDIDATE_PATH = PACKAGE_ROOT / 'output' / 'candidate_pairs.tsv'
INTERMEDIATE = ROOT / 'intermediate' / 'test'

def main() -> None:
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f'Candidate file not found: {CANDIDATE_PATH}')
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    candidate_path = CANDIDATE_PATH.as_posix()
    con = duckdb.connect()
    con.execute('SET threads TO 8')
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"\n        CREATE TEMP TABLE candidates AS\n        SELECT source1_entity_id AS s1_id,\n               trim(unnest(string_split(candidate_entity_ids, ','))) AS sx_id\n        FROM read_csv('{candidate_path}', delim='\\t', header=true, quote='', escape='')\n        WHERE candidate_entity_ids IS NOT NULL AND candidate_entity_ids <> ''\n    ")
    for source, prefix in (('source2', 'S2-'), ('source3', 'S3-')):
        output = (INTERMEDIATE / f'matches_{source}.parquet').as_posix()
        con.execute(f"\n            COPY (\n                SELECT s1_id, sx_id\n                FROM candidates\n                WHERE starts_with(sx_id, '{prefix}')\n            ) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)\n        ")
        count = con.execute(f"SELECT count(*) FROM read_parquet('{output}')").fetchone()[0]
        print(f'{source}: {count:,} candidate pairs -> {output}')
    con.close()
if __name__ == '__main__':
    main()
