from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
import duckdb
from src.config import candidate_parquet, norm_parquet

@dataclass
class PassDiagnostics:
    pass_name: str
    candidate_count: int = 0
    s1_entities_with_candidates: int = 0
    avg_candidates_per_s1: float = 0.0
    p95_candidates_per_s1: float = 0.0
    max_candidates_per_s1: int = 0
    elapsed_seconds: float = 0.0

@dataclass
class BlockingDiagnostics:
    target_source: str
    split: str
    passes: list[PassDiagnostics] = field(default_factory=list)
    union_candidate_count: int = 0
    union_s1_with_candidates: int = 0
    union_avg_per_s1: float = 0.0
    union_p95_per_s1: float = 0.0
    union_max_per_s1: int = 0
    total_elapsed: float = 0.0

def _compute_pass_stats(con: duckdb.DuckDBPyConnection, table: str, pass_name: str, elapsed: float) -> PassDiagnostics:
    stats = con.execute(f'\n        SELECT\n            count(*)                             AS total,\n            count(DISTINCT s1_id)                AS s1_count,\n            avg(cnt)                             AS avg_cnt,\n            percentile_disc(0.95) WITHIN GROUP (ORDER BY cnt) AS p95,\n            max(cnt)                             AS max_cnt\n        FROM (\n            SELECT s1_id, count(*) AS cnt\n            FROM {table}\n            GROUP BY s1_id\n        ) t\n    ').fetchone()
    total = stats[0] or 0
    s1_count = stats[1] or 0
    avg_cnt = stats[2] or 0.0
    p95 = stats[3] or 0.0
    max_cnt = stats[4] or 0
    return PassDiagnostics(pass_name=pass_name, candidate_count=total, s1_entities_with_candidates=s1_count, avg_candidates_per_s1=round(avg_cnt, 2), p95_candidates_per_s1=round(p95, 2), max_candidates_per_s1=max_cnt, elapsed_seconds=round(elapsed, 2))

def _pass_a_compact_name(con: duckdb.DuckDBPyConnection, s1_view: str, sx_view: str) -> str:
    table = '_pass_a'
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE {table} AS\n        SELECT s1.entity_id AS s1_id, sx.entity_id AS sx_id\n        FROM {s1_view} s1\n        JOIN {sx_view} sx\n          ON replace(strip_accents(regexp_replace(s1.name_no_suffix, '\\b(com|org|net|in|co)\\b', '', 'g')), ' ', '') =\n             replace(strip_accents(regexp_replace(sx.name_no_suffix, '\\b(com|org|net|in|co)\\b', '', 'g')), ' ', '')\n         AND s1.country_clean = sx.country_clean\n        WHERE s1.name_no_suffix != '' AND length(s1.name_no_suffix) > 2\n    ")
    return table

def _pass_b_rare_name_token(con: duckdb.DuckDBPyConnection, s1_view: str, sx_view: str, max_freq: int=150) -> str:
    table = '_pass_b'
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE _s1_name_tokens AS\n        SELECT entity_id, country_clean, unnest(string_split(strip_accents(name_no_suffix), ' ')) AS token\n        FROM {s1_view}\n        WHERE name_no_suffix != ''\n    ")
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE _sx_name_tokens AS\n        SELECT entity_id, country_clean, unnest(string_split(strip_accents(name_no_suffix), ' ')) AS token\n        FROM {sx_view}\n        WHERE name_no_suffix != ''\n    ")
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE _rare_name_tokens AS\n        SELECT token, count(*) AS sx_cnt\n        FROM _sx_name_tokens\n        WHERE length(token) >= 3\n          AND token NOT IN ('the', 'and', 'for', 'inc', 'llc', 'ltd', 'pvt', 'corp', 'all', 'any', 'one', 'new')\n        GROUP BY token\n        HAVING count(*) <= {max_freq}\n    ")
    con.execute(f'\n        CREATE OR REPLACE TEMP TABLE {table} AS\n        SELECT DISTINCT s1t.entity_id AS s1_id, sxt.entity_id AS sx_id\n        FROM _s1_name_tokens s1t\n        JOIN _rare_name_tokens rt ON s1t.token = rt.token\n        JOIN _sx_name_tokens sxt ON s1t.token = sxt.token AND s1t.country_clean = sxt.country_clean\n        WHERE length(s1t.token) >= 3\n    ')
    return table

def _pass_c1_exact_address(con: duckdb.DuckDBPyConnection, s1_view: str, sx_view: str) -> str:
    table = '_pass_c1'
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE {table} AS\n        SELECT s1.entity_id AS s1_id, sx.entity_id AS sx_id\n        FROM {s1_view} s1\n        JOIN {sx_view} sx\n          ON s1.addr_clean = sx.addr_clean\n         AND s1.country_clean = sx.country_clean\n        WHERE s1.addr_clean != '' AND length(s1.addr_clean) > 8\n    ")
    return table

def _pass_c2_street_number_name(con: duckdb.DuckDBPyConnection, s1_view: str, sx_view: str) -> str:
    table = '_pass_c2'
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE {table} AS\n        SELECT s1.entity_id AS s1_id, sx.entity_id AS sx_id\n        FROM {s1_view} s1\n        JOIN {sx_view} sx\n          ON regexp_extract(s1.addr_clean, '\\b(\\d+)\\b', 1) = regexp_extract(sx.addr_clean, '\\b(\\d+)\\b', 1)\n         AND regexp_extract(s1.addr_clean, '\\b(\\d+)\\b\\s+([a-z]+)', 2) = regexp_extract(sx.addr_clean, '\\b(\\d+)\\b\\s+([a-z]+)', 2)\n         AND s1.country_clean = sx.country_clean\n        WHERE regexp_extract(s1.addr_clean, '\\b(\\d+)\\b', 1) != ''\n          AND regexp_extract(s1.addr_clean, '\\b(\\d+)\\b\\s+([a-z]+)', 2) NOT IN (\n              '', 'po', 'box', 'suite', 'floor', 'pvt', 'ltd', 'road', 'street', 'avenue', 'lane'\n          )\n          AND length(regexp_extract(s1.addr_clean, '\\b(\\d+)\\b\\s+([a-z]+)', 2)) >= 4\n    ")
    return table

def _pass_c3_rare_address_token(con: duckdb.DuckDBPyConnection, s1_view: str, sx_view: str, max_freq: int=40) -> str:
    table = '_pass_c3'
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE _s1_addr_tokens AS\n        SELECT entity_id, country_clean, unnest(string_split(addr_clean, ' ')) AS token\n        FROM {s1_view}\n        WHERE addr_clean != ''\n    ")
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE _sx_addr_tokens AS\n        SELECT entity_id, country_clean, unnest(string_split(addr_clean, ' ')) AS token\n        FROM {sx_view}\n        WHERE addr_clean != ''\n    ")
    con.execute(f"\n        CREATE OR REPLACE TEMP TABLE _rare_addr_tokens AS\n        SELECT token, count(*) AS cnt\n        FROM _sx_addr_tokens\n        WHERE length(token) >= 5\n          AND token NOT IN (\n              'street', 'avenue', 'road', 'suite', 'floor', 'building', 'highway', 'parkway',\n              'india', 'delhi', 'mumbai', 'bangalore', 'chennai', 'hyderabad', 'texas',\n              'california', 'florida', 'north', 'south', 'nagar', 'colony', 'block',\n              'sector', 'village', 'pradesh', 'karnataka', 'maharashtra', 'tamil',\n              'district', 'county'\n          )\n        GROUP BY token\n        HAVING count(*) BETWEEN 1 AND {max_freq}\n    ")
    con.execute(f'\n        CREATE OR REPLACE TEMP TABLE {table} AS\n        SELECT DISTINCT s1t.entity_id AS s1_id, sxt.entity_id AS sx_id\n        FROM _s1_addr_tokens s1t\n        JOIN _rare_addr_tokens df ON s1t.token = df.token\n        JOIN _sx_addr_tokens sxt ON s1t.token = sxt.token AND s1t.country_clean = sxt.country_clean\n    ')
    return table

def run_blocking(split: str='train', target_source: str='source2') -> BlockingDiagnostics:
    s1_pq = norm_parquet(split, 'source1')
    sx_pq = norm_parquet(split, target_source)
    if not s1_pq.exists():
        raise FileNotFoundError(f'Normalised S1 not found: {s1_pq}')
    if not sx_pq.exists():
        raise FileNotFoundError(f'Normalised {target_source} not found: {sx_pq}')
    con = duckdb.connect()
    con.execute('SET threads TO 8')
    con.execute("SET memory_limit = '8GB'")
    s1_safe = str(s1_pq).replace('\\', '/')
    sx_safe = str(sx_pq).replace('\\', '/')
    con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{s1_safe}')")
    con.execute(f"CREATE VIEW sx AS SELECT * FROM read_parquet('{sx_safe}')")
    s1_count = con.execute('SELECT count(*) FROM s1').fetchone()[0]
    sx_count = con.execute('SELECT count(*) FROM sx').fetchone()[0]
    print(f"\n{'=' * 60}")
    print(f'  BLOCKING: S1 ({s1_count:,}) -> {target_source} ({sx_count:,}) [{split}]')
    print(f"{'=' * 60}")
    diag = BlockingDiagnostics(target_source=target_source, split=split)
    total_t0 = time.perf_counter()
    print('  Pass A  -- compact unaccented name ...')
    t0 = time.perf_counter()
    pass_a = _pass_a_compact_name(con, 's1', 'sx')
    elapsed_a = time.perf_counter() - t0
    stats_a = _compute_pass_stats(con, pass_a, 'pass_a', elapsed_a)
    diag.passes.append(stats_a)
    print(f'    -> {stats_a.candidate_count:,} candidates, {stats_a.s1_entities_with_candidates:,} S1 entities  ({elapsed_a:.1f}s)')
    print('  Pass B  -- rare name tokens ...')
    t0 = time.perf_counter()
    pass_b = _pass_b_rare_name_token(con, 's1', 'sx')
    elapsed_b = time.perf_counter() - t0
    stats_b = _compute_pass_stats(con, pass_b, 'pass_b', elapsed_b)
    diag.passes.append(stats_b)
    print(f'    -> {stats_b.candidate_count:,} candidates, {stats_b.s1_entities_with_candidates:,} S1 entities  ({elapsed_b:.1f}s)')
    print('  Pass C1 -- exact address ...')
    t0 = time.perf_counter()
    pass_c1 = _pass_c1_exact_address(con, 's1', 'sx')
    elapsed_c1 = time.perf_counter() - t0
    stats_c1 = _compute_pass_stats(con, pass_c1, 'pass_c1', elapsed_c1)
    diag.passes.append(stats_c1)
    print(f'    -> {stats_c1.candidate_count:,} candidates, {stats_c1.s1_entities_with_candidates:,} S1 entities  ({elapsed_c1:.1f}s)')
    print('  Pass C2 -- street number + street token ...')
    t0 = time.perf_counter()
    pass_c2 = _pass_c2_street_number_name(con, 's1', 'sx')
    elapsed_c2 = time.perf_counter() - t0
    stats_c2 = _compute_pass_stats(con, pass_c2, 'pass_c2', elapsed_c2)
    diag.passes.append(stats_c2)
    print(f'    -> {stats_c2.candidate_count:,} candidates, {stats_c2.s1_entities_with_candidates:,} S1 entities  ({elapsed_c2:.1f}s)')
    print('  Pass C3 -- rare address tokens ...')
    t0 = time.perf_counter()
    pass_c3 = _pass_c3_rare_address_token(con, 's1', 'sx')
    elapsed_c3 = time.perf_counter() - t0
    stats_c3 = _compute_pass_stats(con, pass_c3, 'pass_c3', elapsed_c3)
    diag.passes.append(stats_c3)
    print(f'    -> {stats_c3.candidate_count:,} candidates, {stats_c3.s1_entities_with_candidates:,} S1 entities  ({elapsed_c3:.1f}s)')
    print('  Unioning all candidate passes ...')
    con.execute(f'\n        CREATE OR REPLACE TEMP TABLE _all_candidates AS\n        SELECT s1_id, sx_id FROM {pass_a}\n        UNION\n        SELECT s1_id, sx_id FROM {pass_b}\n        UNION\n        SELECT s1_id, sx_id FROM {pass_c1}\n        UNION\n        SELECT s1_id, sx_id FROM {pass_c2}\n        UNION\n        SELECT s1_id, sx_id FROM {pass_c3}\n    ')
    union_stats = _compute_pass_stats(con, '_all_candidates', 'union', time.perf_counter() - total_t0)
    diag.union_candidate_count = union_stats.candidate_count
    diag.union_s1_with_candidates = union_stats.s1_entities_with_candidates
    diag.union_avg_per_s1 = union_stats.avg_candidates_per_s1
    diag.union_p95_per_s1 = union_stats.p95_candidates_per_s1
    diag.union_max_per_s1 = union_stats.max_candidates_per_s1
    out_path = candidate_parquet(split, target_source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe_out = str(out_path).replace('\\', '/')
    con.execute(f"\n        COPY (\n            SELECT s1_id, sx_id FROM _all_candidates\n            ORDER BY s1_id, sx_id\n        ) TO '{safe_out}' (FORMAT PARQUET, COMPRESSION ZSTD)\n    ")
    diag.total_elapsed = round(time.perf_counter() - total_t0, 2)
    con.close()
    print(f'\n  +------------------------------------------+')
    print(f'  | UNION CANDIDATES:  {diag.union_candidate_count:>10,}')
    print(f'  | S1 with candidates:{diag.union_s1_with_candidates:>10,}')
    print(f'  | Avg cands/S1:      {diag.union_avg_per_s1:>10.1f}')
    print(f'  | P95 cands/S1:      {diag.union_p95_per_s1:>10.0f}')
    print(f'  | Max cands/S1:      {diag.union_max_per_s1:>10,}')
    print(f'  | Total Time:        {diag.total_elapsed:>10.1f}s')
    print(f'  +------------------------------------------+')
    print(f'  Saved candidate Parquet to: {out_path.name}')
    diag_path = out_path.parent / f'blocking_diag_{target_source}.json'
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(diag), f, indent=2)
    print(f'  Saved diagnostics JSON to: {diag_path.name}')
    return diag

def main() -> int:
    parser = argparse.ArgumentParser(description='Multi-pass blocking candidate generation.')
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--target', choices=['source2', 'source3'], default=None, help='Target source to block against (default: both)')
    args = parser.parse_args()
    targets = [args.target] if args.target else ['source2', 'source3']
    for target in targets:
        run_blocking(args.split, target)
    return 0
if __name__ == '__main__':
    sys.exit(main())
