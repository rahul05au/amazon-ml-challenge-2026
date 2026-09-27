from __future__ import annotations
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import duckdb
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import norm_parquet
NUM_RE = re.compile('\\b\\d+\\b')
POSTAL_RE = re.compile('\\b(\\d{5,6})\\b')
NON_LATIN_RE = re.compile('[^\\x00-\\x7F]')

def s1_norm_path() -> str:
    return str(norm_parquet('train', 'source1')).replace('\\', '/')

def sx_norm_path(src: str) -> str:
    return str(norm_parquet('train', src)).replace('\\', '/')

def gt_path() -> str:
    return str(ROOT / 'intermediate/train/ground_truth.parquet').replace('\\', '/')

def build_learned_transliteration_dict(src: str) -> dict[str, str]:
    print(f'Mining learned transliteration token dictionary from train_gate for {src}...')
    prefix = 'S2-' if src == 'source2' else 'S3-'
    h_expr = "abs(hash(s1.entity_id || 'v1')) % 1000"
    train_gate_cond = f'{h_expr} >= 10 AND {h_expr} < 25'
    con = duckdb.connect()
    aligned_pairs = con.execute(f"\n        WITH unnested AS (\n            SELECT source1_entity_id AS s1_id,\n                   trim(unnest(string_split(matched_entity_ids, ','))) AS sx_id\n            FROM read_parquet('{gt_path()}')\n            WHERE matched_entity_ids != ''\n        )\n        SELECT s1.name_clean AS s1_name, sx.raw_name AS sx_name\n        FROM unnested u\n        JOIN read_parquet('{s1_norm_path()}') s1 ON u.s1_id = s1.entity_id\n        JOIN read_parquet('{sx_norm_path(src)}') sx ON u.sx_id = sx.entity_id\n        WHERE {train_gate_cond}\n          AND u.sx_id LIKE '{prefix}%'\n    ").df()
    con.close()
    token_alignments: dict[str, Counter[str]] = defaultdict(Counter)
    for _, row in aligned_pairs.iterrows():
        s1_words = str(row['s1_name']).lower().split()
        sx_words = str(row['sx_name']).lower().split()
        if len(s1_words) == len(sx_words) and any((NON_LATIN_RE.search(w) for w in sx_words)):
            for sx_w, s1_w in zip(sx_words, s1_words):
                if NON_LATIN_RE.search(sx_w) and (not NON_LATIN_RE.search(s1_w)):
                    token_alignments[sx_w][s1_w] += 1
    translit_map = {}
    for non_latin_tok, counter in token_alignments.items():
        top_latin, count = counter.most_common(1)[0]
        if count >= 2:
            translit_map[non_latin_tok] = top_latin
    print(f'Learned {len(translit_map):,} high-confidence token transliterations (e.g. {list(translit_map.items())[:5]})')
    return translit_map

def transliterate_text(text: str, translit_map: dict[str, str]) -> str:
    words = str(text).lower().split()
    translated = [translit_map.get(w, w) for w in words]
    res = ' '.join(translated)
    res_ascii = unicodedata.normalize('NFKD', res).encode('ascii', 'ignore').decode('ascii')
    return res_ascii.strip()

def evaluate_candidate_expansion(src: str, split_name: str, translit_map: dict[str, str]):
    print(f"\n{'-' * 60}\nEVALUATING CANDIDATE EXPANSION: {src.upper()} | {split_name.upper()}\n{'-' * 60}")
    prefix = 'S2-' if src == 'source2' else 'S3-'
    h_expr_s1 = "abs(hash(s1.entity_id || 'v1')) % 1000"
    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == 'val1':
        where_cond_s1 = f'{h_expr_s1} >= 0 AND {h_expr_s1} < 5'
        where_cond = f'{h_expr} >= 0 AND {h_expr} < 5'
    elif split_name == 'val2':
        where_cond_s1 = f'{h_expr_s1} >= 25 AND {h_expr_s1} < 30'
        where_cond = f'{h_expr} >= 25 AND {h_expr} < 30'
    else:
        raise ValueError(f'Unknown split: {split_name}')
    con = duckdb.connect()
    gt_df = con.execute(f"\n        WITH unnested AS (\n            SELECT source1_entity_id AS s1_id,\n                   trim(unnest(string_split(matched_entity_ids, ','))) AS sx_id\n            FROM read_parquet('{gt_path()}')\n            WHERE matched_entity_ids != ''\n        )\n        SELECT s1_id, sx_id\n        FROM unnested u\n        JOIN read_parquet('{s1_norm_path()}') s1 ON u.s1_id = s1.entity_id\n        WHERE {where_cond_s1}\n          AND u.sx_id LIKE '{prefix}%'\n    ").df()
    gt_set = set(zip(gt_df['s1_id'], gt_df['sx_id']))
    total_gold = len(gt_set)
    cache_path = ROOT / f'intermediate/v2_cache/v2_{src}_{split_name}.parquet'
    v2_df = pd.read_parquet(cache_path, columns=['s1_id', 'sx_id'])
    v2_set = set(zip(v2_df['s1_id'], v2_df['sx_id']))
    v2_recall = len(v2_set & gt_set) / max(total_gold, 1)
    print(f'Existing V2 Candidates: {len(v2_set):,} | Gold covered: {len(v2_set & gt_set):,} / {total_gold:,} ({v2_recall * 100:.2f}%)')
    print('Testing Channel A: Postal + House Number Precision Blocker...')
    t0 = time.time()
    con.execute(f"\n        CREATE TEMP TABLE s1_records AS\n        SELECT entity_id AS s1_id, country_clean AS country, postal_code,\n               regexp_extract(addr_clean, '\\b(\\d+)\\b', 1) AS house_num\n        FROM read_parquet('{s1_norm_path()}')\n        WHERE {where_cond}\n          AND postal_code IS NOT NULL AND postal_code != ''\n    ")
    con.execute(f"\n        CREATE TEMP TABLE sx_records AS\n        SELECT entity_id AS sx_id, country_clean AS country, postal_code,\n               regexp_extract(addr_clean, '\\b(\\d+)\\b', 1) AS house_num\n        FROM read_parquet('{sx_norm_path(src)}')\n        WHERE postal_code IS NOT NULL AND postal_code != ''\n    ")
    addr_pairs = con.execute("\n        SELECT s1.s1_id, sx.sx_id\n        FROM s1_records s1\n        JOIN sx_records sx\n          ON s1.country = sx.country\n         AND s1.postal_code = sx.postal_code\n         AND s1.house_num = sx.house_num\n        WHERE s1.house_num IS NOT NULL AND s1.house_num != ''\n    ").df()
    addr_set = set(zip(addr_pairs['s1_id'], addr_pairs['sx_id']))
    print(f'  Channel A generated {len(addr_set):,} pairs in {time.time() - t0:.2f}s')
    comb_set = v2_set | addr_set
    comb_rec = len(comb_set & gt_set) / max(total_gold, 1)
    recovered = len(comb_set & gt_set) - len(v2_set & gt_set)
    added_cands = len(comb_set) - len(v2_set)
    efficiency = added_cands / max(recovered, 1)
    print(f'  V2 + Channel A Union:')
    print(f'    Recall:             {v2_recall * 100:.2f}% -> {comb_rec * 100:.2f}% (+{(comb_rec - v2_recall) * 100:.2f}%)')
    print(f'    Gold pairs added:   +{recovered:,}')
    print(f'    Total candidates:   {len(v2_set):,} -> {len(comb_set):,} (+{added_cands:,})')
    print(f'    False/Gold Ratio:   {efficiency:.1f} candidates per recovered gold')
    con.close()
    return {'src': src, 'split': split_name, 'v2_recall': v2_recall, 'new_recall': comb_rec, 'recovered_gold': recovered, 'added_candidates': added_cands, 'efficiency': efficiency}

def main():
    print('=' * 80)
    print('V12.1 STEP 3-6: CANDIDATE RECALL EXPANSION EXPERIMENT')
    print('=' * 80)
    for src in ['source2', 'source3']:
        translit_map = build_learned_transliteration_dict(src)
        evaluate_candidate_expansion(src, 'val1', translit_map)
        evaluate_candidate_expansion(src, 'val2', translit_map)
if __name__ == '__main__':
    main()
