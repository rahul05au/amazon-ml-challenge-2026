from __future__ import annotations
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import get_clean_gold, compute_pairwise_features, engineer_competition_features, get_v12_feature_names, apply_target_exclusivity, s1_norm_path, sx_norm_path, MODELS_DIR, V2_CACHE
from experiments.v12_1_collective_matcher import compute_collective_sibling_features, get_v12_1_feature_names
from experiments.v12_1_candidate_expansion import build_learned_transliteration_dict
OCR_DIGIT_MAP = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '5': 's', '8': 'b'})

def normalize_clean(text: str) -> str:
    if not text:
        return ''
    norm = unicodedata.normalize('NFC', str(text).lower())
    cleaned = re.sub('[^\\w\\s]', ' ', norm)
    return re.sub('\\s+', ' ', cleaned).strip()

def normalize_ocr(text: str) -> str:
    return normalize_clean(text).translate(OCR_DIGIT_MAP)

def run_v13_retrieval_pairs(src: str, split_name: str) -> set[tuple[str, str]]:
    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == 'train_gate':
        where_cond = f'{h_expr} >= 10 AND {h_expr} < 25'
    elif split_name == 'val1':
        where_cond = f'{h_expr} >= 0 AND {h_expr} < 5'
    elif split_name == 'val2':
        where_cond = f'{h_expr} >= 25 AND {h_expr} < 30'
    else:
        raise ValueError(split_name)
    tmap = build_learned_transliteration_dict(src)
    tmap['प्रा'] = 'private'
    tmap['लि'] = 'limited'

    def py_translit(name):
        tokens = re.sub('[^\\w\\s]', ' ', str(name).lower()).split()
        return ' '.join([tmap.get(t, t) for t in tokens])
    con = duckdb.connect()
    con.create_function('translit', py_translit, return_type='VARCHAR')
    con.create_function('norm_clean', normalize_clean, return_type='VARCHAR')
    con.create_function('norm_ocr', normalize_ocr, return_type='VARCHAR')
    con.execute(f"\n        CREATE TEMP TABLE s1_records AS\n        SELECT entity_id AS s1_id, country_clean AS country,\n               norm_clean(raw_name) AS name_native,\n               norm_clean(translit(raw_name)) AS name_translit,\n               norm_ocr(raw_name) AS name_ocr,\n               norm_clean(addr_clean) AS addr_clean\n        FROM read_parquet('{s1_norm_path()}')\n        WHERE {where_cond}\n    ")
    con.execute(f"\n        CREATE TEMP TABLE sx_records AS\n        SELECT entity_id AS sx_id, country_clean AS country,\n               norm_clean(raw_name) AS name_native,\n               norm_clean(translit(raw_name)) AS name_translit,\n               norm_ocr(raw_name) AS name_ocr,\n               norm_clean(addr_clean) AS addr_clean\n        FROM read_parquet('{sx_norm_path(src)}')\n        WHERE country_clean IN (SELECT DISTINCT country FROM s1_records)\n    ")
    ocr_df = con.execute("\n        WITH s1_tokens AS (\n            SELECT s1_id, country,\n                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,\n                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2\n            FROM s1_records WHERE length(name_ocr) >= 5\n        ),\n        sx_tokens AS (\n            SELECT sx_id, country,\n                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,\n                   regexp_extract(name_ocr, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2\n            FROM sx_records WHERE length(name_ocr) >= 5\n        )\n        SELECT s1.s1_id, sx.sx_id\n        FROM s1_tokens s1 JOIN sx_tokens sx\n          ON s1.country = sx.country AND s1.t1 = sx.t1 AND s1.t2 = sx.t2\n        WHERE s1.t1 NOT IN ('the', 'and', 'company', 'enterprises', 'group', 'services')\n        QUALIFY row_number() OVER (PARTITION BY s1.s1_id) <= 5\n    ").df()
    trans_df = con.execute("\n        WITH s1_tokens AS (\n            SELECT s1_id, country,\n                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,\n                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2\n            FROM s1_records WHERE length(name_translit) >= 5\n        ),\n        sx_tokens AS (\n            SELECT sx_id, country,\n                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 1) AS t1,\n                   regexp_extract(name_translit, '^([a-z0-9]+)\\s+([a-z0-9]+)', 2) AS t2\n            FROM sx_records WHERE length(name_translit) >= 5\n        )\n        SELECT s1.s1_id, sx.sx_id\n        FROM s1_tokens s1 JOIN sx_tokens sx\n          ON s1.country = sx.country AND s1.t1 = sx.t1 AND s1.t2 = sx.t2\n        WHERE s1.t1 NOT IN ('the', 'and', 'company', 'enterprises', 'group', 'services')\n        QUALIFY row_number() OVER (PARTITION BY s1.s1_id) <= 5\n    ").df()
    addr_df = con.execute("\n        WITH s1_addr AS (\n            SELECT s1_id, country,\n                   regexp_extract(addr_clean, '(\\b\\d+[a-z]?\\b)', 1) AS num,\n                   regexp_extract(addr_clean, '(\\b[a-z]{5,}\\b)', 1) AS word\n            FROM s1_records WHERE addr_clean != ''\n        ),\n        sx_addr AS (\n            SELECT sx_id, country,\n                   regexp_extract(addr_clean, '(\\b\\d+[a-z]?\\b)', 1) AS num,\n                   regexp_extract(addr_clean, '(\\b[a-z]{5,}\\b)', 1) AS word\n            FROM sx_records WHERE addr_clean != ''\n        )\n        SELECT s1.s1_id, sx.sx_id\n        FROM s1_addr s1 JOIN sx_addr sx\n          ON s1.country = sx.country AND s1.num = sx.num AND s1.word = sx.word\n        WHERE s1.num != '' AND s1.word != ''\n          AND s1.word NOT IN ('street', 'avenue', 'nagar', 'floor', 'suite', 'india')\n        QUALIFY row_number() OVER (PARTITION BY s1.s1_id) <= 5\n    ").df()
    con.close()
    return set(zip(ocr_df['s1_id'], ocr_df['sx_id'])) | set(zip(trans_df['s1_id'], trans_df['sx_id'])) | set(zip(addr_df['s1_id'], addr_df['sx_id']))

def audit_and_evaluate_split(src: str, split_name: str):
    print(f"\n{'=' * 75}\nV13 DIAGNOSTIC AUDIT & SCORING: {src.upper()} | {split_name.upper()}\n{'=' * 75}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    total_gold = len(gt_set)
    val_cache_path = V2_CACHE / f'v2_{src}_{split_name}.parquet'
    v2_raw = pd.read_parquet(val_cache_path)
    old_cands = set(zip(v2_raw['s1_id'], v2_raw['sx_id']))
    old_gold_covered = old_cands & gt_set
    old_recall = len(old_gold_covered) / total_gold
    print('Generating new V13 multi-view retrieval pairs...')
    new_retrieved_pairs = run_v13_retrieval_pairs(src, split_name)
    added_pairs = new_retrieved_pairs - old_cands
    v13_cands = old_cands | new_retrieved_pairs
    new_gold_covered = v13_cands & gt_set
    recovered_gold = new_gold_covered - old_gold_covered
    added_non_gold = added_pairs - recovered_gold
    new_recall = len(new_gold_covered) / total_gold
    print(f'\n--- SECTION 1: CANDIDATE SET COMPARISON ---')
    print(f'  Old Candidate Count (V12.1): {len(old_cands):,}')
    print(f'  New Candidate Count (V13):   {len(v13_cands):,}')
    print(f'  Added Candidate Count:       +{len(added_pairs):,} (+{len(added_pairs) / len(old_cands) * 100:.2f}%)')
    print(f'  Recovered Gold Pairs:        +{len(recovered_gold):,}')
    print(f'  Added Non-Gold Pairs:        +{len(added_non_gold):,}')
    print(f'  Candidate Recall:            {old_recall * 100:.2f}% -> {new_recall * 100:.2f}% (+{(new_recall - old_recall) * 100:.2f}%)')
    if len(recovered_gold) > 0:
        print(f'  Efficiency:                  {len(added_pairs) / len(recovered_gold):.1f} candidates per recovered gold')
    print(f'\nFetching text attributes for {len(added_pairs):,} newly added candidate pairs...')
    con = duckdb.connect()
    added_df_input = pd.DataFrame(list(added_pairs), columns=['s1_id', 'sx_id'])
    con.register('added_pairs_tbl', added_df_input)
    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == 'val1':
        where_cond = f'{h_expr} >= 0 AND {h_expr} < 5'
    elif split_name == 'val2':
        where_cond = f'{h_expr} >= 25 AND {h_expr} < 30'
    else:
        raise ValueError(split_name)
    enriched_added = con.execute(f"\n        SELECT a.s1_id, a.sx_id,\n               0.0 AS lgb_score, 0.0 AS tfidf_score,\n               s1.name_clean AS s1_name, sx.name_clean AS sx_name,\n               s1.addr_clean AS s1_addr, sx.addr_clean AS sx_addr,\n               s1.postal_code AS s1_post, sx.postal_code AS sx_post\n        FROM added_pairs_tbl a\n        JOIN read_parquet('{s1_norm_path()}') s1 ON a.s1_id = s1.entity_id\n        JOIN read_parquet('{sx_norm_path(src)}') sx ON a.sx_id = sx.entity_id\n    ").df()
    con.close()
    v13_full_raw = pd.concat([v2_raw, enriched_added], ignore_index=True)
    print(f'Combined candidate graph: {len(v13_full_raw):,} pairs')
    print('Computing pairwise features on complete V13 candidate graph...')
    t0 = time.time()
    pair_feats = compute_pairwise_features(v13_full_raw)
    v13_full = pd.concat([v13_full_raw, pair_feats], axis=1)
    v13_full = engineer_competition_features(v13_full)
    print(f'Pairwise and competition features computed in {time.time() - t0:.1f}s')
    stage1_model = lgb.Booster(model_file=str(MODELS_DIR / f'v12_1_stage1_{src}.txt'))
    stage2_model = lgb.Booster(model_file=str(MODELS_DIR / f'v12_1_stage2_{src}.txt'))
    base_cols = get_v12_feature_names()
    stage2_cols = get_v12_1_feature_names()
    print('Scoring Stage-1 on complete V13 candidate graph...')
    v13_full['stage1_score'] = stage1_model.predict(v13_full[base_cols])
    print('Computing collective sibling features on complete V13 candidate graph...')
    col_feats = compute_collective_sibling_features(v13_full, score_col='stage1_score')
    v13_full = pd.concat([v13_full, col_feats], axis=1)
    print('Scoring Stage-2 on complete V13 candidate graph...')
    v13_full['stage2_score'] = stage2_model.predict(v13_full[stage2_cols])
    v13_full['final_score'] = 0.2 * v13_full['stage1_score'] + 0.8 * v13_full['stage2_score']
    print(f'\n--- SECTION 2: TRACING RECOVERED GOLD PAIRS ---')
    recov_set = recovered_gold
    gold_mask = v13_full.apply(lambda r: (r['s1_id'], r['sx_id']) in recov_set, axis=1)
    recov_df = v13_full[gold_mask].copy()
    old_gold_mask = v13_full.apply(lambda r: (r['s1_id'], r['sx_id']) in old_gold_covered, axis=1)
    old_gold_df = v13_full[old_gold_mask].copy()
    print(f'Total recovered gold pairs traced: {len(recov_df):,}')
    print(f'Recovered Gold Score Distribution:')
    print(f"  Stage-1 Mean Score: {recov_df['stage1_score'].mean():.4f} (Old Gold Mean: {old_gold_df['stage1_score'].mean():.4f})")
    print(f"  Stage-2 Mean Score: {recov_df['stage2_score'].mean():.4f} (Old Gold Mean: {old_gold_df['stage2_score'].mean():.4f})")
    print(f"  Final Blended Mean: {recov_df['final_score'].mean():.4f} (Old Gold Mean: {old_gold_df['final_score'].mean():.4f})")
    print(f"  Fraction with Stage-2 Score >= 0.40: {(recov_df['stage2_score'] >= 0.4).mean() * 100:.1f}%")
    print(f"  Fraction with Stage-2 Score >= 0.47: {(recov_df['stage2_score'] >= 0.47).mean() * 100:.1f}%")
    th = 0.47
    passed_th = recov_df[recov_df['final_score'] >= th]
    print(f'\nRecovered Gold Decoder Breakdown (at threshold {th:.2f}):')
    print(f'  A. Passed Threshold:       {len(passed_th):,} / {len(recov_df):,}')
    print(f'  B. Rejected by Threshold:   {len(recov_df) - len(passed_th):,} / {len(recov_df):,}')
    cand_preds = v13_full[v13_full['final_score'] >= th][['s1_id', 'sx_id', 'final_score']].copy()
    cand_preds['v12_score'] = cand_preds['final_score']
    cand_excl = apply_target_exclusivity(cand_preds, target_margin_thresh=0.0)
    v13_pred_set = set(zip(cand_excl['s1_id'], cand_excl['sx_id']))
    recov_accepted = recov_set & v13_pred_set
    recov_rejected_by_excl = set(zip(passed_th['s1_id'], passed_th['sx_id'])) - v13_pred_set
    print(f'  C. Accepted into Final Submission: {len(recov_accepted):,} / {len(recov_df):,}')
    print(f'  D. Rejected by Target Exclusivity: {len(recov_rejected_by_excl):,} / {len(recov_df):,}')
    old_full = v13_full.iloc[:len(v2_raw)].copy()
    old_cand_preds = old_full[old_full['final_score'] >= th][['s1_id', 'sx_id', 'final_score']].copy()
    old_cand_preds['v12_score'] = old_cand_preds['final_score']
    old_cand_excl = apply_target_exclusivity(old_cand_preds, target_margin_thresh=0.0)
    v12_pred_set = set(zip(old_cand_excl['s1_id'], old_cand_excl['sx_id']))
    unique_v12 = v12_pred_set - v13_pred_set
    unique_v13 = v13_pred_set - v12_pred_set
    common_preds = v12_pred_set & v13_pred_set
    true_pairs_added = unique_v13 & gt_set
    false_pos_added = unique_v13 - gt_set
    print(f'\n--- SECTION 3: PREDICTION SETS COMPARISON ---')
    print(f'  V12.1 Predicted Pairs: {len(v12_pred_set):,}')
    print(f'  V13 Predicted Pairs:   {len(v13_pred_set):,}')
    print(f'  Intersection Size:     {len(common_preds):,}')
    print(f'  Unique to V12.1:       {len(unique_v12):,}')
    print(f'  Unique to V13:         {len(unique_v13):,}')
    print(f'  Recovered True Pairs:  +{len(true_pairs_added):,}')
    print(f'  False Positives Added: +{len(false_pos_added):,}')
    m_old = calculate_macro_f05(old_cand_excl, gt_df, s1_all)
    m_v13 = calculate_macro_f05(cand_excl, gt_df, s1_all)
    print(f'\n--- SECTION 4: CANONICAL METRICS ---')
    print(f"  Old Candidate Graph: Macro F0.5 = {m_old['macro_f05']:.5f} (P={m_old['macro_precision']:.4f}, R={m_old['macro_recall']:.4f})")
    print(f"  V13 Expanded Graph:  Macro F0.5 = {m_v13['macro_f05']:.5f} (P={m_v13['macro_precision']:.4f}, R={m_v13['macro_recall']:.4f})")
    print(f"  Net Macro F0.5 Delta: {m_v13['macro_f05'] - m_old['macro_f05']:+.5f}")
    return {'src': src, 'split': split_name, 'old_candidates': len(old_cands), 'new_candidates': len(v13_cands), 'recovered_gold': len(recovered_gold), 'gold_passed_threshold': len(passed_th), 'gold_accepted_final': len(recov_accepted), 'gold_rejected_by_threshold': len(recov_df) - len(passed_th), 'gold_rejected_by_exclusivity': len(recov_rejected_by_excl), 'true_pairs_added': len(true_pairs_added), 'false_positives_added': len(false_pos_added), 'macro_f05_old': m_old['macro_f05'], 'macro_f05_v13': m_v13['macro_f05']}

def main():
    res = audit_and_evaluate_split('source2', 'val1')
    out_file = ROOT / 'experiments' / 'v13_audit_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
    print(f'\nAudit complete. Saved to {out_file}')
if __name__ == '__main__':
    main()
