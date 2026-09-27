from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import norm_parquet
from src.matcher import calculate_macro_f05
MODELS_DIR = ROOT / 'models'
V2_CACHE = ROOT / 'intermediate' / 'v2_cache'
OPERATIONAL_THRESHOLDS = {'source2': 0.7, 'source3': 0.65}
NUM_RE = re.compile('\\b\\d+\\b')
POSTAL_RE = re.compile('\\b(\\d{5,6})\\b')

def s1_norm_path() -> str:
    return str(norm_parquet('train', 'source1')).replace('\\', '/')

def sx_norm_path(src: str) -> str:
    return str(norm_parquet('train', src)).replace('\\', '/')

def gt_path() -> str:
    return str(ROOT / 'intermediate/train/ground_truth.parquet').replace('\\', '/')

def get_clean_gold(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    prefix = 'S2-' if src == 'source2' else 'S3-'
    h_expr = "abs(hash(entity_id || 'v1')) % 1000"
    if split_name == 'train_gate':
        where_cond = f'{h_expr} >= 10 AND {h_expr} < 25'
    elif split_name == 'val1':
        where_cond = f'{h_expr} >= 0 AND {h_expr} < 5'
    elif split_name == 'val2':
        where_cond = f'{h_expr} >= 25 AND {h_expr} < 30'
    else:
        raise ValueError(f'Unknown split: {split_name}')
    con = duckdb.connect()
    con.execute(f"\n        CREATE TEMP TABLE s1_split AS\n        SELECT entity_id AS s1_id\n        FROM read_parquet('{s1_norm_path()}')\n        WHERE {where_cond}\n    ")
    s1_all = set(con.execute('SELECT s1_id FROM s1_split').df()['s1_id'])
    gt_df = con.execute(f"\n        WITH unnested AS (\n            SELECT source1_entity_id AS s1_id,\n                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id\n            FROM read_parquet('{gt_path()}')\n            WHERE matched_entity_ids != ''\n              AND source1_entity_id IN (SELECT s1_id FROM s1_split)\n        )\n        SELECT s1_id, true_match_id\n        FROM unnested\n        WHERE true_match_id LIKE '{prefix}%'\n    ").df()
    con.close()
    gt_set = set(zip(gt_df['s1_id'], gt_df['true_match_id']))
    return (s1_all, gt_set, gt_df)

def compute_pairwise_features(df: pd.DataFrame) -> pd.DataFrame:
    s1_names = df['s1_name'].fillna('').astype(str).tolist()
    sx_names = df['sx_name'].fillna('').astype(str).tolist()
    s1_addrs = df['s1_addr'].fillna('').astype(str).tolist()
    sx_addrs = df['sx_addr'].fillna('').astype(str).tolist()
    s1_posts = df['s1_post'].fillna('').astype(str).tolist()
    sx_posts = df['sx_post'].fillna('').astype(str).tolist()
    n_rows = len(df)
    feats = {}
    s1_name_tok_lists = [s.split() for s in s1_names]
    sx_name_tok_lists = [s.split() for s in sx_names]
    s1_name_tok_sets = [set(t) for t in s1_name_tok_lists]
    sx_name_tok_sets = [set(t) for t in sx_name_tok_lists]
    s1_addr_tok_lists = [s.split() for s in s1_addrs]
    sx_addr_tok_lists = [s.split() for s in sx_addrs]
    s1_addr_tok_sets = [set(t) for t in s1_addr_tok_lists]
    sx_addr_tok_sets = [set(t) for t in sx_addr_tok_lists]
    s1_num_sets = [set(NUM_RE.findall(s)) for s in s1_addrs]
    sx_num_sets = [set(NUM_RE.findall(s)) for s in sx_addrs]
    s1_first_nums = [NUM_RE.search(s).group(0) if NUM_RE.search(s) else '' for s in s1_addrs]
    sx_first_nums = [NUM_RE.search(s).group(0) if NUM_RE.search(s) else '' for s in sx_addrs]
    feats['name_exact'] = np.array([1.0 if n1 and n1 == n2 else 0.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['name_ratio'] = np.array([fuzz.ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['name_wratio'] = np.array([fuzz.WRatio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['name_jaro_winkler'] = np.array([distance.JaroWinkler.similarity(n1, n2) for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['name_levenshtein'] = np.array([distance.Levenshtein.normalized_similarity(n1, n2) for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['name_token_sort'] = np.array([fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['name_token_set'] = np.array([fuzz.token_set_ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    name_jacc = []
    for t1, t2 in zip(s1_name_tok_sets, sx_name_tok_sets):
        union_len = len(t1 | t2)
        name_jacc.append(len(t1 & t2) / union_len if union_len else 0.0)
    feats['name_token_jaccard'] = np.array(name_jacc, dtype=np.float32)
    name_3g_jacc = []
    for n1, n2 in zip(s1_names, sx_names):
        g1 = {n1[i:i + 3] for i in range(len(n1) - 2)} if len(n1) >= 3 else {n1}
        g2 = {n2[i:i + 3] for i in range(len(n2) - 2)} if len(n2) >= 3 else {n2}
        union_len = len(g1 | g2)
        name_3g_jacc.append(len(g1 & g2) / union_len if union_len else 0.0)
    feats['name_char_3gram_jaccard'] = np.array(name_3g_jacc, dtype=np.float32)
    s1_nlens = [len(n) for n in s1_names]
    sx_nlens = [len(n) for n in sx_names]
    feats['name_len_diff'] = np.array([abs(l1 - l2) for l1, l2 in zip(s1_nlens, sx_nlens)], dtype=np.float32)
    feats['name_len_ratio'] = np.array([min(l1, l2) / max(l1, l2, 1) for l1, l2 in zip(s1_nlens, sx_nlens)], dtype=np.float32)
    feats['name_first_token_match'] = np.array([1.0 if t1 and t2 and (t1[0] == t2[0]) else 0.0 for t1, t2 in zip(s1_name_tok_lists, sx_name_tok_lists)], dtype=np.float32)
    feats['name_first_token_conflict'] = np.array([1.0 if t1 and t2 and (t1[0] != t2[0]) else 0.0 for t1, t2 in zip(s1_name_tok_lists, sx_name_tok_lists)], dtype=np.float32)
    feats['name_translit_sim'] = np.array([fuzz.partial_ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    feats['addr_exact'] = np.array([1.0 if a1 and a1 == a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    feats['addr_ratio'] = np.array([fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    feats['addr_jaro_winkler'] = np.array([distance.JaroWinkler.similarity(a1, a2) if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    feats['addr_levenshtein'] = np.array([distance.Levenshtein.normalized_similarity(a1, a2) if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    feats['addr_token_sort'] = np.array([fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    feats['addr_token_set'] = np.array([fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    addr_jacc = []
    for a1, a2 in zip(s1_addr_tok_sets, sx_addr_tok_sets):
        union_len = len(a1 | a2)
        addr_jacc.append(len(a1 & a2) / union_len if union_len else 0.0)
    feats['addr_token_jaccard'] = np.array(addr_jacc, dtype=np.float32)
    feats['addr_num_shared'] = np.array([len(n1 & n2) for n1, n2 in zip(s1_num_sets, sx_num_sets)], dtype=np.float32)
    feats['addr_house_num_exact'] = np.array([1.0 if fn1 and fn1 == fn2 else 0.0 if fn1 and fn2 else -1.0 for fn1, fn2 in zip(s1_first_nums, sx_first_nums)], dtype=np.float32)
    feats['addr_house_num_conflict'] = np.array([1.0 if fn1 and fn2 and (fn1 != fn2) else 0.0 for fn1, fn2 in zip(s1_first_nums, sx_first_nums)], dtype=np.float32)
    feats['postal_exact'] = np.array([1.0 if p1 and p2 and (p1 == p2) else 0.0 if p1 and p2 else -1.0 for p1, p2 in zip(s1_posts, sx_posts)], dtype=np.float32)
    feats['postal_prefix_match'] = np.array([1.0 if p1 and p2 and (p1[:3] == p2[:3]) else 0.0 if p1 and p2 else -1.0 for p1, p2 in zip(s1_posts, sx_posts)], dtype=np.float32)
    feats['addr_present_both'] = np.array([1.0 if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    s1_alens = [len(a) for a in s1_addrs]
    sx_alens = [len(a) for a in sx_addrs]
    feats['addr_len_ratio'] = np.array([min(l1, l2) / max(l1, l2, 1) for l1, l2 in zip(s1_alens, sx_alens)], dtype=np.float32)
    feats['name_addr_prod'] = feats['name_wratio'] * feats['addr_ratio']
    feats['name_addr_sum'] = feats['name_wratio'] + feats['addr_ratio']
    feats['name_addr_max'] = np.maximum(feats['name_wratio'], feats['addr_ratio'])
    feats['name_addr_min'] = np.minimum(feats['name_wratio'], feats['addr_ratio'])
    feats['name_exact_addr_missing'] = np.array([1.0 if ne == 1.0 and (not a1 or not a2) else 0.0 for ne, a1, a2 in zip(feats['name_exact'], s1_addrs, sx_addrs)], dtype=np.float32)
    feats['name_addr_combined_sim'] = np.array([fuzz.ratio(n1 + ' ' + a1, n2 + ' ' + a2) / 100.0 for n1, a1, n2, a2 in zip(s1_names, s1_addrs, sx_names, sx_addrs)], dtype=np.float32)
    feats['name_toks_only_s1'] = np.array([len(t1 - t2) for t1, t2 in zip(s1_name_tok_sets, sx_name_tok_sets)], dtype=np.float32)
    feats['name_toks_only_sx'] = np.array([len(t2 - t1) for t1, t2 in zip(s1_name_tok_sets, sx_name_tok_sets)], dtype=np.float32)
    feats['addr_toks_only_s1'] = np.array([len(a1 - a2) for a1, a2 in zip(s1_addr_tok_sets, sx_addr_tok_sets)], dtype=np.float32)
    feats['addr_toks_only_sx'] = np.array([len(a2 - a1) for a1, a2 in zip(s1_addr_tok_sets, sx_addr_tok_sets)], dtype=np.float32)
    feats['num_toks_diff'] = np.array([len(n1 ^ n2) for n1, n2 in zip(s1_num_sets, sx_num_sets)], dtype=np.float32)
    feats['name_tok_count_diff'] = np.array([abs(len(t1) - len(t2)) for t1, t2 in zip(s1_name_tok_lists, sx_name_tok_lists)], dtype=np.float32)
    feats['addr_tok_count_diff'] = np.array([abs(len(a1) - len(a2)) for a1, a2 in zip(s1_addr_tok_lists, sx_addr_tok_lists)], dtype=np.float32)
    feat_df = pd.DataFrame(feats, index=df.index)
    return feat_df

def engineer_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_score_col = 'lgb_score' if 'lgb_score' in df.columns else 'r50_score'
    df = df.sort_values(by=['s1_id', base_score_col], ascending=[True, False]).reset_index(drop=True)
    df['cand_rank'] = df.groupby('s1_id').cumcount() + 1
    df['cand_count'] = df.groupby('s1_id')[base_score_col].transform('count')
    top1_scores = df.groupby('s1_id')[base_score_col].transform('first')
    df['top1_score'] = top1_scores
    df['score_diff_top1'] = df[base_score_col] - top1_scores
    df['score_ratio_top1'] = df[base_score_col] / (top1_scores + 1e-06)
    mean_scores = df.groupby('s1_id')[base_score_col].transform('mean')
    df['score_diff_mean'] = df[base_score_col] - mean_scores
    df['score_percentile'] = np.where(df['cand_count'] > 1, (df['cand_count'] - df['cand_rank']) / (df['cand_count'] - 1), 1.0)
    df = df.sort_values(by=['sx_id', base_score_col], ascending=[True, False]).reset_index(drop=True)
    df['target_candidate_count'] = df.groupby('sx_id')[base_score_col].transform('count')
    df['target_rank'] = df.groupby('sx_id').cumcount() + 1
    target_top1_scores = df.groupby('sx_id')[base_score_col].transform('first')
    df['target_best_base_score'] = target_top1_scores
    rank2_targets = df[df['target_rank'] == 2][['sx_id', base_score_col]].drop_duplicates('sx_id').rename(columns={base_score_col: 'target_second_best_base_score'})
    df = df.merge(rank2_targets, on='sx_id', how='left')
    df['target_second_best_base_score'] = df['target_second_best_base_score'].fillna(0.0)
    df['target_margin'] = df['target_best_base_score'] - df['target_second_best_base_score']
    df['target_score_ratio'] = df[base_score_col] / (df['target_best_base_score'] + 1e-06)
    df['target_is_best_candidate'] = (df['target_rank'] == 1).astype(np.float32)
    df = df.sort_values(by=['s1_id', base_score_col], ascending=[True, False]).reset_index(drop=True)
    return df

def get_v12_feature_names() -> list[str]:
    pairwise_feats = ['name_exact', 'name_ratio', 'name_wratio', 'name_jaro_winkler', 'name_levenshtein', 'name_token_sort', 'name_token_set', 'name_token_jaccard', 'name_char_3gram_jaccard', 'name_len_diff', 'name_len_ratio', 'name_first_token_match', 'name_first_token_conflict', 'name_translit_sim', 'addr_exact', 'addr_ratio', 'addr_jaro_winkler', 'addr_levenshtein', 'addr_token_sort', 'addr_token_set', 'addr_token_jaccard', 'addr_num_shared', 'addr_house_num_exact', 'addr_house_num_conflict', 'postal_exact', 'postal_prefix_match', 'addr_present_both', 'addr_len_ratio', 'name_addr_prod', 'name_addr_sum', 'name_addr_max', 'name_addr_min', 'name_exact_addr_missing', 'name_addr_combined_sim', 'name_toks_only_s1', 'name_toks_only_sx', 'addr_toks_only_s1', 'addr_toks_only_sx', 'num_toks_diff', 'name_tok_count_diff', 'addr_tok_count_diff']
    competition_feats = ['lgb_score', 'tfidf_score', 'cand_rank', 'cand_count', 'top1_score', 'score_diff_top1', 'score_ratio_top1', 'score_diff_mean', 'score_percentile', 'target_candidate_count', 'target_rank', 'target_best_base_score', 'target_second_best_base_score', 'target_margin', 'target_score_ratio', 'target_is_best_candidate']
    return pairwise_feats + competition_feats

def mine_training_data(df: pd.DataFrame, gt_set: set[tuple[str, str]], max_hard_neg: int=6) -> pd.DataFrame:
    df['is_gold'] = [(s, x) in gt_set for s, x in zip(df['s1_id'], df['sx_id'])]
    df['label'] = df['is_gold'].astype(np.int32)
    pos_df = df[df['is_gold']].copy()
    neg_df = df[~df['is_gold']].copy()
    neg_df = neg_df.sort_values(by=['s1_id', 'lgb_score'], ascending=[True, False])
    hard_negs = neg_df.groupby('s1_id').head(max_hard_neg)
    train_data = pd.concat([pos_df, hard_negs], ignore_index=True)
    train_data = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return train_data

def apply_target_exclusivity(pred_df: pd.DataFrame, target_margin_thresh: float=0.0) -> pd.DataFrame:
    if len(pred_df) == 0:
        return pred_df
    sorted_df = pred_df.sort_values(by=['sx_id', 'v12_score'], ascending=[True, False]).reset_index(drop=True)
    sorted_df['target_rank_final'] = sorted_df.groupby('sx_id').cumcount() + 1
    top1_df = sorted_df[sorted_df['target_rank_final'] == 1].copy()
    if target_margin_thresh > 0:
        runner_up = sorted_df[sorted_df['target_rank_final'] == 2][['sx_id', 'v12_score']].rename(columns={'v12_score': 'v12_score_2'})
        top1_df = top1_df.merge(runner_up, on='sx_id', how='left')
        top1_df['v12_score_2'] = top1_df['v12_score_2'].fillna(0.0)
        top1_df['final_target_margin'] = top1_df['v12_score'] - top1_df['v12_score_2']
        top1_df = top1_df[(top1_df['final_target_margin'] >= target_margin_thresh) | (top1_df['v12_score_2'] == 0.0)]
    return top1_df[['s1_id', 'sx_id', 'v12_score']]

def train_v12_matcher(src: str) -> tuple[lgb.Booster, list[str]]:
    print(f"\n{'=' * 70}\nTRAINING V12 MATCHER FOR {src.upper()}\n{'=' * 70}")
    train_cache_path = V2_CACHE / f'v2_{src}_train_gate.parquet'
    print(f'Loading cached training pairs from: {train_cache_path.name}')
    raw_train_df = pd.read_parquet(train_cache_path)
    _, gt_set, _ = get_clean_gold(src, 'train_gate')
    print(f'Total candidate pairs: {len(raw_train_df):,} | Confirmed gold pairs in train_gate: {len(gt_set):,}')
    print('Computing 41 pairwise features...')
    t0 = time.time()
    pair_feats = compute_pairwise_features(raw_train_df)
    train_df = pd.concat([raw_train_df, pair_feats], axis=1)
    print(f'Pairwise features computed in {time.time() - t0:.1f}s')
    print('Computing forward S1 & reverse target competition features...')
    train_df = engineer_competition_features(train_df)
    print('Preparing training dataset from full candidate graph...')
    train_df['is_gold'] = [(s, x) in gt_set for s, x in zip(train_df['s1_id'], train_df['sx_id'])]
    train_df['label'] = train_df['is_gold'].astype(np.int32)
    pos_count = train_df['label'].sum()
    neg_count = len(train_df) - pos_count
    print(f'Training set: {len(train_df):,} pairs (Positives: {pos_count:,}, Negatives: {neg_count:,}, Ratio: 1:{neg_count / max(pos_count, 1):.1f})')
    feature_cols = get_v12_feature_names()
    print(f'Total features: {len(feature_cols)}')
    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df['label'])
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'n_estimators': 250, 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7, 'subsample': 0.85, 'colsample_bytree': 0.85, 'scale_pos_weight': 3.0, 'random_state': 42, 'verbose': -1, 'n_jobs': 6}
    print('Fitting LightGBM model...')
    t0 = time.time()
    model = lgb.train(params, dtrain)
    print(f'Model training completed in {time.time() - t0:.1f}s')
    model_save_path = MODELS_DIR / f'v12_lgb_{src}.txt'
    model.save_model(str(model_save_path))
    print(f'Saved model to: {model_save_path}')
    return (model, feature_cols)

def evaluate_v12_pipeline(src: str, split_name: str, model: lgb.Booster, feature_cols: list[str]) -> dict[str, Any]:
    print(f"\n{'-' * 60}\nEVALUATING {src.upper()} | {split_name.upper()}\n{'-' * 60}")
    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    val_cache_path = V2_CACHE / f'v2_{src}_{split_name}.parquet'
    val_df = pd.read_parquet(val_cache_path)
    print('Computing features for validation split...')
    pair_feats = compute_pairwise_features(val_df)
    full_val = pd.concat([val_df, pair_feats], axis=1)
    full_val = engineer_competition_features(full_val)
    r50_thresh = OPERATIONAL_THRESHOLDS[src]
    r50_preds = full_val[full_val['lgb_score'] >= r50_thresh][['s1_id', 'sx_id']].copy()
    r50_metrics = calculate_macro_f05(r50_preds, gt_df, s1_all)
    phase3_path = MODELS_DIR / f'phase3_reranker_{src}.txt'
    p3_score_avail = False
    if phase3_path.exists():
        p3_bst = lgb.Booster(model_file=str(phase3_path))
        p3_feats = ['r50_score', 'cand_rank', 'cand_count', 'top1_score', 'score_diff_top1', 'score_ratio_top1', 'score_diff_mean']
        full_val['r50_score'] = full_val['lgb_score']
        p3_scores = p3_bst.predict(full_val[p3_feats])
        full_val['p3_score'] = p3_scores
        p3_preds = full_val[full_val['p3_score'] >= 0.5][['s1_id', 'sx_id']].copy()
        p3_metrics = calculate_macro_f05(p3_preds, gt_df, s1_all)
        p3_score_avail = True
    else:
        p3_metrics = r50_metrics
    v12_scores = model.predict(full_val[feature_cols])
    full_val['v12_score'] = v12_scores
    best_thresh = 0.7
    best_f05 = -1.0
    best_metrics = {}
    best_excl_metrics = {}
    threshold_candidates = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    best_cand_excl = pd.DataFrame()
    for th in threshold_candidates:
        cand_preds = full_val[full_val['v12_score'] >= th][['s1_id', 'sx_id', 'v12_score']].copy()
        m_direct = calculate_macro_f05(cand_preds, gt_df, s1_all)
        cand_excl = apply_target_exclusivity(cand_preds, target_margin_thresh=0.0)
        m_excl = calculate_macro_f05(cand_excl, gt_df, s1_all)
        if m_excl['macro_f05'] > best_f05:
            best_f05 = m_excl['macro_f05']
            best_thresh = th
            best_metrics = m_direct
            best_excl_metrics = m_excl
            best_cand_excl = cand_excl.copy()
    gold_pairs_df = full_val[[(s, x) in gt_set for s, x in zip(full_val['s1_id'], full_val['sx_id'])]].copy()
    s1_gold_counts = gold_pairs_df.groupby('s1_id')['sx_id'].transform('count')
    sec_gold = gold_pairs_df[s1_gold_counts > 1]
    sec_gold_set = set(zip(sec_gold['s1_id'], sec_gold['sx_id']))
    v12_pred_set = set(zip(best_cand_excl['s1_id'], best_cand_excl['sx_id']))
    r50_pred_set = set(zip(r50_preds['s1_id'], r50_preds['sx_id']))
    sec_rec_r50 = len(sec_gold_set & r50_pred_set) / max(len(sec_gold_set), 1)
    sec_rec_v12 = len(sec_gold_set & v12_pred_set) / max(len(sec_gold_set), 1)
    print(f'Results for {src.upper()} {split_name.upper()}:')
    print(f"  R50 Baseline:     Macro F0.5 = {r50_metrics['macro_f05']:.5f} (P={r50_metrics['macro_precision']:.4f}, R={r50_metrics['macro_recall']:.4f}, Preds={r50_metrics['predicted_matches']:,})")
    if p3_score_avail:
        print(f"  Phase-3 Control:  Macro F0.5 = {p3_metrics['macro_f05']:.5f} (P={p3_metrics['macro_precision']:.4f}, R={p3_metrics['macro_recall']:.4f}, Preds={p3_metrics['predicted_matches']:,})")
    print(f"  V12 Base (th={best_thresh}): Macro F0.5 = {best_metrics['macro_f05']:.5f} (P={best_metrics['macro_precision']:.4f}, R={best_metrics['macro_recall']:.4f}, Preds={best_metrics['predicted_matches']:,})")
    print(f"  V12 + Exclusivity: Macro F0.5 = {best_excl_metrics['macro_f05']:.5f} (P={best_excl_metrics['macro_precision']:.4f}, R={best_excl_metrics['macro_recall']:.4f}, Preds={best_excl_metrics['predicted_matches']:,})")
    print(f'  Secondary Recall: R50 = {sec_rec_r50:.4f} | V12 = {sec_rec_v12:.4f}')
    return {'src': src, 'split': split_name, 'r50': r50_metrics, 'phase3': p3_metrics, 'v12_base': best_metrics, 'v12_excl': best_excl_metrics, 'best_thresh': best_thresh, 'sec_rec_r50': round(sec_rec_r50, 4), 'sec_rec_v12': round(sec_rec_v12, 4)}

def main():
    parser = argparse.ArgumentParser(description='V12 Assignment-Aware Evidence LightGBM')
    parser.add_argument('--sources', nargs='+', default=['source2', 'source3'], help='Sources to train and evaluate')
    args = parser.parse_args()
    results = {}
    for src in args.sources:
        model, feature_cols = train_v12_matcher(src)
        res_v1 = evaluate_v12_pipeline(src, 'val1', model, feature_cols)
        res_v2 = evaluate_v12_pipeline(src, 'val2', model, feature_cols)
        results[f'{src}_val1'] = res_v1
        results[f'{src}_val2'] = res_v2
    results_path = ROOT / 'experiments' / 'v12_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll evaluations finished. Results saved to: {results_path}')
if __name__ == '__main__':
    main()
