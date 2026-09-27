from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.matcher import calculate_macro_f05
from experiments.v12_assignment_lgb import get_clean_gold, compute_pairwise_features, engineer_competition_features, get_v12_feature_names, apply_target_exclusivity, MODELS_DIR, V2_CACHE
from experiments.v12_1_collective_matcher import compute_collective_sibling_features, get_v12_1_feature_names
from experiments.v13_1_pipeline import enrich_and_score_r50, build_expanded_candidate_df
V12_1_SCORES = {('source2', 'val1'): 0.89901, ('source2', 'val2'): 0.90037, ('source3', 'val1'): 0.90198, ('source3', 'val2'): 0.8999}
print('=' * 75)
print('PHASE 1: FROZEN R50 RESCORE GATE — S2 VAL1 RECOVERED GOLD PAIRS')
print('=' * 75)
src_gate, split_gate = ('source2', 'val1')
s1_all_g, gt_set_g, gt_df_g = get_clean_gold(src_gate, split_gate)
old_df_g = pd.read_parquet(V2_CACHE / f'v2_{src_gate}_{split_gate}.parquet')
old_pairs_g = set(zip(old_df_g['s1_id'], old_df_g['sx_id']))
from experiments.v13_candidate_audit_and_eval import run_v13_retrieval_pairs
v13_pairs_g = run_v13_retrieval_pairs(src_gate, split_gate)
added_g = v13_pairs_g - old_pairs_g
recovered_gold_g = added_g & gt_set_g
print(f'\n  V13 retrieval pairs (total):  {len(v13_pairs_g):,}')
print(f'  Net new pairs (not in V2):    {len(added_g):,}')
print(f'  Recovered gold pairs:         {len(recovered_gold_g):,}')
if len(recovered_gold_g) == 0:
    print('\n  !! NO recovered gold pairs found. Cannot perform gate check.')
    print('  !! Check run_v13_retrieval_pairs() — may need data.')
    sys.exit(1)
enriched_g = enrich_and_score_r50(recovered_gold_g, src_gate, split_gate)
r50_scores = enriched_g['lgb_score'].values
print(f'\n  Frozen R50 Score Statistics on {len(r50_scores):,} Recovered Gold Pairs:')
print(f'    Count:               {len(r50_scores):,}')
print(f'    Mean:                {np.mean(r50_scores):.4f}')
print(f'    Median:              {np.median(r50_scores):.4f}')
print(f'    P90:                 {np.percentile(r50_scores, 90):.4f}')
print(f'    P95:                 {np.percentile(r50_scores, 95):.4f}')
print(f'    Max:                 {np.max(r50_scores):.4f}')
print(f'    Fraction > 0.05:     {(r50_scores > 0.05).mean() * 100:.1f}%  ({(r50_scores > 0.05).sum():,}/{len(r50_scores):,})')
print(f'    Fraction > 0.10:     {(r50_scores > 0.1).mean() * 100:.1f}%  ({(r50_scores > 0.1).sum():,}/{len(r50_scores):,})')
print(f'    Fraction > 0.30:     {(r50_scores > 0.3).mean() * 100:.1f}%  ({(r50_scores > 0.3).sum():,}/{len(r50_scores):,})')
print(f'    Fraction > 0.50:     {(r50_scores > 0.5).mean() * 100:.1f}%  ({(r50_scores > 0.5).sum():,}/{len(r50_scores):,})')
print(f'\n  Sample of recovered gold pairs with R50 scores:')
for i, (_, row) in enumerate(enriched_g.head(10).iterrows()):
    s1n = str(row.get('s1_name', ''))[:30]
    sxn = str(row.get('sx_name', ''))[:30]
    print(f"    [{i + 1:2d}] {s1n:<30} | {sxn:<30} | R50={row['lgb_score']:.4f}")
nonzero_frac = (r50_scores > 0.0).mean()
print(f'\n  Non-zero R50 fraction: {nonzero_frac * 100:.1f}%')
if nonzero_frac < 0.01:
    print('\n  !! GATE FAILED: >99% of recovered gold pairs have R50=0.')
    print('  !! Feature construction bug still present. STOPPING.')
    print('  !! Root cause: _process_chunk_worker() may be receiving wrong column names.')
    sys.exit(1)
else:
    print(f'\n  ✅ GATE PASSED: {nonzero_frac * 100:.1f}% of recovered gold pairs have non-zero R50 scores.')
    print('  → PROCEEDING to Phase 2-6: Full V13.1 Pipeline\n')
all_results = {}
for src in ['source2', 'source3']:
    prefix = 'S2-' if src == 'source2' else 'S3-'
    print('\n' + '=' * 75)
    print(f'V13.1 FULL PIPELINE: {src.upper()}')
    print('=' * 75)
    print(f'\n[PHASE 2] Building expanded training candidate graph for {src.upper()}...')
    t0 = time.time()
    _, gt_train, _ = get_clean_gold(src, 'train_gate')
    train_exp_df, train_added = build_expanded_candidate_df(src, 'train_gate')
    print(f'  train_gate: {len(train_exp_df):,} total pairs | {len(train_added):,} new from V13')
    print(f'\n[PHASE 3] Computing pairwise features on expanded training graph...')
    t_feat = time.time()
    train_pw = compute_pairwise_features(train_exp_df)
    train_full = pd.concat([train_exp_df, train_pw], axis=1)
    train_full = engineer_competition_features(train_full)
    train_full['label'] = [(s, x) in gt_train for s, x in zip(train_full['s1_id'], train_full['sx_id'])]
    train_full['label'] = train_full['label'].astype(np.int32)
    print(f"  Features done in {time.time() - t_feat:.1f}s | Pairs: {len(train_full):,} | Positives: {train_full['label'].sum():,}")
    v12_cols = get_v12_feature_names()
    v12_cols_no_r50 = [c for c in v12_cols if c != 'lgb_score']
    lgb_params_s1 = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'num_boost_round': 300, 'learning_rate': 0.05, 'num_leaves': 63, 'max_depth': 7, 'subsample': 0.85, 'colsample_bytree': 0.85, 'scale_pos_weight': 3.0, 'random_state': 42, 'verbose': -1, 'n_jobs': 6}
    n_rounds = lgb_params_s1.pop('num_boost_round')
    print(f'\n  Training Stage-1 Variant A (WITH R50 lgb_score)...')
    dtrain_a = lgb.Dataset(train_full[v12_cols], label=train_full['label'])
    model_s1_a = lgb.train(lgb_params_s1, dtrain_a, num_boost_round=n_rounds)
    print(f'  Training Stage-1 Variant B (WITHOUT R50 lgb_score)...')
    dtrain_b = lgb.Dataset(train_full[v12_cols_no_r50], label=train_full['label'])
    model_s1_b = lgb.train(lgb_params_s1, dtrain_b, num_boost_round=n_rounds)
    print(f'\n  Ablation check on val1 (th=0.70)...')
    val1_exp, _ = build_expanded_candidate_df(src, 'val1')
    s1_all_v1, _, gt_df_v1 = get_clean_gold(src, 'val1')
    val1_pw = compute_pairwise_features(val1_exp)
    val1_full = pd.concat([val1_exp, val1_pw], axis=1)
    val1_full = engineer_competition_features(val1_full)
    val1_full['score_a'] = model_s1_a.predict(val1_full[v12_cols])
    val1_full['score_b'] = model_s1_b.predict(val1_full[v12_cols_no_r50])
    for variant, score_col in [('A (with R50)', 'score_a'), ('B (no R50)', 'score_b')]:
        cols_v = v12_cols if 'a' in score_col else v12_cols_no_r50
        preds = val1_full[val1_full[score_col] >= 0.7][['s1_id', 'sx_id', score_col]].rename(columns={score_col: 'v12_score'})
        m = calculate_macro_f05(apply_target_exclusivity(preds), gt_df_v1, s1_all_v1)
        print(f"    Variant {variant}: F0.5={m['macro_f05']:.5f} P={m['macro_precision']:.4f} R={m['macro_recall']:.4f}")
    use_r50 = model_s1_a.predict(val1_full[v12_cols]).mean() != model_s1_b.predict(val1_full[v12_cols_no_r50]).mean()
    preds_a = val1_full[val1_full['score_a'] >= 0.7][['s1_id', 'sx_id', 'score_a']].rename(columns={'score_a': 'v12_score'})
    preds_b = val1_full[val1_full['score_b'] >= 0.7][['s1_id', 'sx_id', 'score_b']].rename(columns={'score_b': 'v12_score'})
    m_a = calculate_macro_f05(apply_target_exclusivity(preds_a), gt_df_v1, s1_all_v1)
    m_b = calculate_macro_f05(apply_target_exclusivity(preds_b), gt_df_v1, s1_all_v1)
    if m_a['macro_f05'] >= m_b['macro_f05']:
        selected_model_s1 = model_s1_a
        selected_cols_s1 = v12_cols
        selected_variant = 'WITH_R50'
    else:
        selected_model_s1 = model_s1_b
        selected_cols_s1 = v12_cols_no_r50
        selected_variant = 'WITHOUT_R50'
    print(f'  → Selected Stage-1 variant: {selected_variant}')
    stage1_path = MODELS_DIR / f'v13_1_stage1_{src}.txt'
    selected_model_s1.save_model(str(stage1_path))
    print(f'  Stage-1 model saved: {stage1_path.name}')
    print(f'\n[PHASE 4] 3-fold entity-level OOF Stage-1 predictions...')
    s1_ents = train_full['s1_id'].unique()
    fold_map = {e: abs(hash(e + '_oof_v13')) % 3 for e in s1_ents}
    train_full['fold'] = train_full['s1_id'].map(fold_map)
    oof_scores = np.zeros(len(train_full), dtype=np.float32)
    lgb_params_oof = dict(lgb_params_s1)
    for fold_id in range(3):
        tr_mask = train_full['fold'] != fold_id
        va_mask = train_full['fold'] == fold_id
        d_tr = lgb.Dataset(train_full.loc[tr_mask, selected_cols_s1], label=train_full.loc[tr_mask, 'label'])
        bst_oof = lgb.train(lgb_params_oof, d_tr, num_boost_round=n_rounds)
        fold_preds = bst_oof.predict(train_full.loc[va_mask, selected_cols_s1])
        oof_scores[va_mask] = fold_preds
        print(f"  Fold {fold_id + 1}/3: val mean={fold_preds.mean():.4f} max={fold_preds.max():.4f} pos_rate={fold_preds[train_full.loc[va_mask, 'label'] == 1].mean():.4f}")
    train_full['stage1_score'] = oof_scores
    print(f'\n[PHASE 5] Retraining Stage-2 collective sibling model...')
    col_feats_tr = compute_collective_sibling_features(train_full, score_col='stage1_score')
    train_full = pd.concat([train_full, col_feats_tr], axis=1)
    stage2_cols = [c for c in get_v12_1_feature_names() if c in train_full.columns]
    print(f'  Stage-2 feature columns: {len(stage2_cols)}')
    lgb_params_s2 = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'learning_rate': 0.04, 'num_leaves': 45, 'max_depth': 6, 'subsample': 0.85, 'colsample_bytree': 0.8, 'scale_pos_weight': 2.5, 'random_state': 42, 'verbose': -1, 'n_jobs': 6}
    d_s2 = lgb.Dataset(train_full[stage2_cols], label=train_full['label'])
    model_s2 = lgb.train(lgb_params_s2, d_s2, num_boost_round=250)
    stage2_path = MODELS_DIR / f'v13_1_stage2_{src}.txt'
    model_s2.save_model(str(stage2_path))
    print(f'  Stage-2 model saved: {stage2_path.name}')
    print(f'\n[PHASE 6] Full validation on all splits (threshold sweep)...')
    for split in ['val1', 'val2']:
        print(f'\n  ─── {src.upper()} {split.upper()} ───')
        s1_all, gt_set, gt_df = get_clean_gold(src, split)
        v12_1_score = V12_1_SCORES[src, split]
        val_exp, val_added = build_expanded_candidate_df(src, split)
        val_pw = compute_pairwise_features(val_exp)
        val_full = pd.concat([val_exp, val_pw], axis=1)
        val_full = engineer_competition_features(val_full)
        val_full['stage1_score'] = selected_model_s1.predict(val_full[selected_cols_s1])
        col_feats_val = compute_collective_sibling_features(val_full, score_col='stage1_score')
        val_full = pd.concat([val_full, col_feats_val], axis=1)
        val_full['stage2_score'] = model_s2.predict(val_full[stage2_cols])
        val_full['final_score'] = 0.2 * val_full['stage1_score'] + 0.8 * val_full['stage2_score']
        expanded_set = set(zip(val_exp['s1_id'], val_exp['sx_id']))
        old_set = set(zip(pd.read_parquet(V2_CACHE / f'v2_{src}_{split}.parquet')['s1_id'], pd.read_parquet(V2_CACHE / f'v2_{src}_{split}.parquet')['sx_id']))
        new_in_val = expanded_set - old_set
        recovered_gold_val = new_in_val & gt_set
        cand_recall = len(expanded_set & gt_set) / max(len(gt_set), 1)
        print(f'  Expanded candidates: {len(val_exp):,} | New: +{len(new_in_val):,} | Cand recall: {cand_recall:.4f} | Recovered gold: {len(recovered_gold_val):,}')
        best_f05, best_th, best_m = (-1.0, 0.5, {})
        print(f"\n  {'Threshold':<10} {'F0.5':<10} {'Prec':<10} {'Rec':<10} {'Preds':<8} {'vs V12.1'}")
        print(f"  {'-' * 60}")
        for th in [0.3, 0.35, 0.4, 0.43, 0.45, 0.47, 0.5, 0.52, 0.55, 0.6]:
            preds = val_full[val_full['final_score'] >= th][['s1_id', 'sx_id', 'final_score']].rename(columns={'final_score': 'v12_score'})
            excl = apply_target_exclusivity(preds)
            m = calculate_macro_f05(excl, gt_df, s1_all)
            delta = m['macro_f05'] - v12_1_score
            flag = '✅' if m['macro_f05'] > v12_1_score else ''
            print(f"  th={th:.2f}    {m['macro_f05']:.5f}   {m['macro_precision']:.4f}   {m['macro_recall']:.4f}   {m['predicted_matches']:>6,}   {delta:+.5f} {flag}")
            if m['macro_f05'] > best_f05:
                best_f05, best_th, best_m = (m['macro_f05'], th, m)
        improvement = best_f05 - v12_1_score
        target_crossed = '✅ CROSSED 0.91' if best_f05 >= 0.91 else '🎯 CLOSE' if best_f05 >= 0.905 else '❌ BELOW'
        print(f'\n  Best: th={best_th:.2f} → F0.5={best_f05:.5f} ({improvement:+.5f} vs V12.1) {target_crossed}')
        best_preds = val_full[val_full['final_score'] >= best_th][['s1_id', 'sx_id', 'final_score']].rename(columns={'final_score': 'v12_score'})
        best_excl = apply_target_exclusivity(best_preds)
        best_pred_set = set(zip(best_excl['s1_id'], best_excl['sx_id']))
        recov_accepted = recovered_gold_val & best_pred_set
        recov_rate = len(recov_accepted) / max(len(recovered_gold_val), 1)
        if len(recovered_gold_val) > 0:
            recov_scores = val_full[val_full.apply(lambda r: (r['s1_id'], r['sx_id']) in recovered_gold_val, axis=1)]['final_score'].values if len(recovered_gold_val) < 500 else np.array([])
            recov_df = pd.DataFrame(list(recovered_gold_val), columns=['s1_id', 'sx_id'])
            recov_scored = val_full.merge(recov_df, on=['s1_id', 'sx_id'])
            print(f'\n  Recovered Gold Trace ({len(recovered_gold_val):,} pairs):')
            print(f'    Accepted in final prediction: {len(recov_accepted):,} / {len(recovered_gold_val):,} ({recov_rate * 100:.1f}%)')
            if len(recov_scored) > 0:
                rs = recov_scored['final_score'].values
                r50s = recov_scored['lgb_score'].values
                s1s = recov_scored['stage1_score'].values
                s2s = recov_scored['stage2_score'].values
                print(f'    R50 scores:    mean={r50s.mean():.4f} p50={np.median(r50s):.4f} p90={np.percentile(r50s, 90):.4f} max={r50s.max():.4f}')
                print(f'    Stage-1 scores: mean={s1s.mean():.4f} p50={np.median(s1s):.4f} p90={np.percentile(s1s, 90):.4f} max={s1s.max():.4f}')
                print(f'    Stage-2 scores: mean={s2s.mean():.4f} p50={np.median(s2s):.4f} p90={np.percentile(s2s, 90):.4f} max={s2s.max():.4f}')
                print(f'    Final scores:   mean={rs.mean():.4f} p50={np.median(rs):.4f} p90={np.percentile(rs, 90):.4f} max={rs.max():.4f}')
                rejected = recovered_gold_val - recov_accepted
                if len(rejected) > 0:
                    rej_df = val_full.merge(pd.DataFrame(list(rejected), columns=['s1_id', 'sx_id']), on=['s1_id', 'sx_id'])
                    if len(rej_df) > 0:
                        print(f"    Rejected ({len(rejected):,}): threshold_reject={(rej_df['final_score'] < best_th).sum():,} exclusivity_reject={(rej_df['final_score'] >= best_th).sum():,}")
        all_results[f'{src}_{split}'] = {'v12_1_f05': v12_1_score, 'v13_1_f05': best_f05, 'best_threshold': best_th, 'precision': best_m.get('macro_precision', 0.0), 'recall': best_m.get('macro_recall', 0.0), 'singleton_accuracy': best_m.get('singleton_accuracy', 0.0), 'candidate_recall': round(cand_recall, 6), 'new_candidates': len(new_in_val), 'recovered_gold': len(recovered_gold_val), 'recovered_gold_accepted': len(recov_accepted), 'recovered_acceptance_rate': round(recov_rate, 4), 'stage1_variant': selected_variant, 'improvement': round(best_f05 - v12_1_score, 6)}
print('\n\n' + '=' * 75)
print('V13.1 FINAL RESULTS SUMMARY')
print('=' * 75)
print(f"\n{'Split':<18} {'V12.1':>8} {'V13.1':>8} {'Delta':>8} {'RecGold':>8} {'Accepted':>9} {'Status'}")
print('-' * 75)
all_v13 = []
all_v12 = []
for key, r in all_results.items():
    all_v13.append(r['v13_1_f05'])
    all_v12.append(r['v12_1_f05'])
    status = '✅ >0.91' if r['v13_1_f05'] >= 0.91 else '🎯 >0.905' if r['v13_1_f05'] >= 0.905 else '  <0.91'
    print(f"  {key:<16} {r['v12_1_f05']:>8.5f} {r['v13_1_f05']:>8.5f} {r['improvement']:>+8.5f} {r['recovered_gold']:>7,} {r['recovered_gold_accepted']:>8,}   {status}")
avg_v13 = np.mean(all_v13)
avg_v12 = np.mean(all_v12)
print('-' * 75)
print(f"  {'AVERAGE':<16} {avg_v12:>8.5f} {avg_v13:>8.5f} {avg_v13 - avg_v12:>+8.5f}")
print('=' * 75)
print(f'\n📊 Average V13.1: {avg_v13:.5f} | Target: 0.91000 | V12.1: {avg_v12:.5f}')
if avg_v13 >= 0.95:
    decision = '🔥 STRONG (>=0.95) — Prepare as THIRD SUBMISSION CANDIDATE immediately.'
elif avg_v13 >= 0.93:
    decision = '💪 PREFERRED (>=0.93) — Prepare as THIRD SUBMISSION CANDIDATE.'
elif avg_v13 >= 0.91:
    decision = '✅ TARGET MET (>=0.91) — Prepare as THIRD SUBMISSION CANDIDATE.'
elif avg_v13 >= 0.905:
    decision = '🎯 CLOSE — Apply cheap threshold optimization, then submit if improves.'
elif avg_v13 > avg_v12:
    decision = f'📈 BETTER than V12.1 but below 0.91 (+{avg_v13 - avg_v12:+.5f}). Apply cheapest optimization then submit best.'
else:
    decision = f'⬇️ WORSE than V12.1. ROLLBACK to V12.1 for submission.'
print(f'\n  Decision: {decision}')
results_path = ROOT / 'experiments' / 'v13_1_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2)
print(f'\n  Results saved to: {results_path}')
