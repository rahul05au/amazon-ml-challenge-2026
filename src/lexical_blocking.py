from __future__ import annotations
import argparse
import sys
import time
import tracemalloc
from typing import Any
import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import INTERMEDIATE_DIR, VAL_FRACTION, VAL_SEED, candidate_parquet, norm_parquet

def retrieve_tfidf_candidates(target_source: str='source2', split: str='train', k: int=20, batch_size: int=1000, s1_ids: set[str] | None=None, return_scores: bool=False) -> pd.DataFrame:
    s1_pq = norm_parquet(split, 'source1')
    sx_pq = norm_parquet(split, target_source)
    s1_safe = str(s1_pq).replace('\\', '/')
    sx_safe = str(sx_pq).replace('\\', '/')
    con = duckdb.connect()
    if s1_ids is not None:
        con.register('_s1_ids_filter', pd.DataFrame({'entity_id': list(s1_ids)}))
        val_s1 = con.execute(f"\n            SELECT s1.entity_id AS s1_id, s1.country_clean,\n                   trim(COALESCE(s1.name_clean, '') || ' ' || COALESCE(s1.addr_clean, '')) AS text\n            FROM read_parquet('{s1_safe}') s1\n            JOIN _s1_ids_filter f ON s1.entity_id = f.entity_id\n        ").df()
    else:
        val_s1 = con.execute(f"\n            SELECT entity_id AS s1_id, country_clean,\n                   trim(COALESCE(name_clean, '') || ' ' || COALESCE(addr_clean, '')) AS text\n            FROM read_parquet('{s1_safe}')\n            WHERE abs(hash(entity_id || '{VAL_SEED}')) % 1000 < {int(VAL_FRACTION * 1000)}\n        ").df()
    target_df = con.execute(f"\n        SELECT entity_id AS sx_id, country_clean,\n                   trim(COALESCE(name_clean, '') || ' ' || COALESCE(addr_clean, '')) AS text\n        FROM read_parquet('{sx_safe}')\n    ").df()
    con.close()
    vec = TfidfVectorizer(min_df=2, max_df=0.005, stop_words='english', sublinear_tf=True, dtype=np.float32)
    X_target = vec.fit_transform(target_df['text'])
    X_query = vec.transform(val_s1['text'])
    del target_df['text'], val_s1['text']
    retrieved_s1: list[str] = []
    retrieved_sx: list[str] = []
    retrieved_scores: list[float] = []
    for country in val_s1['country_clean'].unique():
        q_mask = (val_s1['country_clean'] == country).values
        t_mask = (target_df['country_clean'] == country).values
        if not t_mask.any() or not q_mask.any():
            continue
        q_sub = X_query[q_mask]
        t_sub = X_target[t_mask]
        sub_s1_ids = val_s1['s1_id'].values[q_mask]
        sub_target_ids = target_df['sx_id'].values[t_mask]
        t_sub_T = t_sub.T.tocsr()
        n_queries = q_sub.shape[0]
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            sims = (q_sub[start_idx:end_idx] @ t_sub_T).tocsr()
            for i in range(sims.shape[0]):
                r_start = sims.indptr[i]
                r_end = sims.indptr[i + 1]
                n_nonzeros = r_end - r_start
                if n_nonzeros == 0:
                    continue
                c_indices = sims.indices[r_start:r_end]
                c_data = sims.data[r_start:r_end]
                if n_nonzeros <= k:
                    chosen_idx = c_indices
                    chosen_scores = c_data
                else:
                    top_k_pos = np.argpartition(c_data, -k)[-k:]
                    chosen_idx = c_indices[top_k_pos]
                    chosen_scores = c_data[top_k_pos]
                s1_curr = sub_s1_ids[start_idx + i]
                for sx_idx, score in zip(chosen_idx, chosen_scores):
                    retrieved_s1.append(s1_curr)
                    retrieved_sx.append(sub_target_ids[sx_idx])
                    if return_scores:
                        retrieved_scores.append(float(score))
            del sims
    if return_scores:
        res = pd.DataFrame({'s1_id': retrieved_s1, 'sx_id': retrieved_sx, 'tfidf_score': retrieved_scores})
        return res.sort_values('tfidf_score', ascending=False).drop_duplicates(['s1_id', 'sx_id'])
    return pd.DataFrame({'s1_id': retrieved_s1, 'sx_id': retrieved_sx}).drop_duplicates()

def evaluate_retrieval_experiment(target_source: str='source2', split: str='train', k: int=20) -> dict[str, Any]:
    tracemalloc.start()
    t0 = time.perf_counter()
    cand_df = retrieve_tfidf_candidates(target_source=target_source, split=split, k=k)
    prefix = 'S2-' if target_source == 'source2' else 'S3-'
    gt_pq = str(INTERMEDIATE_DIR / split / 'ground_truth.parquet').replace('\\', '/')
    base_pq = str(candidate_parquet(split, target_source)).replace('\\', '/')
    con = duckdb.connect()
    con.execute('SET threads TO 6')
    con.execute("SET memory_limit = '6GB'")
    con.execute(f"\n        CREATE TEMP TABLE _val_s1 AS\n        SELECT source1_entity_id AS s1_id\n        FROM read_parquet('{gt_pq}')\n        WHERE abs(hash(source1_entity_id || '{VAL_SEED}')) % 1000 < {int(VAL_FRACTION * 1000)}\n    ")
    con.execute(f"\n        CREATE TEMP TABLE _gt_target AS\n        SELECT\n            gt.source1_entity_id AS s1_id,\n            trim(unnest(string_split(gt.matched_entity_ids, ','))) AS true_match_id\n        FROM read_parquet('{gt_pq}') gt\n        WHERE gt.source1_entity_id IN (SELECT s1_id FROM _val_s1)\n          AND gt.matched_entity_ids != ''\n    ")
    con.execute(f"\n        CREATE TEMP TABLE _gt_filtered AS\n        SELECT s1_id, true_match_id\n        FROM _gt_target\n        WHERE true_match_id LIKE '{prefix}%'\n    ")
    con.execute(f"\n        CREATE TEMP TABLE _baseline AS\n        SELECT s1_id, sx_id\n        FROM read_parquet('{base_pq}')\n        WHERE s1_id IN (SELECT s1_id FROM _val_s1)\n    ")
    con.register('_retrieval', cand_df)
    con.execute('\n        CREATE TEMP TABLE _union AS\n        SELECT s1_id, sx_id FROM _baseline\n        UNION\n        SELECT s1_id, sx_id FROM _retrieval\n    ')
    total_s1 = con.execute('SELECT count(*) FROM _val_s1').fetchone()[0]
    total_true = con.execute('SELECT count(*) FROM _gt_filtered').fetchone()[0]

    def _eval_cands(table_name: str) -> dict[str, Any]:
        found = con.execute(f'\n            SELECT count(*)\n            FROM _gt_filtered gt\n            JOIN {table_name} c ON gt.s1_id = c.s1_id AND gt.true_match_id = c.sx_id\n        ').fetchone()[0]
        stats = con.execute(f'\n            SELECT\n                sum(cnt) AS total,\n                count(*) AS s1_count,\n                avg(cnt) AS avg_cnt,\n                percentile_disc(0.95) WITHIN GROUP (ORDER BY cnt) AS p95,\n                max(cnt) AS max_cnt\n            FROM (\n                SELECT s1_id, count(*) AS cnt\n                FROM {table_name}\n                GROUP BY s1_id\n            ) t\n        ').fetchone()
        total_cands = int(stats[0]) if stats[0] is not None else 0
        s1_cov = int(stats[1]) if stats[1] is not None else 0
        recall = found / total_true if total_true > 0 else 0.0
        return {'found': found, 'total_true': total_true, 'recall': round(recall, 6), 'candidate_count': total_cands, 's1_covered': s1_cov, 's1_coverage_pct': round(s1_cov / total_s1 * 100.0, 2) if total_s1 > 0 else 0.0, 'avg_per_s1': round(float(stats[2]), 2) if stats[2] is not None else 0.0, 'p95_per_s1': int(stats[3]) if stats[3] is not None else 0, 'max_per_s1': int(stats[4]) if stats[4] is not None else 0}
    base_metrics = _eval_cands('_baseline')
    retrieval_metrics = _eval_cands('_retrieval')
    union_metrics = _eval_cands('_union')
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    newly_recovered = union_metrics['found'] - base_metrics['found']
    added_candidates = union_metrics['candidate_count'] - base_metrics['candidate_count']
    efficiency = added_candidates / newly_recovered if newly_recovered > 0 else float('inf')
    con.close()
    result = {'target_source': target_source, 'k': k, 'runtime_seconds': round(elapsed, 1), 'peak_memory_mb': round(peak_bytes / (1024 * 1024), 1), 'baseline': base_metrics, 'retrieval': retrieval_metrics, 'union': union_metrics, 'recovered_true_matches': newly_recovered, 'added_candidates': added_candidates, 'candidates_per_recovered': round(efficiency, 1) if efficiency != float('inf') else None}
    _print_report(result)
    return result

def _print_report(res: dict[str, Any]) -> None:
    target = res['target_source'].upper()
    b = res['baseline']
    r = res['retrieval']
    u = res['union']
    print(f"\n{'=' * 60}")
    print(f'  V2 EXP 1 -- LEXICAL RETRIEVAL BLOCKING: {target}')
    print(f"{'=' * 60}")
    print(f"  Baseline recall:         {b['recall']:>8.2%} ({b['found']:,} / {b['total_true']:,})")
    print(f"  Retrieval-only recall:   {r['recall']:>8.2%} ({r['found']:,} / {r['total_true']:,})")
    print(f"  Union recall:            {u['recall']:>8.2%} ({u['found']:,} / {u['total_true']:,})")
    print(f'  --------------------------------------------------')
    print(f"  Baseline candidates:     {b['candidate_count']:>10,}")
    print(f"  Retrieval candidates:    {r['candidate_count']:>10,}")
    print(f"  Union candidates:        {u['candidate_count']:>10,}")
    print(f"  Added candidates:        {res['added_candidates']:>10,}")
    print(f'  --------------------------------------------------')
    print(f"  Baseline avg cands/S1:   {b['avg_per_s1']:>10.1f}")
    print(f"  Union avg cands/S1:      {u['avg_per_s1']:>10.1f}")
    print(f"  Baseline P95:            {b['p95_per_s1']:>10}")
    print(f"  Union P95:               {u['p95_per_s1']:>10}")
    print(f'  --------------------------------------------------')
    print(f"  Recovered true matches:  {res['recovered_true_matches']:>10,}")
    eff_str = f"{res['candidates_per_recovered']:.1f}" if res['candidates_per_recovered'] is not None else 'N/A'
    print(f'  Added cands / recovered: {eff_str:>10}')
    print(f'  --------------------------------------------------')
    print(f"  Runtime:                 {res['runtime_seconds']:>9.1f}s")
    print(f"  Peak memory:             {res['peak_memory_mb']:>9.1f} MB")
    print(f"{'=' * 60}\n")

def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate additive TF-IDF lexical blocking.')
    parser.add_argument('--target', choices=['source2', 'source3', 'both'], default='both')
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()
    targets = ['source2', 'source3'] if args.target == 'both' else [args.target]
    for target in targets:
        evaluate_retrieval_experiment(target_source=target, k=args.k)
    return 0
if __name__ == '__main__':
    sys.exit(main())
