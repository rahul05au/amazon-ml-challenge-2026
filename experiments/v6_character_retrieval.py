"""V6 Phase 1: Character N-Gram Additive Retrieval Experiment.

Evaluates:
  1. Additive character TF-IDF on business_name (3-5 grams)
  2. top-k = 20, then top-k = 40
  3. Union with frozen V2 candidates (deduplicated)
  4. Blocking metrics on val1 and val2 for S2 and S3:
     - baseline blocking recall
     - character retrieval recall
     - union recall
     - candidate counts (baseline, char, union, added)
     - avg candidates/S1, P95 candidates/S1
     - recovered true matches, added cands / recovered true match
  5. Downstream R50 matcher evaluation:
     - Macro F0.5
     - precision
     - recall
     - singleton accuracy
  6. Phase 1 Decision: KEEP or DISCARD.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet
from src.features import compute_features_parallel
from src.matcher import MODELS_DIR, calculate_macro_f05

_V2_THRESH = {"source2": 0.70, "source3": 0.65}
_CACHE_DIR = ROOT / "intermediate" / "v2_cache"

_H = "abs(hash(entity_id || 'v1')) % 1000"
_WHERE = {
    "val1": f"{_H} >= 0  AND {_H} < 5",
    "val2": f"{_H} >= 25 AND {_H} < 30",
}


def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET threads TO 6; SET memory_limit = '6GB'")
    return c


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def get_split_s1_and_gt(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    prefix = "S2-" if src == "source2" else "S3-"
    c = _con()
    c.execute(f"""
        CREATE TEMP TABLE _s1 AS
        SELECT entity_id AS s1_id
        FROM read_parquet('{_s1pq()}')
        WHERE {_WHERE[split_name]}
    """)
    s1_all = set(c.execute("SELECT s1_id FROM _s1").df()["s1_id"])
    gt_df = c.execute(f"""
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
        FROM read_parquet('{_gtpq()}')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
          AND source1_entity_id IN (SELECT s1_id FROM _s1)
    """).df()
    c.close()
    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
    return s1_all, gt_set, gt_df


def retrieve_char_tfidf_candidates(
    target_source: str,
    s1_ids: set[str],
    k: int = 20,
    batch_size: int = 1000,
) -> tuple[pd.DataFrame, float, float]:
    """Retrieve top-k candidates per S1 entity using character 3-5 gram TF-IDF on business_name."""
    tracemalloc.start()
    t0 = time.time()

    c = _con()
    c.register("_s1_filter", pd.DataFrame({"entity_id": list(s1_ids)}))
    val_s1 = c.execute(f"""
        SELECT s1.entity_id AS s1_id, COALESCE(s1.country_clean, '') AS country_clean,
               COALESCE(s1.name_clean, '') AS name_clean
        FROM read_parquet('{_s1pq()}') s1
        JOIN _s1_filter f ON s1.entity_id = f.entity_id
    """).df()

    target_df = c.execute(f"""
        SELECT entity_id AS sx_id, COALESCE(country_clean, '') AS country_clean,
               COALESCE(name_clean, '') AS name_clean
        FROM read_parquet('{_sxpq(target_source)}')
    """).df()
    c.close()

    # Fit char 3-5 gram TF-IDF on target corpus
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_target = vec.fit_transform(target_df["name_clean"])
    X_query = vec.transform(val_s1["name_clean"])

    retrieved_s1: list[str] = []
    retrieved_sx: list[str] = []

    # Country-partitioned sparse multiplication
    for country in val_s1["country_clean"].unique():
        q_mask = (val_s1["country_clean"] == country).values
        t_mask = (target_df["country_clean"] == country).values

        if not t_mask.any() or not q_mask.any():
            continue

        q_sub = X_query[q_mask]
        t_sub = X_target[t_mask]
        sub_s1_ids = val_s1["s1_id"].values[q_mask]
        sub_target_ids = target_df["sx_id"].values[t_mask]

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
                else:
                    top_k_pos = np.argpartition(c_data, -k)[-k:]
                    chosen_idx = c_indices[top_k_pos]

                s1_curr = sub_s1_ids[start_idx + i]
                for sx_idx in chosen_idx:
                    retrieved_s1.append(s1_curr)
                    retrieved_sx.append(sub_target_ids[sx_idx])
            del sims

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    dt = time.time() - t0

    char_cands = pd.DataFrame({"s1_id": retrieved_s1, "sx_id": retrieved_sx}).drop_duplicates()
    return char_cands, dt, peak_mb


def run_phase_1():
    print("=" * 80)
    print("PHASE 1: CHARACTER N-GRAM ADDITIVE RETRIEVAL EXPERIMENTS")
    print("=" * 80)

    results = []

    for src in ["source2", "source3"]:
        print(f"\nTarget Source: {src.upper()}")
        print("-" * 60)
        bst = lgb.Booster(model_file=str(MODELS_DIR / f"lgb_{src}.txt"))
        thresh = _V2_THRESH[src]

        for split in ["val1", "val2"]:
            s1_all, gt_set, gt_df = get_split_s1_and_gt(src, split)
            total_gold = len(gt_set)

            # Load baseline cache
            base_cache = pd.read_parquet(_CACHE_DIR / f"v2_{src}_{split}.parquet")
            base_cands = base_cache[["s1_id", "sx_id"]].drop_duplicates()
            base_cand_set = set(zip(base_cands["s1_id"], base_cands["sx_id"]))

            # Baseline metrics
            base_present = gt_set & base_cand_set
            base_recall = len(base_present) / total_gold if total_gold > 0 else 0.0

            # Active R50 predictions on baseline
            c = _con()
            c.register("_c", base_cache[["s1_id", "sx_id"]])
            ns = c.execute(f"""
                SELECT c.s1_id, c.sx_id,
                       s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf
                FROM _c c
                JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
                JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
            """).df()
            c.close()
            feat_base = base_cache.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({"s1_no_suf": "", "sx_no_suf": ""})
            X_base = compute_features_parallel(feat_base, n_workers=6)
            scores_base = bst.predict(X_base)
            base_preds = base_cands[scores_base >= thresh]
            base_eval = calculate_macro_f05(base_preds, gt_df, s1_all)

            print(f"\n>>> {src.upper()} - {split.upper()} Baseline:")
            print(f"    Candidates: {len(base_cand_set):,} | Gold recall: {base_recall:.4f} ({len(base_present)}/{total_gold})")
            print(f"    R50 F0.5: {base_eval['macro_f05']:.5f} | P: {base_eval['macro_precision']:.5f} | R: {base_eval['macro_recall']:.5f}")

            for k in [20, 40]:
                print(f"\n  Testing Character TF-IDF (top-k={k})...")
                char_cands, dt, peak_mb = retrieve_char_tfidf_candidates(src, s1_all, k=k)
                char_cand_set = set(zip(char_cands["s1_id"], char_cands["sx_id"]))
                char_present = gt_set & char_cand_set
                char_recall = len(char_present) / total_gold if total_gold > 0 else 0.0

                # Union with baseline
                union_cand_set = base_cand_set | char_cand_set
                union_present = gt_set & union_cand_set
                union_recall = len(union_present) / total_gold if total_gold > 0 else 0.0

                # Newly added candidates
                added_cand_set = char_cand_set - base_cand_set
                recovered_gold = union_present - base_present
                ratio = len(added_cand_set) / max(len(recovered_gold), 1)

                # Distribution of candidates per S1
                union_df = pd.DataFrame(list(union_cand_set), columns=["s1_id", "sx_id"])
                per_s1 = union_df.groupby("s1_id").size()
                avg_cands = float(per_s1.mean())
                p95_cands = float(np.percentile(per_s1, 95))

                print(f"    Char-only candidates: {len(char_cand_set):,} | recall: {char_recall:.4f} ({len(char_present)}/{total_gold})")
                print(f"    Union candidates:     {len(union_cand_set):,} (+{len(added_cand_set):,}) | recall: {union_recall:.4f} (+{len(recovered_gold)} gold)")
                print(f"    Added cands / recovered gold: {ratio:.1f} | Avg cands/S1: {avg_cands:.1f} | P95 cands/S1: {p95_cands:.1f}")
                print(f"    Runtime: {dt:.1f}s | Peak RAM: {peak_mb:.1f}MB")

                # Evaluate downstream R50 on newly added candidates
                if added_cand_set:
                    added_df = pd.DataFrame(list(added_cand_set), columns=["s1_id", "sx_id"])
                    c = _con()
                    c.register("_a", added_df)
                    added_full = c.execute(f"""
                        SELECT a.s1_id, a.sx_id,
                               s1.name_clean     AS s1_name,   sx.name_clean     AS sx_name,
                               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
                               s1.addr_clean     AS s1_addr,   sx.addr_clean     AS sx_addr,
                               COALESCE(s1.postal_code, '') AS s1_post,
                               COALESCE(sx.postal_code, '') AS sx_post
                        FROM _a a
                        JOIN read_parquet('{_s1pq()}') s1 ON a.s1_id = s1.entity_id
                        JOIN read_parquet('{_sxpq(src)}') sx ON a.sx_id = sx.entity_id
                    """).df()
                    c.close()

                    X_added = compute_features_parallel(added_full, n_workers=6)
                    scores_added = bst.predict(X_added)
                    added_preds = added_full[scores_added >= thresh][["s1_id", "sx_id"]]
                    union_preds = pd.concat([base_preds, added_preds], ignore_index=True).drop_duplicates()
                else:
                    union_preds = base_preds

                union_eval = calculate_macro_f05(union_preds, gt_df, s1_all)
                f05_diff = union_eval["macro_f05"] - base_eval["macro_f05"]

                print(f"    Downstream Union R50: F0.5 = {union_eval['macro_f05']:.5f} ({f05_diff:+.5f}) | P = {union_eval['macro_precision']:.5f} | R = {union_eval['macro_recall']:.5f}")

                results.append({
                    "source": src,
                    "split": split,
                    "k": k,
                    "base_recall": base_recall,
                    "char_recall": char_recall,
                    "union_recall": union_recall,
                    "base_cands": len(base_cand_set),
                    "char_cands": len(char_cand_set),
                    "union_cands": len(union_cand_set),
                    "added_cands": len(added_cand_set),
                    "recovered_gold": len(recovered_gold),
                    "added_per_gold": ratio,
                    "avg_cands_s1": avg_cands,
                    "p95_cands_s1": p95_cands,
                    "base_f05": base_eval["macro_f05"],
                    "union_f05": union_eval["macro_f05"],
                    "f05_diff": f05_diff,
                    "union_p": union_eval["macro_precision"],
                    "union_r": union_eval["macro_recall"],
                    "runtime_s": dt,
                    "peak_mb": peak_mb,
                })

    print("\n" + "=" * 90)
    print("PHASE 1 SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Source':<8} {'Split':<5} {'k':<3} {'BaseRec':<8} {'CharRec':<8} {'UnionRec':<9} {'AddedCands':<11} {'RecovGold':<10} {'BaseF0.5':<9} {'UnionF0.5':<10} {'F0.5Diff':<9}")
    print("-" * 90)
    for r in results:
        print(f"{r['source']:<8} {r['split']:<5} {r['k']:<3} {r['base_recall']:<8.4f} {r['char_recall']:<8.4f} {r['union_recall']:<9.4f} "
              f"{r['added_cands']:<11,} {r['recovered_gold']:<10,} {r['base_f05']:<9.5f} {r['union_f05']:<10.5f} {r['f05_diff']:<+9.5f}")

    return results


if __name__ == "__main__":
    run_phase_1()
