"""
V8: Ranking Diagnostics Only.

Read-only diagnostic experiment evaluating whether the R50 matcher fails
due to ranking error, thresholding/competition, or feature representation limits.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import norm_parquet
from src.features import FEATURE_NAMES, compute_features_parallel
from src.matcher import MODELS_DIR

_CACHE_DIR = ROOT / "intermediate" / "v2_cache"
_MODELS_DIR = ROOT / "models"
_OPERATIONAL_THRESHOLDS = {"source2": 0.70, "source3": 0.65}

_H = "abs(hash(entity_id || 'v1')) % 1000"
_SPLIT_WHERE = {
    "val1": f"{_H} >= 0 AND {_H} < 5",
    "val2": f"{_H} >= 25 AND {_H} < 30",
}


def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")


def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")


def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")


def get_clean_gold(src: str, split_name: str) -> tuple[set[str], set[tuple[str, str]], pd.DataFrame]:
    """Retrieve clean, unpolluted ground truth for the target source."""
    prefix = "S2-" if src == "source2" else "S3-"
    where_clause = _SPLIT_WHERE[split_name]

    c = duckdb.connect()
    c.execute(f"""
        CREATE TEMP TABLE _s1 AS
        SELECT entity_id AS s1_id
        FROM read_parquet('{_s1pq()}')
        WHERE {where_clause}
    """)
    s1_all = set(c.execute("SELECT s1_id FROM _s1").df()["s1_id"])

    gt_df = c.execute(f"""
        WITH unnested AS (
            SELECT source1_entity_id AS s1_id,
                   trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
            FROM read_parquet('{_gtpq()}')
            WHERE matched_entity_ids != ''
              AND source1_entity_id IN (SELECT s1_id FROM _s1)
        )
        SELECT s1_id, true_match_id
        FROM unnested
        WHERE true_match_id LIKE '{prefix}%'
    """).df()
    c.close()

    gt_set = set(zip(gt_df["s1_id"], gt_df["true_match_id"]))
    return s1_all, gt_set, gt_df


def load_scored_candidates(src: str, split_name: str) -> tuple[pd.DataFrame, np.ndarray, lgb.Booster]:
    """Load cached candidates, compute 15 features, and score with R50 booster."""
    cache_path = _CACHE_DIR / f"v2_{src}_{split_name}.parquet"
    cache = pd.read_parquet(cache_path)

    c = duckdb.connect()
    c.register("_c", cache[["s1_id", "sx_id"]])
    ns = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
               s1.raw_country AS s1_ctry, sx.raw_country AS sx_ctry,
               s1.raw_name AS s1_raw_name, sx.raw_name AS sx_raw_name,
               s1.raw_address AS s1_raw_addr, sx.raw_address AS sx_raw_addr
        FROM _c c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()

    feat_df = cache.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({
        "s1_no_suf": "", "sx_no_suf": "",
        "s1_raw_name": "", "sx_raw_name": "",
        "s1_raw_addr": "", "sx_raw_addr": "",
    })

    X = compute_features_parallel(feat_df, n_workers=6)
    for f_idx, fname in enumerate(FEATURE_NAMES):
        feat_df[fname] = X[:, f_idx]

    bst = lgb.Booster(model_file=str(_MODELS_DIR / f"lgb_{src}.txt"))
    scores = bst.predict(X)
    feat_df["r50_score"] = scores
    return feat_df, X, bst


def analyze_split(src: str, split_name: str) -> dict:
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC RUN: {src.upper()} | {split_name.upper()}")
    print(f"{'='*70}")
    t0 = time.time()

    s1_all, gt_set, gt_df = get_clean_gold(src, split_name)
    total_gold = len(gt_set)
    total_s1 = len(s1_all)
    threshold = _OPERATIONAL_THRESHOLDS[src]

    feat_df, X, bst = load_scored_candidates(src, split_name)
    print(f"Loaded and scored {len(feat_df):,} candidate pairs in {time.time()-t0:.1f}s")

    # Flag true matches
    feat_df["is_gold"] = [(s, x) in gt_set for s, x in zip(feat_df["s1_id"], feat_df["sx_id"])]

    # ---------------------------------------------------------
    # 1. RANK CANDIDATES WITHIN EACH S1
    # ---------------------------------------------------------
    # Sort by s1_id and r50_score descending
    feat_df.sort_values(by=["s1_id", "r50_score"], ascending=[True, False], inplace=True)
    feat_df["rank"] = feat_df.groupby("s1_id").cumcount() + 1

    # Candidate set size per S1
    cand_counts = feat_df.groupby("s1_id")["sx_id"].count().to_dict()

    # Top scores per S1
    grouped_scores = feat_df.groupby("s1_id")["r50_score"].apply(list).to_dict()

    # ---------------------------------------------------------
    # 2. GOLD RANK DISTRIBUTION
    # ---------------------------------------------------------
    gold_in_cands = feat_df[feat_df["is_gold"]].copy()
    present_gold_count = len(gold_in_cands)
    missed_gold_count = total_gold - present_gold_count

    gold_ranks = gold_in_cands["rank"].values

    top_1 = np.sum(gold_ranks <= 1) / total_gold
    top_3 = np.sum(gold_ranks <= 3) / total_gold
    top_5 = np.sum(gold_ranks <= 5) / total_gold
    top_10 = np.sum(gold_ranks <= 10) / total_gold
    top_20 = np.sum(gold_ranks <= 20) / total_gold
    top_50 = np.sum(gold_ranks <= 50) / total_gold
    top_100 = np.sum(gold_ranks <= 100) / total_gold

    med_rank = float(np.median(gold_ranks)) if len(gold_ranks) else 0.0
    p75_rank = float(np.percentile(gold_ranks, 75)) if len(gold_ranks) else 0.0
    p90_rank = float(np.percentile(gold_ranks, 90)) if len(gold_ranks) else 0.0
    p95_rank = float(np.percentile(gold_ranks, 95)) if len(gold_ranks) else 0.0
    p99_rank = float(np.percentile(gold_ranks, 99)) if len(gold_ranks) else 0.0
    max_rank = int(np.max(gold_ranks)) if len(gold_ranks) else 0

    print("\n--- GOLD RANK DISTRIBUTION ---")
    print(f"Total Gold: {total_gold:,} | Present in Cands: {present_gold_count:,} ({present_gold_count/total_gold*100:.2f}%)")
    print(f"Top-1:   {top_1:.4f} ({top_1*100:.2f}%)")
    print(f"Top-3:   {top_3:.4f} ({top_3*100:.2f}%)")
    print(f"Top-5:   {top_5:.4f} ({top_5*100:.2f}%)")
    print(f"Top-10:  {top_10:.4f} ({top_10*100:.2f}%)")
    print(f"Top-20:  {top_20:.4f} ({top_20*100:.2f}%)")
    print(f"Top-50:  {top_50:.4f} ({top_50*100:.2f}%)")
    print(f"Top-100: {top_100:.4f} ({top_100*100:.2f}%)")
    print(f"Median: {med_rank:.1f} | P75: {p75_rank:.1f} | P90: {p90_rank:.1f} | P95: {p95_rank:.1f} | P99: {p99_rank:.1f} | Max: {max_rank}")

    # ---------------------------------------------------------
    # 3. FALSE POSITIVE RANK & MARGIN ANALYSIS
    # ---------------------------------------------------------
    accepted = feat_df[feat_df["r50_score"] >= threshold].copy()
    tp_df = accepted[accepted["is_gold"]]
    fp_df = accepted[~accepted["is_gold"]].copy()

    # Precompute top-1 score per S1 for FP margin analysis
    top1_map = {s1: scores[0] for s1, scores in grouped_scores.items()}
    fp_df["top1_score"] = fp_df["s1_id"].map(top1_map)
    fp_df["margin"] = fp_df["top1_score"] - fp_df["r50_score"]

    fp_ranks = fp_df["rank"].values
    fp_scores = fp_df["r50_score"].values
    fp_margins = fp_df["margin"].values

    fp_stats = {
        "count": len(fp_df),
        "tp_count": len(tp_df),
        "rank_p50": float(np.percentile(fp_ranks, 50)) if len(fp_ranks) else 0.0,
        "rank_p75": float(np.percentile(fp_ranks, 75)) if len(fp_ranks) else 0.0,
        "rank_p90": float(np.percentile(fp_ranks, 90)) if len(fp_ranks) else 0.0,
        "rank_p95": float(np.percentile(fp_ranks, 95)) if len(fp_ranks) else 0.0,
        "rank_p99": float(np.percentile(fp_ranks, 99)) if len(fp_ranks) else 0.0,
        "score_p50": float(np.percentile(fp_scores, 50)) if len(fp_scores) else 0.0,
        "score_p75": float(np.percentile(fp_scores, 75)) if len(fp_scores) else 0.0,
        "score_p90": float(np.percentile(fp_scores, 90)) if len(fp_scores) else 0.0,
        "score_p95": float(np.percentile(fp_scores, 95)) if len(fp_scores) else 0.0,
        "score_p99": float(np.percentile(fp_scores, 99)) if len(fp_scores) else 0.0,
        "margin_p50": float(np.percentile(fp_margins, 50)) if len(fp_margins) else 0.0,
        "margin_p75": float(np.percentile(fp_margins, 75)) if len(fp_margins) else 0.0,
        "margin_p90": float(np.percentile(fp_margins, 90)) if len(fp_margins) else 0.0,
        "margin_p95": float(np.percentile(fp_margins, 95)) if len(fp_margins) else 0.0,
        "margin_p99": float(np.percentile(fp_margins, 99)) if len(fp_margins) else 0.0,
        "at_rank_1": int(np.sum(fp_ranks == 1)),
        "at_rank_2": int(np.sum(fp_ranks == 2)),
        "at_rank_3": int(np.sum(fp_ranks == 3)),
    }

    print("\n--- FALSE POSITIVE ANALYSIS ---")
    print(f"Accepted TP: {len(tp_df):,} | Accepted FP: {len(fp_df):,}")
    print(f"FP at Rank 1: {fp_stats['at_rank_1']:,} ({fp_stats['at_rank_1']/len(fp_df)*100:.1f}%) | Rank 2: {fp_stats['at_rank_2']:,} | Rank 3: {fp_stats['at_rank_3']:,}")
    print(f"FP Rank P50/P75/P90/P95/P99: {fp_stats['rank_p50']:.1f} / {fp_stats['rank_p75']:.1f} / {fp_stats['rank_p90']:.1f} / {fp_stats['rank_p95']:.1f} / {fp_stats['rank_p99']:.1f}")
    print(f"FP Score P50/P75/P90/P95/P99: {fp_stats['score_p50']:.4f} / {fp_stats['score_p75']:.4f} / {fp_stats['score_p90']:.4f} / {fp_stats['score_p95']:.4f} / {fp_stats['score_p99']:.4f}")
    print(f"FP Margin P50/P75/P90/P95/P99: {fp_stats['margin_p50']:.4f} / {fp_stats['margin_p75']:.4f} / {fp_stats['margin_p90']:.4f} / {fp_stats['margin_p95']:.4f} / {fp_stats['margin_p99']:.4f}")

    # ---------------------------------------------------------
    # 4. CANDIDATE-COMPETITION ANALYSIS ACROSS S1 GROUPS
    # ---------------------------------------------------------
    # Map S1 gold count
    s1_gold_map = gt_df.groupby("s1_id")["true_match_id"].apply(set).to_dict()

    # Classify each S1 into one of the 4 groups:
    # 1. Correct Top-1: top-1 candidate is in s1_gold_map
    # 2. Correct Candidate not Top-1: has gold matches, but top-1 is not in gold
    # 3. True Singleton with FP: has 0 gold matches, but >= 1 candidate accepted
    # 4. Multi-Match: has >= 2 gold matches
    # Build per-S1 metrics table
    s1_metrics = []
    top1_pairs = feat_df[feat_df["rank"] == 1].set_index("s1_id")

    for s1 in s1_all:
        g_set = s1_gold_map.get(s1, set())
        scores = grouped_scores.get(s1, [])
        c_count = cand_counts.get(s1, 0)
        t1 = scores[0] if len(scores) >= 1 else 0.0
        t2 = scores[1] if len(scores) >= 2 else 0.0
        t3 = scores[2] if len(scores) >= 3 else 0.0
        m12 = t1 - t2
        m13 = t1 - t3
        n_above = sum(s >= threshold for s in scores)

        has_top1 = s1 in top1_pairs.index
        top1_sx = top1_pairs.loc[s1, "sx_id"] if has_top1 else None
        top1_is_gold = top1_sx in g_set if top1_sx else False

        is_multi = len(g_set) >= 2
        is_correct_top1 = len(g_set) > 0 and top1_is_gold
        is_wrong_top1 = len(g_set) > 0 and not top1_is_gold
        is_singleton_fp = len(g_set) == 0 and n_above > 0

        s1_metrics.append({
            "s1_id": s1,
            "gold_count": len(g_set),
            "cand_count": c_count,
            "top1_score": t1,
            "top2_score": t2,
            "top3_score": t3,
            "margin_1_2": m12,
            "margin_1_3": m13,
            "n_above": n_above,
            "is_correct_top1": is_correct_top1,
            "is_wrong_top1": is_wrong_top1,
            "is_singleton_fp": is_singleton_fp,
            "is_multi": is_multi,
        })

    s1_mdf = pd.DataFrame(s1_metrics)

    print("\n--- CANDIDATE COMPETITION BY GROUP ---")
    group_stats = {}
    for grp_name, mask in [
        ("Entities with correct top-1", s1_mdf["is_correct_top1"]),
        ("Entities where correct cand is not top-1", s1_mdf["is_wrong_top1"]),
        ("True singletons with FP prediction", s1_mdf["is_singleton_fp"]),
        ("Multi-match entities", s1_mdf["is_multi"]),
    ]:
        sub = s1_mdf[mask]
        group_stats[grp_name] = {
            "n_entities": len(sub),
            "cand_count_mean": sub["cand_count"].mean() if len(sub) else 0.0,
            "cand_count_med": sub["cand_count"].median() if len(sub) else 0.0,
            "top1_score_mean": sub["top1_score"].mean() if len(sub) else 0.0,
            "top1_score_med": sub["top1_score"].median() if len(sub) else 0.0,
            "top2_score_mean": sub["top2_score"].mean() if len(sub) else 0.0,
            "top2_score_med": sub["top2_score"].median() if len(sub) else 0.0,
            "margin_1_2_mean": sub["margin_1_2"].mean() if len(sub) else 0.0,
            "margin_1_2_med": sub["margin_1_2"].median() if len(sub) else 0.0,
            "n_above_mean": sub["n_above"].mean() if len(sub) else 0.0,
        }
        print(f"[{grp_name}] (N={len(sub):,}):")
        print(f"   Cand count: mean={sub['cand_count'].mean():.1f}, med={sub['cand_count'].median():.0f}")
        print(f"   Top-1 score: mean={sub['top1_score'].mean():.4f}, med={sub['top1_score'].median():.4f}")
        print(f"   Top-2 score: mean={sub['top2_score'].mean():.4f}, med={sub['top2_score'].median():.4f}")
        print(f"   Margin 1-2:  mean={sub['margin_1_2'].mean():.4f}, med={sub['margin_1_2'].median():.4f}")
        print(f"   Above thresh: mean={sub['n_above'].mean():.2f}")

    # Overall S1 competition metrics
    overall_comp = {
        "top1_score_med": float(s1_mdf["top1_score"].median()),
        "top2_score_med": float(s1_mdf["top2_score"].median()),
        "top3_score_med": float(s1_mdf["top3_score"].median()),
        "margin_1_2_med": float(s1_mdf["margin_1_2"].median()),
        "margin_1_3_med": float(s1_mdf["margin_1_3"].median()),
        "cand_count_med": float(s1_mdf["cand_count"].median()),
    }

    # ---------------------------------------------------------
    # 5. FEATURE SEPARABILITY ANALYSIS
    # ---------------------------------------------------------
    print("\n--- FEATURE SEPARABILITY ANALYSIS (TRUE vs FALSE CANDIDATES) ---")
    # For every feature, compute distribution on true vs false candidate pairs
    true_mask = feat_df["is_gold"].values
    false_mask = ~true_mask

    feat_analysis = []
    for f_idx, fname in enumerate(FEATURE_NAMES):
        vals = X[:, f_idx]
        t_vals = vals[true_mask]
        f_vals = vals[false_mask]

        t_mean, t_med = float(np.mean(t_vals)), float(np.median(t_vals))
        t_p25, t_p75, t_p90 = float(np.percentile(t_vals, 25)), float(np.percentile(t_vals, 75)), float(np.percentile(t_vals, 90))

        f_mean, f_med = float(np.mean(f_vals)), float(np.median(f_vals))
        f_p25, f_p75, f_p90 = float(np.percentile(f_vals, 25)), float(np.percentile(f_vals, 75)), float(np.percentile(f_vals, 90))

        # Separation metric: Cohen's d style or median difference
        pooled_std = np.sqrt(0.5 * (np.var(t_vals) + np.var(f_vals))) + 1e-6
        separation_d = (t_mean - f_mean) / pooled_std
        separation_med = t_med - f_med

        feat_analysis.append({
            "feature": fname,
            "true_mean": t_mean, "true_med": t_med, "true_p25": t_p25, "true_p75": t_p75, "true_p90": t_p90,
            "false_mean": f_mean, "false_med": f_med, "false_p25": f_p25, "false_p75": f_p75, "false_p90": f_p90,
            "separation_d": float(separation_d),
            "separation_med": float(separation_med),
        })

    feat_df_analysis = pd.DataFrame(feat_analysis).sort_values(by="separation_d", ascending=False)
    for row in feat_df_analysis.itertuples():
        print(f"  {row.feature:18s} | True med={row.true_med:.4f} (mean={row.true_mean:.4f}) | False med={row.false_med:.4f} (mean={row.false_mean:.4f}) | sep_d={row.separation_d:.4f} | sep_med={row.separation_med:.4f}")

    top5_features = feat_df_analysis.head(5)["feature"].tolist()
    print(f"\nTop 5 separating features: {top5_features}")

    # ---------------------------------------------------------
    # 6. HARD-CASE ERROR BREAKDOWN
    # ---------------------------------------------------------
    print("\n--- HARD-CASE ERROR BREAKDOWN ---")
    # Subgroup A: Gold pairs ranked below Top-1
    gold_below_top1 = feat_df[feat_df["is_gold"] & (feat_df["rank"] > 1)].copy()
    print(f"Total gold pairs ranked below Top-1: {len(gold_below_top1):,} ({len(gold_below_top1)/total_gold*100:.2f}%)")

    # Subgroup B: False positives ranked Top-1
    fp_at_top1 = feat_df[(~feat_df["is_gold"]) & (feat_df["rank"] == 1) & (feat_df["r50_score"] >= threshold)].copy()
    print(f"Total false positives ranked Top-1: {len(fp_at_top1):,}")

    # Subgroup C: False positives with score >= 0.95
    fp_high_score = feat_df[(~feat_df["is_gold"]) & (feat_df["r50_score"] >= 0.95)].copy()
    print(f"Total false positives with R50 score >= 0.95: {len(fp_high_score):,}")

    # Deterministic categorization for high-scoring false positives:
    # Inspect a deterministic sample of up to 500 high-scoring FPs
    c_sample = fp_high_score.head(500).copy() if len(fp_high_score) > 0 else fp_df.head(500).copy()
    
    categories = {
        "same_near_identical_name": 0,
        "cross_script_collision": 0,
        "alias_domain_variant": 0,
        "different_location": 0,
        "address_noise": 0,
        "short_common_name": 0,
        "other": 0,
    }

    import re
    domain_re = re.compile(r"\b(com|org|net|io|in|co|biz|info)\b")
    non_ascii_re = re.compile(r"[^\x00-\x7F]")

    for row in c_sample.itertuples():
        n1 = str(row.s1_raw_name).strip().lower()
        n2 = str(row.sx_raw_name).strip().lower()
        a1 = str(row.s1_raw_addr).strip().lower()
        a2 = str(row.sx_raw_addr).strip().lower()

        is_cross_script = bool(non_ascii_re.search(n1)) != bool(non_ascii_re.search(n2))
        has_domain = bool(domain_re.search(n1)) or bool(domain_re.search(n2))
        is_short = len(n1) <= 5 or len(n2) <= 5

        # Check name vs address mismatch
        if is_cross_script:
            categories["cross_script_collision"] += 1
        elif has_domain and ("." in n1 or "." in n2):
            categories["alias_domain_variant"] += 1
        elif is_short:
            categories["short_common_name"] += 1
        elif row.name_exact == 1.0 or row.name_ratio >= 0.95:
            if row.addr_ratio < 0.40:
                categories["different_location"] += 1
            else:
                categories["same_near_identical_name"] += 1
        elif row.addr_ratio < 0.30:
            categories["different_location"] += 1
        elif abs(len(a1) - len(a2)) > 30:
            categories["address_noise"] += 1
        else:
            categories["other"] += 1

    total_cat = sum(categories.values())
    cat_pct = {k: v / total_cat * 100 if total_cat else 0.0 for k, v in categories.items()}
    print("Error Categories distribution on sample:", cat_pct)

    return {
        "src": src,
        "split": split_name,
        "total_gold": total_gold,
        "present_gold": present_gold_count,
        "top_1": top_1,
        "top_3": top_3,
        "top_5": top_5,
        "top_10": top_10,
        "top_20": top_20,
        "top_50": top_50,
        "top_100": top_100,
        "med_rank": med_rank,
        "p75_rank": p75_rank,
        "p90_rank": p90_rank,
        "p95_rank": p95_rank,
        "p99_rank": p99_rank,
        "max_rank": max_rank,
        "fp_stats": fp_stats,
        "overall_comp": overall_comp,
        "group_stats": group_stats,
        "feat_analysis": feat_df_analysis,
        "top5_features": top5_features,
        "categories": categories,
        "cat_pct": cat_pct,
        "elapsed": time.time() - t0,
    }


def main():
    tracemalloc.start()
    t_start = time.time()

    print("=" * 80)
    print("V8: RANKING DIAGNOSTICS EXPERIMENT")
    print("Operational R50 Matcher — Dual Untouched Validation Splits")
    print("=" * 80)

    results = {}
    for src in ["source2", "source3"]:
        for split in ["val1", "val2"]:
            results[(src, split)] = analyze_split(src, split)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - t_start

    print("\n" + "=" * 80)
    print("ALL V8 DIAGNOSTICS COMPLETED")
    print(f"Total Runtime: {total_time:.1f}s | Peak RAM: {peak / 1024 / 1024:.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    main()
