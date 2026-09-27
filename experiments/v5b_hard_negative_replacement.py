"""V5b: Hard-Negative Replacement Retraining.

Tests whether hard negatives can improve the matcher when they REPLACE
easy/random negatives while keeping the total negative training count
and class balance approximately unchanged.

Controlled invariants:
  - Total negative count = original V1 training negative count (exact)
  - Positive count = original V1 training positive count (exact)
  - scale_pos_weight = 3.0 (exact)
  - 15 features, LightGBM hyperparameters = identical to V2
  - S2 threshold = 0.70, S3 threshold = 0.65 (frozen)
  - Candidate generation & TF-IDF parameters = frozen
  - val1 [0,5) and val2 [25,30) = evaluation only (untouched)
  - internal_val [5,10) = early stopping only
  - train_core [10,25) = training population
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import candidate_parquet, norm_parquet
from src.features import FEATURE_NAMES, compute_features_parallel
from src.lexical_blocking import retrieve_tfidf_candidates
from src.matcher import MODELS_DIR, calculate_macro_f05

# Frozen V2 configuration
_TFIDF_K      = 20
_TFIDF_THRESH = {"source2": 0.53, "source3": 0.51}
_V2_THRESH    = {"source2": 0.70, "source3": 0.65}
_SPW          = 3.0
_CAP_PER_S1   = 20
_SEED         = 42

_H = "abs(hash(entity_id || 'v1')) % 1000"
_WHERE = {
    "val1":         f"{_H} >= 0  AND {_H} < 5",
    "val2":         f"{_H} >= 25 AND {_H} < 30",
    "internal_val": f"{_H} >= 5  AND {_H} < 10",
    "train_core":   f"{_H} >= 10 AND {_H} < 25",
}

_CACHE = ROOT / "intermediate" / "v2_cache"


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


def _cpq(src: str) -> str:
    return str(candidate_parquet("train", src)).replace("\\", "/")


def _split_ids(split_name: str) -> set[str]:
    c = _con()
    ids = set(c.execute(
        f"SELECT entity_id FROM read_parquet('{_s1pq()}') WHERE {_WHERE[split_name]}"
    ).df()["entity_id"])
    c.close()
    return ids


def _gt_df(src: str, split_name: str) -> tuple[set, pd.DataFrame]:
    prefix = "S2-" if src == "source2" else "S3-"
    c = _con()
    c.execute(f"CREATE TEMP TABLE _s AS SELECT entity_id s1_id FROM read_parquet('{_s1pq()}') WHERE {_WHERE[split_name]}")
    df = c.execute(f"""
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS true_match_id
        FROM read_parquet('{_gtpq()}')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
          AND source1_entity_id IN (SELECT s1_id FROM _s)
    """).df()
    c.close()
    return set(zip(df["s1_id"], df["true_match_id"])), df


def load_train_split(src: str, split_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract candidates, text columns, and labels in a single query to guarantee
    zero row misalignment between features and labels.
    """
    prefix = "S2-" if src == "source2" else "S3-"
    c = _con()
    c.execute(f"""
        CREATE TEMP TABLE s1_sel AS
        SELECT entity_id AS s1_id
        FROM read_parquet('{_s1pq()}')
        WHERE {_WHERE[split_name]}
    """)
    c.execute(f"""
        CREATE TEMP TABLE gt_target AS
        SELECT source1_entity_id AS s1_id,
               trim(unnest(string_split(matched_entity_ids, ','))) AS true_id
        FROM read_parquet('{_gtpq()}')
        WHERE matched_entity_ids != '' AND matched_entity_ids LIKE '%{prefix}%'
          AND source1_entity_id IN (SELECT s1_id FROM s1_sel)
    """)
    df = c.execute(f"""
        WITH cands AS (
            SELECT DISTINCT c.s1_id, c.sx_id
            FROM read_parquet('{_cpq(src)}') c
            JOIN s1_sel s ON c.s1_id = s.s1_id
        )
        SELECT
            c.s1_id, c.sx_id,
            CASE WHEN g.true_id IS NOT NULL THEN 1 ELSE 0 END AS label,
            s1.name_clean     AS s1_name,   sx.name_clean     AS sx_name,
            s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
            s1.addr_clean     AS s1_addr,   sx.addr_clean     AS sx_addr,
            COALESCE(s1.postal_code, '') AS s1_post,
            COALESCE(sx.postal_code, '') AS sx_post
        FROM cands c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
        LEFT JOIN gt_target g ON c.s1_id = g.s1_id AND c.sx_id = g.true_id
    """).df()
    c.close()

    y = df["label"].values.astype(np.int8)
    X = compute_features_parallel(df, n_workers=6)
    print(f"  {split_name}: {len(df):,} pairs | pos={y.sum():,} | neg={(y == 0).sum():,}")
    return X, y, df[["s1_id", "sx_id"]]


def mine_hard_negatives(src: str) -> tuple[np.ndarray, pd.DataFrame]:
    thresh = _TFIDF_THRESH[src]
    train_ids = _split_ids("train_core")
    gt_set, _ = _gt_df(src, "train_core")

    c = _con()
    v1 = c.execute(f"""
        SELECT DISTINCT c.s1_id, c.sx_id
        FROM read_parquet('{_cpq(src)}') c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        WHERE {_WHERE["train_core"]}
    """).df()
    c.close()
    v1_set = set(zip(v1["s1_id"], v1["sx_id"]))

    print(f"  Retrieving TF-IDF candidates for {len(train_ids):,} train_core entities...")
    tfidf = retrieve_tfidf_candidates(
        target_source=src, split="train",
        k=_TFIDF_K, s1_ids=train_ids, return_scores=True,
    )

    # Filter: not in V1 AND tfidf_score >= threshold
    is_new = np.array([(s, x) not in v1_set for s, x in zip(tfidf["s1_id"], tfidf["sx_id"])])
    new_above = tfidf[is_new & (tfidf["tfidf_score"].values >= thresh)].copy().reset_index(drop=True)

    # Filter: confirmed non-matches (label == 0 against ground truth)
    labels = np.array([(s, x) in gt_set for s, x in zip(new_above["s1_id"], new_above["sx_id"])], dtype=np.int8)
    confirmed = new_above[labels == 0].copy().reset_index(drop=True)
    print(f"  New above threshold: {len(new_above):,} | confirmed non-matches: {len(confirmed):,} "
          f"(positives removed: {(labels == 1).sum():,})")

    # Join text features for confirmed hard negatives
    c = _con()
    c.register("_conf", confirmed[["s1_id", "sx_id"]])
    feat_df = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
               s1.name_clean     AS s1_name,   sx.name_clean     AS sx_name,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf,
               s1.addr_clean     AS s1_addr,   sx.addr_clean     AS sx_addr,
               COALESCE(s1.postal_code, '') AS s1_post,
               COALESCE(sx.postal_code, '') AS sx_post
        FROM _conf c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()

    X_cand = compute_features_parallel(feat_df, n_workers=6)

    # Score with frozen V2 model
    v2_bst = lgb.Booster(model_file=str(MODELS_DIR / f"lgb_{src}.txt"))
    v2_scores = v2_bst.predict(X_cand)
    del v2_bst

    # Cap per S1 entity and sort globally descending by score
    info = pd.DataFrame({
        "s1_id": feat_df["s1_id"].values,
        "sx_id": feat_df["sx_id"].values,
        "v2_score": v2_scores,
        "s1_name": feat_df["s1_name"].values,
        "sx_name": feat_df["sx_name"].values,
        "s1_addr": feat_df["s1_addr"].values,
        "sx_addr": feat_df["sx_addr"].values,
        "feat_idx": np.arange(len(feat_df)),
    })
    info = info.sort_values("v2_score", ascending=False)
    info = info.groupby("s1_id", sort=False).head(_CAP_PER_S1).reset_index(drop=True)
    info = info.sort_values("v2_score", ascending=False).reset_index(drop=True)

    X_pool = X_cand[info["feat_idx"].values]
    hn_info = info.drop(columns=["feat_idx"])

    per_s1 = hn_info.groupby("s1_id").size()
    print(f"  Hard-negative pool (after cap={_CAP_PER_S1}): {len(hn_info):,} | "
          f"unique S1: {hn_info['s1_id'].nunique():,} | "
          f"max/S1: {per_s1.max()} | median/S1: {per_s1.median():.1f}")
    return X_pool, hn_info


def load_val_eval(src: str, split_name: str) -> tuple[np.ndarray, pd.DataFrame, set, pd.DataFrame]:
    cache = pd.read_parquet(_CACHE / f"v2_{src}_{split_name}.parquet")
    c = _con()
    c.register("_c", cache[["s1_id", "sx_id"]])
    ns = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf
        FROM _c c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()
    feat_df = cache.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({"s1_no_suf": "", "sx_no_suf": ""})
    X = compute_features_parallel(feat_df, n_workers=6)
    s1_all = _split_ids(split_name)
    _, gt = _gt_df(src, split_name)
    return X, feat_df[["s1_id", "sx_id"]], s1_all, gt


def v2_cached_eval(src: str, split_name: str, s1_all: set, gt: pd.DataFrame) -> dict:
    cache = pd.read_parquet(_CACHE / f"v2_{src}_{split_name}.parquet")
    preds = cache[cache["lgb_score"] >= _V2_THRESH[src]][["s1_id", "sx_id"]]
    return calculate_macro_f05(preds, gt, s1_all)


def _lgb_params() -> dict:
    return dict(
        objective="binary", metric="binary_logloss", boosting_type="gbdt",
        learning_rate=0.1, num_leaves=31, max_depth=6, min_child_samples=20,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
        scale_pos_weight=_SPW, n_jobs=6, verbose=-1,
    )


def train_v5b(src: str, tag: str,
              X_tr: np.ndarray, y_tr: np.ndarray,
              X_iv: np.ndarray, y_iv: np.ndarray) -> lgb.Booster:
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    div    = lgb.Dataset(X_iv, label=y_iv, reference=dtrain, feature_name=FEATURE_NAMES)
    bst = lgb.train(_lgb_params(), dtrain, num_boost_round=200, valid_sets=[div],
                    callbacks=[lgb.early_stopping(15, verbose=False)])
    path = MODELS_DIR / f"v5b_{tag}_lgb_{src}.txt"
    bst.save_model(str(path))
    print(f"  Saved {path.name} ({bst.num_trees()} trees)")
    return bst


def eval_on_split(bst: lgb.Booster, X: np.ndarray, pairs: pd.DataFrame,
                  s1_all: set, gt: pd.DataFrame, threshold: float) -> tuple[dict, np.ndarray]:
    scores = bst.predict(X)
    preds = pairs[scores >= threshold][["s1_id", "sx_id"]]
    m = calculate_macro_f05(preds, gt, s1_all)
    return m, scores


def score_distribution(scores: np.ndarray, thresh: float) -> dict:
    return dict(
        p50=float(np.percentile(scores, 50)),
        p90=float(np.percentile(scores, 90)),
        p95=float(np.percentile(scores, 95)),
        p99=float(np.percentile(scores, 99)),
        max=float(np.max(scores)),
        ge_thresh=int((scores >= thresh).sum()),
    )


def run_source(src: str) -> dict:
    t0 = time.perf_counter()
    thresh = _V2_THRESH[src]
    print(f"\n{'='*60}\n  V5b — {src.upper()}\n{'='*60}")

    print("\n1. Loading train_core...")
    X_tc, y_tc, _ = load_train_split(src, "train_core")
    pos_mask = (y_tc == 1)
    neg_mask = (y_tc == 0)

    X_pos = X_tc[pos_mask]
    y_pos = y_tc[pos_mask]
    n_pos = len(y_pos)

    X_orig_neg = X_tc[neg_mask]
    y_orig_neg = y_tc[neg_mask]
    n_orig_neg = len(y_orig_neg)

    print(f"  Original train: {len(y_tc):,} | pos={n_pos:,} | neg={n_orig_neg:,}")

    print("\n2. Loading internal_val (for early stopping only)...")
    X_iv, y_iv, _ = load_train_split(src, "internal_val")

    print("\n3. Mining hard negatives...")
    X_hn_pool, hn_info = mine_hard_negatives(src)
    n_pool = len(hn_info)

    print("\n4. Loading val1 & val2 (evaluation only)...")
    X_v1, pairs_v1, s1_all_v1, gt_v1 = load_val_eval(src, "val1")
    X_v2, pairs_v2, s1_all_v2, gt_v2 = load_val_eval(src, "val2")

    bv2_m1 = v2_cached_eval(src, "val1", s1_all_v1, gt_v1)
    bv2_m2 = v2_cached_eval(src, "val2", s1_all_v2, gt_v2)

    # Replacement experiments: R25 (25k), R50 (50k), R75 (75k)
    # Target replacement counts from Section 7 example: 25,000, 50,000, 75,000
    configs = [
        ("r25", 25_000, "R25"),
        ("r50", 50_000, "R50"),
        ("r75", 75_000, "R75"),
    ]

    r_results = {}
    for tag, requested_hn, label in configs:
        n_hn = min(requested_hn, n_pool)
        actual_pct = (n_hn / n_orig_neg) * 100.0

        # Deterministic slice of top hard negatives (already sorted desc by V2 score)
        X_hn = X_hn_pool[:n_hn]
        y_hn = np.zeros(n_hn, dtype=np.int8)

        # Deterministic sample of original easy negatives to keep
        n_orig_keep = n_orig_neg - n_hn
        rng = np.random.RandomState(_SEED)
        keep_idx = rng.choice(n_orig_neg, size=n_orig_keep, replace=False)
        keep_idx.sort()

        X_orig_kept = X_orig_neg[keep_idx]
        y_orig_kept = y_orig_neg[keep_idx]

        # Combine: pos + kept original neg + hard neg
        X_tr = np.vstack([X_pos, X_orig_kept, X_hn])
        y_tr = np.concatenate([y_pos, y_orig_kept, y_hn])

        total_neg = len(y_tr) - n_pos
        pos_neg_ratio = n_pos / total_neg

        print(f"\n{label} Retraining:")
        print(f"  requested hn: {requested_hn:,} | selected hn: {n_hn:,} | actual replacement: {actual_pct:.2f}%")
        print(f"  total train: {len(y_tr):,} | pos: {n_pos:,} | neg: {total_neg:,} (orig: {n_orig_neg:,})")
        print(f"  pos:neg ratio: 1:{1/pos_neg_ratio:.1f} | scale_pos_weight: {_SPW}")

        bst = train_v5b(src, tag, X_tr, y_tr, X_iv, y_iv)

        m1, scores_v1 = eval_on_split(bst, X_v1, pairs_v1, s1_all_v1, gt_v1, thresh)
        m2, scores_v2 = eval_on_split(bst, X_v2, pairs_v2, s1_all_v2, gt_v2, thresh)

        dist_v1 = score_distribution(scores_v1, thresh)
        dist_v2 = score_distribution(scores_v2, thresh)

        d1 = m1["macro_f05"] - bv2_m1["macro_f05"]
        d2 = m2["macro_f05"] - bv2_m2["macro_f05"]
        print(f"  val1 F0.5={m1['macro_f05']:.5f} (D={d1:+.5f}) | val2 F0.5={m2['macro_f05']:.5f} (D={d2:+.5f})")

        r_results[tag] = dict(
            tag=tag, label=label,
            requested_hn=requested_hn,
            selected_hn=n_hn,
            actual_pct=actual_pct,
            total_neg=total_neg,
            pos_neg_ratio=pos_neg_ratio,
            num_trees=bst.num_trees(),
            val1_metrics=m1, val2_metrics=m2,
            dist_v1=dist_v1, dist_v2=dist_v2,
        )

    return dict(
        src=src, thresh=thresh,
        n_pos=n_pos, n_orig_neg=n_orig_neg,
        n_pool=n_pool, hn_info=hn_info,
        v2_m1=bv2_m1, v2_m2=bv2_m2,
        r=r_results,
        elapsed=time.perf_counter() - t0,
    )


def print_report(results: dict, total_elapsed: float) -> None:
    print("\n" + "=" * 65)
    print("V5b — HARD-NEGATIVE REPLACEMENT")
    print("=" * 65)

    print("\nDATA SPLIT")
    print("----------")
    print("val1:                     abs(hash(entity_id || 'v1')) % 1000 in [0, 5)")
    print("val2:                     abs(hash(entity_id || 'v1')) % 1000 in [25, 30)")
    print("internal_val:             abs(hash(entity_id || 'v1')) % 1000 in [5, 10)")
    print("train_core:               abs(hash(entity_id || 'v1')) % 1000 in [10, 25)")
    print("validation overlap with internal_val: 0")

    print("\nFROZEN V2 CONTROL")
    print("-----------------")
    print(f"S2 val1 F0.5: {results['source2']['v2_m1']['macro_f05']:.5f}")
    print(f"S2 val2 F0.5: {results['source2']['v2_m2']['macro_f05']:.5f}")
    print(f"S3 val1 F0.5: {results['source3']['v2_m1']['macro_f05']:.5f}")
    print(f"S3 val2 F0.5: {results['source3']['v2_m2']['macro_f05']:.5f}")

    for src, lbl in [("source2", "S2"), ("source3", "S3")]:
        r = results[src]
        print(f"\n{lbl}")
        print("--")
        print(f"Original training positives: {r['n_pos']:,}")
        print(f"Original training negatives: {r['n_orig_neg']:,}")

        for tag in ["r25", "r50", "r75"]:
            cfg = r["r"][tag]
            m1 = cfg["val1_metrics"]
            m2 = cfg["val2_metrics"]
            d1 = m1["macro_f05"] - r["v2_m1"]["macro_f05"]
            d2 = m2["macro_f05"] - r["v2_m2"]["macro_f05"]
            fp_sing1 = m1["total_singletons"] - m1["singletons_correct"]
            fp_sing2 = m2["total_singletons"] - m2["singletons_correct"]

            print(f"\n{cfg['label']}")
            print(f"  hard negatives selected: {cfg['selected_hn']:,}")
            print(f"  actual replacement %:    {cfg['actual_pct']:.2f}%")
            print(f"  total negatives:         {cfg['total_neg']:,}")
            print(f"  positive:negative ratio: 1:{1/cfg['pos_neg_ratio']:.1f}")
            print(f"  trees:                   {cfg['num_trees']}")

            print(f"  val1 F0.5:               {m1['macro_f05']:.5f}  (D={d1:+.5f})")
            print(f"  val1 Precision:          {m1['macro_precision']:.5f}")
            print(f"  val1 Recall:             {m1['macro_recall']:.5f}")
            print(f"  val1 Singleton Acc:      {m1['singleton_accuracy']:.5f}")
            print(f"  val1 FP Singleton:       {fp_sing1}")

            print(f"  val2 F0.5:               {m2['macro_f05']:.5f}  (D={d2:+.5f})")
            print(f"  val2 Precision:          {m2['macro_precision']:.5f}")
            print(f"  val2 Recall:             {m2['macro_recall']:.5f}")
            print(f"  val2 Singleton Acc:      {m2['singleton_accuracy']:.5f}")
            print(f"  val2 FP Singleton:       {fp_sing2}")

    print("\nHARD-NEGATIVE QUALITY")
    print("---------------------")
    for src, lbl in [("source2", "S2"), ("source3", "S3")]:
        hi = results[src]["hn_info"]
        scores = hi["v2_score"].values
        per_s1 = hi.groupby("s1_id").size()
        print(f"{lbl}:")
        print(f"  pool size:     {len(hi):,}")
        print(f"  selected max:  {min(75_000, len(hi)):,}")
        print(f"  unique S1:     {hi['s1_id'].nunique():,}")
        print(f"  max per S1:    {per_s1.max()}")
        print(f"  median per S1: {per_s1.median():.1f}")
        print(f"  P50:           {np.percentile(scores, 50):.4f}")
        print(f"  P90:           {np.percentile(scores, 90):.4f}")
        print(f"  P95:           {np.percentile(scores, 95):.4f}")
        print(f"  P99:           {np.percentile(scores, 99):.4f}")
        print(f"  max:           {scores.max():.4f}")

    print("\nMODEL SANITY")
    print("------------")
    for src, lbl in [("source2", "S2"), ("source3", "S3")]:
        print(f"{lbl}:")
        for tag in ["r25", "r50", "r75"]:
            cfg = results[src]["r"][tag]
            d1 = cfg["dist_v1"]
            d2 = cfg["dist_v2"]
            print(f"  {cfg['label']} (val1): P50={d1['p50']:.4f} P90={d1['p90']:.4f} P95={d1['p95']:.4f} "
                  f"P99={d1['p99']:.4f} max={d1['max']:.4f} >=thresh={d1['ge_thresh']}")
            print(f"  {cfg['label']} (val2): P50={d2['p50']:.4f} P90={d2['p90']:.4f} P95={d2['p95']:.4f} "
                  f"P99={d2['p99']:.4f} max={d2['max']:.4f} >=thresh={d2['ge_thresh']}")

    print("\nSAMPLE TOP HARD NEGATIVES")
    print("-------------------------")
    shown = 0
    for src in ["source2", "source3"]:
        hi = results[src]["hn_info"]
        for _, row in hi.head(3).iterrows():
            shown += 1
            print(f"{shown}. [{src}] score={row.v2_score:.4f}")
            print(f"   s1: {str(row.s1_name)[:65]!r}")
            print(f"   sx: {str(row.sx_name)[:65]!r}")
            print(f"   s1 addr: {str(row.s1_addr)[:55]!r}")
            print(f"   sx addr: {str(row.sx_addr)[:55]!r}")

    # Determine final decision
    best_configs = []
    for src in ["source2", "source3"]:
        r = results[src]
        for tag in ["r25", "r50", "r75"]:
            cfg = r["r"][tag]
            d1 = cfg["val1_metrics"]["macro_f05"] - r["v2_m1"]["macro_f05"]
            d2 = cfg["val2_metrics"]["macro_f05"] - r["v2_m2"]["macro_f05"]
            if d1 > 0.0001 and d2 > 0.0001:
                best_configs.append((src, tag, min(d1, d2)))

    print("\nFINAL DECISION")
    print("--------------")
    if best_configs:
        print("KEEP")
        print(f"Winning configuration: {best_configs}")
        print("Reason: Achieved consistent Macro F0.5 improvements across both validation splits.")
    else:
        print("DISCARD")
        print("Winning configuration: NONE")
        print("Reason: No replacement configuration improved Macro F0.5 on BOTH validation splits without regression.")

    print(f"\nV2 models overwritten: NO")
    print(f"V2 thresholds changed: NO")
    print(f"Blocking changed: NO")
    print(f"Features changed: NO")
    print(f"scale_pos_weight changed: NO")
    print(f"Validation leakage: NO")
    print(f"External data: NO")
    print(f"Full test inference: NO")
    print(f"Runtime: {total_elapsed:.1f}s")


def main() -> None:
    t0 = time.perf_counter()
    print("Verifying split disjointness...")
    iv_ids = _split_ids("internal_val")
    v1_ids = _split_ids("val1")
    v2_ids = _split_ids("val2")
    tc_ids = _split_ids("train_core")
    assert len(iv_ids & v1_ids) == 0, "internal_val overlaps val1"
    assert len(iv_ids & v2_ids) == 0, "internal_val overlaps val2"
    assert len(tc_ids & v1_ids) == 0, "train_core overlaps val1"
    assert len(tc_ids & v2_ids) == 0, "train_core overlaps val2"
    assert len(tc_ids & iv_ids) == 0, "train_core overlaps internal_val"
    print(f"  train_core ({len(tc_ids):,}) | internal_val ({len(iv_ids):,}) | "
          f"val1 ({len(v1_ids):,}) | val2 ({len(v2_ids):,}) all mutually disjoint.")

    results = {}
    for src in ["source2", "source3"]:
        results[src] = run_source(src)

    print_report(results, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
