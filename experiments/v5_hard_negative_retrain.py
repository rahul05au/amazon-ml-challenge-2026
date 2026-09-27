"""V5: Clean Hard-Negative Retraining.

Single controlled variable: additional confirmed hard-negative training examples.
Everything else is identical to V2 (features, architecture, SPW=3.0, thresholds).

Data splits — all use  abs(hash(entity_id || 'v1')) % 1000:
  val1:          [0,  5)   evaluation only — never touched
  val2:          [25, 30)  evaluation only — never touched
  internal_val:  [5,  10)  early stopping only — not used for model selection
  train_core:    [10, 25)  training data

  Disjointness guaranteed by non-overlapping integer ranges of the same hash.

Hard-negative selection (per source):
  1. Retrieve TF-IDF candidates (frozen k=20) for train_core S1 entities.
  2. Keep only: not in V1 AND tfidf_score >= frozen prune threshold.
  3. Label against GT: discard any pair in GT matches (confirmed non-match).
  4. Score with frozen V2 model to rank by difficulty.
  5. Apply per-entity cap (CAP_PER_S1 = 20) to prevent domination.
  6. Sort globally by V2 score descending — deterministic.

H configurations (additional confirmed hard negatives):
  H1 = 0.5 × |V1 negatives in train_core|
  H2 = 1.0 × |V1 negatives in train_core|
  H3 = 2.0 × |V1 negatives in train_core|

Model config — unchanged from V2:
  scale_pos_weight = 3.0
  S2 threshold = 0.70
  S3 threshold = 0.65
  all LightGBM hyperparameters identical
  same 15 features

V2 models never overwritten.
V5 models: models/v5_{h1|h2|h3}_lgb_{source}.txt
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

# Frozen V2 config — never modified
_TFIDF_K      = 20
_TFIDF_THRESH = {"source2": 0.53, "source3": 0.51}
_V2_THRESH    = {"source2": 0.70, "source3": 0.65}
_SPW          = 3.0
_CAP_PER_S1   = 20

# Hash expression used for all splits
_H = "abs(hash(entity_id || 'v1')) % 1000"

# Split definitions — non-overlapping ranges, same hash
_WHERE = {
    "val1":         f"{_H} >= 0  AND {_H} < 5",
    "val2":         f"{_H} >= 25 AND {_H} < 30",
    "internal_val": f"{_H} >= 5  AND {_H} < 10",
    "train_core":   f"{_H} >= 10 AND {_H} < 25",
}

_CACHE = ROOT / "intermediate" / "v2_cache"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _s1pq() -> str:
    return str(norm_parquet("train", "source1")).replace("\\", "/")

def _sxpq(src: str) -> str:
    return str(norm_parquet("train", src)).replace("\\", "/")

def _gtpq() -> str:
    return str(ROOT / "intermediate/train/ground_truth.parquet").replace("\\", "/")

def _cpq(src: str) -> str:
    return str(candidate_parquet("train", src)).replace("\\", "/")

def _con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET threads TO 6; SET memory_limit = '6GB'")
    return c

def _split_ids(split_name: str) -> set[str]:
    where = _WHERE[split_name].replace("entity_id", "s.entity_id")
    c = _con()
    ids = set(c.execute(
        f"SELECT entity_id FROM read_parquet('{_s1pq()}') s WHERE {_WHERE[split_name]}"
    ).df()["entity_id"])
    c.close()
    return ids

def _gt_df(src: str, split_name: str) -> tuple[set, pd.DataFrame]:
    """GT pairs for split as (gt_set, DataFrame(s1_id, true_match_id))."""
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

def _label(df: pd.DataFrame, gt_set: set) -> np.ndarray:
    return np.array([(s, x) in gt_set for s, x in zip(df["s1_id"], df["sx_id"])], dtype=np.int8)

def _join_all(cands: pd.DataFrame, src: str) -> pd.DataFrame:
    """Full feature-column join for training candidates."""
    c = _con()
    c.register("_c", cands[["s1_id", "sx_id"]])
    df = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
            s1.name_clean      AS s1_name,   sx.name_clean      AS sx_name,
            s1.name_no_suffix  AS s1_no_suf,  sx.name_no_suffix  AS sx_no_suf,
            s1.addr_clean      AS s1_addr,   sx.addr_clean      AS sx_addr,
            COALESCE(s1.postal_code,'') AS s1_post,
            COALESCE(sx.postal_code,'') AS sx_post
        FROM _c c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()
    return df

def _add_no_suffix(df: pd.DataFrame, src: str) -> pd.DataFrame:
    """Add name_no_suffix columns missing from the V2 cache."""
    c = _con()
    c.register("_c", df[["s1_id", "sx_id"]])
    ns = c.execute(f"""
        SELECT c.s1_id, c.sx_id,
               s1.name_no_suffix AS s1_no_suf, sx.name_no_suffix AS sx_no_suf
        FROM _c c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{_sxpq(src)}') sx ON c.sx_id = sx.entity_id
    """).df()
    c.close()
    return df.merge(ns, on=["s1_id", "sx_id"], how="left").fillna({"s1_no_suf": "", "sx_no_suf": ""})


# ---------------------------------------------------------------------------
# Training data: V1 train_core candidates
# ---------------------------------------------------------------------------

def build_v1_train(src: str) -> tuple[np.ndarray, np.ndarray]:
    """V1 candidates for train_core [10,25), features + labels."""
    gt_set, _ = _gt_df(src, "train_core")
    c = _con()
    df = c.execute(f"""
        SELECT c.s1_id, c.sx_id
        FROM read_parquet('{_cpq(src)}') c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        WHERE {_WHERE["train_core"]}
    """).df().drop_duplicates()
    c.close()
    y = _label(df, gt_set)
    X = compute_features_parallel(_join_all(df, src), n_workers=6)
    print(f"  V1 train_core: {len(y):,} pairs  pos={y.sum():,}  neg={(y==0).sum():,}")
    return X, y


# ---------------------------------------------------------------------------
# Internal validation: V1 internal_val candidates for early stopping
# ---------------------------------------------------------------------------

def build_internal_val(src: str) -> tuple[np.ndarray, np.ndarray]:
    """V1 candidates for internal_val [5,10), features + labels."""
    gt_set, _ = _gt_df(src, "internal_val")
    c = _con()
    df = c.execute(f"""
        SELECT c.s1_id, c.sx_id
        FROM read_parquet('{_cpq(src)}') c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        WHERE {_WHERE["internal_val"]}
    """).df().drop_duplicates()
    c.close()
    y = _label(df, gt_set)
    X = compute_features_parallel(_join_all(df, src), n_workers=6)
    print(f"  internal_val:  {len(y):,} pairs  pos={y.sum():,}")
    return X, y


# ---------------------------------------------------------------------------
# Hard-negative mining
# ---------------------------------------------------------------------------

def mine_hard_negatives(src: str) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Returns (X_pool, hn_info):
      X_pool:  feature matrix, shape (N_pool, 15), sorted by V2 score desc
      hn_info: DataFrame(s1_id, sx_id, v2_score, s1_name, sx_name, s1_addr, sx_addr)
    """
    thresh = _TFIDF_THRESH[src]
    train_ids = _split_ids("train_core")
    gt_set, _ = _gt_df(src, "train_core")

    # V1 candidate set for deduplication
    c = _con()
    v1 = c.execute(f"""
        SELECT c.s1_id, c.sx_id
        FROM read_parquet('{_cpq(src)}') c
        JOIN read_parquet('{_s1pq()}') s1 ON c.s1_id = s1.entity_id
        WHERE {_WHERE["train_core"]}
    """).df().drop_duplicates()
    c.close()
    v1_set = set(zip(v1["s1_id"], v1["sx_id"]))

    print(f"  Retrieving TF-IDF candidates for {len(train_ids):,} train_core entities...")
    tfidf = retrieve_tfidf_candidates(
        target_source=src, split="train",
        k=_TFIDF_K, s1_ids=train_ids, return_scores=True,
    )

    # Filter: new (not in V1) AND above frozen threshold
    is_new = np.array([(s, x) not in v1_set for s, x in zip(tfidf["s1_id"], tfidf["sx_id"])])
    new_above = tfidf[is_new & (tfidf["tfidf_score"].values >= thresh)].copy().reset_index(drop=True)

    # Confirmed non-matches: keep only label==0
    labels = _label(new_above, gt_set)
    confirmed = new_above[labels == 0].copy().reset_index(drop=True)
    print(f"  New above threshold: {len(new_above):,}  confirmed non-matches: {len(confirmed):,}  "
          f"(positives removed: {(labels==1).sum():,})")

    # Join feature text columns (needed for features AND error sample)
    feat_df = _join_all(confirmed, src)
    X_cand = compute_features_parallel(feat_df, n_workers=6)

    # Score with frozen V2 model
    v2_bst = lgb.Booster(model_file=str(MODELS_DIR / f"lgb_{src}.txt"))
    v2_scores = v2_bst.predict(X_cand)
    del v2_bst

    # Per-S1 cap: top CAP_PER_S1 by V2 score per entity, then sort globally
    info = pd.DataFrame({
        "s1_id":   feat_df["s1_id"].values,
        "sx_id":   feat_df["sx_id"].values,
        "v2_score": v2_scores,
        "s1_name":  feat_df["s1_name"].values,
        "sx_name":  feat_df["sx_name"].values,
        "s1_addr":  feat_df["s1_addr"].values,
        "sx_addr":  feat_df["sx_addr"].values,
        "feat_idx": np.arange(len(feat_df)),
    })
    info = info.sort_values("v2_score", ascending=False)
    info = info.groupby("s1_id", sort=False).head(_CAP_PER_S1).reset_index(drop=True)
    info = info.sort_values("v2_score", ascending=False).reset_index(drop=True)

    X_pool = X_cand[info["feat_idx"].values]
    hn_info = info.drop(columns=["feat_idx"])

    per_s1 = hn_info.groupby("s1_id").size()
    print(f"  Hard-negative pool (after cap={_CAP_PER_S1}): {len(hn_info):,}  "
          f"unique S1: {hn_info['s1_id'].nunique():,}  "
          f"max/S1: {per_s1.max()}  median/S1: {per_s1.median():.1f}")
    return X_pool, hn_info


# ---------------------------------------------------------------------------
# Validation feature loading (from V2 cache; reused for all H configs)
# ---------------------------------------------------------------------------

def load_val_features(src: str, split_name: str) -> tuple[np.ndarray, pd.DataFrame, set, pd.DataFrame]:
    """V2 candidate pool for val split: feature matrix + GT metadata."""
    cache = pd.read_parquet(_CACHE / f"v2_{src}_{split_name}.parquet")
    feat_df = _add_no_suffix(cache, src)
    print(f"  {split_name}: {len(feat_df):,} pairs")
    X = compute_features_parallel(feat_df, n_workers=6)
    s1_all = _split_ids(split_name)
    _, gt = _gt_df(src, split_name)
    return X, feat_df[["s1_id", "sx_id"]], s1_all, gt


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def _lgb_params() -> dict:
    return dict(
        objective="binary", metric="binary_logloss", boosting_type="gbdt",
        learning_rate=0.1, num_leaves=31, max_depth=6, min_child_samples=20,
        feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
        scale_pos_weight=_SPW, n_jobs=6, verbose=-1,
    )

def train_v5(src: str, tag: str,
             X_tr: np.ndarray, y_tr: np.ndarray,
             X_iv: np.ndarray, y_iv: np.ndarray) -> lgb.Booster:
    """Train one V5 model on (X_tr, y_tr), early-stop on internal_val."""
    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    div    = lgb.Dataset(X_iv, label=y_iv, reference=dtrain, feature_name=FEATURE_NAMES)
    bst = lgb.train(_lgb_params(), dtrain, num_boost_round=200, valid_sets=[div],
                    callbacks=[lgb.early_stopping(15, verbose=False)])
    path = MODELS_DIR / f"v5_{tag}_lgb_{src}.txt"
    bst.save_model(str(path))
    print(f"  Saved {path.name} ({bst.num_trees()} trees)")
    return bst

def eval_on_split(bst: lgb.Booster, X: np.ndarray, pairs: pd.DataFrame,
                  s1_all: set, gt: pd.DataFrame, threshold: float) -> dict:
    scores = bst.predict(X)
    preds = pairs[scores >= threshold][["s1_id", "sx_id"]]
    return calculate_macro_f05(preds, gt, s1_all)

def v2_cached_eval(src: str, split_name: str, s1_all: set, gt: pd.DataFrame) -> dict:
    cache = pd.read_parquet(_CACHE / f"v2_{src}_{split_name}.parquet")
    preds = cache[cache["lgb_score"] >= _V2_THRESH[src]][["s1_id", "sx_id"]]
    return calculate_macro_f05(preds, gt, s1_all)


# ---------------------------------------------------------------------------
# Per-source orchestration
# ---------------------------------------------------------------------------

def run_source(src: str) -> dict:
    t0 = time.perf_counter()
    thresh = _V2_THRESH[src]
    print(f"\n{'='*60}\n  V5 — {src.upper()}\n{'='*60}")

    # Build V1 training data (train_core)
    print("\nBuilding train_core (V1 candidates) ...")
    X_v1, y_v1 = build_v1_train(src)
    n_v1_neg = int((y_v1 == 0).sum())
    n_v1_pos = int(y_v1.sum())

    # Build internal validation (for early stopping only)
    print("\nBuilding internal_val (for early stopping) ...")
    X_iv, y_iv = build_internal_val(src)

    # Mine hard negatives (once; reused for H1/H2/H3)
    print("\nMining hard negatives ...")
    X_pool, hn_info = mine_hard_negatives(src)
    n_pool = len(hn_info)

    # Load val1/val2 features once — NEVER used for training decisions
    print("\nLoading val1 features (evaluation only) ...")
    X_v1f, pairs_v1, s1_all_v1, gt_v1 = load_val_features(src, "val1")
    print("Loading val2 features (evaluation only) ...")
    X_v2f, pairs_v2, s1_all_v2, gt_v2 = load_val_features(src, "val2")

    # V2 baselines (from cache)
    bv2_m1 = v2_cached_eval(src, "val1", s1_all_v1, gt_v1)
    bv2_m2 = v2_cached_eval(src, "val2", s1_all_v2, gt_v2)

    # H1/H2/H3 training runs
    h_results: dict[str, dict] = {}
    for tag, ratio in [("h1", 0.5), ("h2", 1.0), ("h3", 2.0)]:
        n_take = min(int(n_v1_neg * ratio), n_pool)
        X_hn = X_pool[:n_take]          # top-n_take by V2 score (pre-sorted)
        y_hn = np.zeros(n_take, dtype=np.int8)

        X_tr = np.vstack([X_v1, X_hn])
        y_tr = np.concatenate([y_v1, y_hn])

        print(f"\n{tag.upper()} (ratio={ratio}, hn={n_take:,}) — train={len(y_tr):,} ...")
        bst = train_v5(src, tag, X_tr, y_tr, X_iv, y_iv)

        m1 = eval_on_split(bst, X_v1f, pairs_v1, s1_all_v1, gt_v1, thresh)
        m2 = eval_on_split(bst, X_v2f, pairs_v2, s1_all_v2, gt_v2, thresh)
        h_results[tag] = {"n_hn": n_take, "ratio": ratio, "val1": m1, "val2": m2}
        d1 = m1["macro_f05"] - bv2_m1["macro_f05"]
        d2 = m2["macro_f05"] - bv2_m2["macro_f05"]
        print(f"  val1 F0.5={m1['macro_f05']:.5f} (D={d1:+.5f})  "
              f"val2 F0.5={m2['macro_f05']:.5f} (D={d2:+.5f})")

    return dict(
        src=src, thresh=thresh,
        n_v1_pos=n_v1_pos, n_v1_neg=n_v1_neg,
        n_pool=n_pool,
        hn_info=hn_info,
        v2_m1=bv2_m1, v2_m2=bv2_m2,
        h=h_results,
        elapsed=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def _mrow(m: dict, mv2: dict, label: str) -> None:
    d = m["macro_f05"] - mv2["macro_f05"]
    dp = m["macro_precision"] - mv2["macro_precision"]
    dr = m["macro_recall"] - mv2["macro_recall"]
    ds = m["singleton_accuracy"] - mv2["singleton_accuracy"]
    fp_sing = m["total_singletons"] - m["singletons_correct"]
    print(f"  {label} F0.5:          {m['macro_f05']:.5f}  (D={d:+.5f})")
    print(f"  {label} Precision:      {m['macro_precision']:.5f}  (D={dp:+.5f})")
    print(f"  {label} Recall:         {m['macro_recall']:.5f}  (D={dr:+.5f})")
    print(f"  {label} Singleton Acc:  {m['singleton_accuracy']:.5f}  (D={ds:+.5f})")
    print(f"  {label} FP Singleton:   {fp_sing}")
    print(f"  {label} Predicted:      {m['predicted_matches']}")


def print_report(results: dict, total_elapsed: float) -> None:
    print("\n" + "=" * 65)
    print("V5 — CLEAN HARD-NEGATIVE RETRAINING")
    print("=" * 65)

    print("\nDATA SPLIT")
    print("----------")
    print("val1:                     abs(hash(entity_id || 'v1')) % 1000 in [0, 5)")
    print("val2:                     abs(hash(entity_id || 'v1')) % 1000 in [25, 30)")
    print("internal training holdout: abs(hash(entity_id || 'v1')) % 1000 in [5, 10)")
    print("train_core:               abs(hash(entity_id || 'v1')) % 1000 in [10, 25)")
    print("overlap internal_val ∩ val1: 0  (ranges [5,10) and [0,5) are disjoint)")
    print("overlap internal_val ∩ val2: 0  (ranges [5,10) and [25,30) are disjoint)")

    for src, lbl in [("source2", "S2"), ("source3", "S3")]:
        r = results[src]
        print(f"\n{lbl}")
        print("--")
        print(f"Frozen V2:")
        print(f"  val1 F0.5: {r['v2_m1']['macro_f05']:.5f}")
        print(f"  val2 F0.5: {r['v2_m2']['macro_f05']:.5f}")

        for tag, label in [("h1", "H1 — 0.5x"), ("h2", "H2 — 1.0x"), ("h3", "H3 — 2.0x")]:
            hr = r["h"][tag]
            print(f"\n{label}")
            print(f"  hard negatives: {hr['n_hn']:,}")
            _mrow(hr["val1"], r["v2_m1"], "val1")
            _mrow(hr["val2"], r["v2_m2"], "val2")

    print("\nHARD-NEGATIVE QUALITY")
    print("---------------------")
    for src, lbl in [("source2", "S2"), ("source3", "S3")]:
        hi = results[src]["hn_info"]
        scores = hi["v2_score"].values
        per_s1 = hi.groupby("s1_id").size()
        print(f"{lbl}:")
        print(f"  count:         {len(hi):,}")
        print(f"  unique S1:     {hi['s1_id'].nunique():,}")
        print(f"  max per S1:    {per_s1.max()}")
        print(f"  median per S1: {per_s1.median():.1f}")
        print(f"  P50:           {np.percentile(scores, 50):.4f}")
        print(f"  P90:           {np.percentile(scores, 90):.4f}")
        print(f"  P95:           {np.percentile(scores, 95):.4f}")
        print(f"  P99:           {np.percentile(scores, 99):.4f}")
        print(f"  max:           {scores.max():.4f}")

    print("\nERROR SAMPLE")
    print("------------")
    print("Top hard negatives by V2 score (cross-source, combined):")
    shown = 0
    for src in ["source2", "source3"]:
        hi = results[src]["hn_info"]
        for i, row in hi.head(5).iterrows():
            if shown >= 6:
                break
            print(f"{shown+1}. [{src}]  score={row.v2_score:.4f}")
            print(f"   s1: {str(row.s1_name)[:70]!r}")
            print(f"   sx: {str(row.sx_name)[:70]!r}")
            print(f"   s1 addr: {str(row.s1_addr)[:60]!r}")
            print(f"   sx addr: {str(row.sx_addr)[:60]!r}")
            shown += 1

    print("\nFINAL DECISION")
    print("--------------")
    best_per_src: dict[str, tuple[str | None, float]] = {}
    for src in ["source2", "source3"]:
        r = results[src]
        best_tag, best_d = None, -999.0
        for tag in ["h1", "h2", "h3"]:
            hr = r["h"][tag]
            d1 = hr["val1"]["macro_f05"] - r["v2_m1"]["macro_f05"]
            d2 = hr["val2"]["macro_f05"] - r["v2_m2"]["macro_f05"]
            min_d = min(d1, d2)
            if min_d > 0.0001 and min_d > best_d:
                best_d, best_tag = min_d, tag
        best_per_src[src] = (best_tag, best_d)

    s2_win, s2_d = best_per_src["source2"]
    s3_win, s3_d = best_per_src["source3"]

    if s2_win or s3_win:
        print("KEEP")
        if s2_win:
            print(f"S2 winning configuration: {s2_win.upper()}  min-D F0.5={s2_d:+.5f}")
        if s3_win:
            print(f"S3 winning configuration: {s3_win.upper()}  min-D F0.5={s3_d:+.5f}")
    else:
        print("DISCARD")
        print("Winning configuration: NONE")
        print("Reason: No H configuration improves Macro F0.5 on BOTH val splits.")

    print(f"\nV2 models overwritten: NO")
    print("V2 thresholds changed: NO")
    print("Blocking changed: NO")
    print("Features changed: NO")
    print("Validation leakage: NO")
    print("External data: NO")
    print("Full test inference: NO")
    print(f"Runtime: {total_elapsed:.1f}s")
    print("Tests: py -3 -m pytest tests/ -q")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()

    # Verify disjointness at runtime (sanity check)
    print("Verifying split disjointness...")
    iv_ids = _split_ids("internal_val")
    v1_ids = _split_ids("val1")
    v2_ids = _split_ids("val2")
    assert len(iv_ids & v1_ids) == 0, "BUG: internal_val overlaps val1"
    assert len(iv_ids & v2_ids) == 0, "BUG: internal_val overlaps val2"
    print(f"  internal_val ({len(iv_ids):,} entities)  ∩  val1 ({len(v1_ids):,}): 0")
    print(f"  internal_val ({len(iv_ids):,} entities)  ∩  val2 ({len(v2_ids):,}): 0")

    results = {}
    for src in ["source2", "source3"]:
        results[src] = run_source(src)

    print_report(results, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
