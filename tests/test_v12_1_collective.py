"""Unit tests for V12.1 Collective Sibling Matcher and Exclusivity."""

import numpy as np
import pandas as pd
import pytest

from experiments.v12_1_collective_matcher import (
    compute_collective_sibling_features,
    get_v12_1_feature_names,
)
from experiments.v12_assignment_lgb import apply_target_exclusivity


def test_v12_1_feature_names():
    cols = get_v12_1_feature_names()
    assert len(cols) == 69
    assert "stage1_score" in cols
    assert "s1_sibling_high_conf_count" in cols
    assert "cluster_s1_agreement" in cols
    assert "target_stage1_margin" in cols


def test_collective_sibling_features():
    df = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"],
        "sx_id": ["S2-A", "S2-B", "S2-C", "S2-D"],
        "stage1_score": [0.90, 0.80, 0.30, 0.70],
        "sx_name": ["Acme Corp", "Acme Inc", "Alpha Beta", "Beta Corp"],
        "sx_post": ["12345", "12345", "99999", "55555"],
        "s1_country": ["us", "us", "us", "in"],
    })

    feats = compute_collective_sibling_features(df, score_col="stage1_score")
    assert len(feats) == len(df)
    assert "s1_sibling_high_conf_count" in feats.columns
    assert "s1_best_sibling_score" in feats.columns
    assert "cluster_s1_agreement" in feats.columns
    assert "target_stage1_margin" in feats.columns

    # Check that for S1-1 top candidate (score=0.90), sibling high-conf count is 1 (S2-B with 0.80 >= 0.60)
    top_cand = feats.iloc[0]
    assert top_cand["s1_sibling_high_conf_count"] == 1.0


def test_target_exclusivity_structural_constraint():
    # Target S2-A has 2 competing S1s: S1-1 (0.85) and S1-2 (0.75)
    pred_df = pd.DataFrame({
        "s1_id": ["S1-1", "S1-2", "S1-3"],
        "sx_id": ["S2-A", "S2-A", "S2-B"],
        "v12_score": [0.85, 0.75, 0.60],
    })

    excl_df = apply_target_exclusivity(pred_df, target_margin_thresh=0.05)
    # S2-A must only be assigned to S1-1 because 0.85 > 0.75 and margin 0.10 >= 0.05
    assert len(excl_df) == 2
    assigned_targets = excl_df["sx_id"].tolist()
    assert len(assigned_targets) == len(set(assigned_targets))
    assert ("S1-1", "S2-A") in set(zip(excl_df["s1_id"], excl_df["sx_id"]))
    assert ("S1-2", "S2-A") not in set(zip(excl_df["s1_id"], excl_df["sx_id"]))
