"""Unit tests for V12 Assignment-Aware Evidence LightGBM Matcher."""

import numpy as np
import pandas as pd
import pytest

from experiments.v12_assignment_lgb import (
    apply_target_exclusivity,
    compute_pairwise_features,
    engineer_competition_features,
    get_v12_feature_names,
    mine_training_data,
)


def test_compute_pairwise_features():
    df = pd.DataFrame({
        "s1_name": ["Acme Corp", "Beta LLC"],
        "sx_name": ["Acme Corporation", "Gamma Inc"],
        "s1_addr": ["123 Main St", "456 Oak Rd"],
        "sx_addr": ["123 Main Street Suite 100", "789 Pine Ave"],
        "s1_post": ["12345", "99999"],
        "sx_post": ["12345-6789", "88888"],
    })

    feats = compute_pairwise_features(df)
    assert len(feats) == 2
    assert "name_exact" in feats.columns
    assert "name_wratio" in feats.columns
    assert "addr_house_num_exact" in feats.columns
    assert "name_addr_prod" in feats.columns
    assert "num_toks_diff" in feats.columns

    # Pair 1: house number 123 matches
    assert feats.loc[0, "addr_house_num_exact"] == 1.0
    # Pair 2: house number 456 vs 789 conflicts
    assert feats.loc[1, "addr_house_num_conflict"] == 1.0


def test_engineer_competition_features():
    df = pd.DataFrame({
        "s1_id": ["S1-A", "S1-A", "S1-B"],
        "sx_id": ["S2-X", "S2-Y", "S2-X"],  # S2-X is proposed by both S1-A and S1-B
        "lgb_score": [0.85, 0.40, 0.90],
    })

    out = engineer_competition_features(df)
    assert "cand_rank" in out.columns
    assert "target_candidate_count" in out.columns
    assert "target_rank" in out.columns
    assert "target_margin" in out.columns

    # S2-X was proposed by both S1-A (0.85) and S1-B (0.90)
    s2_x_rows = out[out["sx_id"] == "S2-X"].set_index("s1_id")
    assert s2_x_rows.loc["S1-B", "target_rank"] == 1
    assert s2_x_rows.loc["S1-A", "target_rank"] == 2
    assert s2_x_rows.loc["S1-B", "target_is_best_candidate"] == 1.0
    assert s2_x_rows.loc["S1-A", "target_is_best_candidate"] == 0.0
    assert np.isclose(s2_x_rows.loc["S1-B", "target_margin"], 0.90 - 0.85)


def test_apply_target_exclusivity():
    # S2-X is claimed by both S1-A (0.75) and S1-B (0.92)
    # S2-Y is claimed by S1-A (0.80)
    pred_df = pd.DataFrame({
        "s1_id": ["S1-A", "S1-B", "S1-A"],
        "sx_id": ["S2-X", "S2-X", "S2-Y"],
        "v12_score": [0.75, 0.92, 0.80],
    })

    excl = apply_target_exclusivity(pred_df, target_margin_thresh=0.05)
    # S2-X should be exclusively assigned to S1-B
    assert len(excl[excl["sx_id"] == "S2-X"]) == 1
    assert excl[excl["sx_id"] == "S2-X"]["s1_id"].values[0] == "S1-B"
    # S1-A still keeps S2-Y (multi-match safety across different targets)
    assert len(excl[excl["sx_id"] == "S2-Y"]) == 1
    assert excl[excl["sx_id"] == "S2-Y"]["s1_id"].values[0] == "S1-A"


def test_mine_training_data():
    df = pd.DataFrame({
        "s1_id": ["S1-A", "S1-A", "S1-A", "S1-A"],
        "sx_id": ["S2-1", "S2-2", "S2-3", "S2-4"],
        "lgb_score": [0.90, 0.85, 0.60, 0.20],
    })
    # S2-1 is gold positive, others are negatives
    gt_set = {("S1-A", "S2-1")}
    sampled = mine_training_data(df, gt_set, max_hard_neg=2)

    assert len(sampled) == 3  # 1 positive + 2 top hard negatives (0.85 and 0.60)
    assert ("S1-A", "S2-1") in set(zip(sampled["s1_id"], sampled["sx_id"]))
    assert ("S1-A", "S2-2") in set(zip(sampled["s1_id"], sampled["sx_id"]))
    assert ("S1-A", "S2-3") in set(zip(sampled["s1_id"], sampled["sx_id"]))
    assert ("S1-A", "S2-4") not in set(zip(sampled["s1_id"], sampled["sx_id"]))
