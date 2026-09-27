"""Unit tests for V11 GPU Cross-Encoder and Calibrated Fusion."""

import numpy as np
import pandas as pd
import pytest

from experiments.v11_crossencoder import (
    calculate_macro_f05,
    engineer_fusion_features,
    format_cross_pair,
    mine_crossencoder_training_data,
    serialize_entity,
)


def test_serialize_entity():
    # Standard values
    text = serialize_entity("Acme Corp", "123 Main St", "US")
    assert text == "NAME: Acme Corp ADDRESS: 123 Main St COUNTRY: US"

    # Missing / None values
    text_missing = serialize_entity(None, "", pd.NA)
    assert text_missing == "NAME: MISSING ADDRESS: MISSING COUNTRY: MISSING"


def test_format_cross_pair():
    pair = format_cross_pair("Acme", "123 Main", "US", "Acme LLC", "123 Main Street", "US")
    assert pair.startswith("[ENTITY_A] NAME: Acme ADDRESS: 123 Main COUNTRY: US")
    assert "[ENTITY_B] NAME: Acme LLC ADDRESS: 123 Main Street COUNTRY: US" in pair


def test_mine_crossencoder_training_data():
    df = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-1", "S1-1", "S1-2", "S1-2"],
        "sx_id": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6"],
        "lgb_score": [0.90, 0.85, 0.60, 0.40, 0.70, 0.30],
        "s1_name": ["A", "A", "A", "A", "B", "B"],
        "s1_addr": ["Addr A", "Addr A", "Addr A", "Addr A", "Addr B", "Addr B"],
        "sx_name": ["A1", "A2", "A3", "A4", "B1", "B2"],
        "sx_addr": ["Addr A1", "Addr A2", "Addr A3", "Addr A4", "Addr B1", "Addr B2"],
    })
    # S1-1 matches S2-1 (positive), others are negatives
    # S1-2 matches S2-5 (positive)
    gt_set = {("S1-1", "S2-1"), ("S1-2", "S2-5")}

    mined = mine_crossencoder_training_data(df, gt_set, h_neg=2)
    assert len(mined[mined["label"] == 1.0]) == 2
    # S1-1 negatives: up to 2 top scored (S2-2 with 0.85, S2-3 with 0.60)
    assert len(mined[mined["label"] == 0.0]) <= 3
    assert "pair_text" in mined.columns


def test_engineer_fusion_features():
    df = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-2"],
        "sx_id": ["S2-1", "S2-2", "S2-3"],
        "lgb_score": [0.80, 0.60, 0.75],
        "cross_prob": [0.95, 0.30, 0.85],
    })
    feat_df = engineer_fusion_features(df)
    assert "cand_rank" in feat_df.columns
    assert "top1_score" in feat_df.columns
    assert "cross_margin_top1" in feat_df.columns
    assert "cross_rank_within_candidates" in feat_df.columns
    # Top-1 cross margin should be 0.0 for highest candidate
    assert feat_df.loc[feat_df["sx_id"] == "S2-1", "cross_margin_top1"].values[0] == 0.0


def test_calculate_macro_f05():
    preds = pd.DataFrame({
        "s1_id": ["S1-1", "S1-2"],
        "sx_id": ["S2-1", "S2-2"],
    })
    trues = pd.DataFrame({
        "s1_id": ["S1-1"],
        "true_match_id": ["S2-1"],
    })
    all_s1 = {"S1-1", "S1-2", "S1-3"}
    # S1-1: correct match -> F0.5 = 1.0
    # S1-2: false positive -> F0.5 = 0.0
    # S1-3: true singleton correctly predicted empty -> F0.5 = 1.0
    res = calculate_macro_f05(preds, trues, all_s1)
    assert res["total_s1"] == 3
    assert res["total_singletons"] == 2
    assert res["singletons_correct"] == 1
    assert np.isclose(res["macro_f05"], 2.0 / 3.0, atol=1e-3)
