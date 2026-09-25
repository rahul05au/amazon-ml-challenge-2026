"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, compute_features_parallel


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 15
    assert "name_exact" in FEATURE_NAMES
    assert "name_wratio" in FEATURE_NAMES
    assert "postal_exact" in FEATURE_NAMES


def test_compute_features_parallel_empty():
    empty_df = pd.DataFrame()
    res = compute_features_parallel(empty_df, n_workers=2)
    assert res.shape == (0, 15)


def test_compute_features_parallel_values():
    df = pd.DataFrame({
        "s1_name": ["Acme Corp", "Beta LLC"],
        "sx_name": ["Acme Corporation", "Gamma Inc"],
        "s1_no_suf": ["Acme", "Beta"],
        "sx_no_suf": ["Acme", "Gamma"],
        "s1_addr": ["123 Main St", "456 Oak Rd"],
        "sx_addr": ["123 Main Street", "789 Pine Ave"],
        "s1_post": ["90210", ""],
        "sx_post": ["90210", "10001"],
    })

    features = compute_features_parallel(df, n_workers=2)
    assert features.shape == (2, 15)
    assert features.dtype == np.float32

    # Acme pair: exact base name, same street number, same postal
    assert features[0, 1] == 1.0  # name_no_suf_exact
    assert features[0, 12] == 1.0  # number_exact
    assert features[0, 13] == 1.0  # postal_exact

    # Beta/Gamma pair: different names
    assert features[1, 0] == 0.0  # name_exact
    assert features[1, 1] == 0.0  # name_no_suf_exact
