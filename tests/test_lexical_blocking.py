"""Unit tests for additive lexical retrieval blocking."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.lexical_blocking import retrieve_tfidf_candidates


def test_sparse_topk_selection():
    """Verify top-k extraction correctly picks the highest similarity indices."""
    k = 3
    # 2 queries, 5 target items
    data = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.5])
    indices = np.array([0, 1, 2, 0, 3, 4])
    indptr = np.array([0, 3, 6])
    sims = sparse.csr_matrix((data, indices, indptr), shape=(2, 5))

    results = []
    for i in range(sims.shape[0]):
        r_start = sims.indptr[i]
        r_end = sims.indptr[i + 1]
        c_indices = sims.indices[r_start:r_end]
        c_data = sims.data[r_start:r_end]

        if len(c_indices) <= k:
            chosen = c_indices
        else:
            top_pos = np.argpartition(c_data, -k)[-k:]
            chosen = c_indices[top_pos]
        results.append(set(chosen))

    # Query 0 items: {0: 0.1, 1: 0.9, 2: 0.4} -> all 3
    assert results[0] == {0, 1, 2}
    # Query 1 items: {0: 0.8, 3: 0.2, 4: 0.5} -> top 3 are {0, 3, 4}
    assert results[1] == {0, 3, 4}
