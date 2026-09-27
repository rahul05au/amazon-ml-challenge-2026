from __future__ import annotations
import multiprocessing as mp
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
FEATURE_NAMES: list[str] = ['name_exact', 'name_no_suf_exact', 'name_ratio', 'name_wratio', 'name_token_set', 'name_len_diff', 'name_len_ratio', 'name_jaccard', 'addr_exact', 'addr_ratio', 'addr_token_set', 'addr_jaccard', 'number_exact', 'postal_exact', 'name_addr_mean']
_NUM_RE = re.compile('\\b\\d+\\b')

def _process_chunk_worker(chunk: pd.DataFrame) -> np.ndarray:
    s1_names = chunk['s1_name'].fillna('').tolist()
    sx_names = chunk['sx_name'].fillna('').tolist()
    s1_no_suf = chunk['s1_no_suf'].fillna('').tolist()
    sx_no_suf = chunk['sx_no_suf'].fillna('').tolist()
    s1_addrs = chunk['s1_addr'].fillna('').tolist()
    sx_addrs = chunk['sx_addr'].fillna('').tolist()
    s1_posts = chunk['s1_post'].fillna('').tolist()
    sx_posts = chunk['sx_post'].fillna('').tolist()
    s1_name_toks = [set(s.split()) for s in s1_names]
    sx_name_toks = [set(s.split()) for s in sx_names]
    s1_addr_toks = [set(s.split()) for s in s1_addrs]
    sx_addr_toks = [set(s.split()) for s in sx_addrs]
    name_exact = np.array([1.0 if n1 and n1 == n2 else 0.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    name_no_suf_exact = np.array([1.0 if s1 and s1 == s2 else 0.0 for s1, s2 in zip(s1_no_suf, sx_no_suf)], dtype=np.float32)
    name_ratio = np.array([fuzz.ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    name_wratio = np.array([fuzz.WRatio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    name_token_set = np.array([fuzz.token_set_ratio(n1, n2) / 100.0 for n1, n2 in zip(s1_names, sx_names)], dtype=np.float32)
    s1_lens = [len(s) for s in s1_names]
    sx_lens = [len(s) for s in sx_names]
    name_len_diff = np.array([abs(l1 - l2) for l1, l2 in zip(s1_lens, sx_lens)], dtype=np.float32)
    name_len_ratio = np.array([min(l1, l2) / max(l1, l2, 1) for l1, l2 in zip(s1_lens, sx_lens)], dtype=np.float32)
    name_jaccard = np.array([len(t1 & t2) / len(t1 | t2) if t1 | t2 else 0.0 for t1, t2 in zip(s1_name_toks, sx_name_toks)], dtype=np.float32)
    addr_exact = np.array([1.0 if a1 and a1 == a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    addr_ratio = np.array([fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    addr_token_set = np.array([fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0 for a1, a2 in zip(s1_addrs, sx_addrs)], dtype=np.float32)
    addr_jaccard = np.array([len(t1 & t2) / len(t1 | t2) if t1 | t2 else 0.0 for t1, t2 in zip(s1_addr_toks, sx_addr_toks)], dtype=np.float32)
    num_exact = []
    for a1, a2 in zip(s1_addrs, sx_addrs):
        m1 = _NUM_RE.search(a1)
        m2 = _NUM_RE.search(a2)
        if m1 and m2:
            num_exact.append(1.0 if m1.group(0) == m2.group(0) else 0.0)
        else:
            num_exact.append(-1.0)
    number_exact = np.array(num_exact, dtype=np.float32)
    post_exact = []
    for p1, p2 in zip(s1_posts, sx_posts):
        if p1 and p2:
            post_exact.append(1.0 if p1 == p2 else 0.0)
        else:
            post_exact.append(-1.0)
    postal_exact = np.array(post_exact, dtype=np.float32)
    name_addr_mean = (name_wratio + addr_ratio) / 2.0
    return np.column_stack([name_exact, name_no_suf_exact, name_ratio, name_wratio, name_token_set, name_len_diff, name_len_ratio, name_jaccard, addr_exact, addr_ratio, addr_token_set, addr_jaccard, number_exact, postal_exact, name_addr_mean])

def compute_features_parallel(df: pd.DataFrame, n_workers: int=6) -> np.ndarray:
    if len(df) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    chunk_size = (len(df) + n_workers - 1) // n_workers
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size) if i < len(df)]
    with mp.Pool(n_workers) as pool:
        results = pool.map(_process_chunk_worker, chunks)
    return np.vstack(results)
