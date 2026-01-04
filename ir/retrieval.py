from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from tqdm import tqdm


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def topk_cosine_retrieval(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    doc_ids: Sequence[str],
    k: int,
    query_batch: int = 256,
) -> Dict[int, List[str]]:
    """
    Returns mapping from query index -> ranked doc_ids (top-k), using cosine similarity.

    Assumes inputs are already L2-normalized.
    """
    assert query_emb.ndim == 2 and doc_emb.ndim == 2
    assert query_emb.shape[1] == doc_emb.shape[1]
    doc_ids_arr = np.array(list(doc_ids), dtype=object)

    out: Dict[int, List[str]] = {}
    for start in tqdm(range(0, query_emb.shape[0], query_batch), desc="Retrieval"):
        q = query_emb[start : start + query_batch]  # (b, d)
        scores = q @ doc_emb.T  # (b, n_docs)
        kk = min(k, scores.shape[1])
        idx_part = np.argpartition(-scores, kth=max(kk - 1, 0), axis=1)[:, :kk]
        part_scores = np.take_along_axis(scores, idx_part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        for bi in range(idx_sorted.shape[0]):
            qi = start + bi
            out[qi] = doc_ids_arr[idx_sorted[bi]].tolist()
    return out


