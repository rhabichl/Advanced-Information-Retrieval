from __future__ import annotations

import math
from typing import Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    if not retrieved:
        return 0.0
    topk = retrieved[:k]
    return sum(1 for d in topk if d in rel) / float(k)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & rel) / float(len(rel))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """
    Binary relevance nDCG@k.
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


