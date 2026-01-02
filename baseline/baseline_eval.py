#!/usr/bin/env python3
"""
Baseline retrieval evaluation (no fine-tuning).

Implements:
- Load Austrian dataset from Hugging Face: krapfi/Advanced-Information-Retrieval
- Build corpus (unique docs by `id`) and qrels (each query maps to its doc id)
- 70/30 split on queries (easy, row-level)
- Embed corpus + queries with a HF Transformers encoder model
- Retrieve top-k by cosine similarity
- Report Precision@k, Recall@k, nDCG@k

Notes:
- This is intentionally simple and self-contained (no external IR framework).
- For big embedding models (e.g. Qwen3-Embedding-8B), run on a GPU box.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


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
    # ideal DCG for binary relevance is all relevant docs ranked at top, capped by k
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(frozen=True)
class DatasetIR:
    corpus_ids: List[str]
    corpus_texts: List[str]
    query_ids: List[str]
    query_texts: List[str]
    qrels: Dict[str, List[str]]  # qid -> list of relevant corpus_ids


def load_austrian_dataset_ir(dataset_id: str = "krapfi/Advanced-Information-Retrieval") -> DatasetIR:
    """
    Austrian dataset format:
      - id: document id (not unique across rows)
      - query: query string
      - document: document text
    """
    d = load_dataset(dataset_id, split="train")

    # Build corpus from unique document ids. Keep first occurrence's document text.
    corpus_map: Dict[str, str] = {}
    for row in tqdm(d, desc="Building corpus (unique docs)"):
        doc_id = row["id"]
        if doc_id not in corpus_map:
            corpus_map[doc_id] = row["document"]

    corpus_ids = list(corpus_map.keys())
    corpus_texts = [corpus_map[i] for i in corpus_ids]

    query_ids: List[str] = []
    query_texts: List[str] = []
    qrels: Dict[str, List[str]] = {}

    for i, row in enumerate(tqdm(d, desc="Building queries + qrels")):
        qid = f"q{i}"
        query_ids.append(qid)
        query_texts.append(row["query"])
        qrels[qid] = [row["id"]]

    return DatasetIR(
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )


def split_queries(
    query_ids: Sequence[str],
    train_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    qids = np.array(list(query_ids), dtype=object)
    rng.shuffle(qids)
    split = int(round(len(qids) * train_ratio))
    train = qids[:split].tolist()
    test = qids[split:].tolist()
    return train, test


def load_encoder(model_name_or_path: str):
    """
    Loads a Hugging Face encoder model and tokenizer.

    This expects a model usable with mean pooling on last_hidden_state.
    (If your model has a special embedding API, adapt `encode_texts`.)
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def mean_pool(last_hidden_state, attention_mask):
    """
    Standard mean pooling for transformer token embeddings.
    """
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch

    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded)
            pooled = mean_pool(out.last_hidden_state, encoded["attention_mask"])
        embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs)


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
        # argpartition for top-k, then sort within top-k
        idx_part = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, idx_part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx_sorted = np.take_along_axis(idx_part, order, axis=1)
        for bi in range(idx_sorted.shape[0]):
            qi = start + bi
            out[qi] = doc_ids_arr[idx_sorted[bi]].tolist()
    return out


def evaluate(
    test_qids: Sequence[str],
    qid_to_ranked: Dict[str, List[str]],
    qrels: Dict[str, List[str]],
    ks: Sequence[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        p = []
        r = []
        n = []
        for qid in test_qids:
            ranked = qid_to_ranked.get(qid, [])
            rel = qrels.get(qid, [])
            p.append(precision_at_k(ranked, rel, k))
            r.append(recall_at_k(ranked, rel, k))
            n.append(ndcg_at_k(ranked, rel, k))
        metrics[f"P@{k}"] = float(np.mean(p)) if p else 0.0
        metrics[f"R@{k}"] = float(np.mean(r)) if r else 0.0
        metrics[f"nDCG@{k}"] = float(np.mean(n)) if n else 0.0
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="krapfi/Advanced-Information-Retrieval")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_queries", type=int, default=20000, help="Cap queries for faster runs (0 = no cap)")
    ap.add_argument("--k", type=str, default="1,3,5,10")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--query_batch_retrieval", type=int, default=256)
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    ks = sorted(set(ks))
    if not ks:
        raise SystemExit("Provide at least one k in --k")

    data = load_austrian_dataset_ir(args.dataset)
    print(f"Corpus docs: {len(data.corpus_ids)}")
    print(f"Queries: {len(data.query_ids)}")

    train_qids, test_qids = split_queries(data.query_ids, args.train_ratio, args.seed)
    print(f"Split queries: train={len(train_qids)} test={len(test_qids)} (ratio={args.train_ratio})")

    # Optionally cap queries for faster experiments
    if args.max_queries and args.max_queries > 0:
        # cap test first (baseline eval focus), then train
        test_qids = test_qids[: min(len(test_qids), args.max_queries)]
        print(f"Capped test queries to: {len(test_qids)}")

    qid_to_text = dict(zip(data.query_ids, data.query_texts))
    test_texts = [qid_to_text[qid] for qid in test_qids]

    tokenizer, model, device = load_encoder(args.model)
    print(f"Model device: {device}")

    doc_emb = encode_texts(
        data.corpus_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    qry_emb = encode_texts(
        test_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    doc_emb = l2_normalize(doc_emb)
    qry_emb = l2_normalize(qry_emb)

    ranked_by_index = topk_cosine_retrieval(
        qry_emb,
        doc_emb,
        data.corpus_ids,
        k=max(ks),
        query_batch=args.query_batch_retrieval,
    )

    qid_to_ranked = {test_qids[i]: ranked_by_index[i] for i in range(len(test_qids))}
    metrics = evaluate(test_qids, qid_to_ranked, data.qrels, ks)

    print("\n=== Baseline metrics ===")
    for key in sorted(metrics.keys(), key=lambda s: (s.split("@")[0], int(s.split("@")[1]))):
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    # Quiet down tokenizer parallelism warning noise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()


