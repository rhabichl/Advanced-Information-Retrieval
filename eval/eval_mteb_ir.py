#!/usr/bin/env python3
"""
Evaluate an embedding model on an MTEB-style IR dataset (corpus/queries/qrels).

Example:
  python3 eval/eval_mteb_ir.py --dataset mteb/GerDaLIR --model Qwen/Qwen3-Embedding-8B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Allow running as `python3 eval/eval_mteb_ir.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from ir.datasets import load_mteb_ir
from ir.embedding import encode_texts, load_encoder
from ir.metrics import ndcg_at_k, precision_at_k, recall_at_k
from ir.retrieval import l2_normalize, topk_cosine_retrieval


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
    ap.add_argument("--dataset", default="mteb/GerDaLIR")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=str, default="1,3,5,10")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--query_batch_retrieval", type=int, default=256)
    ap.add_argument("--max_queries", type=int, default=0, help="Cap queries for speed (0 = no cap)")
    args = ap.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    ks = sorted(set(ks))
    if not ks:
        raise SystemExit("Provide at least one k in --k")

    data = load_mteb_ir(args.dataset, split=args.split)
    print(f"Dataset: {args.dataset} split={args.split}")
    print(f"Corpus docs: {len(data.corpus_ids)}")
    print(f"Queries: {len(data.query_ids)}")

    qids = data.query_ids
    if args.max_queries and args.max_queries > 0:
        qids = qids[: min(len(qids), args.max_queries)]
        print(f"Capped queries to: {len(qids)}")

    qid_to_text = dict(zip(data.query_ids, data.query_texts))
    query_texts = [qid_to_text[qid] for qid in qids]

    tokenizer, model, device = load_encoder(args.model)
    print(f"Model device: {device}")

    doc_emb = encode_texts(
        data.corpus_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        desc="Encoding corpus",
    )
    qry_emb = encode_texts(
        query_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        desc="Encoding queries",
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
    qid_to_ranked = {qids[i]: ranked_by_index[i] for i in range(len(qids))}
    metrics = evaluate(qids, qid_to_ranked, data.qrels, ks)

    print("\n=== Metrics ===")
    for key in sorted(metrics.keys(), key=lambda s: (s.split("@")[0], int(s.split("@")[1]))):
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()


