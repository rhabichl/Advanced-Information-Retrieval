from __future__ import annotations

from typing import Dict

from datasets import load_dataset
from tqdm import tqdm

from ir.types import DatasetIR


def load_austrian_paired_ir(dataset_id: str = "krapfi/Advanced-Information-Retrieval") -> DatasetIR:
    """
    Austrian dataset format:
      - id: document id (not unique across rows)
      - query: query string
      - document: document text

    We build:
      - corpus: unique docs by `id`
      - queries: one per row
      - qrels: each query has exactly 1 relevant doc-id (the row's `id`)
    """
    d = load_dataset(dataset_id, split="train")

    corpus_map: Dict[str, str] = {}
    for row in tqdm(d, desc="Building corpus (unique docs)"):
        doc_id = row["id"]
        if doc_id not in corpus_map:
            corpus_map[doc_id] = row["document"]

    corpus_ids = list(corpus_map.keys())
    corpus_texts = [corpus_map[i] for i in corpus_ids]

    query_ids = []
    query_texts = []
    qrels = {}
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


def load_mteb_ir(dataset_id: str, split: str = "test") -> DatasetIR:
    """
    Loads MTEB-style IR datasets that expose:
      - config 'corpus' with fields id/text/title
      - config 'queries' with fields id/text
      - config 'qrels' with fields query-id/corpus-id/score

    Example:
      - mteb/GerDaLIR
    """
    def _load(cfg: str):
        # Some environments can end up with partial caches; try a forced redownload once.
        try:
            return load_dataset(dataset_id, cfg, split=split)
        except Exception:
            return load_dataset(dataset_id, cfg, split=split, download_mode="force_redownload")

    try:
        corpus = _load("corpus")
        queries = _load("queries")
        qrels_ds = _load("qrels")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MTEB-style dataset configs for {dataset_id} (need corpus/queries/qrels). "
            f"Original error: {e}"
        ) from e

    corpus_ids = [row["id"] for row in corpus]
    corpus_texts = [row.get("text", "") for row in corpus]

    query_ids = [row["id"] for row in queries]
    query_texts = [row.get("text", "") for row in queries]

    qrels: Dict[str, list] = {}
    for row in qrels_ds:
        if float(row.get("score", 0)) <= 0:
            continue
        qid = row["query-id"]
        did = row["corpus-id"]
        qrels.setdefault(qid, []).append(did)

    return DatasetIR(
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
    )


