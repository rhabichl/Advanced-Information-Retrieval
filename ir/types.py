from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class DatasetIR:
    corpus_ids: List[str]
    corpus_texts: List[str]
    query_ids: List[str]
    query_texts: List[str]
    qrels: Dict[str, List[str]]  # qid -> list of relevant corpus_ids


