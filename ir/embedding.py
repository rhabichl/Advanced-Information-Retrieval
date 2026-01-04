from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def load_encoder(model_name_or_path: str):
    """
    Loads a Hugging Face encoder model and tokenizer.

    We keep this generic:
    - trust_remote_code=True for models like Qwen3-Embedding
    - pooling handled in `get_sentence_embeddings`
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
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def get_sentence_embeddings(model_output, attention_mask):
    """
    Try a few common conventions used by embedding models.
    """
    # Some remote-code models return dict-like outputs
    if isinstance(model_output, dict):
        if "sentence_embedding" in model_output:
            return model_output["sentence_embedding"]
        if "embeddings" in model_output:
            return model_output["embeddings"]
        if "pooler_output" in model_output:
            return model_output["pooler_output"]
        if "last_hidden_state" in model_output:
            return mean_pool(model_output["last_hidden_state"], attention_mask)

    # Standard HF ModelOutput
    if hasattr(model_output, "sentence_embedding"):
        return model_output.sentence_embedding
    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output
    if hasattr(model_output, "last_hidden_state"):
        return mean_pool(model_output.last_hidden_state, attention_mask)

    raise ValueError("Unsupported model output format; add a pooling adapter in ir/embedding.py")


def encode_texts(
    texts: Sequence[str],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int,
    desc: str = "Encoding",
) -> np.ndarray:
    import torch

    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
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
            pooled = get_sentence_embeddings(out, encoded.get("attention_mask"))
        embs.append(pooled.detach().cpu().numpy())
    return np.vstack(embs)


def move_to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


