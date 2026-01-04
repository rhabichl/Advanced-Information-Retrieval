#!/usr/bin/env python3
"""
Fine-tune an embedding model on German (anchor, positive) pairs using in-batch negatives.

Default training dataset:
  malteos/ger-da-lir-anchor-positives-pairs  (splits: train/validation/test)

This script is intentionally simple:
- encode anchors and positives
- similarity matrix (cosine) over the batch
- cross-entropy loss with diagonal labels (InfoNCE / MultipleNegativesRankingLoss style)

For large models (e.g. Qwen3-Embedding-8B) you typically want:
- GPU
- small batch_size + gradient_accumulation
- optional LoRA (`--use_lora`)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Allow running as `python3 finetune/finetune_german_pairs.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def l2_normalize_torch(x, eps: float = 1e-12):
    import torch

    denom = torch.linalg.norm(x, dim=1, keepdim=True)
    denom = torch.clamp(denom, min=eps)
    return x / denom


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def get_sentence_embeddings(model_output, attention_mask):
    # mirror ir/embedding.py logic but keep this script self-contained on GPU boxes
    if isinstance(model_output, dict):
        if "sentence_embedding" in model_output:
            return model_output["sentence_embedding"]
        if "embeddings" in model_output:
            return model_output["embeddings"]
        if "pooler_output" in model_output:
            return model_output["pooler_output"]
        if "last_hidden_state" in model_output:
            return mean_pool(model_output["last_hidden_state"], attention_mask)
    if hasattr(model_output, "sentence_embedding"):
        return model_output.sentence_embedding
    if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
        return model_output.pooler_output
    if hasattr(model_output, "last_hidden_state"):
        return mean_pool(model_output.last_hidden_state, attention_mask)
    raise ValueError("Unsupported model output format; add pooling logic")


@dataclass
class Batch:
    anchors: List[str]
    positives: List[str]


def iter_pairs(dataset_id: str, split: str, max_pairs: int = 0):
    """
    Yields (anchor, positive) where 'positives' is a list and we take the first one.
    """
    ds = load_dataset(dataset_id, split=split)
    n = ds.num_rows
    limit = n if not max_pairs or max_pairs <= 0 else min(n, max_pairs)
    for i in range(limit):
        row = ds[i]
        anchor = row["anchor"]
        positives = row["positives"]
        if not positives:
            continue
        yield anchor, positives[0]


def batcher(it, batch_size: int):
    buf_a: List[str] = []
    buf_p: List[str] = []
    for a, p in it:
        buf_a.append(a)
        buf_p.append(p)
        if len(buf_a) >= batch_size:
            yield Batch(buf_a, buf_p)
            buf_a, buf_p = [], []
    if buf_a:
        yield Batch(buf_a, buf_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="malteos/ger-da-lir-anchor-positives-pairs")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--eval_split", default="validation")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--output_dir", default="outputs/finetuned-german")
    ap.add_argument("--max_train_pairs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max_eval_pairs", type=int, default=2000, help="0 = no cap")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    if args.use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise SystemExit(
                f"--use_lora requested but peft is not available: {e}. Install peft on the training machine."
            )
        # generic target modules; users can adjust if needed
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, lora_cfg)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Build streaming-ish iterators (indexing is fine at these sizes)
    train_pairs = list(iter_pairs(args.dataset, args.train_split, args.max_train_pairs))
    eval_pairs = list(iter_pairs(args.dataset, args.eval_split, args.max_eval_pairs))

    steps_per_epoch = int(np.ceil(len(train_pairs) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(0.05 * total_steps), 1)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def encode_batch(texts: List[str]) -> torch.Tensor:
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        emb = get_sentence_embeddings(out, enc.get("attention_mask"))
        emb = l2_normalize_torch(emb)
        return emb

    def batch_loss(a_emb: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        # cosine similarity because embeddings are normalized
        sim = (a_emb @ p_emb.T) / args.temperature  # (b, b)
        labels = torch.arange(sim.size(0), device=sim.device)
        return torch.nn.functional.cross_entropy(sim, labels)

    @torch.no_grad()
    def eval_once() -> float:
        model.eval()
        losses = []
        for b in batcher(iter(eval_pairs), args.batch_size):
            a = encode_batch(b.anchors)
            p = encode_batch(b.positives)
            loss = batch_loss(a, p)
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float("nan")

    global_step = 0
    running = []

    print(f"Device: {device}")
    print(f"Train pairs: {len(train_pairs)}  Eval pairs: {len(eval_pairs)}")
    print(f"Total steps: {total_steps}  Warmup: {warmup_steps}")

    for epoch in range(args.epochs):
        rng = np.random.default_rng(42 + epoch)
        order = rng.permutation(len(train_pairs))
        it = (train_pairs[i] for i in order)

        pbar = tqdm(batcher(it, args.batch_size), total=steps_per_epoch, desc=f"epoch {epoch+1}/{args.epochs}")
        optim.zero_grad(set_to_none=True)

        for b in pbar:
            a_emb = encode_batch(b.anchors)
            p_emb = encode_batch(b.positives)
            loss = batch_loss(a_emb, p_emb) / float(args.grad_accum)
            loss.backward()

            running.append(loss.item() * float(args.grad_accum))
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            global_step += 1
            if args.log_every and global_step % args.log_every == 0:
                avg = float(np.mean(running[-args.log_every :]))
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

            if args.eval_every and global_step % args.eval_every == 0 and eval_pairs:
                ev = eval_once()
                print(f"\n[step {global_step}] eval_loss={ev:.4f}")

    # Save
    try:
        model.save_pretrained(args.output_dir)
    except Exception:
        # PEFT models wrap base model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


