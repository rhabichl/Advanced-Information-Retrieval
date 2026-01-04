# Advanced-Information-Retrieval

This repo currently contains the project design document plus a small baseline evaluation script.

## Baseline (no fine-tuning)

Baseline = run retrieval with a **pretrained** embedding model and report **Precision@k / Recall@k / nDCG@k**.

### Setup

```bash
python3 -m pip install -r requirements.txt
```

### Run on Austrian dataset

Dataset: [`krapfi/Advanced-Information-Retrieval`](https://huggingface.co/datasets/krapfi/Advanced-Information-Retrieval)

```bash
python3 baseline/baseline_eval.py \
  --dataset krapfi/Advanced-Information-Retrieval \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --train_ratio 0.7 \
  --max_queries 5000 \
  --k 1,3,5,10
```

Notes:

- `--train_ratio 0.7` creates a 70/30 split over queries (easy row-level split).
- `--max_queries` is just for speed while debugging; set `0` to evaluate all test queries.
- For your real baseline, swap `--model` to your intended model (e.g. `Qwen/Qwen3-Embedding-8B`) and run on a GPU machine.
