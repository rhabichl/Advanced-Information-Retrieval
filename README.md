# Advanced-Information-Retrieval

This repo currently contains the project design document plus a small baseline evaluation script.

## Baseline (no fine-tuning)

Baseline = run retrieval with a **pretrained** embedding model and report **Precision@k / Recall@k / nDCG@k**.

### Setup

```bash
python3 -m pip install -r requirements.txt
```

## Evaluation

Dataset: [`mteb/GerDaLIRSmall`](https://huggingface.co/datasets/mteb/GerDaLIRSmall), [`krapfi/Advanced-Information-Retrieval`](https://huggingface.co/datasets/krapfi/Advanced-Information-Retrieval), [`mteb/LeCaRDv2`](https://huggingface.co/datasets/mteb/LeCaRDv2)

```bash
python3 compute_sim.py
```

## Fine-tuning (German pairs)

Dataset (pairs): [`mteb/GerDaLIR`](https://huggingface.co/datasets/mteb/GerDaLIR)

On a GPU box, a safe first smoke run:

```bash
python3 finetune_lora.py 
```

After fine-tuning, evaluate the model

```bash
python3 compute_sim.py
```

## Presentation (Marp)

The presentation lives in `presentation/presentation.md` and is exported to `presentation/presentation.pdf`.

To compile the PDF from the Markdown using Marp:

```bash
marp presentation/presentation.md --pdf --allow-local-files --output presentation/presentation.pdf
```
