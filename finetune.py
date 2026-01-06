from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from transformers import BitsAndBytesConfig
from transformers.integrations import TensorBoardCallback
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from sentence_transformers.evaluation import InformationRetrievalEvaluator


# 1. Load a model to finetune with 2. (Optional) model card data
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={
        "quantization_config": bnb_config,
        "device_map": "auto",
    },
)

# Load the three subsets
qrels = load_dataset("mteb/GerDaLIR", "qrels", split="test")
corpus = load_dataset("mteb/GerDaLIR", "corpus", split="test")
queries = load_dataset("mteb/GerDaLIR", "queries", split="test")

# Create lookup dictionaries for fast access
corpus_dict = {doc['id']: doc for doc in corpus}
queries_dict = {q['id']: q for q in queries}

# Build the dataset for CachedMultipleNegativesRankingLoss
data = []
for qrel in qrels:
    query_id = qrel['query-id']
    corpus_id = qrel['corpus-id']
    
    query_data = queries_dict[query_id]
    corpus_data = corpus_dict[corpus_id]
    
    # Extract text fields (adjust field names if needed)
    query_text = query_data.get('text', query_data.get('query', ''))
    corpus_text = corpus_data.get('text', corpus_data.get('document', ''))
    
    data.append({
        'anchor': query_text,
        'positive': corpus_text
    })


# Create dataset
st_dataset = Dataset.from_list(data)
st_dataset = st_dataset.select_columns(['anchor', 'positive'])

st_dataset = st_dataset.map(
    lambda example: {
        'anchor': example['anchor'][:512],
        'positive': example['positive'][:512]
    }
)


# Create 80/20 train/test split
split_dataset = st_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")
print(f"\nTrain example: {train_dataset[0]}")
print(f"Eval example: {eval_dataset[0]}")

# 4. Define a loss function
loss = CachedMultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/qwen3-8b-embedding-ger-legal",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    report_to="tensorboard",
    run_name="qwen3-8b-embedding-ger-legal",
)

# 6. t
eval_dataset = eval_dataset.map(
    lambda example, idx: {
        'anchor_id': f'q{idx}',
        'positive_id': f'd{idx}'
    },
    with_indices=True
)

# Then create evaluator
queries = {f'q{i}': anchor for i, anchor in enumerate(eval_dataset['anchor'])}
corpus = {f'd{i}': positive for i, positive in enumerate(eval_dataset['positive'])}
relevant_docs = {f'q{i}': {f'd{i}'} for i in range(len(eval_dataset))}

dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs
)

dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
    callbacks=[TensorBoardCallback()]
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs
)
test_evaluator(model)

# 8. Save the trained model
model.save_pretrained("models/qwen3-8b-embedding-ger-legal/final")
