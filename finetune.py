from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from peft import LoraConfig, get_peft_model

# 1. Load model WITHOUT quantization for training
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={
        "device_map": "auto",
    },
)

# 2. Add LoRA for efficient training (if memory is an issue)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)
# model = model.to_peft(peft_config)  # If using sentence-transformers with PEFT support

# Load datasets
qrels = load_dataset("mteb/GerDaLIR", "qrels", split="test")
corpus = load_dataset("mteb/GerDaLIR", "corpus", split="test")
queries = load_dataset("mteb/GerDaLIR", "queries", split="test")

corpus_dict = {doc['id']: doc for doc in corpus}
queries_dict = {q['id']: q for q in queries}

# Build training data
data = []
for qrel in qrels:
    query_id = qrel['query-id']
    corpus_id = qrel['corpus-id']
    
    query_data = queries_dict[query_id]
    corpus_data = corpus_dict[corpus_id]
    
    query_text = query_data.get('text', query_data.get('query', ''))
    corpus_text = corpus_data.get('text', corpus_data.get('document', ''))
    
    data.append({
        'anchor': query_text,
        'positive': corpus_text
    })

st_dataset = Dataset.from_list(data)

# Don't truncate - let the tokenizer handle it
split_dataset = st_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# 3. Setup proper evaluation
# Build proper relevant_docs mapping from qrels
eval_queries = {}
eval_corpus = {}
eval_relevant_docs = {}

for qrel in qrels:
    query_id = qrel['query-id']
    corpus_id = qrel['corpus-id']
    
    # Add query
    if query_id not in eval_queries:
        eval_queries[query_id] = queries_dict[query_id].get('text', '')
    
    # Add corpus doc
    if corpus_id not in eval_corpus:
        eval_corpus[corpus_id] = corpus_dict[corpus_id].get('text', '')
    
    # Add to relevant docs
    if query_id not in eval_relevant_docs:
        eval_relevant_docs[query_id] = set()
    eval_relevant_docs[query_id].add(corpus_id)

dev_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name="gerdalir-test"
)

# 4. Loss function
loss = CachedMultipleNegativesRankingLoss(model)

# 5. Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/qwen3-8b-embedding-ger-legal",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Reduce if OOM
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Use fp16 instead of 4-bit
    max_grad_norm=1.0,  # Gradient clipping
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_cosine_ndcg@10",
    report_to="tensorboard",
)

# 6. Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)

trainer.train()

# 7. Save
model.save_pretrained("models/qwen3-8b-embedding-ger-legal/final")