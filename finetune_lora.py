from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch
import os

# Set environment variable for memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear any existing cache
torch.cuda.empty_cache()

# 1. Load model with 8-bit quantization (NOT 4-bit, 8-bit is trainable)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 8-bit is trainable, unlike 4-bit
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={
        "quantization_config": bnb_config,
        "device_map": "auto",
    },
)

# 2. Prepare for training and add LoRA
model[0].auto_model = prepare_model_for_kbit_training(model[0].auto_model)

peft_config = LoraConfig(
    r=4,  # Keep this small
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

from peft import get_peft_model
model[0].auto_model = get_peft_model(model[0].auto_model, peft_config)
model[0].auto_model.print_trainable_parameters()

# Enable gradient checkpointing
model[0].auto_model.gradient_checkpointing_enable()

# 3. Load data
qrels = load_dataset("mteb/GerDaLIR", "qrels", split="test")
corpus = load_dataset("mteb/GerDaLIR", "corpus", split="test")
queries = load_dataset("mteb/GerDaLIR", "queries", split="test")

corpus_dict = {doc['id']: doc for doc in corpus}
queries_dict = {q['id']: q for q in queries}

# Build training pairs
data = []
for qrel in qrels:
    query_id = qrel['query-id']
    corpus_id = qrel['corpus-id']
    
    query_data = queries_dict.get(query_id)
    corpus_data = corpus_dict.get(corpus_id)
    
    if query_data and corpus_data:
        query_text = query_data.get('text', query_data.get('query', ''))
        corpus_text = corpus_data.get('text', corpus_data.get('document', ''))
        
        if query_text and corpus_text:
            data.append({
                'anchor': query_text,
                'positive': corpus_text
            })

print(f"Total training pairs: {len(data)}")

st_dataset = Dataset.from_list(data)
split_dataset = st_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# 4. Setup lightweight evaluation
# Use only a small subset for evaluation during training
eval_queries = {}
eval_corpus = {}
eval_relevant_docs = {}

# Limit evaluation set to 500 samples
eval_limit = min(500, len(qrels))
eval_qrels = qrels.select(range(eval_limit))

for qrel in eval_qrels:
    query_id = qrel['query-id']
    corpus_id = qrel['corpus-id']
    
    if query_id not in eval_queries:
        eval_queries[query_id] = queries_dict[query_id].get('text', '')
    
    if corpus_id not in eval_corpus:
        eval_corpus[corpus_id] = corpus_dict[corpus_id].get('text', '')
    
    if query_id not in eval_relevant_docs:
        eval_relevant_docs[query_id] = set()
    eval_relevant_docs[query_id].add(corpus_id)

dev_evaluator = InformationRetrievalEvaluator(
    queries=eval_queries,
    corpus=eval_corpus,
    relevant_docs=eval_relevant_docs,
    name="gerdalir-dev",
    truncate_dim=None,
    score_functions={"cosine": lambda x, y: x @ y.T},  # Simpler scoring
)

# 5. Loss function with smaller cache
loss = CachedMultipleNegativesRankingLoss(
    model,
    mini_batch_size=8,  # Reduced cache size
)

# 6. EXTREME memory-saving training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/qwen3-8b-embedding-ger-legal",
    num_train_epochs=5,
    per_device_train_batch_size=1,  # Minimum batch size
    per_device_eval_batch_size=1,  # Minimum eval batch
    gradient_accumulation_steps=64,  # Effective batch = 32
    gradient_checkpointing=True,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    bf16=True,
    max_grad_norm=0.3,  # Lower for 8-bit training
    optim="adamw_8bit",  # 8-bit optimizer saves memory
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=500,  # Less frequent evaluation
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,  # Only keep 1 checkpoint
    logging_steps=50,
    load_best_model_at_end=False,  # Disable to save memory
    report_to="tensorboard",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # No parallel loading
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)

# 7. Train with error handling
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)

# Clear cache one more time
torch.cuda.empty_cache()

try:
    trainer.train()
except torch.cuda.OutOfMemoryError:
    print("\n⚠️ Still OOM! Try these emergency measures:")
    print("1. Set per_device_train_batch_size=1 and gradient_accumulation_steps=64")
    print("2. Reduce LoRA r from 8 to 4")
    print("3. Set mini_batch_size=8 in loss function")
    print("4. Disable evaluation during training (eval_strategy='no')")
    raise

# 8. Save only LoRA adapters
model[0].auto_model.save_pretrained("models/qwen3-8b-embedding-ger-legal/lora_adapters")
print("✅ Training complete! LoRA adapters saved.")