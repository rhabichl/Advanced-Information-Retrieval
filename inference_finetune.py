from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig


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
model.load_adapter("krapfi/Qwen3-Embedding-8B-Ger-Legal")
embeddings = model.encode(["This is an example sentence", "Each sentence is converted"])



e = model.encode(["Hello world"])

print(e)