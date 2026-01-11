from datasets import load_dataset
import numpy as np
import pandas as pd
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

def chi_document(model):
    dataset = load_dataset("mteb/LeCaRDv2", "corpus", split="corpus")
    documents = dataset["text"]
    ids = dataset["_id"]
    print(documents)
    truncated_list = [s[:1024] for s in documents]
    document_embeddings = model.encode_document(
        truncated_list,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Create DataFrame with IDs and embeddings
    df = pd.DataFrame({
        '_id': ids,
        'embedding': list(document_embeddings)
    })
    
    # Save both the DataFrame and the raw embeddings
    df.to_pickle("./embeddings/chinese_documents_1024chars.pkl")
    
    return df


def chi_query(model):
    dataset = load_dataset("mteb/LeCaRDv2", "queries", split="queries")
    querys = dataset["text"]
    ids = dataset["_id"]
    
    truncated_list = [s[:1024] for s in querys]
    query_embeddings = model.encode_query(
        truncated_list,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt_name="query",
    )
    
    # Create DataFrame with IDs and embeddings
    df = pd.DataFrame({
        '_id': ids,
        'embedding': list(query_embeddings)
    })
    
    # Save both the DataFrame and the raw embeddings
    df.to_pickle("./embeddings/chinese_query_1024.pkl")
    return df


def aut_document(model):
    dataset = load_dataset("krapfi/Advanced-Information-Retrieval", split="train")
    documents = dataset["document"]
    ids = dataset["id"]
    
    truncated_list = [s[:1024] for s in documents]
    document_embeddings = model.encode_document(
        truncated_list,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Create DataFrame with IDs and embeddings
    df = pd.DataFrame({
        '_id': ids,
        'embedding': list(document_embeddings)
    })
    
    # Save both the DataFrame and the raw embeddings
    df.to_pickle("./embeddings/finetuned_documents_1024chars_aut.pkl")
    
    return df

def aut_query(model):
    dataset = load_dataset("krapfi/Advanced-Information-Retrieval", split="train")
    querys = dataset["query"]
    ids = dataset["id"]
    
    truncated_list = [s[80:1024] for s in querys]
    query_embeddings = model.encode_query(
        truncated_list,
        show_progress_bar=True,
        batch_size=4,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt_name="query",
    )
    
    # Create DataFrame with IDs and embeddings
    df = pd.DataFrame({
        '_id': ids,
        'embedding': list(query_embeddings)
    })
    
    # Save both the DataFrame and the raw embeddings
    df.to_pickle("./embeddings/finetuned_query_1024chars_aut.pkl")
    return df



def ger_document(model):
    dataset = load_dataset("mteb/GerDaLIRSmall", "corpus", split="corpus")
    documents = dataset["text"]
    ids = dataset["_id"]
    
    truncated_list = [s[:1024] for s in documents]
    document_embeddings = model.encode_document(
        truncated_list,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Create DataFrame with IDs and embeddings
    df = pd.DataFrame({
        '_id': ids,
        'embedding': list(document_embeddings)
    })
    
    # Save both the DataFrame and the raw embeddings
    df.to_pickle("./embeddings/finetuned_documents_1024chars.pkl")
    
    return df

def ger_query(model):
    dataset = load_dataset("mteb/GerDaLIRSmall", "queries", split="queries")
    querys = dataset["text"]
    ids = dataset["_id"]
    
    truncated_list = [s[:1024] for s in querys]
    query_embeddings = model.encode_query(
        truncated_list,
        show_progress_bar=True,
        batch_size=4,
        convert_to_numpy=True,
        normalize_embeddings=True,
        prompt_name="query",
    )
    
    # Create DataFrame with IDs and embeddings
    df = pd.DataFrame({
        '_id': ids,
        'embedding': list(query_embeddings)
    })
    
    # Save both the DataFrame and the raw embeddings
    df.to_pickle("./embeddings/finetuned_query_1024.pkl")
    return df

def main():
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
    #model.load_adapter("krapfi/Qwen3-Embedding-8B-Ger-Legal")
    
    #ger_document(model)
    #ger_query_df = query(model)
    #aut_query(model)
    #aut_document(model)
    chi_document(model)
    chi_query(model)


    #print(f"Query DataFrame shape: {query_df.shape}")
    #print(query_df.head())

if __name__ == '__main__':
    main()
