from datasets import load_dataset

dataset = load_dataset("parquet", data_files="dataset.parquet")

dataset.push_to_hub("krapfi/Advanced-Information-Retrieval")
