import numpy as np
import pandas as pd

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

def dcg_at_k(relevance_scores, k):
    """
    Calculate Discounted Cumulative Gain at k
    :param relevance_scores: list of relevance scores (1 for relevant, 0 for not)
    :param k: cutoff position
    """
    relevance_scores = np.array(relevance_scores[:k])
    if relevance_scores.size == 0:
        return 0.0
    
    # DCG formula: sum(rel_i / log2(i+2)) for i in range(k)
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return np.sum(relevance_scores / discounts)

def ndcg_at_k(relevance_scores, k):
    """
    Calculate Normalized Discounted Cumulative Gain at k
    :param relevance_scores: list of relevance scores in ranked order
    :param k: cutoff position
    """
    dcg = dcg_at_k(relevance_scores, k)
    
    # Ideal DCG: sort relevance scores in descending order
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def chi():
    mappings = load_dataset("mteb/LeCaRDv2", split="test")
    queries_qer = pd.read_pickle("./embeddings/chinese_query_1024.pkl")
    documents_ger = pd.read_pickle("./embeddings/chinese_documents_1024chars.pkl")

    # Convert HuggingFace dataset to pandas DataFrame
    qrels_df = pd.DataFrame(mappings)

    # Merge with query embeddings
    merged_df = qrels_df.merge(
        queries_qer,
        left_on='query-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from queries
    merged_df = merged_df.rename(columns={'embedding': 'query_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

    # Merge with document embeddings
    merged_df = merged_df.merge(
        documents_ger,
        left_on='corpus-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from documents
    merged_df = merged_df.rename(columns={'embedding': 'document_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column










    TESTSIZE = 2000
    k_values = [1, 5, 10, 20, 50, 100]

    # Get random sample of query-ids
    all_query_ids = merged_df["query-id"].unique()
    sampled_query_ids = random.sample(list(all_query_ids), min(TESTSIZE, len(all_query_ids)))

    # Filter merged_df to only include sampled queries
    test_data = merged_df[merged_df["query-id"].isin(sampled_query_ids)]

    print(f"Testing on {len(sampled_query_ids)} queries")
    print(f"Total query-document pairs in test set: {len(test_data)}")

    # Calculate recall
    recall = {k: [] for k in k_values}
    precision = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    for query_id, group in tqdm(test_data.groupby("query-id"), total=len(sampled_query_ids)):
        # Get the query embedding
        query_embedding = group.iloc[0]["query_embedding"]

        # Get all correct document IDs for this query
        correct_doc_ids = set(group["corpus-id"].values)
        num_relevant = len(correct_doc_ids)

        # Get top results - search in ALL documents
        result = get_k_results_from_dot_ger(query_embedding, merged_df, max(k_values))

        # Create relevance list for the ranking (1 if relevant, 0 otherwise)
        relevance_scores = [1 if doc_idx in correct_doc_ids else 0 for score, doc_idx in result]

        # Calculate metrics for each k
        for k in k_values:
            # Get top-k results
            top_k_docs = {doc_idx for score, doc_idx in result[:k]}

            # Recall@k: proportion of relevant docs found
            num_relevant_found = len(top_k_docs.intersection(correct_doc_ids))
            recall[k].append(num_relevant_found / num_relevant if num_relevant > 0 else 0)

            # Precision@k: proportion of retrieved docs that are relevant
            precision[k].append(num_relevant_found / k)

            # nDCG@k
            ndcg[k].append(ndcg_at_k(relevance_scores, k))

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\n{'Metric':<15} " + " ".join([f"@{k:<6}" for k in k_values]))
    print("-"*60)

    # Recall
    recall_str = f"{'Recall':<15} "
    for k in k_values:
        recall_str += f"{np.mean(recall[k]):<7.4f} "
    print(recall_str)

    # Precision
    precision_str = f"{'Precision':<15} "
    for k in k_values:
        precision_str += f"{np.mean(precision[k]):<7.4f} "
    print(precision_str)

    # nDCG
    ndcg_str = f"{'nDCG':<15} "
    for k in k_values:
        ndcg_str += f"{np.mean(ndcg[k]):<7.4f} "
    print(ndcg_str)

    print(f"\nNumber of queries evaluated: {len(sampled_query_ids)}")
    mappings_old = mappings
    queries_qer_old = queries_qer
    documents_ger_old = documents_ger
    merged_df_old = merged_df

    mappings = load_dataset("mteb/LeCaRDv2", split="test")
    queries_qer = pd.read_pickle("./embeddings/finetune_chinese_query_1024.pkl")
    documents_ger = pd.read_pickle("./embeddings/finetune_chinese_documents_1024chars.pkl")

        # Convert HuggingFace dataset to pandas DataFrame
    qrels_df = pd.DataFrame(mappings)

    # Merge with query embeddings
    merged_df = qrels_df.merge(
        queries_qer,
        left_on='query-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from queries
    merged_df = merged_df.rename(columns={'embedding': 'query_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

    # Merge with document embeddings
    merged_df = merged_df.merge(
        documents_ger,
        left_on='corpus-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from documents
    merged_df = merged_df.rename(columns={'embedding': 'document_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

        

    TESTSIZE = 2000
    k_values = [1, 5, 10, 20, 50, 100]
    
    recall = {k: [] for k in k_values}
    precision = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    for query_id, group in tqdm(test_data.groupby("query-id"), total=len(sampled_query_ids)):
        # Get the query embedding
        query_embedding = group.iloc[0]["query_embedding"]

        # Get all correct document IDs for this query
        correct_doc_ids = set(group["corpus-id"].values)
        num_relevant = len(correct_doc_ids)

        # Get top results - search in ALL documents
        result = get_k_results_from_dot_ger(query_embedding, merged_df, max(k_values))

        # Create relevance list for the ranking (1 if relevant, 0 otherwise)
        relevance_scores = [1 if doc_idx in correct_doc_ids else 0 for score, doc_idx in result]

        # Calculate metrics for each k
        for k in k_values:
            # Get top-k results
            top_k_docs = {doc_idx for score, doc_idx in result[:k]}

            # Recall@k: proportion of relevant docs found
            num_relevant_found = len(top_k_docs.intersection(correct_doc_ids))
            recall[k].append(num_relevant_found / num_relevant if num_relevant > 0 else 0)

            # Precision@k: proportion of retrieved docs that are relevant
            precision[k].append(num_relevant_found / k)

            # nDCG@k
            ndcg[k].append(ndcg_at_k(relevance_scores, k))

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\n{'Metric':<15} " + " ".join([f"@{k:<6}" for k in k_values]))
    print("-"*60)

    # Recall
    recall_str = f"{'Recall':<15} "
    for k in k_values:
        recall_str += f"{np.mean(recall[k]):<7.4f} "
    print(recall_str)

    # Precision
    precision_str = f"{'Precision':<15} "
    for k in k_values:
        precision_str += f"{np.mean(precision[k]):<7.4f} "
    print(precision_str)

    # nDCG
    ndcg_str = f"{'nDCG':<15} "
    for k in k_values:
        ndcg_str += f"{np.mean(ndcg[k]):<7.4f} "
    print(ndcg_str)

    print(f"\nNumber of queries evaluated: {len(sampled_query_ids)}")



def ger():
    mappings = load_dataset("mteb/GerDaLIRSmall", split="test")
    queries_qer = pd.read_pickle("./embeddings/query_1024.pkl")
    documents_ger = pd.read_pickle("./embeddings/documents_1024chars.pkl")

    # Convert HuggingFace dataset to pandas DataFrame
    qrels_df = pd.DataFrame(mappings)

    # Merge with query embeddings
    merged_df = qrels_df.merge(
        queries_qer,
        left_on='query-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from queries
    merged_df = merged_df.rename(columns={'embedding': 'query_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

    # Merge with document embeddings
    merged_df = merged_df.merge(
        documents_ger,
        left_on='corpus-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from documents
    merged_df = merged_df.rename(columns={'embedding': 'document_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

        

    TESTSIZE = 2000
    k_values = [1, 5, 10, 20, 50, 100]

    # Get random sample of query-ids
    all_query_ids = merged_df["query-id"].unique()
    sampled_query_ids = random.sample(list(all_query_ids), min(TESTSIZE, len(all_query_ids)))

    # Filter merged_df to only include sampled queries
    test_data = merged_df[merged_df["query-id"].isin(sampled_query_ids)]

    print(f"Testing on {len(sampled_query_ids)} queries")
    print(f"Total query-document pairs in test set: {len(test_data)}")

    # Calculate recall
    recall = {k: [] for k in k_values}
    precision = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    for query_id, group in tqdm(test_data.groupby("query-id"), total=len(sampled_query_ids)):
        # Get the query embedding
        query_embedding = group.iloc[0]["query_embedding"]

        # Get all correct document IDs for this query
        correct_doc_ids = set(group["corpus-id"].values)
        num_relevant = len(correct_doc_ids)

        # Get top results - search in ALL documents
        result = get_k_results_from_dot_ger(query_embedding, merged_df, max(k_values))

        # Create relevance list for the ranking (1 if relevant, 0 otherwise)
        relevance_scores = [1 if doc_idx in correct_doc_ids else 0 for score, doc_idx in result]

        # Calculate metrics for each k
        for k in k_values:
            # Get top-k results
            top_k_docs = {doc_idx for score, doc_idx in result[:k]}

            # Recall@k: proportion of relevant docs found
            num_relevant_found = len(top_k_docs.intersection(correct_doc_ids))
            recall[k].append(num_relevant_found / num_relevant if num_relevant > 0 else 0)

            # Precision@k: proportion of retrieved docs that are relevant
            precision[k].append(num_relevant_found / k)

            # nDCG@k
            ndcg[k].append(ndcg_at_k(relevance_scores, k))

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\n{'Metric':<15} " + " ".join([f"@{k:<6}" for k in k_values]))
    print("-"*60)

    # Recall
    recall_str = f"{'Recall':<15} "
    for k in k_values:
        recall_str += f"{np.mean(recall[k]):<7.4f} "
    print(recall_str)

    # Precision
    precision_str = f"{'Precision':<15} "
    for k in k_values:
        precision_str += f"{np.mean(precision[k]):<7.4f} "
    print(precision_str)

    # nDCG
    ndcg_str = f"{'nDCG':<15} "
    for k in k_values:
        ndcg_str += f"{np.mean(ndcg[k]):<7.4f} "
    print(ndcg_str)

    print(f"\nNumber of queries evaluated: {len(sampled_query_ids)}")

    mappings_old = mappings
    queries_qer_old = queries_qer
    documents_ger_old = documents_ger
    merged_df_old = merged_df

    mappings = load_dataset("mteb/GerDaLIRSmall", split="test")
    queries_qer = pd.read_pickle("./embeddings/finetuned_query_1024.pkl")
    documents_ger = pd.read_pickle("./embeddings/finetuned_documents_1024chars.pkl")

        # Convert HuggingFace dataset to pandas DataFrame
    qrels_df = pd.DataFrame(mappings)

    # Merge with query embeddings
    merged_df = qrels_df.merge(
        queries_qer,
        left_on='query-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from queries
    merged_df = merged_df.rename(columns={'embedding': 'query_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

    # Merge with document embeddings
    merged_df = merged_df.merge(
        documents_ger,
        left_on='corpus-id',
        right_on='_id',
        how='left'
    )

    # Rename the embedding column to be clear it's from documents
    merged_df = merged_df.rename(columns={'embedding': 'document_embedding'})
    merged_df = merged_df.drop(columns=['_id'])  # Remove redundant _id column

        

    TESTSIZE = 2000
    k_values = [1, 5, 10, 20, 50, 100]
    
    # Get random sample of query-ids
    all_query_ids = merged_df["query-id"].unique()
    sampled_query_ids = random.sample(list(all_query_ids), min(TESTSIZE, len(all_query_ids)))
    
    # Filter merged_df to only include sampled queries
    test_data = merged_df[merged_df["query-id"].isin(sampled_query_ids)]
    
    print(f"Testing on {len(sampled_query_ids)} queries")
    print(f"Total query-document pairs in test set: {len(test_data)}")
    
    # Calculate recall
    recall = {k: [] for k in k_values}
    precision = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    for query_id, group in tqdm(test_data.groupby("query-id"), total=len(sampled_query_ids)):
        # Get the query embedding
        query_embedding = group.iloc[0]["query_embedding"]

        # Get all correct document IDs for this query
        correct_doc_ids = set(group["corpus-id"].values)
        num_relevant = len(correct_doc_ids)

        # Get top results - search in ALL documents
        result = get_k_results_from_dot_ger(query_embedding, merged_df, max(k_values))

        # Create relevance list for the ranking (1 if relevant, 0 otherwise)
        relevance_scores = [1 if doc_idx in correct_doc_ids else 0 for score, doc_idx in result]

        # Calculate metrics for each k
        for k in k_values:
            # Get top-k results
            top_k_docs = {doc_idx for score, doc_idx in result[:k]}

            # Recall@k: proportion of relevant docs found
            num_relevant_found = len(top_k_docs.intersection(correct_doc_ids))
            recall[k].append(num_relevant_found / num_relevant if num_relevant > 0 else 0)

            # Precision@k: proportion of retrieved docs that are relevant
            precision[k].append(num_relevant_found / k)

            # nDCG@k
            ndcg[k].append(ndcg_at_k(relevance_scores, k))

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\n{'Metric':<15} " + " ".join([f"@{k:<6}" for k in k_values]))
    print("-"*60)

    # Recall
    recall_str = f"{'Recall':<15} "
    for k in k_values:
        recall_str += f"{np.mean(recall[k]):<7.4f} "
    print(recall_str)

    # Precision
    precision_str = f"{'Precision':<15} "
    for k in k_values:
        precision_str += f"{np.mean(precision[k]):<7.4f} "
    print(precision_str)

    # nDCG
    ndcg_str = f"{'nDCG':<15} "
    for k in k_values:
        ndcg_str += f"{np.mean(ndcg[k]):<7.4f} "
    print(ndcg_str)

    print(f"\nNumber of queries evaluated: {len(sampled_query_ids)}")


def get_k_results_from_dot_ger(query, documents, k):
    """
    Get the indices for the top k matching vectors
    :param query: np.array 1D (single query embedding)
    :param documents: DataFrame with 'document_embedding' and 'corpus-id' columns
    :param k: top results to return
    :return: list of (score, corpus_id) tuples for top k results
    """
    indices = []
    for idx, row in documents.iterrows():
        indices.append((np.dot(query, row["document_embedding"]), row["corpus-id"]))
    
    # Sort by score (first element) in descending order
    indices.sort(reverse=True, key=lambda x: x[0])
    return indices[0:k]



def get_k_results_from_dot(query, documents, k):
    """
    Get the indices for the top k matching vectors
    :param query: np.array 1D
    :param documents: list/array of np.array
    :param k: top results to return
    :return: list of (score, idx) tuples for top k results
    """
    indices = []
    for idx, doc in enumerate(documents):
        indices.append((np.dot(query, doc), idx))
    
    # Sort by score (first element) in descending order
    indices.sort(reverse=True, key=lambda x: x[0])
    return indices[0:k]

def aut():
    querie_aut = pd.read_pickle("./embeddings/query_1024chars_aut.pkl")
    document_aut = pd.read_pickle("./embeddings/documents_1024chars_aut.pkl")
    
    # Reset indices to ensure proper alignment
    querie_aut = querie_aut.reset_index(drop=True)
    document_aut = document_aut.reset_index(drop=True)
    
    # Create merged dataframe using row index alignment
    merged_austrian = pd.DataFrame({
        'query_embedding': querie_aut['embedding'].values,
        'document_embedding': document_aut['embedding'].values,
        'score': 1.0,
        'query-id': querie_aut.index,
        'corpus-id': document_aut.index
    })
    
    TESTSIZE = 500
    print(merged_austrian.head())
    
    testset = random.sample(list(merged_austrian.index), TESTSIZE)
    test_data = merged_austrian.iloc[testset].reset_index(drop=True)
    print(test_data)
    
    k_values = [1, 5, 10, 20, 50, 100]
    recall = {k: [] for k in k_values}
    precision = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    
    # Convert to list for get_k_results_from_dotvev
    all_docs = list(document_aut['embedding'].values)
    
    # For pair-level: each query has exactly 1 relevant document
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        query_embedding = row["query_embedding"]
        correct_doc_id = row["corpus-id"]

        result = get_k_results_from_dot_ger(query_embedding, merged_austrian, max(k_values))

        # Relevance list
        relevance_scores = [1 if doc_idx == correct_doc_id else 0 for score, doc_idx in result]

        # Find position of correct document
        position = None
        for pos, (score, doc_idx) in enumerate(result):
            if doc_idx == correct_doc_id:
                position = pos
                break
            
        for k in k_values:
            # Recall and Precision are same for single relevant doc
            if position is not None and position < k:
                recall[k].append(1)
                precision[k].append(1 / k)  # Only 1 relevant doc out of k
            else:
                recall[k].append(0)
                precision[k].append(0)

            ndcg[k].append(ndcg_at_k(relevance_scores, k))  
            
    recall_str = f"{'Recall':<15} "
    for k in k_values:
        recall_str += f"{np.mean(recall[k]):<7.4f} "
    print(recall_str)

    # Precision
    precision_str = f"{'Precision':<15} "
    for k in k_values:
        precision_str += f"{np.mean(precision[k]):<7.4f} "
    print(precision_str)

    # nDCG
    ndcg_str = f"{'nDCG':<15} "
    for k in k_values:
        ndcg_str += f"{np.mean(ndcg[k]):<7.4f} "
    print(ndcg_str)

    print(f"\nNumber of queries evaluated: {len(testset)}")

     # ========================================
    # FINETUNED AUSTRIAN DATASET
    # ========================================
    querie_ft = pd.read_pickle("./embeddings/finetuned_query_1024chars_aut.pkl")
    document_ft = pd.read_pickle("./embeddings/finetuned_documents_1024chars_aut.pkl")  


        # Reset indices to ensure proper alignment
    querie_ft = querie_ft.reset_index(drop=True)
    document_ft = document_ft.reset_index(drop=True)
    
    # Create merged dataframe using row index alignment
    merged_austrian_ft = pd.DataFrame({
        'query_embedding': querie_ft['embedding'].values,
        'document_embedding': document_ft['embedding'].values,
        'score': 1.0,
        'query-id': querie_ft.index,
        'corpus-id': document_ft.index
    })
    
    testset = random.sample(list(merged_austrian_ft.index), TESTSIZE)
    test_data = merged_austrian_ft.iloc[testset].reset_index(drop=True)
    print()    
    k_values = [1, 5, 10, 20, 50, 100]
    recall = {k: [] for k in k_values}
    precision = {k: [] for k in k_values}
    ndcg = {k: [] for k in k_values}
    
    # Convert to list for get_k_results_from_dotvev
    all_docs = list(document_aut['embedding'].values)
    
    # For pair-level: each query has exactly 1 relevant document
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
        query_embedding = row["query_embedding"]
        correct_doc_id = row["corpus-id"]

        result = get_k_results_from_dot_ger(query_embedding, merged_austrian, max(k_values))

        # Relevance list
        relevance_scores = [1 if doc_idx == correct_doc_id else 0 for score, doc_idx in result]

        # Find position of correct document
        position = None
        for pos, (score, doc_idx) in enumerate(result):
            if doc_idx == correct_doc_id:
                position = pos
                break
            
        for k in k_values:
            # Recall and Precision are same for single relevant doc
            if position is not None and position < k:
                recall[k].append(1)
                precision[k].append(1 / k)  # Only 1 relevant doc out of k
            else:
                recall[k].append(0)
                precision[k].append(0)

            ndcg[k].append(ndcg_at_k(relevance_scores, k))  
            
    recall_str = f"{'Recall':<15} "
    for k in k_values:
        recall_str += f"{np.mean(recall[k]):<7.4f} "
    print(recall_str)

    # Precision
    precision_str = f"{'Precision':<15} "
    for k in k_values:
        precision_str += f"{np.mean(precision[k]):<7.4f} "
    print(precision_str)

    # nDCG
    ndcg_str = f"{'nDCG':<15} "
    for k in k_values:
        ndcg_str += f"{np.mean(ndcg[k]):<7.4f} "
    print(ndcg_str)

    print(f"\nNumber of queries evaluated: {len(testset)}")


chi()
#ger()
#aut()