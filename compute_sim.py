import numpy as np
import pandas as pd

from datasets import load_dataset


from pymilvus import MilvusClient, DataType
import pandas as pd
import numpy as np

def upload_df_to_milvus(
    df: pd.DataFrame,
    collection_name: str,
    vector_field: str,
    vector_dim: int,
    id_field: str = None,
    host: str = "localhost",
    port: str = "19530",
    drop_existing: bool = False
):
    """
    Upload a pandas DataFrame to Milvus.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to upload
    collection_name : str
        Name of the Milvus collection
    vector_field : str
        Column name containing the vector embeddings (should be list/array)
    vector_dim : int
        Dimension of the vectors
    id_field : str, optional
        Column to use as primary key. If None, auto-generates IDs
    host : str
        Milvus server host
    port : str
        Milvus server port
    drop_existing : bool
        Whether to drop existing collection with same name
    
    Returns:
    --------
    Collection object
    """
    df = df.drop('query_embedding', axis=1)
    df = df.drop('corpus-id', axis=1)
    df = df.drop('score', axis=1)

    df = df.drop("query-id", axis=1)
    #print(df.to_dict())

    data = [{"id": i[0], "document_embedding": i[1]["document_embedding"]} for i in df.iterrows() ]
    print(data)
    # Connect to Milvus
    print(f"Connecting to Milvus at {host}:{port}...")
    client = MilvusClient(
        uri=f"http://{host}:{port}",
    )

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )


    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="document_embedding", datatype=DataType.FLOAT16_VECTOR, dim=vector_dim)

    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )

    res = client.insert(collection_name=collection_name, data=data)

    # Drop existing collection if requested



def calculate_metrics_at_k(merged_df, k_values=[1, 5, 10, 20, 50, 100]):
    """
    Calculate Recall@k, Precision@k, and NDCG@k for information retrieval
    
    Parameters:
    - merged_df: DataFrame with columns ['query-id', 'corpus-id', 'score', 'query_embedding', 'document_embedding']
    - k_values: list of k values to evaluate
    """
    
    # Calculate dot product (since embeddings are normalized)
    merged_df['predicted_score'] = merged_df.apply(
        lambda row: np.dot(row['query_embedding'], row['document_embedding']),
        axis=1
    )
    
    # Group by query
    queries = merged_df.groupby('query-id')
    
    results = {k: {'recall': [], 'precision': [], 'ndcg': []} for k in k_values}
    
    for query_id, group in queries:
        # Sort by predicted scores (descending)
        group_sorted = group.sort_values('predicted_score', ascending=False)
        
        # Get relevant documents (those with score > 0, adjust threshold as needed)
        relevant_docs = set(group[group['score'] > 0]['corpus-id'].values)
        total_relevant = len(relevant_docs)
        
        if total_relevant == 0:
            continue  # Skip queries with no relevant documents
        
        for k in k_values:
            # Get top-k predictions
            top_k = group_sorted.head(k)
            top_k_docs = set(top_k['corpus-id'].values)
            
            # Calculate metrics
            true_positives = len(top_k_docs.intersection(relevant_docs))
            
            # Recall@k
            recall = true_positives / total_relevant if total_relevant > 0 else 0
            results[k]['recall'].append(recall)
            
            # Precision@k
            precision = true_positives / k
            results[k]['precision'].append(precision)
            
            # NDCG@k
            # Get relevance scores for top-k documents
            dcg = 0
            for i, (_, row) in enumerate(top_k.iterrows()):
                rel = row['score']  # Use the ground truth score
                dcg += rel / np.log2(i + 2)  # i+2 because rank starts at 1
            
            # Calculate ideal DCG (sort by actual relevance)
            ideal_sorted = group.nlargest(k, 'score')
            idcg = 0
            for i, (_, row) in enumerate(ideal_sorted.iterrows()):
                rel = row['score']
                idcg += rel / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            results[k]['ndcg'].append(ndcg)
    
    # Calculate mean metrics
    metrics_summary = []
    for k in k_values:
        metrics_summary.append({
            'k': k,
            'Recall@k': np.mean(results[k]['recall']),
            'Precision@k': np.mean(results[k]['precision']),
            'NDCG@k': np.mean(results[k]['ndcg']),
            'Num_queries': len(results[k]['recall'])
        })
    
    return pd.DataFrame(metrics_summary)

def top_k_l2(query_emb, embeddings_series, k):
    # Convert Series[list/np.array] → (n, d) array
    embeddings = np.vstack(embeddings_series.to_numpy())

    query_emb = np.asarray(query_emb)

    dists = np.linalg.norm(embeddings - query_emb, axis=1)
    top_k_idx = np.argsort(dists)[:k]

    return top_k_idx, dists[top_k_idx]

def faiss_top_k_l2(query_emb, index, k):
    query_emb = np.asarray(query_emb, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    return indices[0], distances[0]



def calculate_metrics_aut(merged_df, k_values=[1, 5, 10, 20, 50, 100]):
    import faiss

    d = merged_df["document_embedding"].shape[1]  # 4096

    index = faiss.IndexFlatL2(d)  # exact Euclidean search
    index.add(embeddings)

    print("Number of vectors in index:", index.ntotal)


    for m in merged_df.iterrows():
        idxs, dists = faiss_top_k_l2(
            m[1]["query_embedding"],
            index,
            k=10
        )
        print(idxs)
    metrics_summary = []
    for k in k_values:
        metrics_summary.append({
            'k': k,
            'Recall@k': np.mean(results[k]['recall']),
            'Precision@k': np.mean(results[k]['precision']),
            'NDCG@k': np.mean(results[k]['ndcg']),
            'Num_queries': len(results[k]['recall'])
        })
    
    return pd.DataFrame(metrics_summary)

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

    print(calculate_metrics_at_k(merged_df))

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

    print(merged_df.head())
    print(merged_df_old.head())

    print(calculate_metrics_at_k(merged_df))

def create_merged_dataframe(queries_df, documents_df, sample_docs=1000):
    """
    For each query, create pairs with:
    - The matching document (score=1)
    - Random sample of other documents (score=0)
    """
    data = []
    
    # Get unique IDs and create embedding lookups
    unique_query_ids = queries_df['_id'].unique()
    unique_doc_ids = documents_df['_id'].unique()
    
    query_emb_dict = queries_df.groupby('_id')['embedding'].first().to_dict()
    doc_emb_dict = documents_df.groupby('_id')['embedding'].first().to_dict()
    
    print(f"Processing {len(unique_query_ids)} queries...")
    
    for i, query_id in enumerate(unique_query_ids):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(unique_query_ids)} queries...")
        
        q_emb = query_emb_dict[query_id]
        
        # Select documents for this query
        # Include the matching document + random sample of others
        if query_id in doc_emb_dict:
            # Add the matching document (positive)
            data.append({
                'query-id': query_id,
                'corpus-id': query_id,
                'query_embedding': q_emb,
                'document_embedding': doc_emb_dict[query_id],
                'score': 1.0
            })
        
        # Sample negative documents
        other_doc_ids = [did for did in unique_doc_ids if did != query_id]
        sampled_doc_ids = np.random.choice(
            other_doc_ids,
            size=min(sample_docs - 1, len(other_doc_ids)),
            replace=False
        )
        
        for doc_id in sampled_doc_ids:
            data.append({
                'query-id': query_id,
                'corpus-id': doc_id,
                'query_embedding': q_emb,
                'document_embedding': doc_emb_dict[doc_id],
                'score': 0.0
            })
    
    return pd.DataFrame(data)




def aut():
    print("\n" + "="*50)
    print("AUSTRIAN DATASET EVALUATION")
    print("="*50)   

    querie_aut = pd.read_pickle("./embeddings/query_1024chars_aut.pkl")
    document_aut = pd.read_pickle("./embeddings/documents_1024chars_aut.pkl")   

    print(f"Query shape: {querie_aut.shape}")
    print(f"Document shape: {document_aut.shape}")  

    # Check if they have the same number of rows
    if len(querie_aut) != len(document_aut):
        print(f"WARNING: Row counts don't match! Queries: {len(querie_aut)}, Documents: {len(document_aut)}")
        # Truncate to the minimum length
        min_len = min(len(querie_aut), len(document_aut))
        querie_aut = querie_aut.iloc[:min_len].reset_index(drop=True)
        document_aut = document_aut.iloc[:min_len].reset_index(drop=True)
        print(f"Truncated both to {min_len} rows")  

    # Reset indices to ensure proper alignment
    querie_aut = querie_aut.reset_index(drop=True)
    document_aut = document_aut.reset_index(drop=True)  

    # Create merged dataframe using row index alignment
    merged_austrian = pd.DataFrame({
        'query_embedding': querie_aut['embedding'].values,
        'document_embedding': document_aut['embedding'].values,
        'score': 1.0,  # All pairs are positive (matching pairs)
        'query-id': querie_aut.index,  # Use index as ID
        'corpus-id': document_aut.index  # Use index as ID
    })  

    print(f"Austrian merged shape: {merged_austrian.shape}")
    print(merged_austrian.head())   


    upload_df_to_milvus(merged_austrian, "aut_new", "document_embedding", 4096)
    exit(0)
    # Calculate metrics
    metrics_austrian = calculate_metrics_aut(merged_austrian, k_values=[1, 5, 10, 20, 50, 100])
    print("\nAustrian Dataset Metrics:")
    print(metrics_austrian) 


    # ========================================
    # FINETUNED AUSTRIAN DATASET
    # ========================================
    print("\n" + "="*50)
    print("FINETUNED AUSTRIAN DATASET EVALUATION")
    print("="*50)   

    querie_ft = pd.read_pickle("./embeddings/finetuned_query_1024chars_aut.pkl")
    document_ft = pd.read_pickle("./embeddings/finetuned_documents_1024chars_aut.pkl")  

    # Same process as above
    min_len = min(len(querie_ft), len(document_ft))
    querie_ft = querie_ft.iloc[:min_len].reset_index(drop=True)
    document_ft = document_ft.iloc[:min_len].reset_index(drop=True) 

    merged_austrian_ft = pd.DataFrame({
        'query_embedding': querie_ft['embedding'].values,
        'document_embedding': document_ft['embedding'].values,
        'score': 1.0,
        'query-id': querie_ft.index,
        'corpus-id': document_ft.index
    })  

    print(f"Finetuned Austrian merged shape: {merged_austrian_ft.shape}") 

    metrics_austrian_ft = calculate_metrics_aut(merged_austrian_ft, k_values=[1, 5, 10, 20, 50, 100])
    print("\nFinetuned Austrian Dataset Metrics:")
    print(metrics_austrian_ft)  


    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "="*50)
    print("AUSTRIAN: BASE vs FINETUNED COMPARISON")
    print("="*50)   

    comparison = pd.DataFrame({
        'k': metrics_austrian['k'],
        'Base_Recall': metrics_austrian['Recall@k'],
        'FT_Recall': metrics_austrian_ft['Recall@k'],
        'Recall_Δ': metrics_austrian_ft['Recall@k'] - metrics_austrian['Recall@k'],
        'Base_NDCG': metrics_austrian['NDCG@k'],
        'FT_NDCG': metrics_austrian_ft['NDCG@k'],
        'NDCG_Δ': metrics_austrian_ft['NDCG@k'] - metrics_austrian['NDCG@k']
    })
    print(comparison.to_string(index=False))


#ger()
aut()