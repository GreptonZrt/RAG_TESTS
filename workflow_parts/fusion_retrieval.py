"""
Fusion Retrieval: Combining Vector and BM25 Search

This module implements fusion retrieval that combines semantic vector search
with keyword-based BM25 retrieval for improved document retrieval.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional


def _normalize_embedding(embedding):
    """
    Normalize embedding to ensure it's a list of floats.
    
    Handles both raw lists and EmbeddingItem objects.
    """
    if hasattr(embedding, 'embedding'):
        # It's an EmbeddingItem object
        return embedding.embedding
    return embedding


class SimpleVectorStore:
    """A simple vector store implementation using NumPy for similarity search."""
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store text content
        self.metadata = []  # List to store metadata
    
    def add_item(self, text: str, embedding: List[float], metadata: Optional[Dict] = None):
        """
        Add an item to the vector store.
        
        Args:
            text (str): The text content
            embedding (List[float]): The embedding vector
            metadata (Dict, optional): Additional metadata
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def add_items(self, items: List[Dict], embeddings: List[List[float]]):
        """
        Add multiple items to the vector store.
        
        Args:
            items (List[Dict]): List of text items with optional metadata
            embeddings (List[List[float]]): List of embedding vectors
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            text = item if isinstance(item, str) else item.get("text", "")
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            metadata["index"] = i
            # Normalize embedding in case it's an EmbeddingItem
            normalized_embedding = _normalize_embedding(embedding)
            self.add_item(text=text, embedding=normalized_embedding, metadata=metadata)
    
    def similarity_search_with_scores(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Find the most similar items to a query embedding with similarity scores.
        
        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of results to return
            
        Returns:
            List[Dict]: Top k most similar items with scores
        """
        if not self.vectors:
            return []
        
        # Normalize query embedding in case it's an EmbeddingItem
        normalized_query = _normalize_embedding(query_embedding)
        query_vector = np.array(normalized_query)
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        
        return results
    
    def get_all_documents(self) -> List[Dict]:
        """
        Get all documents in the store.
        
        Returns:
            List[Dict]: All documents with text and metadata
        """
        return [{"text": text, "metadata": meta} for text, meta in zip(self.texts, self.metadata)]


def create_bm25_index(chunks: List) -> BM25Okapi:
    """
    Create a BM25 index from the given chunks.
    
    Args:
        chunks (List): List of text chunks (str or dict)
        
    Returns:
        BM25Okapi: A BM25 index
    """
    # Extract text from each chunk
    texts = []
    for chunk in chunks:
        if isinstance(chunk, str):
            texts.append(chunk)
        else:
            texts.append(chunk.get("text", ""))
    
    # Tokenize each document by splitting on whitespace
    tokenized_docs = [text.split() for text in texts]
    
    # Create the BM25 index using the tokenized documents
    bm25 = BM25Okapi(tokenized_docs)
    
    return bm25


def bm25_search(bm25: BM25Okapi, chunks: List, query: str, k: int = 5) -> List[Dict]:
    """
    Search the BM25 index with a query.
    
    Args:
        bm25 (BM25Okapi): BM25 index
        chunks (List): List of text chunks
        query (str): Query string
        k (int): Number of results to return
        
    Returns:
        List[Dict]: Top k results with scores
    """
    # Tokenize the query by splitting it into individual words
    query_tokens = query.split()
    
    # Get BM25 scores for the query tokens against the indexed documents
    scores = bm25.get_scores(query_tokens)
    
    results = []
    
    # Iterate over the scores and corresponding chunks
    for i, score in enumerate(scores):
        chunk = chunks[i]
        text = chunk if isinstance(chunk, str) else chunk.get("text", "")
        metadata = {} if isinstance(chunk, str) else chunk.get("metadata", {}).copy()
        metadata["index"] = i
        
        results.append({
            "text": text,
            "metadata": metadata,
            "bm25_score": float(score)
        })
    
    # Sort the results by BM25 score in descending order
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    
    # Return the top k results
    return results[:k]


def fusion_retrieval(query: str, chunks: List, vector_store: SimpleVectorStore, 
                    bm25_index: BM25Okapi, embedding_fn, k: int = 5, 
                    alpha: float = 0.5) -> List[Dict]:
    """
    Perform fusion retrieval combining vector-based and BM25 search.
    
    Args:
        query (str): Query string
        chunks (List): Original text chunks
        vector_store (SimpleVectorStore): Vector store
        bm25_index (BM25Okapi): BM25 index
        embedding_fn: Function to create embeddings
        k (int): Number of results to return
        alpha (float): Weight for vector scores (0-1), where 1-alpha is BM25 weight
        
    Returns:
        List[Dict]: Top k results based on combined scores
    """
    # Define small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Get vector search results
    query_embedding = embedding_fn(query)
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))
    
    # Get BM25 search results
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))
    
    # Create dictionaries to map document index to score
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}
    
    # Ensure all documents have scores for both methods
    all_docs = vector_store.get_all_documents()
    combined_results = []
    
    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)
        bm25_score = bm25_scores_dict.get(i, 0.0)
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })
    
    # Extract scores as arrays
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])
    
    # Normalize scores
    if np.max(vector_scores) > np.min(vector_scores):
        norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    else:
        norm_vector_scores = np.zeros_like(vector_scores)
    
    if np.max(bm25_scores) > np.min(bm25_scores):
        norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    else:
        norm_bm25_scores = np.zeros_like(bm25_scores)
    
    # Compute combined scores
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores
    
    # Add combined scores to results
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)
    
    # Sort by combined score (descending)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Return top k results
    return combined_results[:k]


def retrieve_vector_only(query: str, vector_store: SimpleVectorStore, 
                        embedding_fn, k: int = 5) -> List[Dict]:
    """
    Retrieve documents using only vector-based similarity search.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        embedding_fn: Function to create embeddings
        k (int): Number of documents to retrieve
        
    Returns:
        List[Dict]: Retrieved documents with similarity scores
    """
    query_embedding = embedding_fn(query)
    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)
    return retrieved_docs


def retrieve_bm25_only(query: str, chunks: List, bm25_index: BM25Okapi, 
                      k: int = 5) -> List[Dict]:
    """
    Retrieve documents using only BM25-based keyword search.
    
    Args:
        query (str): User query
        chunks (List): Text chunks
        bm25_index (BM25Okapi): BM25 index
        k (int): Number of documents to retrieve
        
    Returns:
        List[Dict]: Retrieved documents with BM25 scores
    """
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)
    return retrieved_docs
