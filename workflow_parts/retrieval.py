"""
Retrieval - Workflow Part

Semantic search and chunk retrieval for RAG workflows.
"""

import numpy as np
from typing import List


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def semantic_search(
    query: str,
    chunks: List[str],
    embeddings: List,
    k: int = 5
) -> List[str]:
    """
    Find top-k chunks most similar to query using semantic search.
    
    Args:
        query: The search query string
        chunks: List of text chunks to search through
        embeddings: List of EmbeddingItem objects for chunks
        k: Number of top results to return
        
    Returns:
        List[str]: Top-k most similar chunks
    """
    from workflow_parts.embedding import create_embeddings
    
    # Create embedding for query
    query_embeddings = create_embeddings(query)
    query_emb = np.array(query_embeddings[0].embedding)
    
    # Calculate similarity scores
    similarity_scores = []
    for i, chunk_emb in enumerate(embeddings):
        score = cosine_similarity(query_emb, np.array(chunk_emb.embedding))
        similarity_scores.append((i, score))
    
    # Sort by similarity (descending)
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k indices
    top_indices = [idx for idx, _ in similarity_scores[:k]]
    
    # Return top-k chunks
    return [chunks[i] for i in top_indices]
