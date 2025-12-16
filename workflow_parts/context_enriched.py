"""
Context-Enriched RAG Module

Enhances retrieval by including neighboring chunks alongside the most relevant chunk.
This provides broader context for the LLM to generate better responses.
"""

import numpy as np
from typing import List


def context_enriched_search(
    query: str,
    text_chunks: List[str],
    embeddings: List,
    get_embedding_fn,
    k: int = 1,
    context_size: int = 1
) -> List[str]:
    """
    Retrieves the most relevant chunk along with its neighboring chunks.
    
    This function performs semantic search to find the top-k relevant chunks,
    then expands the result by including neighboring chunks for context.

    Args:
        query (str): Search query.
        text_chunks (List[str]): List of text chunks.
        embeddings (List): List of pre-computed chunk embeddings (not used directly, kept for API consistency).
        get_embedding_fn: Function to generate embeddings for the query.
        k (int): Number of relevant chunks to retrieve (default: 1).
        context_size (int): Number of neighboring chunks to include on each side (default: 1).

    Returns:
        List[str]: Relevant text chunks with contextual information.
        
    Example:
        >>> chunks = ["text 1", "text 2", "text 3", "text 4"]
        >>> results = context_enriched_search("query", chunks, [], get_embedding_fn)
        >>> # Returns the best match + neighbors
    """
    
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        if isinstance(vec1, float) or isinstance(vec2, float):
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    # Get query embedding
    try:
        query_embedding_response = get_embedding_fn(query.strip())
        # Extract the embedding vector
        if hasattr(query_embedding_response, 'embedding'):
            query_embedding = np.array(query_embedding_response.embedding)
        elif isinstance(query_embedding_response, list):
            query_embedding = np.array(query_embedding_response)
        else:
            query_embedding = np.array(query_embedding_response)
    except Exception as e:
        print(f"  [WARNING] Failed to embed query: {e}")
        return []
    
    similarity_scores = []

    # Compute similarity scores between query and each text chunk
    for i, chunk in enumerate(text_chunks):
        try:
            chunk_embedding_response = get_embedding_fn(chunk.strip())
            # Extract the embedding vector
            if hasattr(chunk_embedding_response, 'embedding'):
                chunk_embedding = np.array(chunk_embedding_response.embedding)
            elif isinstance(chunk_embedding_response, list):
                chunk_embedding = np.array(chunk_embedding_response)
            else:
                chunk_embedding = np.array(chunk_embedding_response)
            
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            similarity_scores.append((i, similarity_score))
        except Exception:
            # If embedding fails, assign low score
            similarity_scores.append((i, 0.0))

    # Sort chunks by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Collect the top-k chunks and their neighbors
    result_indices = set()
    for top_k_idx in range(min(k, len(similarity_scores))):
        top_index = similarity_scores[top_k_idx][0]
        
        # Add the main chunk
        result_indices.add(top_index)
        
        # Add neighboring chunks
        for offset in range(-context_size, context_size + 1):
            neighbor_idx = top_index + offset
            if 0 <= neighbor_idx < len(text_chunks):
                result_indices.add(neighbor_idx)
    
    # Return chunks in original order
    return [text_chunks[i] for i in sorted(result_indices)]
