"""
Contextual Chunk Headers RAG Module

Generates descriptive headers for each chunk using an LLM, then uses both
headers and text content for semantic search. This improves retrieval by
allowing the model to understand chunk context.
"""

import numpy as np
from typing import List, Callable, Any
from dataclasses import dataclass


@dataclass
class ChunkWithHeader:
    """Represents a chunk with its LLM-generated header."""
    header: str
    text: str
    text_embedding: List[float] = None
    header_embedding: List[float] = None


def generate_chunk_headers(
    chunks: List[str],
    get_embedding_fn: Callable,
    generate_header_fn: Callable
) -> List[ChunkWithHeader]:
    """
    Generates headers for chunks and computes their embeddings.

    Args:
        chunks (List[str]): List of text chunks.
        get_embedding_fn: Function that returns a callable for single embedding.
        generate_header_fn: Function to generate header for a chunk (e.g., LLM call).

    Returns:
        List[ChunkWithHeader]: Chunks with generated headers and embeddings.
    """
    # If get_embedding_fn is a callable that returns another callable (like our get_embedding_fn),
    # invoke it first
    if callable(get_embedding_fn):
        try:
            # Test if it returns a callable
            embed_test = get_embedding_fn("test")
            if not callable(embed_test) and hasattr(embed_test, 'embedding'):
                # It's a single embedding, so get_embedding_fn is our function
                embed_fn = get_embedding_fn
            else:
                # It might be a factory, try calling it
                embed_fn = get_embedding_fn()
        except TypeError:
            # If it fails, assume it's a factory function
            embed_fn = get_embedding_fn()
    else:
        embed_fn = get_embedding_fn
    
    chunked_with_headers = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Generate header for the chunk using LLM
            header = generate_header_fn(chunk)
            
            # Generate embeddings for both header and text
            try:
                text_embedding_response = embed_fn(chunk.strip())
                if hasattr(text_embedding_response, 'embedding'):
                    text_embedding = text_embedding_response.embedding
                elif isinstance(text_embedding_response, list):
                    text_embedding = text_embedding_response
                else:
                    text_embedding = list(text_embedding_response)
            except Exception as e:
                print(f"  [WARNING] Failed to embed chunk text {i}: {e}")
                text_embedding = None
            
            try:
                header_embedding_response = embed_fn(header.strip())
                if hasattr(header_embedding_response, 'embedding'):
                    header_embedding = header_embedding_response.embedding
                elif isinstance(header_embedding_response, list):
                    header_embedding = header_embedding_response
                else:
                    header_embedding = list(header_embedding_response)
            except Exception as e:
                print(f"  [WARNING] Failed to embed chunk header {i}: {e}")
                header_embedding = None
            
            chunked_with_headers.append(ChunkWithHeader(
                header=header,
                text=chunk,
                text_embedding=text_embedding,
                header_embedding=header_embedding
            ))
            
        except Exception as e:
            print(f"  [WARNING] Failed to generate header for chunk {i}: {e}")
            # Fallback: use first 100 chars as header
            fallback_header = chunk[:100] + "..."
            chunked_with_headers.append(ChunkWithHeader(
                header=fallback_header,
                text=chunk,
                text_embedding=None,
                header_embedding=None
            ))
    
    return chunked_with_headers


def cosine_similarity(vec1: Any, vec2: Any) -> float:
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1: First vector (list or numpy array).
        vec2: Second vector (list or numpy array).

    Returns:
        float: Cosine similarity score.
    """
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    except Exception:
        return 0.0


def semantic_search_with_headers(
    query: str,
    chunks_with_headers: List[ChunkWithHeader],
    get_embedding_fn: Callable,
    k: int = 5,
    header_weight: float = 0.5
) -> List[ChunkWithHeader]:
    """
    Searches for the most relevant chunks based on both header and text similarity.

    Args:
        query (str): User query.
        chunks_with_headers (List[ChunkWithHeader]): List of chunks with headers and embeddings.
        get_embedding_fn: Function to generate embeddings for the query.
        k (int): Number of top results (default: 5).
        header_weight (float): Weight for header similarity (0.0-1.0). 
                              Text weight is (1.0 - header_weight) (default: 0.5).

    Returns:
        List[ChunkWithHeader]: Top-k most relevant chunks with headers.
    """
    # Get query embedding
    try:
        query_embedding_response = get_embedding_fn(query.strip())
        if hasattr(query_embedding_response, 'embedding'):
            query_embedding = np.array(query_embedding_response.embedding)
        elif isinstance(query_embedding_response, list):
            query_embedding = np.array(query_embedding_response)
        else:
            query_embedding = np.array(query_embedding_response)
    except Exception as e:
        print(f"  [WARNING] Failed to embed query: {e}")
        return []

    similarities = []

    # Compute similarity scores
    for chunk in chunks_with_headers:
        # Skip chunks without embeddings
        if chunk.text_embedding is None or chunk.header_embedding is None:
            similarities.append((chunk, 0.0))
            continue
        
        try:
            # Compute similarity with text embedding
            sim_text = cosine_similarity(query_embedding, chunk.text_embedding)
            # Compute similarity with header embedding
            sim_header = cosine_similarity(query_embedding, chunk.header_embedding)
            
            # Weighted average: header_weight * header + (1 - header_weight) * text
            text_weight = 1.0 - header_weight
            avg_similarity = (header_weight * sim_header) + (text_weight * sim_text)
            
            similarities.append((chunk, avg_similarity))
        except Exception as e:
            print(f"  [WARNING] Failed to compute similarity: {e}")
            similarities.append((chunk, 0.0))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top-k results
    return [chunk for chunk, _ in similarities[:k]]
