"""
Semantic Chunking - Workflow Part

Splits text into semantic chunks based on sentence embeddings and similarity scores.
Unlike sliding window chunking, semantic chunking respects semantic boundaries.

Methods:
- percentile: Finds the Xth percentile of all similarity differences
- standard_deviation: Splits where similarity drops more than X standard deviations below mean
- interquartile: Uses IQR (Q3 - Q1) to determine split points
"""

import numpy as np
from typing import List, Callable, Optional
from workflow_parts.embedding import create_embeddings as create_embedding_vectors


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def split_sentences(text: str, delimiter: str = ". ") -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Text to split
        delimiter: Sentence delimiter (default: ". ")
        
    Returns:
        List of sentences
    """
    sentences = text.split(delimiter)
    # Rejoin delimiter except for the last sentence
    return [s + delimiter if i < len(sentences) - 1 else s for i, s in enumerate(sentences)]


def compute_sentence_similarities(sentences: List[str], get_embedding_fn: Callable) -> List[float]:
    """
    Compute cosine similarity between consecutive sentences.
    
    Args:
        sentences: List of sentences
        get_embedding_fn: Function to generate embeddings for a sentence
        
    Returns:
        List of similarity scores between consecutive sentences
    """
    if len(sentences) < 2:
        return []
    
    # Get embeddings for all sentences
    embeddings = []
    for sentence in sentences:
        try:
            embedding_response = get_embedding_fn(sentence.strip())
            # Extract the embedding vector - might be EmbeddingItem or list
            if hasattr(embedding_response, 'embedding'):
                embedding = np.array(embedding_response.embedding)
            elif isinstance(embedding_response, list):
                embedding = np.array(embedding_response)
            else:
                embedding = np.array(embedding_response)
            embeddings.append(embedding)
        except Exception as e:
            print(f"  [WARNING] Failed to embed sentence: {e}")
            # Use zero vector as fallback
            embeddings.append(np.zeros(1536))  # Default OpenAI embedding size
    
    # Compute similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
    
    return similarities


def compute_breakpoints(
    similarities: List[float],
    method: str = "percentile",
    threshold: float = 90
) -> List[int]:
    """
    Compute chunk breakpoints based on similarity drops.
    
    Args:
        similarities: Similarity scores between consecutive sentences
        method: 'percentile', 'standard_deviation', or 'interquartile'
        threshold: Threshold value (percentile % for percentile, std devs for std_dev)
        
    Returns:
        List of sentence indices where chunks should break
    """
    if not similarities:
        return []
    
    similarities_arr = np.array(similarities)
    
    if method == "percentile":
        # Find the Xth percentile of similarity scores
        threshold_value = np.percentile(similarities_arr, threshold)
    elif method == "standard_deviation":
        # Find points where similarity drops more than X std devs below mean
        mean = np.mean(similarities_arr)
        std_dev = np.std(similarities_arr)
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # Use IQR to find outlier points
        q1, q3 = np.percentile(similarities_arr, [25, 75])
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        raise ValueError(f"Invalid method '{method}'. Choose: 'percentile', 'standard_deviation', 'interquartile'")
    
    # Return indices where similarity drops below threshold
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]


def split_into_semantic_chunks(
    sentences: List[str],
    breakpoints: List[int]
) -> List[str]:
    """
    Split sentences into chunks at breakpoint indices.
    
    Args:
        sentences: List of sentences
        breakpoints: Indices where chunks should split
        
    Returns:
        List of text chunks
    """
    if not breakpoints:
        # No breakpoints - return original text
        return ["".join(sentences)]
    
    chunks = []
    start = 0
    
    for bp in sorted(breakpoints):
        # Create chunk from start to breakpoint
        chunk = "".join(sentences[start:bp + 1])
        if chunk.strip():
            chunks.append(chunk)
        start = bp + 1
    
    # Add remaining sentences as final chunk
    if start < len(sentences):
        chunk = "".join(sentences[start:])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks if chunks else ["".join(sentences)]


def chunk_text_semantic(
    text: str,
    method: str = "percentile",
    threshold: float = 90,
    get_embedding_fn: Optional[Callable] = None,
    **kwargs
) -> List[str]:
    """
    Split text into semantic chunks based on sentence embeddings.
    
    This method:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Calculates similarity between consecutive sentences
    4. Finds semantic breakpoints based on similarity drops
    5. Creates chunks respecting semantic boundaries
    
    Args:
        text: Text to chunk
        method: Breakpoint method ('percentile', 'standard_deviation', 'interquartile')
        threshold: Threshold value for the method
        get_embedding_fn: Function to create embeddings (uses default if not provided)
        **kwargs: Additional arguments (ignored, for compatibility with other chunkers)
        
    Returns:
        List of semantic chunks
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = text.split(". ")
    if not sentences:
        return [text]
    
    # If only one sentence, return as-is
    if len(sentences) < 2:
        return sentences
    
    # Use provided embedding function or create one
    if get_embedding_fn is None:
        # Use the embedding function from workflow_parts.embedding
        from workflow_parts.embedding import create_embeddings as embed_fn
        
        def get_embedding_fn(sentence):
            """Wrapper to get embedding for a single sentence"""
            embeddings = embed_fn([sentence])
            if embeddings:
                return embeddings[0]
            return np.zeros(1536)
    
    # Compute similarities between consecutive sentences
    print(f"  [Semantic] Computing embeddings for {len(sentences)} sentences...")
    similarities = compute_sentence_similarities(sentences, get_embedding_fn)
    
    if not similarities:
        return sentences
    
    # Find breakpoints
    print(f"  [Semantic] Finding breakpoints (method={method}, threshold={threshold})...")
    breakpoints = compute_breakpoints(similarities, method=method, threshold=threshold)
    print(f"  [Semantic] Found {len(breakpoints)} breakpoints")
    
    # Split into chunks
    chunks = split_into_semantic_chunks(sentences, breakpoints)
    
    return chunks
