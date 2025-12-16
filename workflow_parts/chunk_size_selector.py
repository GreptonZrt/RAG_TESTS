"""
Chunk Size Selector Module

Provides multiple chunking strategies with different sizes.
Allows dynamic selection of chunk size parameters for RAG pipelines.
"""

from typing import List, Dict, Callable, Tuple


def chunk_text_variable_size(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Chunks text with variable size parameter.
    
    This is a flexible chunking function that allows adjusting chunk size
    without hardcoding it into the pipeline.

    Args:
        text (str): The text to chunk.
        chunk_size (int): Size of each chunk in characters (default: 1000).
        overlap (int): Overlapping characters between chunks (default: 200).

    Returns:
        List[str]: List of text chunks.
        
    Example:
        >>> chunks = chunk_text_variable_size("Long text...", chunk_size=512, overlap=50)
        >>> len(chunks)  # Will have more chunks with smaller size
    """
    chunks = []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    step = chunk_size - overlap
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def get_chunk_size_stats(
    text: str,
    chunk_sizes: List[int],
    overlap_ratio: float = 0.2
) -> Dict[int, Dict]:
    """
    Calculate statistics for different chunk sizes.
    
    Useful for understanding the impact of chunk size on retrieval.

    Args:
        text (str): The text to analyze.
        chunk_sizes (List[int]): List of chunk sizes to evaluate.
        overlap_ratio (float): Overlap as a ratio of chunk size (default: 0.2 = 20%).

    Returns:
        Dict: Statistics for each chunk size including count, avg size, etc.
        
    Example:
        >>> stats = get_chunk_size_stats("text", [256, 512, 1024])
        >>> print(stats[512]["count"])  # Number of chunks for size 512
    """
    stats = {}
    
    for size in chunk_sizes:
        overlap = int(size * overlap_ratio)
        chunks = chunk_text_variable_size(text, chunk_size=size, overlap=overlap)
        
        stats[size] = {
            "count": len(chunks),
            "chunk_size": size,
            "overlap": overlap,
            "avg_chunk_length": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            "total_text_length": len(text),
            "coverage_ratio": (len(chunks) * size) / len(text) if len(text) > 0 else 0
        }
    
    return stats


def create_variable_chunker(chunk_size: int = 1000, overlap: int = 200) -> Callable:
    """
    Creates a chunker function with fixed parameters.
    
    Useful for creating retriever-compatible chunker functions
    that can be passed to orchestration functions.

    Args:
        chunk_size (int): Size of each chunk (default: 1000).
        overlap (int): Overlap between chunks (default: 200).

    Returns:
        Callable: A chunker function that takes text and returns chunks.
        
    Example:
        >>> chunker = create_variable_chunker(512, 100)
        >>> chunks = chunker("Long text...")
    """
    def chunker(text: str, **kwargs) -> List[str]:
        # Allow override via kwargs
        size = kwargs.get("chunk_size", chunk_size)
        ovlp = kwargs.get("overlap", overlap)
        return chunk_text_variable_size(text, chunk_size=size, overlap=ovlp)
    
    return chunker
