"""
Chunking - Workflow Part

Different text chunking strategies for RAG workflows.
"""

from typing import List


def chunk_text_sliding_window(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks using sliding window approach.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks
