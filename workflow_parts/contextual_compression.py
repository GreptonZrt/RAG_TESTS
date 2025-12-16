"""
Contextual Compression Module

Implements LLM-based compression of retrieved chunks.
Reduces context length by filtering out irrelevant parts while preserving
information necessary to answer the query.
"""

from typing import List, Callable, Tuple, Optional


def compress_chunk(
    chunk: str,
    query: str,
    get_generation_fn: Callable,
    compression_type: str = "selective"
) -> Tuple[str, float]:
    """
    Compresses a chunk by keeping only query-relevant parts.

    Args:
        chunk (str): Text chunk to compress.
        query (str): User query.
        get_generation_fn: Function that calls LLM for compression.
        compression_type (str): "selective" (keep relevant sentences),
                                "summary" (brief summary), or
                                "extraction" (exact quotes)
                                (default: "selective").

    Returns:
        Tuple[str, float]: Compressed chunk and compression ratio (0-100).
        
    Example:
        >>> chunk = "Long text about AI and cooking..."
        >>> compressed, ratio = compress_chunk(chunk, "What is AI?", llm_fn)
        >>> # Returns compressed text with ~60% compression ratio
    """
    
    # Define system prompts
    system_prompts = {
        "selective": """You are an expert at filtering information.
Extract ONLY sentences/paragraphs directly relevant to the query.
- Include only query-relevant text
- Preserve exact wording
- Maintain original order
- NO paraphrasing or commentary""",
        
        "summary": """You are an expert at summarization.
Create a concise summary focusing ONLY on query-relevant information.
- Be brief but comprehensive
- Focus exclusively on query-relevant content
- Omit irrelevant details
- Neutral, factual tone""",
        
        "extraction": """You are an expert at information extraction.
Extract ONLY exact sentences containing query-relevant information.
- Use direct quotes only
- Preserve original wording
- Include only directly relevant sentences
- Separate by newlines
- NO commentary"""
    }
    
    system_prompt = system_prompts.get(compression_type, system_prompts["selective"])
    
    user_prompt = f"""Query: {query}

Document Chunk:
{chunk}

Extract only the content relevant to answering this query."""
    
    try:
        response = get_generation_fn(system_prompt, user_prompt)
        compressed_chunk = response.get("content", "").strip()
        
        # Calculate compression ratio
        original_length = len(chunk)
        compressed_length = len(compressed_chunk)
        
        if original_length > 0:
            compression_ratio = ((original_length - compressed_length) / original_length) * 100
        else:
            compression_ratio = 0.0
        
        return compressed_chunk, compression_ratio
        
    except Exception as e:
        print(f"  [WARNING] Compression failed: {e}")
        return chunk, 0.0  # Return original on error


def compress_chunks_batch(
    chunks: List[str],
    query: str,
    get_generation_fn: Callable,
    compression_type: str = "selective"
) -> Tuple[List[str], List[float]]:
    """
    Compresses multiple chunks efficiently.

    Args:
        chunks (List[str]): List of chunks to compress.
        query (str): User query.
        get_generation_fn: Function that calls LLM.
        compression_type (str): Compression type (default: "selective").

    Returns:
        Tuple[List[str], List[float]]: Compressed chunks and ratios.
    """
    compressed_chunks = []
    ratios = []
    
    for i, chunk in enumerate(chunks):
        compressed, ratio = compress_chunk(
            chunk, query, get_generation_fn, compression_type
        )
        compressed_chunks.append(compressed)
        ratios.append(ratio)
    
    return compressed_chunks, ratios
