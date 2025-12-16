"""
Reranker Module

Implements LLM-based reranking of search results.
After semantic search retrieves initial candidates, the reranker uses an LLM
to score and reorder results based on relevance to the query.
"""

import re
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass


@dataclass
class RankedResult:
    """Represents a ranked search result."""
    text: str
    similarity_score: float
    relevance_score: Optional[float] = None


def rerank_with_llm(
    query: str,
    chunks: List[str],
    get_generation_fn: Callable,
    k: int = 3
) -> List[str]:
    """
    Reranks chunks using LLM relevance scoring.
    
    The LLM evaluates each chunk on how well it answers the query,
    providing a more nuanced ranking than pure semantic similarity.

    Args:
        query (str): User query.
        chunks (List[str]): Initial retrieved chunks to rerank.
        get_generation_fn: Function that calls LLM for scoring.
        k (int): Number of results to return (default: 3).

    Returns:
        List[str]: Reranked chunk texts (top-k).
        
    Example:
        >>> chunks = ["text 1", "text 2", "text 3"]
        >>> ranked = rerank_with_llm("query", chunks, llm_fn, k=2)
        >>> # Returns 2 best chunks ranked by relevance
    """
    if not chunks:
        return []
    
    print(f"  Reranking {len(chunks)} chunks...")
    
    scored_results = []
    
    system_prompt = """You are an expert at evaluating document relevance for queries.
Rate documents on a scale 0-10 based on how well they answer the given query.

Guidelines:
- 0-2: Completely irrelevant
- 3-5: Somewhat relevant, limited information
- 6-8: Relevant, partially answers query
- 9-10: Highly relevant, directly answers query

RESPOND WITH ONLY A SINGLE INTEGER (0-10). NO OTHER TEXT."""
    
    for i, chunk in enumerate(chunks):
        try:
            user_prompt = f"""Query: {query}

Document:
{chunk[:1000]}  

Relevance score (0-10):"""
            
            response = get_generation_fn(system_prompt, user_prompt)
            score_text = response.get("content", "").strip()
            
            # Extract score
            score_match = re.search(r'\b(10|[0-9])\b', score_text)
            if score_match:
                score = float(score_match.group(1))
            else:
                print(f"    [WARNING] Could not extract score from: {score_text}")
                score = 0.0
            
            scored_results.append((chunk, score))
            
        except Exception as e:
            print(f"    [WARNING] Reranking failed for chunk {i}: {e}")
            scored_results.append((chunk, 0.0))
    
    # Sort by relevance score (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k texts
    return [text for text, _ in scored_results[:k]]


def rerank_by_keyword_matching(
    query: str,
    chunks: List[str],
    k: int = 5
) -> List[str]:
    """
    Simple keyword-based reranking (fast, no LLM calls).
    
    Scores chunks based on how many query keywords they contain.
    Useful as a lightweight alternative to LLM-based reranking.

    Args:
        query (str): User query.
        chunks (List[str]): Chunks to rerank.
        k (int): Number of results to return (default: 5).

    Returns:
        List[str]: Reranked chunks by keyword coverage.
    """
    keywords = set(query.lower().split())
    keywords = {kw for kw in keywords if len(kw) > 3}  # Filter short words
    
    scored_results = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Count keyword matches
        match_count = sum(1 for kw in keywords if kw in chunk_lower)
        # Calculate coverage ratio
        coverage = match_count / len(keywords) if keywords else 0
        scored_results.append((chunk, coverage))
    
    # Sort by coverage
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    return [text for text, _ in scored_results[:k]]
