"""
Feedback Loop RAG Module

Implements a feedback-based retrieval system that:
1. Tracks relevance feedback on chunks
2. Adjusts retrieval based on feedback history
3. Adapts search parameters based on past positive feedback
"""

import json
from typing import List, Dict, Any, Callable
import numpy as np


class FeedbackStore:
    """Stores and manages feedback data for RAG improvement."""
    
    def __init__(self):
        """Initialize feedback storage."""
        self.feedback_history = []
        self.chunk_relevance_scores = {}  # chunk_text -> relevance score
        self.chunk_feedback_count = {}     # chunk_text -> feedback count
    
    def add_feedback(self, query: str, response: str, chunks: List[str], 
                     relevance_score: float, quality_score: float, 
                     comments: str = "") -> Dict[str, Any]:
        """
        Record feedback for a query result.
        
        Args:
            query: User query
            response: Generated response
            chunks: Retrieved chunks that were used
            relevance_score: Relevance rating (0-1)
            quality_score: Response quality rating (0-1)
            comments: Optional feedback comments
            
        Returns:
            Feedback dictionary with metadata
        """
        feedback_entry = {
            "query": query,
            "response": response,
            "chunks": chunks,
            "relevance_score": float(relevance_score),
            "quality_score": float(quality_score),
            "comments": comments,
            "timestamp": None
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Update chunk-level relevance scores
        if relevance_score > 0.5:  # Consider as positive feedback
            for chunk in chunks:
                chunk_key = chunk[:100]  # Use first 100 chars as key
                if chunk_key not in self.chunk_relevance_scores:
                    self.chunk_relevance_scores[chunk_key] = []
                    self.chunk_feedback_count[chunk_key] = 0
                
                self.chunk_relevance_scores[chunk_key].append(relevance_score)
                self.chunk_feedback_count[chunk_key] += 1
        
        return feedback_entry
    
    def get_average_relevance(self, chunk_text: str) -> float:
        """
        Get average relevance score for a chunk.
        
        Args:
            chunk_text: Text of the chunk
            
        Returns:
            Average relevance score (0-1)
        """
        chunk_key = chunk_text[:100]
        if chunk_key not in self.chunk_relevance_scores:
            return 1.0  # Default to 1.0 if no feedback
        
        scores = self.chunk_relevance_scores[chunk_key]
        return sum(scores) / len(scores) if scores else 1.0
    
    def get_feedback_count(self, chunk_text: str) -> int:
        """Get number of feedback instances for a chunk."""
        chunk_key = chunk_text[:100]
        return self.chunk_feedback_count.get(chunk_key, 0)


def assess_feedback_relevance(query: str, doc_text: str, feedback: Dict[str, Any],
                             generation_fn: Callable) -> bool:
    """
    Use LLM to assess if past feedback is relevant to current query/document.
    
    Args:
        query: Current user query
        doc_text: Document text being evaluated
        feedback: Previous feedback entry
        generation_fn: Function to call LLM
        
    Returns:
        True if feedback is relevant, False otherwise
    """
    system_prompt = """You are an AI system that determines if past feedback is relevant to a current query and document.
    Answer with ONLY 'yes' or 'no'. Your job is strictly to determine relevance, not to provide explanations."""
    
    doc_preview = doc_text[:500] + ("... [truncated]" if len(doc_text) > 500 else "")
    response_preview = feedback.get("response", "")[:500]
    if len(feedback.get("response", "")) > 500:
        response_preview += "... [truncated]"
    
    user_prompt = f"""
    Current query: {query}
    Past query that received feedback: {feedback['query']}
    Document content: {doc_preview}
    Past response that received feedback: {response_preview}

    Is this past feedback relevant to the current query and document? (yes/no)
    """
    
    response = generation_fn(system_prompt, user_prompt)
    answer = response.get("content", "").lower()
    return 'yes' in answer


def adjust_chunk_score_with_feedback(chunk: str, base_score: float, 
                                     feedback_store: FeedbackStore) -> float:
    """
    Adjust a chunk's score based on feedback history.
    
    Args:
        chunk: Text chunk
        base_score: Original similarity score
        feedback_store: FeedbackStore instance
        
    Returns:
        Adjusted relevance score
    """
    avg_relevance = feedback_store.get_average_relevance(chunk)
    feedback_count = feedback_store.get_feedback_count(chunk)
    
    # Weight feedback based on count (more feedback = higher confidence)
    confidence_weight = min(feedback_count / 5.0, 1.0)  # Max out at 5 feedbacks
    
    # Blend base score with feedback-based score
    adjusted_score = (base_score * (1 - confidence_weight) + 
                     avg_relevance * confidence_weight)
    
    return adjusted_score


def retrieve_with_feedback(query: str, chunks: List[str], embeddings: List[Any],
                          feedback_store: FeedbackStore, k: int = None) -> List[str]:
    """
    Retrieve chunks with feedback-based adjustment.
    
    Core retrieval function that incorporates feedback scores.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects with .embedding attribute
        feedback_store: FeedbackStore with feedback history
        k: Number of results to return
        
    Returns:
        Top-k chunks adjusted by feedback scores
    """
    if k is None:
        k = 5
    
    if not chunks or not embeddings:
        return []
    
    # Compute query embedding
    from workflow_parts.embedding import get_embedding_fn
    query_emb_obj = get_embedding_fn()(query.strip())
    query_vec = np.array(query_emb_obj.embedding)
    
    # Calculate similarity scores
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_vec = np.array(embeddings[i].embedding)
        norm_query = np.linalg.norm(query_vec)
        norm_chunk = np.linalg.norm(chunk_vec)
        
        if norm_query > 0 and norm_chunk > 0:
            base_sim = np.dot(query_vec, chunk_vec) / (norm_query * norm_chunk)
        else:
            base_sim = 0.0
        
        # Adjust with feedback
        adjusted_score = adjust_chunk_score_with_feedback(chunk, base_sim, feedback_store)
        scores.append((i, adjusted_score))
    
    # Sort by adjusted score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k chunks
    result_indices = [idx for idx, _ in scores[:k]]
    return [chunks[idx] for idx in result_indices]


def format_feedback_request(query: str, response: str) -> Dict[str, Any]:
    """
    Format a feedback request for user collection.
    
    Args:
        query: Original query
        response: Generated response
        
    Returns:
        Formatted feedback request
    """
    return {
        "query": query,
        "response": response,
        "request": {
            "relevance": "How relevant is the response to your query? (0-1)",
            "quality": "How good is the overall quality? (0-1)",
            "comments": "Any additional comments? (optional)"
        }
    }


def simulate_feedback(query: str, response: str, 
                     is_good_response: bool = True) -> Dict[str, Any]:
    """
    Simulate user feedback for testing.
    
    Args:
        query: Query string
        response: Response string
        is_good_response: Whether to simulate positive feedback
        
    Returns:
        Simulated feedback data
    """
    if is_good_response:
        relevance = 0.8 + np.random.uniform(0, 0.2)
        quality = 0.8 + np.random.uniform(0, 0.2)
        comments = "Good response, informative and relevant."
    else:
        relevance = 0.2 + np.random.uniform(0, 0.3)
        quality = 0.2 + np.random.uniform(0, 0.3)
        comments = "Response didn't address the query properly."
    
    return {
        "query": query,
        "response": response,
        "relevance_score": relevance,
        "quality_score": quality,
        "comments": comments
    }
