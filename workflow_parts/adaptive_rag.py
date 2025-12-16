"""
Adaptive RAG Module

Implements adaptive retrieval that:
1. Classifies queries by type
2. Adjusts retrieval strategy based on query type
3. Uses different k values and search strategies for different query types
"""

from typing import List, Dict, Any, Callable
import numpy as np


def classify_query(query: str, generation_fn: Callable) -> str:
    """
    Classify a query into one of four categories.
    
    Args:
        query: User query
        generation_fn: Function to call LLM for classification
        
    Returns:
        Query category: 'Factual', 'Analytical', 'Opinion', or 'Contextual'
    """
    system_prompt = """You are an expert at classifying questions. 
    Classify the given query into exactly one of these categories:
    - Factual: Queries seeking specific, verifiable information.
    - Analytical: Queries requiring comprehensive analysis or explanation.
    - Opinion: Queries about subjective matters or seeking diverse viewpoints.
    - Contextual: Queries that depend on user-specific context.

    Return ONLY the category name, without any explanation or additional text."""
    
    user_prompt = f"Classify this query: {query}"
    
    response = generation_fn(system_prompt, user_prompt)
    category = response.get("content", "").strip()
    
    # Validate and normalize category
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    for valid in valid_categories:
        if valid.lower() in category.lower():
            return valid
    
    return "Factual"  # Default


def get_adaptive_parameters(query_category: str) -> Dict[str, Any]:
    """
    Get retrieval parameters based on query category.
    
    Args:
        query_category: Query classification
        
    Returns:
        Dictionary with adaptive parameters
    """
    parameters = {
        "Factual": {
            "k": 3,
            "strategy": "direct",
            "temperature": 0.0,
            "description": "Direct similarity search for specific facts"
        },
        "Analytical": {
            "k": 5,
            "strategy": "multi_perspective",
            "temperature": 0.3,
            "description": "Multi-perspective search for comprehensive coverage"
        },
        "Opinion": {
            "k": 4,
            "strategy": "diverse",
            "temperature": 0.5,
            "description": "Diverse retrieval for multiple viewpoints"
        },
        "Contextual": {
            "k": 6,
            "strategy": "contextual",
            "temperature": 0.3,
            "description": "Extended context for user-dependent queries"
        }
    }
    
    return parameters.get(query_category, parameters["Factual"])


def factual_retrieval_strategy(query: str, chunks: List[str], embeddings: List[Any],
                              generation_fn: Callable, k: int = 3) -> List[str]:
    """
    Direct similarity search for factual queries.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function
        k: Number of results
        
    Returns:
        Top-k chunks
    """
    if not chunks or not embeddings:
        return []
    
    from workflow_parts.embedding import get_embedding_fn
    query_emb_obj = get_embedding_fn()(query.strip())
    query_vec = np.array(query_emb_obj.embedding)
    
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_vec = np.array(embeddings[i].embedding)
        norm_q = np.linalg.norm(query_vec)
        norm_c = np.linalg.norm(chunk_vec)
        
        if norm_q > 0 and norm_c > 0:
            sim = np.dot(query_vec, chunk_vec) / (norm_q * norm_c)
        else:
            sim = 0.0
        scores.append((i, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    result_indices = [idx for idx, _ in scores[:k]]
    return [chunks[idx] for idx in result_indices]


def analytical_retrieval_strategy(query: str, chunks: List[str], embeddings: List[Any],
                                  generation_fn: Callable, k: int = 5) -> List[str]:
    """
    Multi-perspective search for analytical queries.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function
        k: Number of results
        
    Returns:
        Top-k chunks with diverse perspectives
    """
    if not chunks or not embeddings:
        return []
    
    # Generate sub-questions to explore different aspects
    system_prompt = """You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve 
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line."""
    
    user_prompt = f"Generate sub-questions for this analytical query: {query}"
    
    response = generation_fn(system_prompt, user_prompt)
    sub_queries_text = response.get("content", "")
    sub_queries = [q.strip() for q in sub_queries_text.split('\n') if q.strip()][:3]
    
    # Retrieve for each sub-question
    from workflow_parts.embedding import get_embedding_fn
    all_results = []
    seen_texts = set()
    
    for sub_query in sub_queries:
        sub_emb_obj = get_embedding_fn()(sub_query.strip())
        sub_vec = np.array(sub_emb_obj.embedding)
        
        scores = []
        for i, chunk in enumerate(chunks):
            chunk_vec = np.array(embeddings[i].embedding)
            norm_s = np.linalg.norm(sub_vec)
            norm_c = np.linalg.norm(chunk_vec)
            
            if norm_s > 0 and norm_c > 0:
                sim = np.dot(sub_vec, chunk_vec) / (norm_s * norm_c)
            else:
                sim = 0.0
            scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 2 for this sub-query
        for idx, _ in scores[:2]:
            if chunks[idx] not in seen_texts:
                all_results.append(chunks[idx])
                seen_texts.add(chunks[idx])
    
    # Fill remaining with main query results
    if len(all_results) < k:
        main_emb_obj = get_embedding_fn()(query.strip())
        main_vec = np.array(main_emb_obj.embedding)
        
        scores = []
        for i, chunk in enumerate(chunks):
            chunk_vec = np.array(embeddings[i].embedding)
            norm_m = np.linalg.norm(main_vec)
            norm_c = np.linalg.norm(chunk_vec)
            
            if norm_m > 0 and norm_c > 0:
                sim = np.dot(main_vec, chunk_vec) / (norm_m * norm_c)
            else:
                sim = 0.0
            scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, _ in scores:
            if chunks[idx] not in seen_texts and len(all_results) < k:
                all_results.append(chunks[idx])
                seen_texts.add(chunks[idx])
    
    return all_results[:k]


def opinion_retrieval_strategy(query: str, chunks: List[str], embeddings: List[Any],
                               generation_fn: Callable, k: int = 4) -> List[str]:
    """
    Diverse retrieval for opinion queries.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function
        k: Number of results
        
    Returns:
        Top-k chunks with diverse perspectives
    """
    if not chunks or not embeddings:
        return []
    
    # Identify different perspectives
    system_prompt = """You are an expert at identifying different perspectives on a topic.
    For the given query about opinions or viewpoints, identify different perspectives 
    that people might have on this topic.

    Return a list of exactly 3 different viewpoint angles, one per line."""
    
    user_prompt = f"Identify different perspectives on: {query}"
    
    response = generation_fn(system_prompt, user_prompt)
    viewpoints_text = response.get("content", "")
    viewpoints = [v.strip() for v in viewpoints_text.split('\n') if v.strip()][:3]
    
    # Retrieve for each viewpoint
    from workflow_parts.embedding import get_embedding_fn
    results_by_viewpoint = {}
    
    for viewpoint in viewpoints:
        combined_query = f"{query} {viewpoint}"
        emb_obj = get_embedding_fn()(combined_query.strip())
        vec = np.array(emb_obj.embedding)
        
        scores = []
        for i, chunk in enumerate(chunks):
            chunk_vec = np.array(embeddings[i].embedding)
            norm_q = np.linalg.norm(vec)
            norm_c = np.linalg.norm(chunk_vec)
            
            if norm_q > 0 and norm_c > 0:
                sim = np.dot(vec, chunk_vec) / (norm_q * norm_c)
            else:
                sim = 0.0
            scores.append((i, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        results_by_viewpoint[viewpoint] = [chunks[idx] for idx, _ in scores[:2]]
    
    # Select one from each viewpoint if possible
    selected = []
    for viewpoint in viewpoints:
        if viewpoint in results_by_viewpoint and results_by_viewpoint[viewpoint]:
            selected.append(results_by_viewpoint[viewpoint][0])
    
    # Fill remaining slots with highest-scoring results
    if len(selected) < k:
        all_chunks_scores = []
        for viewpoint in viewpoints:
            for chunk in results_by_viewpoint.get(viewpoint, []):
                if chunk not in selected:
                    all_chunks_scores.append(chunk)
        
        for chunk in all_chunks_scores[:k - len(selected)]:
            if chunk not in selected:
                selected.append(chunk)
    
    return selected[:k]


def contextual_retrieval_strategy(query: str, chunks: List[str], embeddings: List[Any],
                                  generation_fn: Callable, k: int = 6) -> List[str]:
    """
    Extended context retrieval for context-dependent queries.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function
        k: Number of results (higher for contextual)
        
    Returns:
        Top-k chunks with extended context
    """
    if not chunks or not embeddings:
        return []
    
    from workflow_parts.embedding import get_embedding_fn
    query_emb_obj = get_embedding_fn()(query.strip())
    query_vec = np.array(query_emb_obj.embedding)
    
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_vec = np.array(embeddings[i].embedding)
        norm_q = np.linalg.norm(query_vec)
        norm_c = np.linalg.norm(chunk_vec)
        
        if norm_q > 0 and norm_c > 0:
            sim = np.dot(query_vec, chunk_vec) / (norm_q * norm_c)
        else:
            sim = 0.0
        scores.append((i, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    result_indices = [idx for idx, _ in scores[:k]]
    return [chunks[idx] for idx in result_indices]


def retrieve_adaptive(query: str, chunks: List[str], embeddings: List[Any],
                     generation_fn: Callable, k: int = None) -> List[str]:
    """
    Retrieve chunks using adaptive strategy based on query classification.
    
    Core retrieval function that determines strategy based on query type.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function
        k: Number of results (overrides adaptive k if provided)
        
    Returns:
        Top-k chunks selected by adaptive strategy
    """
    if not chunks or not embeddings:
        return []
    
    # Classify the query
    query_category = classify_query(query, generation_fn)
    params = get_adaptive_parameters(query_category)
    
    # Use provided k or adaptive k
    k_to_use = k if k is not None else params["k"]
    
    print(f"  [Adaptive] Query type: {query_category}")
    print(f"  [Adaptive] Strategy: {params['strategy']}, k={k_to_use}")
    
    # Apply strategy
    if params["strategy"] == "direct":
        return factual_retrieval_strategy(query, chunks, embeddings, generation_fn, k_to_use)
    elif params["strategy"] == "multi_perspective":
        return analytical_retrieval_strategy(query, chunks, embeddings, generation_fn, k_to_use)
    elif params["strategy"] == "diverse":
        return opinion_retrieval_strategy(query, chunks, embeddings, generation_fn, k_to_use)
    elif params["strategy"] == "contextual":
        return contextual_retrieval_strategy(query, chunks, embeddings, generation_fn, k_to_use)
    else:
        return factual_retrieval_strategy(query, chunks, embeddings, generation_fn, k_to_use)
