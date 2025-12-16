"""
Self-RAG Module

Implements self-reflective RAG that:
1. Determines if retrieval is needed
2. Assesses whether generation is supported by context
3. Iterates if needed for better coverage
"""

from typing import List, Dict, Any, Callable, Tuple
import numpy as np


def determine_if_retrieval_needed(query: str, generation_fn: Callable) -> bool:
    """
    Determines if retrieval is necessary for the query.
    
    Args:
        query: User query
        generation_fn: LLM generation function
        
    Returns:
        True if retrieval is needed, False otherwise
    """
    system_prompt = """You are an AI assistant that determines if retrieval is necessary to answer a query.
    For factual questions, specific information requests, or questions about events, people, or concepts, answer "Yes".
    For opinions, hypothetical scenarios, or simple queries with common knowledge, answer "No".
    Answer with ONLY "Yes" or "No"."""
    
    user_prompt = f"Query: {query}\n\nIs retrieval necessary to answer this query accurately?"
    
    response = generation_fn(system_prompt, user_prompt)
    answer = response.get("content", "").strip().lower()
    
    return "yes" in answer


def assess_relevance(query: str, chunk: str, generation_fn: Callable) -> str:
    """
    Assess if a retrieved chunk is relevant to the query.
    
    Args:
        query: User query
        chunk: Retrieved chunk
        generation_fn: LLM generation function
        
    Returns:
        'relevant', 'partially relevant', or 'irrelevant'
    """
    system_prompt = """You are an AI assistant that assesses if a chunk is relevant to a query.
    Return ONLY one of these three options:
    - relevant: The chunk directly addresses the query
    - partially relevant: The chunk contains some useful information but not fully addressing the query
    - irrelevant: The chunk does not address the query"""
    
    # Truncate chunk if too long
    chunk_preview = chunk[:500] + ("..." if len(chunk) > 500 else "")
    
    user_prompt = f"""Query: {query}

Chunk: {chunk_preview}

Is this chunk relevant to the query?"""
    
    response = generation_fn(system_prompt, user_prompt)
    answer = response.get("content", "").lower().strip()
    
    if "relevant" in answer and "irrelevant" not in answer:
        return "relevant"
    elif "partially" in answer:
        return "partially relevant"
    else:
        return "irrelevant"


def assess_support(response: str, context: str, generation_fn: Callable) -> str:
    """
    Assess how well a response is supported by the context.
    
    Args:
        response: Generated response
        context: Context text
        generation_fn: LLM generation function
        
    Returns:
        'fully supported', 'partially supported', or 'no support'
    """
    system_prompt = """You are an AI assistant that determines if a response is supported by the given context.
    Evaluate if the facts, claims, and information in the response are backed by the context.
    Answer with ONLY one of these three options:
    - fully supported: All information in the response is directly supported by the context.
    - partially supported: Some information in the response is supported by the context, but some is not.
    - no support: The response contains significant information not found in or contradicting the context.
    """
    
    # Truncate context if too long
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [truncated]"
    
    user_prompt = f"""Context:
    {context}

    Response:
    {response}

    How well is this response supported by the context?"""
    
    response_obj = generation_fn(system_prompt, user_prompt)
    answer = response_obj.get("content", "").lower().strip()
    
    if "fully" in answer:
        return "fully supported"
    elif "partially" in answer:
        return "partially supported"
    else:
        return "no support"


def rate_utility(query: str, response: str, generation_fn: Callable) -> int:
    """
    Rate the utility of a response for the query.
    
    Args:
        query: User query
        response: Generated response
        generation_fn: LLM generation function
        
    Returns:
        Utility rating from 1 to 5
    """
    system_prompt = """You are an AI assistant that rates the utility of a response to a query.
    Consider how well the response answers the query, its completeness, correctness, and helpfulness.
    Rate the utility on a scale from 1 to 5, where:
    - 1: Not useful at all
    - 2: Slightly useful
    - 3: Moderately useful
    - 4: Very useful
    - 5: Exceptionally useful
    Answer with ONLY a single number from 1 to 5."""
    
    user_prompt = f"""Query: {query}
    Response:
    {response}

    Rate the utility of this response on a scale from 1 to 5:"""
    
    response_obj = generation_fn(system_prompt, user_prompt)
    rating_text = response_obj.get("content", "").strip()
    
    # Extract number from response
    import re
    match = re.search(r'[1-5]', rating_text)
    if match:
        return int(match.group())
    
    return 3  # Default


def filter_relevant_chunks(query: str, chunks: List[str], 
                          generation_fn: Callable) -> List[str]:
    """
    Filter chunks to keep only relevant ones.
    
    Args:
        query: User query
        chunks: List of chunks
        generation_fn: LLM generation function
        
    Returns:
        Filtered list of relevant chunks
    """
    relevant_chunks = []
    
    for chunk in chunks:
        relevance = assess_relevance(query, chunk, generation_fn)
        if relevance == "relevant":
            relevant_chunks.append(chunk)
        # Optionally include partially relevant chunks
        # elif relevance == "partially relevant":
        #     relevant_chunks.append(chunk)
    
    return relevant_chunks


def retrieve_with_self_reflection(query: str, chunks: List[str], embeddings: List[Any],
                                  generation_fn: Callable, k: int = None,
                                  max_iterations: int = 2) -> Tuple[List[str], Dict[str, Any]]:
    """
    Retrieve chunks with self-reflection and iteration.
    
    Core retrieval function that:
    1. Determines if retrieval is needed
    2. Retrieves initial chunks
    3. Filters for relevance
    4. Optionally iterates if quality is low
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function
        k: Number of results
        max_iterations: Max refinement iterations
        
    Returns:
        Tuple of (retrieved chunks, reflection metadata)
    """
    if k is None:
        k = 5
    
    if not chunks or not embeddings:
        return [], {"iterations": 0, "retrieval_needed": False}
    
    metadata = {
        "iterations": 0,
        "retrieval_needed": False,
        "initial_chunks_count": 0,
        "filtered_chunks_count": 0,
        "relevance_assessments": []
    }
    
    # Step 1: Check if retrieval is needed
    retrieval_needed = determine_if_retrieval_needed(query, generation_fn)
    metadata["retrieval_needed"] = retrieval_needed
    
    if not retrieval_needed:
        return [], metadata
    
    # Step 2: Basic semantic search
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
    initial_chunk_indices = [idx for idx, _ in scores[:k * 2]]  # Get more initially
    initial_chunks = [chunks[idx] for idx in initial_chunk_indices]
    metadata["initial_chunks_count"] = len(initial_chunks)
    
    # Step 3: Filter for relevance
    for iteration in range(max_iterations):
        relevant_chunks = filter_relevant_chunks(query, initial_chunks, generation_fn)
        metadata["filtered_chunks_count"] = len(relevant_chunks)
        
        if len(relevant_chunks) >= k or iteration == max_iterations - 1:
            metadata["iterations"] = iteration + 1
            return relevant_chunks[:k], metadata
        
        # If not enough relevant chunks, expand search
        if iteration < max_iterations - 1:
            k_expanded = int(k * 1.5)
            initial_chunk_indices = [idx for idx, _ in scores[:k_expanded]]
            initial_chunks = [chunks[idx] for idx in initial_chunk_indices]
    
    metadata["iterations"] = max_iterations
    return initial_chunks[:k], metadata


def self_rag_orchestrate(query: str, chunks: List[str], embeddings: List[Any],
                        generation_fn: Callable, response_gen_fn: Callable) -> Dict[str, Any]:
    """
    Full Self-RAG pipeline with reflection and iteration.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM function for reflection
        response_gen_fn: LLM function for response generation
        
    Returns:
        Dictionary with results and reflection metadata
    """
    result = {
        "query": query,
        "retrieval_needed": False,
        "retrieved_chunks": [],
        "response": "",
        "support_level": "no support",
        "utility_rating": 0,
        "metadata": {}
    }
    
    # Determine if retrieval is needed
    retrieval_needed = determine_if_retrieval_needed(query, generation_fn)
    result["retrieval_needed"] = retrieval_needed
    
    if not retrieval_needed:
        result["response"] = "This query can be answered from general knowledge without retrieval."
        return result
    
    # Retrieve with self-reflection
    retrieved_chunks, metadata = retrieve_with_self_reflection(
        query, chunks, embeddings, generation_fn, k=5, max_iterations=2
    )
    result["retrieved_chunks"] = retrieved_chunks
    result["metadata"] = metadata
    
    if not retrieved_chunks:
        result["response"] = "No relevant information could be retrieved for this query."
        return result
    
    # Generate response
    context = "\n\n".join(retrieved_chunks)
    system_prompt = "You are a helpful assistant. Answer the question based on the provided context."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    response_obj = response_gen_fn(system_prompt, user_prompt)
    response = response_obj.get("content", "")
    result["response"] = response
    
    # Assess support level
    support = assess_support(response, context, generation_fn)
    result["support_level"] = support
    
    # Rate utility
    utility = rate_utility(query, response, generation_fn)
    result["utility_rating"] = utility
    
    return result
