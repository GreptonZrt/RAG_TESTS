"""
HyDE RAG Module

Implements Hypothetical Document Embedding:
1. Generates a hypothetical document that would answer the query
2. Uses the hypothetical document as an improved query
3. Retrieves chunks similar to the hypothetical document
4. Generates final response from retrieved context
"""

from typing import List, Dict, Any, Callable
import numpy as np


def generate_hypothetical_document(query: str, generation_fn: Callable,
                                   desired_length: int = 1000) -> str:
    """
    Generate a hypothetical document that would answer the query.
    
    Args:
        query: User query
        generation_fn: LLM generation function
        desired_length: Target length in characters
        
    Returns:
        Generated hypothetical document
    """
    system_prompt = f"""You are an expert document creator. 
    Given a question, generate a detailed document that would directly answer this question.
    The document should be approximately {desired_length} characters long and provide an in-depth, 
    informative answer to the question. Write as if this document is from an authoritative source
    on the subject. Include specific details, facts, and explanations.
    Do not mention that this is a hypothetical document - just write the content directly."""
    
    user_prompt = f"Question: {query}\n\nGenerate a document that fully answers this question:"
    
    response = generation_fn(system_prompt, user_prompt)
    return response.get("content", "")


def hyde_retrieve(query: str, chunks: List[str], embeddings: List[Any],
                  generation_fn: Callable, k: int = None) -> tuple:
    """
    Retrieve chunks using HyDE approach.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM generation function
        k: Number of results
        
    Returns:
        Tuple of (retrieved chunks, hypothetical document)
    """
    if k is None:
        k = 5
    
    if not chunks or not embeddings:
        return [], ""
    
    # Step 1: Generate hypothetical document
    hypothetical_doc = generate_hypothetical_document(query, generation_fn)
    
    # Step 2: Get embedding for hypothetical document
    from workflow_parts.embedding import get_embedding_fn
    hyde_emb_obj = get_embedding_fn()(hypothetical_doc.strip())
    hyde_vec = np.array(hyde_emb_obj.embedding)
    
    # Step 3: Retrieve chunks similar to hypothetical document
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_vec = np.array(embeddings[i].embedding)
        norm_h = np.linalg.norm(hyde_vec)
        norm_c = np.linalg.norm(chunk_vec)
        
        if norm_h > 0 and norm_c > 0:
            sim = np.dot(hyde_vec, chunk_vec) / (norm_h * norm_c)
        else:
            sim = 0.0
        scores.append((i, sim))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-k chunks
    result_indices = [idx for idx, _ in scores[:k]]
    retrieved_chunks = [chunks[idx] for idx in result_indices]
    
    return retrieved_chunks, hypothetical_doc


def compare_hyde_and_standard(query: str, chunks: List[str], embeddings: List[Any],
                             generation_fn: Callable, 
                             response_gen_fn: Callable, k: int = 5) -> Dict[str, Any]:
    """
    Compare HyDE and standard RAG approaches.
    
    Args:
        query: User query
        chunks: Available text chunks
        embeddings: Embedding objects
        generation_fn: LLM generation function
        response_gen_fn: Response generation function
        k: Number of chunks to retrieve
        
    Returns:
        Dictionary with comparison results
    """
    # HyDE retrieval
    hyde_chunks, hypothetical_doc = hyde_retrieve(query, chunks, embeddings, generation_fn, k)
    
    # Standard retrieval (direct query embedding)
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
    standard_chunks = [chunks[idx] for idx, _ in scores[:k]]
    
    # Generate responses for both
    hyde_response_data = response_gen_fn(
        "You are a helpful assistant. Answer the question based on the provided context.",
        f"Context:\n{chr(10).join(hyde_chunks)}\n\nQuestion: {query}"
    )
    hyde_response = hyde_response_data.get("content", "")
    
    standard_response_data = response_gen_fn(
        "You are a helpful assistant. Answer the question based on the provided context.",
        f"Context:\n{chr(10).join(standard_chunks)}\n\nQuestion: {query}"
    )
    standard_response = standard_response_data.get("content", "")
    
    return {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "hyde_chunks": hyde_chunks,
        "hyde_response": hyde_response,
        "standard_chunks": standard_chunks,
        "standard_response": standard_response
    }


def create_hyde_retriever(generation_fn):
    """
    Create a HyDE retriever function.
    
    Args:
        generation_fn: LLM generation function
        
    Returns:
        Retriever function using HyDE
    """
    def retriever(query, chunks, embeddings, k=None):
        """Retrieve using HyDE approach."""
        retrieved_chunks, _ = hyde_retrieve(query, chunks, embeddings, generation_fn, k)
        return retrieved_chunks
    
    return retriever


def create_hyde_with_metadata_retriever(generation_fn):
    """
    Create a HyDE retriever that returns metadata.
    
    Args:
        generation_fn: LLM generation function
        
    Returns:
        Tuple of (retriever function, metadata dict)
    """
    metadata = {}
    
    def retriever(query, chunks, embeddings, k=None):
        """Retrieve using HyDE approach with metadata."""
        retrieved_chunks, hypothetical_doc = hyde_retrieve(
            query, chunks, embeddings, generation_fn, k
        )
        metadata["hypothetical_document"] = hypothetical_doc
        metadata["retrieved_chunks_count"] = len(retrieved_chunks)
        return retrieved_chunks
    
    return retriever, metadata
