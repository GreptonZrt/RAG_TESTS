"""
Proposition Chunking Module

Implements fine-grained retrieval using atomic propositions:
1. Decomposes text into atomic, self-contained propositions
2. Creates embeddings for propositions
3. Retrieves relevant propositions instead of full chunks
4. Reconstructs coherent responses from propositions
"""

from typing import List, Dict, Any, Callable
import json
import re
import numpy as np


def generate_propositions(chunk: str, generation_fn: Callable) -> List[str]:
    """
    Generate atomic, self-contained propositions from a text chunk.
    
    Args:
        chunk: Text chunk to decompose
        generation_fn: LLM generation function
        
    Returns:
        List of propositions
    """
    system_prompt = """Please break down the following text into simple, self-contained propositions. 
    Ensure that each proposition meets the following criteria:

    1. Express a Single Fact: Each proposition should state one specific fact or claim.
    2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
    3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
    4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
    5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses.

    Output ONLY the list of propositions without any additional text or explanations."""
    
    user_prompt = f"Text to convert into propositions:\n\n{chunk}"
    
    response = generation_fn(system_prompt, user_prompt)
    raw_propositions = response.get("content", "").split('\n')
    
    # Clean up propositions
    clean_propositions = []
    for prop in raw_propositions:
        # Remove numbering and bullets
        cleaned = re.sub(r'^\s*(\d+\.|\-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:
            clean_propositions.append(cleaned)
    
    return clean_propositions


def decompose_chunks_into_propositions(chunks: List[str], 
                                      generation_fn: Callable) -> Dict[str, List[str]]:
    """
    Decompose all chunks into propositions.
    
    Args:
        chunks: List of text chunks
        generation_fn: LLM generation function
        
    Returns:
        Dictionary mapping chunk index to propositions
    """
    chunk_propositions = {}
    
    for i, chunk in enumerate(chunks):
        propositions = generate_propositions(chunk, generation_fn)
        chunk_propositions[i] = propositions
        
        if i % 2 == 0:
            print(f"  [Prop] Chunk {i+1}: Generated {len(propositions)} propositions")
    
    return chunk_propositions


def evaluate_proposition(proposition: str, original_text: str, 
                        generation_fn: Callable) -> Dict[str, int]:
    """
    Evaluate a proposition's quality.
    
    Args:
        proposition: Proposition to evaluate
        original_text: Original text for comparison
        generation_fn: LLM generation function
        
    Returns:
        Dictionary with quality scores
    """
    system_prompt = """You are an expert at evaluating the quality of propositions extracted from text.
    Rate the given proposition on the following criteria (scale 1-10):

    - Accuracy: How well the proposition reflects information in the original text
    - Clarity: How easy it is to understand the proposition without additional context
    - Completeness: Whether the proposition includes necessary details (dates, qualifiers, etc.)
    - Conciseness: Whether the proposition is concise without losing important information

    The response must be in valid JSON format with numerical scores for each criterion:
    {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
    """
    
    user_prompt = f"""Proposition: {proposition}

    Original Text: {original_text}

    Please provide your evaluation scores in JSON format."""
    
    response = generation_fn(system_prompt, user_prompt)
    response_text = response.get("content", "{}")
    
    try:
        scores = json.loads(response_text)
        return scores
    except json.JSONDecodeError:
        # Fallback scores
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }


def retrieve_with_propositions(query: str, chunks: List[str], embeddings: List[Any],
                              chunk_propositions: Dict[int, List[str]],
                              proposition_embeddings: Dict[int, List[Any]],
                              k: int = None) -> List[str]:
    """
    Retrieve chunks using proposition-level matching.
    
    Args:
        query: User query
        chunks: Original text chunks
        embeddings: Chunk embeddings (not used for prop retrieval but kept for compatibility)
        chunk_propositions: Dictionary mapping chunk index to propositions
        proposition_embeddings: Dictionary mapping chunk index to proposition embeddings
        k: Number of chunks to return
        
    Returns:
        Top-k relevant chunks reconstructed from propositions
    """
    if k is None:
        k = 5
    
    if not chunks:
        return []
    
    # Compute query embedding
    from workflow_parts.embedding import get_embedding_fn
    query_emb_obj = get_embedding_fn()(query.strip())
    query_vec = np.array(query_emb_obj.embedding)
    
    # Find most relevant propositions
    proposition_scores = []
    
    for chunk_idx, propositions in chunk_propositions.items():
        if chunk_idx not in proposition_embeddings:
            continue
        
        prop_embeds = proposition_embeddings[chunk_idx]
        
        for prop_idx, proposition in enumerate(propositions):
            if prop_idx < len(prop_embeds):
                prop_vec = np.array(prop_embeds[prop_idx].embedding)
                norm_q = np.linalg.norm(query_vec)
                norm_p = np.linalg.norm(prop_vec)
                
                if norm_q > 0 and norm_p > 0:
                    sim = np.dot(query_vec, prop_vec) / (norm_q * norm_p)
                else:
                    sim = 0.0
                
                proposition_scores.append({
                    "chunk_idx": chunk_idx,
                    "prop_idx": prop_idx,
                    "proposition": proposition,
                    "score": sim
                })
    
    # Sort by score
    proposition_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Select top propositions and group by chunk
    selected_chunks_set = set()
    for prop_score in proposition_scores:
        if len(selected_chunks_set) < k:
            selected_chunks_set.add(prop_score['chunk_idx'])
    
    # Return chunks in original order
    result = []
    for chunk_idx in sorted(selected_chunks_set):
        if chunk_idx < len(chunks):
            result.append(chunks[chunk_idx])
    
    return result[:k]


def retrieve_propositions_only(query: str, chunks: List[str], embeddings: List[Any],
                              chunk_propositions: Dict[int, List[str]],
                              proposition_embeddings: Dict[int, List[Any]],
                              k: int = None) -> List[str]:
    """
    Retrieve only the most relevant propositions (not full chunks).
    
    Args:
        query: User query
        chunks: Original text chunks (for reference)
        embeddings: Chunk embeddings
        chunk_propositions: Dictionary mapping chunk index to propositions
        proposition_embeddings: Dictionary mapping chunk index to proposition embeddings
        k: Number of propositions to return
        
    Returns:
        List of relevant propositions
    """
    if k is None:
        k = 5
    
    if not chunks:
        return []
    
    # Compute query embedding
    from workflow_parts.embedding import get_embedding_fn
    query_emb_obj = get_embedding_fn()(query.strip())
    query_vec = np.array(query_emb_obj.embedding)
    
    # Score all propositions
    proposition_scores = []
    
    for chunk_idx, propositions in chunk_propositions.items():
        if chunk_idx not in proposition_embeddings:
            continue
        
        prop_embeds = proposition_embeddings[chunk_idx]
        
        for prop_idx, proposition in enumerate(propositions):
            if prop_idx < len(prop_embeds):
                prop_vec = np.array(prop_embeds[prop_idx].embedding)
                norm_q = np.linalg.norm(query_vec)
                norm_p = np.linalg.norm(prop_vec)
                
                if norm_q > 0 and norm_p > 0:
                    sim = np.dot(query_vec, prop_vec) / (norm_q * norm_p)
                else:
                    sim = 0.0
                
                proposition_scores.append({
                    "proposition": proposition,
                    "score": sim
                })
    
    # Sort by score and return top-k
    proposition_scores.sort(key=lambda x: x['score'], reverse=True)
    return [p['proposition'] for p in proposition_scores[:k]]


def create_proposition_index(chunks: List[str], generation_fn: Callable) -> tuple:
    """
    Create proposition index for all chunks.
    
    Args:
        chunks: List of text chunks
        generation_fn: LLM generation function
        
    Returns:
        Tuple of (chunk_propositions, proposition_embeddings)
    """
    print("  [Propositions] Decomposing chunks into propositions...")
    chunk_propositions = decompose_chunks_into_propositions(chunks, generation_fn)
    
    # Create embeddings for all propositions
    print("  [Propositions] Creating proposition embeddings...")
    from workflow_parts.embedding import get_embedding_fn
    proposition_embeddings = {}
    
    total_props = 0
    for chunk_idx, propositions in chunk_propositions.items():
        if propositions:
            prop_embeds = [get_embedding_fn()(prop.strip()) for prop in propositions]
            proposition_embeddings[chunk_idx] = prop_embeds
            total_props += len(propositions)
    
    print(f"  [Propositions] Created index: {total_props} total propositions")
    
    return chunk_propositions, proposition_embeddings
