"""
Document Augmentation RAG Module

Augments chunks by generating relevant questions that can be answered by each chunk.
This improves retrieval by allowing the model to find chunks via question similarity,
not just direct text similarity.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class AugmentedChunk:
    """Represents a chunk with generated questions and embeddings."""
    text: str
    questions: List[str]
    text_embedding: List[float] = None
    question_embeddings: List[List[float]] = None


def generate_questions(
    chunk: str,
    get_generation_fn: Callable,
    num_questions: int = 5
) -> List[str]:
    """
    Generates relevant questions that can be answered from a chunk.

    Args:
        chunk (str): The text chunk to generate questions from.
        get_generation_fn: Function that calls LLM for generation.
        num_questions (int): Number of questions to generate (default: 5).

    Returns:
        List[str]: List of generated questions.
        
    Example:
        >>> questions = generate_questions("AI is...", llm_fn, num_questions=3)
        >>> # Returns: ["What is AI?", "How does AI work?", ...]
    """
    try:
        system_prompt = (
            "You are an expert at generating relevant questions from text. "
            "Create concise questions that can be answered using only the provided text. "
            "Focus on key information and concepts."
        )
        
        user_prompt = f"""Based on the following text, generate {num_questions} different questions that can be answered using only this text:

{chunk}

Format your response as a numbered list of questions only, with no additional text."""
        
        # Generate questions using LLM
        response = get_generation_fn(system_prompt, user_prompt)
        
        # Extract questions from response
        questions_text = response.get("content", "").strip()
        questions = []
        
        for line in questions_text.split('\n'):
            # Remove numbering and clean up whitespace
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
            if cleaned_line and cleaned_line.endswith('?'):
                questions.append(cleaned_line)
        
        return questions[:num_questions]  # Limit to requested number
        
    except Exception as e:
        print(f"  [WARNING] Failed to generate questions: {e}")
        return []


def augment_chunks_with_questions(
    chunks: List[str],
    get_embedding_fn: Callable,
    get_generation_fn: Callable,
    num_questions: int = 5
) -> List[AugmentedChunk]:
    """
    Augments text chunks by generating questions for each chunk.

    Args:
        chunks (List[str]): List of text chunks.
        get_embedding_fn: Function to generate embeddings.
        get_generation_fn: Function to generate questions via LLM.
        num_questions (int): Number of questions per chunk (default: 5).

    Returns:
        List[AugmentedChunk]: Augmented chunks with questions and embeddings.
    """
    augmented_chunks = []
    
    for i, chunk in enumerate(chunks):
        try:
            # Generate questions for this chunk
            questions = generate_questions(chunk, get_generation_fn, num_questions)
            
            # Generate embedding for chunk text
            try:
                text_embedding_response = get_embedding_fn(chunk.strip())
                if hasattr(text_embedding_response, 'embedding'):
                    text_embedding = text_embedding_response.embedding
                elif isinstance(text_embedding_response, list):
                    text_embedding = text_embedding_response
                else:
                    text_embedding = list(text_embedding_response)
            except Exception as e:
                print(f"  [WARNING] Failed to embed chunk {i}: {e}")
                text_embedding = None
            
            # Generate embeddings for questions
            question_embeddings = []
            for j, question in enumerate(questions):
                try:
                    q_embedding_response = get_embedding_fn(question.strip())
                    if hasattr(q_embedding_response, 'embedding'):
                        q_embedding = q_embedding_response.embedding
                    elif isinstance(q_embedding_response, list):
                        q_embedding = q_embedding_response
                    else:
                        q_embedding = list(q_embedding_response)
                    question_embeddings.append(q_embedding)
                except Exception as e:
                    print(f"  [WARNING] Failed to embed question {j} for chunk {i}: {e}")
                    question_embeddings.append(None)
            
            augmented_chunks.append(AugmentedChunk(
                text=chunk,
                questions=questions,
                text_embedding=text_embedding,
                question_embeddings=question_embeddings
            ))
            
        except Exception as e:
            print(f"  [WARNING] Failed to augment chunk {i}: {e}")
            # Fallback: add chunk without questions
            augmented_chunks.append(AugmentedChunk(
                text=chunk,
                questions=[],
                text_embedding=None,
                question_embeddings=[]
            ))
    
    return augmented_chunks


def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    except Exception:
        return 0.0


def augmented_semantic_search(
    query: str,
    augmented_chunks: List[AugmentedChunk],
    get_embedding_fn: Callable,
    k: int = 5,
    question_weight: float = 0.5
) -> List[str]:
    """
    Searches for relevant chunks using both text and question embeddings.

    Args:
        query (str): User query.
        augmented_chunks (List[AugmentedChunk]): Augmented chunks with questions.
        get_embedding_fn: Function to generate embeddings.
        k (int): Number of top results (default: 5).
        question_weight (float): Weight for question similarity vs text similarity (default: 0.5).

    Returns:
        List[str]: Top-k most relevant chunk texts.
    """
    # Get query embedding
    try:
        query_embedding_response = get_embedding_fn(query.strip())
        if hasattr(query_embedding_response, 'embedding'):
            query_embedding = np.array(query_embedding_response.embedding)
        elif isinstance(query_embedding_response, list):
            query_embedding = np.array(query_embedding_response)
        else:
            query_embedding = np.array(query_embedding_response)
    except Exception as e:
        print(f"  [WARNING] Failed to embed query: {e}")
        return []

    similarities = []

    # Score each chunk
    for chunk in augmented_chunks:
        # Skip if no embeddings
        if chunk.text_embedding is None:
            similarities.append((chunk.text, 0.0))
            continue
        
        try:
            # Similarity with chunk text
            text_sim = cosine_similarity(query_embedding, chunk.text_embedding)
            
            # Similarity with generated questions
            max_question_sim = 0.0
            if chunk.question_embeddings:
                for q_embedding in chunk.question_embeddings:
                    if q_embedding:
                        q_sim = cosine_similarity(query_embedding, q_embedding)
                        max_question_sim = max(max_question_sim, q_sim)
            
            # Combined score
            text_weight = 1.0 - question_weight
            combined_sim = (text_weight * text_sim) + (question_weight * max_question_sim)
            
            similarities.append((chunk.text, combined_sim))
        except Exception as e:
            print(f"  [WARNING] Failed to compute similarity: {e}")
            similarities.append((chunk.text, 0.0))

    # Sort by score
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top-k texts
    return [text for text, _ in similarities[:k]]
