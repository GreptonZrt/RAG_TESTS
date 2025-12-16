"""
RAG Pipeline Orchestration

Generic orchestration functions for building RAG pipelines.
Supports different chunking strategies, retrievers, rerankers, etc.

Pattern:
    results = run_rag_pipeline(
        file_paths,
        chunker=chunk_text_sliding_window,
        retriever=semantic_search,
        reranker=None,  # optional
        **kwargs
    )
"""

import os
from typing import List, Dict, Optional, Union, Callable
from pathlib import Path
from dotenv import load_dotenv

from workflow_parts.data_loading import (
    load_multiple_files,
    combine_documents,
    load_validation_data,
    extract_queries_from_validation_data
)
from workflow_parts.embedding import create_embeddings
from workflow_parts.generation import generate_response
from workflow_parts.evaluation import evaluate_response

# Load environment variables
load_dotenv()


def discover_documents(data_dir: str = "data") -> List[str]:
    """
    Automatically discover all document files in a directory.
    
    Includes: PDF, DOCX, TXT files
    Excludes: JSON files
    
    Args:
        data_dir: Directory to search for documents (default: "data")
        
    Returns:
        List of file paths sorted by filename
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    # File extensions to include
    valid_extensions = {'.pdf', '.docx', '.doc', '.txt'}
    
    # Find all matching files
    files = [
        str(f) for f in data_path.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    # Sort by filename for consistent ordering
    return sorted(files)


def run_rag_pipeline(
    file_paths: Union[str, List[str]],
    query: str,
    chunker: Callable,
    retriever: Callable,
    reranker: Optional[Callable] = None,
    k: int = 5,
    evaluate: bool = True,
    ideal_answer: Optional[str] = None,
    use_ocr: bool = False,
    auto_ocr: bool = True,
    embedding_deployment: Optional[str] = None,
    chat_deployment: Optional[str] = None,
    **chunking_kwargs
) -> Dict:
    """
    Generic RAG pipeline orchestration.
    
    Args:
        file_paths: Single file path or list of file paths
        query: User query
        chunker: Chunking function (e.g., chunk_text_sliding_window)
        retriever: Retrieval function (e.g., semantic_search)
        reranker: Optional reranking function
        k: Number of chunks to retrieve
        evaluate: Whether to evaluate response
        ideal_answer: Ground truth answer for evaluation
        use_ocr: Enable OCR for all PDF pages
        auto_ocr: Automatically try OCR if standard extraction yields no text
        embedding_deployment: Override embedding model
        chat_deployment: Override chat model
        **chunking_kwargs: Additional kwargs for chunker (chunk_size, overlap, etc.)
        
    Returns:
        Dict with query, retrieved_chunks, ai_response, evaluation
    """
    # Override deployments if specified
    if embedding_deployment:
        os.environ["EMBEDDING_DEPLOYMENT"] = embedding_deployment
    if chat_deployment:
        os.environ["CHAT_DEPLOYMENT"] = chat_deployment
    
    # Step 1: Load documents
    print(f"[Step 1] Loading documents...")
    file_paths_list = file_paths if isinstance(file_paths, list) else [file_paths]
    file_results = load_multiple_files(file_paths_list, use_ocr=use_ocr, auto_ocr=auto_ocr)
    text = combine_documents(file_results)
    print(f"  [READY] Loaded {len(text)} characters from {len(file_paths_list)} file(s)")
    
    # Step 2: Chunk text using provided chunker
    print(f"[Step 2] Chunking text...")
    chunks = chunker(text, **chunking_kwargs)
    print(f"  → Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings
    print(f"[Step 3] Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print(f"  → Created {len(embeddings)} embeddings")
    
    # Step 4: Retrieve relevant chunks using provided retriever
    print(f"[Step 4] Retrieving top-{k} chunks...")
    retrieved_chunks = retriever(query, chunks, embeddings, k=k)
    print(f"  → Retrieved {len(retrieved_chunks)} chunks")
    
    # Step 4b: Optional reranking
    if reranker:
        print(f"[Step 4b] Reranking chunks...")
        retrieved_chunks = reranker(query, retrieved_chunks)
        print(f"  → Reranked to {len(retrieved_chunks)} chunks")
    
    # Step 5: Generate response
    print(f"[Step 5] Generating response...")
    gen_result = generate_response(query, retrieved_chunks)
    ai_response = gen_result["content"]
    print(f"  → Generated response ({len(ai_response)} characters)")
    
    result = {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "ai_response": ai_response,
    }
    
    # Step 6: Optional evaluation
    if evaluate:
        print(f"[Step 6] Evaluating response...")
        eval_result = evaluate_response(query, ai_response, ideal_answer)
        result["evaluation"] = eval_result
        print(f"  → Evaluation score: {eval_result.get('score', 'N/A')}")
    
    return result


def run_rag_batch(
    file_paths: Union[str, List[str]],
    queries: List[str],
    chunker: Callable,
    retriever: Callable,
    reranker: Optional[Callable] = None,
    k: int = 5,
    evaluate: bool = True,
    ideal_answers: Optional[Dict[str, str]] = None,
    use_ocr: bool = False,
    auto_ocr: bool = True,
    embedding_deployment: Optional[str] = None,
    chat_deployment: Optional[str] = None,
    **chunking_kwargs
) -> List[Dict]:
    """
    Run RAG pipeline for multiple queries (with caching of chunking/embeddings).
    
    Args:
        file_paths: Single file path or list of file paths
        queries: List of queries to process
        chunker: Chunking function
        retriever: Retrieval function
        reranker: Optional reranking function
        k: Number of chunks to retrieve
        evaluate: Whether to evaluate responses
        ideal_answers: Dict mapping queries to ideal answers
        use_ocr: Enable OCR for all PDF pages
        auto_ocr: Automatically try OCR if standard extraction yields no text
        embedding_deployment: Override embedding model
        chat_deployment: Override chat model
        **chunking_kwargs: Additional kwargs for chunker
        
    Returns:
        List of result dictionaries
    """
    # Override deployments if specified
    if embedding_deployment:
        os.environ["EMBEDDING_DEPLOYMENT"] = embedding_deployment
    if chat_deployment:
        os.environ["CHAT_DEPLOYMENT"] = chat_deployment
    
    ideal_answers = ideal_answers or {}
    
    # Cache: Load documents and create chunks/embeddings once
    print(f"[Batch Init] Loading documents and creating embeddings...")
    file_paths_list = file_paths if isinstance(file_paths, list) else [file_paths]
    file_results = load_multiple_files(file_paths_list, use_ocr=use_ocr, auto_ocr=auto_ocr)
    text = combine_documents(file_results)
    chunks = chunker(text, **chunking_kwargs)
    embeddings = create_embeddings(chunks)
    print(f"  [READY] {len(chunks)} chunks, {len(embeddings)} embeddings\n")
    
    results = []
    for i, query in enumerate(queries):
        try:
            print(f"\n{'='*70}")
            print(f"Query {i+1}/{len(queries)}: {query}")
            print(f"{'='*70}")
            
            # Step 4: Retrieve
            retrieved_chunks = retriever(query, chunks, embeddings, k=k)
            
            # Step 4b: Optional reranking
            if reranker:
                retrieved_chunks = reranker(query, retrieved_chunks)
            
            # Step 5: Generate
            gen_result = generate_response(query, retrieved_chunks)
            ai_response = gen_result["content"]
            
            ideal = ideal_answers.get(query) if ideal_answers else None
            
            result = {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "ai_response": ai_response,
                "ideal_answer": ideal or "",
            }
            
            # Step 6: Optional evaluation
            if evaluate:
                eval_result = evaluate_response(query, ai_response, ideal)
                result["evaluation"] = eval_result
            
            results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "query": query,
                "error": str(e)
            })
    
    return results


def run_rag_from_validation_file(
    file_paths: Union[str, List[str]],
    val_file: str,
    chunker: Callable,
    retriever: Callable,
    reranker: Optional[Callable] = None,
    k: int = 5,
    evaluate: bool = True,
    max_queries: Optional[int] = None,
    use_ocr: bool = False,
    auto_ocr: bool = True,
    embedding_deployment: Optional[str] = None,
    chat_deployment: Optional[str] = None,
    **chunking_kwargs
) -> List[Dict]:
    """
    Run RAG pipeline using queries from a validation file.
    
    Args:
        file_paths: Single file path or list of file paths
        val_file: Path to JSON validation file
        chunker: Chunking function
        retriever: Retrieval function
        reranker: Optional reranking function
        k: Number of chunks to retrieve
        evaluate: Whether to evaluate responses
        max_queries: Maximum number of queries to process (None = all)
        use_ocr: Enable OCR for image-based PDFs
        embedding_deployment: Override embedding model
        chat_deployment: Override chat model
        **chunking_kwargs: Additional kwargs for chunker
        
    Returns:
        List of result dictionaries
    """
    print(f"Loading validation data from: {val_file}")
    validation_data = load_validation_data(val_file)
    
    # Limit queries if specified
    data = validation_data[:max_queries] if max_queries else validation_data
    
    # Extract queries and ideal answers
    queries = extract_queries_from_validation_data(data)
    ideal_answers = {item['question']: item.get('ideal_answer', '') for item in data}
    
    print(f"Processing {len(queries)} queries from validation file\n")
    
    results = run_rag_batch(
        file_paths=file_paths,
        queries=queries,
        chunker=chunker,
        retriever=retriever,
        reranker=reranker,
        k=k,
        evaluate=evaluate,
        ideal_answers=ideal_answers,
        use_ocr=use_ocr,
        auto_ocr=auto_ocr,
        embedding_deployment=embedding_deployment,
        chat_deployment=chat_deployment,
        **chunking_kwargs
    )
    
    return results


def print_result(result: Dict) -> None:
    """Pretty-print a single result."""
    print(f"\n{'-'*70}")
    print(f"Query: {result.get('query', 'N/A')}")
    print(f"{'-'*70}")
    
    if 'error' in result:
        print(f"ERROR: {result['error']}")
        return
    
    # Print retrieved chunks
    print(f"\nRetrieved Chunks ({len(result.get('retrieved_chunks', []))} total):")
    for i, chunk in enumerate(result.get('retrieved_chunks', [])):
        preview = chunk[:150].replace('\n', ' ') + "..." if len(chunk) > 150 else chunk
        print(f"  [{i+1}] {preview}")
    
    # Print AI response
    print(f"\nAI Response:")
    print(result.get('ai_response', 'N/A'))
    
    # Print ideal answer if available
    if result.get('ideal_answer'):
        print(f"\nIdeal Answer:")
        print(result['ideal_answer'])
    
    # Print evaluation if available
    if 'evaluation' in result:
        eval_result = result['evaluation']
        score = eval_result.get('score', 'N/A')
        reasoning = eval_result.get('reasoning', '')
        print(f"\nEvaluation Score: {score}")
        if reasoning:
            print(f"Reasoning: {reasoning}")


def print_results(results: List[Dict]) -> None:
    """Pretty-print all results."""
    for result in results:
        print_result(result)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    successful = sum(1 for r in results if 'error' not in r)
    print(f"Processed: {successful}/{len(results)} queries successfully")
    
    # Average evaluation score
    scores = [r['evaluation'].get('score') 
              for r in results 
              if 'evaluation' in r and r['evaluation'].get('score') is not None]
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Average Evaluation Score: {avg_score:.2f}")
