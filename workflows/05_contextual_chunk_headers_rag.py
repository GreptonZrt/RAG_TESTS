#!/usr/bin/env python3
"""
Workflow 05: Contextual Chunk Headers RAG

Generates descriptive headers for each chunk using an LLM, then uses both
headers and text content for semantic search. This improves retrieval by
allowing the model to understand chunk context at a glance.

Usage:
    python workflows/05_contextual_chunk_headers_rag.py --max 1 --no-eval --header-weight 0.5
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

import argparse
from workflow_parts.orchestration import (
    discover_documents,
    run_rag_from_validation_file,
    print_results
)
from workflow_parts.data_loading import load_multiple_files
from workflow_parts.embedding import get_embedding_fn
from workflow_parts.contextual_headers import (
    generate_chunk_headers,
    semantic_search_with_headers,
    ChunkWithHeader
)
from workflow_parts.output_formatter import UnifiedSummaryFormatter
from workflow_parts.generation import generate_response, get_generation_client
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results


def create_header_generator():
    """
    Creates a function that generates headers for chunks using LLM.
    
    Returns:
        Function that generates a header for a given chunk.
    """
    client = get_generation_client()
    
    def generate_header(chunk: str) -> str:
        """Generate a header for a chunk using LLM."""
        try:
            system_prompt = "Generate a concise and informative title for the given text (max 10 words)."
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk[:500]}  # Limit chunk size for LLM
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [WARNING] Failed to generate header: {e}")
            # Fallback: use first 100 characters
            return chunk[:100].strip() + "..."
    
    return generate_header


def create_header_retriever(header_weight: float = 0.5, k: int = 5):
    """
    Creates a retriever function that uses header-aware semantic search.
    
    Args:
        header_weight: Weight for header similarity (0.0-1.0).
        k: Number of top results to retrieve.
    
    Returns:
        Function that retrieves chunks using header-aware search.
    """
    # Pre-generate headers and embeddings for all chunks
    chunks_cache = {}
    
    def retriever(query, chunks, embeddings, k=None):
        """Retrieve using header-aware semantic search."""
        
        # Generate headers if not already done
        cache_key = hash(tuple(chunks))
        if cache_key not in chunks_cache:
            print("  [Header Gen] Generating headers for chunks...")
            
            header_gen = create_header_generator()
            chunks_with_headers = generate_chunk_headers(
                chunks=chunks,
                get_embedding_fn=get_embedding_fn,
                generate_header_fn=header_gen
            )
            chunks_cache[cache_key] = chunks_with_headers
        else:
            chunks_with_headers = chunks_cache[cache_key]
        
        # Search using header-aware semantic search
        results = semantic_search_with_headers(
            query=query,
            chunks_with_headers=chunks_with_headers,
            get_embedding_fn=get_embedding_fn,
            k=k or 5,
            header_weight=header_weight
        )
        
        # Return just the text parts
        return [chunk.text for chunk in results]
    
    return retriever


def main():
    """Main entry point for workflow 05."""
    start_time = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Contextual Chunk Headers RAG Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflows/05_contextual_chunk_headers_rag.py --max 3
  python workflows/05_contextual_chunk_headers_rag.py --header-weight 0.7 --k 3 --no-eval
  python workflows/05_contextual_chunk_headers_rag.py --query "What is AI?"
        """
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of queries to process (default: all)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top chunks to retrieve (default: 5)")
    parser.add_argument("--header-weight", type=float, default=0.5,
                        help="Weight for header similarity: 0.0 (text-only) to 1.0 (header-only) (default: 0.5)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single custom query instead of from validation file")
    parser.add_argument("--all", action="store_true",
                        help="Process all queries in validation file")
    parser.add_argument("--use-ocr", action="store_true",
                        help="Force OCR for PDF extraction")
    parser.add_argument("--batch", action="store_true", help="Batch mode (minimal output)")
    
    args = parser.parse_args()
    
    # Validate header weight
    if not (0.0 <= args.header_weight <= 1.0):
        print("[ERROR] --header-weight must be between 0.0 and 1.0")
        return 1
    
    # Auto-discover documents
    print("\nAuto-discovered documents:")
    docs = discover_documents("data")
    for doc in docs:
        print(f"  - {doc}")
    print()
    
    # Load validation data
    validation_file = "data/val_multi.json"
    if not os.path.exists(validation_file):
        validation_file = "data/val.json"
    
    print(f"Loading validation data from: {validation_file}")
    
    # Create the retriever with header-aware search
    retriever = create_header_retriever(
        header_weight=args.header_weight,
        k=args.k
    )
    
    # Run the RAG pipeline
    if args.query:
        # Single custom query
        from workflow_parts.chunking import chunk_text_sliding_window
        
        print(f"\n{'='*70}")
        print(f"Query: {args.query}")
        print(f"{'='*70}\n")
        
        # Load and chunk documents
        text = load_multiple_files(
            docs,
            auto_ocr=args.use_ocr or True
        )
        chunks = chunk_text_sliding_window(text, 1000, 200)
        
        # Retrieve relevant chunks
        retrieved_chunks = retriever(chunks, args.query)
        
        # Generate response
        context_text = "\n".join(retrieved_chunks)
        response = generate_response(args.query, context_text)
        
        print(f"Retrieved {len(retrieved_chunks)} chunks (with headers)")
        print(f"\nAI Response:\n{response}\n")
        
    else:
        # Process validation file
        from workflow_parts.chunking import chunk_text_sliding_window
        
        results = run_rag_from_validation_file(
            file_paths=docs,
            val_file=validation_file,
            chunker=chunk_text_sliding_window,
            retriever=retriever,
            k=5,
            evaluate=(not args.no_eval),
            max_queries=args.max,
            auto_ocr=args.use_ocr or True
        )
        
        print_results(results)
        
        # Track results and calculate metrics
        metrics = create_metrics_from_results(results)
        tracker = ResultsTracker()
        tracker.add_result(workflow_id="05", workflow_name="Contextual Chunk Headers RAG", metrics=metrics)
        tracker.save_results()
        
        # Calculate execution time
        total_time = time.time() - start_time
        
        # Print unified summary with only essential metrics
        summary_formatter = UnifiedSummaryFormatter("Contextual Chunk Headers RAG", 5)
        summary = summary_formatter.format_summary(
            queries_processed=len(results),
            total_time=total_time,
            metrics=metrics
        )
        print(summary)


if __name__ == "__main__":
    try:
        exit(main() or 0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
