#!/usr/bin/env python3
"""
Workflow 04: Context-Enriched RAG

Enhances retrieval by including neighboring chunks alongside the most relevant chunk.
This provides broader context for the LLM to generate better responses.

Usage:
    python workflows/04_context_enriched_rag.py --max 1 --no-eval --context-size 1 --k 1
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
from workflow_parts.context_enriched import context_enriched_search
from workflow_parts.output_formatter import UnifiedSummaryFormatter
from workflow_parts.retrieval import semantic_search
from workflow_parts.generation import generate_response
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results


def create_enriched_retriever(context_size: int = 1, k: int = 1):
    """
    Creates a retriever function that uses context-enriched search.
    
    Args:
        context_size: Number of neighboring chunks to include.
        k: Number of top chunks to retrieve before adding context.
    
    Returns:
        Function that retrieves context-enriched chunks.
    """
    def retriever(query, chunks, embeddings, k=None):
        # Use context-enriched search instead of simple semantic search
        return context_enriched_search(
            query=query,
            text_chunks=chunks,
            embeddings=embeddings,
            get_embedding_fn=get_embedding_fn,
            k=context_size,
            context_size=context_size
        )
    return retriever


def main():
    """Main entry point for workflow 04."""
    start_time = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Context-Enriched RAG Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflows/04_context_enriched_rag.py --max 3
  python workflows/04_context_enriched_rag.py --context-size 2 --k 2 --no-eval
  python workflows/04_context_enriched_rag.py --query "What is AI?"
        """
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of queries to process (default: all)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of top chunks to retrieve before context expansion (default: 1)")
    parser.add_argument("--context-size", type=int, default=1,
                        help="Number of neighboring chunks to include (default: 1)")
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
    
    # Create the retriever with context enrichment
    retriever = create_enriched_retriever(
        context_size=args.context_size,
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
        
        print(f"Retrieved {len(retrieved_chunks)} chunks (with context)")
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
        tracker.add_result(workflow_id="04", workflow_name="Context Enriched RAG", metrics=metrics)
        tracker.save_results()
        
        # Calculate execution time
        total_time = time.time() - start_time
        
        # Print unified summary with only essential metrics
        summary_formatter = UnifiedSummaryFormatter("Context Enriched RAG", 4)
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
