#!/usr/bin/env python3
"""
02 - Semantic Chunking RAG

RAG pipeline using semantic chunking instead of sliding window.
Splits text based on semantic boundaries detected through sentence embeddings.

Semantic chunking respects content meaning, potentially creating better retrieval results.

Usage:
    python workflows/02_semantic_chunking.py                    # First query
    python workflows/02_semantic_chunking.py --all              # All queries
    python workflows/02_semantic_chunking.py --query "Question"
    python workflows/02_semantic_chunking.py --max 5            # First 5 queries
    python workflows/02_semantic_chunking.py --method percentile --threshold 90
"""

import argparse
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import (
    run_rag_from_validation_file,
    run_rag_batch,
    print_results,
    discover_documents
)
from workflow_parts.semantic_chunking import chunk_text_semantic
from workflow_parts.output_formatter import UnifiedSummaryFormatter
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results
from workflow_parts.retrieval import semantic_search


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="02 Semantic Chunking RAG")
    parser.add_argument('--files', nargs='+', help='PDF/DOCX files to process (space-separated)')
    parser.add_argument('--pdf', help='(Deprecated) Single PDF file')
    parser.add_argument('--val', default='data/val_multi.json', help='Validation file')
    parser.add_argument('--val-multi', default='data/val_multi.json', help='Validation file for multiple documents')
    parser.add_argument('--query', type=str, help='Custom query')
    parser.add_argument('--all', action='store_true', help='Process all queries')
    parser.add_argument('--max', type=int, default=1, help='Max number of queries')
    parser.add_argument('--k', type=int, default=5, help='Number of top chunks to retrieve')
    
    # Semantic chunking specific arguments
    parser.add_argument('--method', type=str, default='percentile', 
                        choices=['percentile', 'standard_deviation', 'interquartile'],
                        help='Semantic chunking method')
    parser.add_argument('--threshold', type=float, default=90, 
                        help='Threshold value for chunking method (percentile: 0-100, std_dev: sigma units, iqr: N/A)')
    
    parser.add_argument('--use-ocr', action='store_true', help='Enable OCR for image PDFs')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--multi', action='store_true', help='Use multi-document validation file')
    parser.add_argument("--batch", action="store_true", help="Batch mode (minimal output)")
    
    args = parser.parse_args()
    
    if not os.getenv('AZURE_OPENAI_ENDPOINT') and not os.getenv('OPENAI_API_KEY'):
        print("ERROR: No credentials configured. Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + API_VERSION or OPENAI_API_KEY")
        return 1
    
    # Debug: print credential status
    print("\n=== Credential Configuration ===")
    if os.getenv('AZURE_OPENAI_ENDPOINT'):
        print("[OK] Azure OpenAI detected")
        print(f"  Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')[:50]}...")
        print(f"  API Version: {os.getenv('API_VERSION', 'NOT SET')}")
        print(f"  Embedding Deployment: {os.getenv('EMBEDDING_DEPLOYMENT', 'text-embedding-ada-002')}")
        print(f"  Chat Deployment: {os.getenv('CHAT_DEPLOYMENT', 'gpt-4o-mini')}")
    elif os.getenv('OPENAI_API_KEY'):
        print("[OK] OpenAI API Key detected")
    print("="*35 + "\n")
    
    # Determine file paths
    if args.files:
        file_paths = args.files
    elif args.pdf:
        file_paths = args.pdf
    else:
        # Automatically discover all documents in data/ directory
        discovered = discover_documents("data")
        if discovered:
            file_paths = discovered
            print(f"Auto-discovered {len(discovered)} document(s):")
            for f in discovered:
                print(f"  - {Path(f).name}")
        else:
            # Fallback to default single file
            file_paths = 'data/AI_Information.pdf'
    
    # Determine validation file
    if args.multi:
        val_file = args.val_multi
    elif isinstance(file_paths, list) and len(file_paths) > 1:
        val_file = args.val_multi
    else:
        val_file = args.val
    
    # Check if validation file exists
    if not os.path.exists(val_file):
        print(f"Warning: Validation file not found: {val_file}")
        fallback = args.val if val_file == args.val_multi else args.val_multi
        if os.path.exists(fallback):
            print(f"  Using fallback: {fallback}")
            val_file = fallback
    
    # Create a semantic chunker with specified method and threshold
    def chunker_with_params(text, **kwargs):
        """Wrap semantic chunker with specified parameters"""
        return chunk_text_semantic(
            text,
            method=args.method,
            threshold=args.threshold,
            **kwargs
        )
    
    # Handle custom query
    if args.query:
        results = run_rag_batch(
            file_paths=file_paths,
            queries=[args.query],
            chunker=chunker_with_params,
            retriever=semantic_search,
            k=args.k,
            evaluate=not args.no_eval,
            use_ocr=args.use_ocr,
            auto_ocr=True,
            embedding_deployment=os.getenv('EMBEDDING_DEPLOYMENT'),
            chat_deployment=os.getenv('CHAT_DEPLOYMENT')
        )
    else:
        # Run from validation file
        max_q = None if args.all else args.max
        results = run_rag_from_validation_file(
            file_paths=file_paths,
            val_file=val_file,
            chunker=chunker_with_params,
            retriever=semantic_search,
            k=args.k,
            evaluate=not args.no_eval,
            max_queries=max_q,
            use_ocr=args.use_ocr,
            auto_ocr=True,
            embedding_deployment=os.getenv('EMBEDDING_DEPLOYMENT'),
            chat_deployment=os.getenv('CHAT_DEPLOYMENT')
        )
    
    print_results(results)
    
    # Track results and calculate metrics
    metrics = create_metrics_from_results(results)
    tracker = ResultsTracker()
    tracker.add_result(workflow_id="02", workflow_name="Semantic Chunking RAG", metrics=metrics)
    tracker.save_results()
    
    # Calculate execution time
    total_time = time.time() - start_time
    
    # Print unified summary with only essential metrics
    summary_formatter = UnifiedSummaryFormatter("Semantic Chunking RAG", 2)
    summary = summary_formatter.format_summary(
        queries_processed=len(results),
        total_time=total_time,
        metrics=metrics
    )
    print(summary)
    
    return 0


if __name__ == '__main__':
    exit(main())
