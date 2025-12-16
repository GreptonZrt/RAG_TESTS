#!/usr/bin/env python3
"""
XX - [WORKFLOW NAME] RAG

[BRIEF DESCRIPTION]

Usage:
    python workflows/XX_[name].py                   # First query
    python workflows/XX_[name].py --all             # All queries
    python workflows/XX_[name].py --query "Question"
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import (
    run_rag_from_validation_file,
    run_rag_batch,
    print_results
)
# TODO: Import your specific functions here
# from workflow_parts.chunking import chunk_text_xxx
# from workflow_parts.retrieval import xxx_search
# from workflow_parts.reranking import xxx_rerank


def main():
    parser = argparse.ArgumentParser(description="XX [Workflow Name] RAG")
    parser.add_argument('--files', nargs='+', help='PDF/DOCX files to process')
    parser.add_argument('--pdf', help='(Deprecated) Single PDF file')
    parser.add_argument('--val', default='data/val_multi.json', help='Validation file')
    parser.add_argument('--query', type=str, help='Custom query')
    parser.add_argument('--all', action='store_true', help='Process all queries')
    parser.add_argument('--max', type=int, default=1, help='Max number of queries')
    parser.add_argument('--k', type=int, default=5, help='Number of top chunks to retrieve')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--chunk-overlap', type=int, default=200)
    parser.add_argument('--use-ocr', action='store_true', help='Enable OCR for image PDFs')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation')
    
    args = parser.parse_args()
    
    if not os.getenv('AZURE_OPENAI_ENDPOINT') and not os.getenv('OPENAI_API_KEY'):
        print("ERROR: No credentials configured. Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + API_VERSION or OPENAI_API_KEY")
        return 1
    
    # Debug: print credential status
    print("\n=== Credential Configuration ===")
    if os.getenv('AZURE_OPENAI_ENDPOINT'):
        print("[OK] Azure OpenAI detected")
    elif os.getenv('OPENAI_API_KEY'):
        print("[OK] OpenAI API Key detected")
    print("="*35 + "\n")
    
    # Determine file paths
    file_paths = args.files or args.pdf or 'data/AI_Information.pdf'
    
    # TODO: Replace these with your specific functions
    # from workflow_parts.chunking import chunk_text_xxx
    # from workflow_parts.retrieval import xxx_search
    # chunker = chunk_text_xxx
    # retriever = xxx_search
    # reranker = None  # or your reranker function
    
    # Handle custom query
    if args.query:
        results = run_rag_batch(
            file_paths=file_paths,
            queries=[args.query],
            chunker=None,  # TODO: set to your chunker
            retriever=None,  # TODO: set to your retriever
            reranker=None,  # TODO: optional reranker
            k=args.k,
            evaluate=not args.no_eval,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            use_ocr=args.use_ocr
        )
    else:
        # Run from validation file
        max_q = None if args.all else args.max
        results = run_rag_from_validation_file(
            file_paths=file_paths,
            val_file=args.val,
            chunker=None,  # TODO: set to your chunker
            retriever=None,  # TODO: set to your retriever
            reranker=None,  # TODO: optional reranker
            k=args.k,
            evaluate=not args.no_eval,
            max_queries=max_q,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            use_ocr=args.use_ocr
        )
    
    print_results(results)
    return 0


if __name__ == '__main__':
    exit(main())
