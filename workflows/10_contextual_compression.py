#!/usr/bin/env python3
"""
Workflow 10: Contextual Compression RAG

Improves efficiency by compressing retrieved chunks with an LLM.
Removes irrelevant parts from each chunk, reducing context length while preserving
information needed to answer the query.

Usage:
    python workflows/10_contextual_compression.py --max 1 --no-eval --compression-type selective
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
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
from workflow_parts.generation import generate_response, get_generation_client
from workflow_parts.retrieval import semantic_search
from workflow_parts.contextual_compression import compress_chunks_batch
from workflow_parts.output_formatter import UnifiedSummaryFormatter
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results


def create_generation_fn():
    """Creates a function for LLM calls."""
    client = get_generation_client()
    
    def gen_fn(system_prompt: str, user_prompt: str) -> dict:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return {
                "content": response.choices[0].message.content.strip(),
                "response_obj": response
            }
        except Exception as e:
            print(f"  [WARNING] LLM call failed: {e}")
            return {"content": "", "response_obj": None}
    
    return gen_fn


def create_compression_retriever(
    compression_type: str = "selective",
    k: int = 5
):
    """
    Creates a retriever that includes contextual compression.
    
    Args:
        compression_type: "selective" (keep relevant sentences),
                         "summary" (brief summary), or
                         "extraction" (exact quotes).
        k: Number of results to return.
    
    Returns:
        Function that retrieves and compresses chunks.
    """
    # Cache compression results
    compression_cache = {}
    
    def retriever(query, chunks, embeddings, k=None):
        """Retrieve and compress chunks."""
        import numpy as np
        
        # Step 1: Initial semantic search using embeddings
        query_embeddings = get_embedding_fn()(query.strip())
        if hasattr(query_embeddings, 'embedding'):
            query_emb = np.array(query_embeddings.embedding)
        else:
            query_emb = np.array(query_embeddings)
        
        # Calculate similarity scores
        similarity_scores = []
        for i, chunk_emb in enumerate(embeddings):
            emb_vec = np.array(chunk_emb.embedding) if hasattr(chunk_emb, 'embedding') else np.array(chunk_emb)
            score = np.dot(query_emb, emb_vec) / (np.linalg.norm(query_emb) * np.linalg.norm(emb_vec))
            similarity_scores.append((i, score))
        
        # Sort and get top-k
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        initial_results = [chunks[idx] for idx, _ in similarity_scores[:k or 5]]
        
        # Step 2: Compress
        print(f"  [Compress] Compressing {len(initial_results)} chunks...")
        
        gen_fn = create_generation_fn()
        compressed_results, ratios = compress_chunks_batch(
            initial_results, query, gen_fn, compression_type
        )
        
        # Print compression stats
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0
        print(f"  [Compress] Average compression: {avg_ratio:.1f}%")
        
        return compressed_results
    
    return retriever


def main():
    """Main entry point for workflow 10."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Contextual Compression RAG Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflows/10_contextual_compression.py --max 3
  python workflows/10_contextual_compression.py --compression-type summary --no-eval
  python workflows/10_contextual_compression.py --query "What is AI?"
  
Compression types:
  selective  - Keep only relevant sentences
  summary    - Create brief summary
  extraction - Extract relevant quotes
        """
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of queries to process (default: all)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of chunks to retrieve before compression (default: 5)")
    parser.add_argument("--compression-type", choices=["selective", "summary", "extraction"],
                        default="selective",
                        help="Compression strategy (default: selective)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single custom query")
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
    
    # Create retriever with compression
    retriever = create_compression_retriever(
        compression_type=args.compression_type,
        k=args.k
    )
    
    # Run pipeline
    if args.query:
        from workflow_parts.chunking import chunk_text_sliding_window
        
        print(f"\n{'='*70}")
        print(f"Query: {args.query}")
        print(f"{'='*70}\n")
        
        text = load_multiple_files(docs, auto_ocr=args.use_ocr or True)
        chunks = chunk_text_sliding_window(text, 1000, 200)
        
        retrieved_chunks = retriever(args.query, chunks, [])
        response = generate_response(args.query, retrieved_chunks)
        
        print(f"Retrieved and compressed {len(retrieved_chunks)} chunks")
        print(f"\nAI Response:\n{response['content']}\n")
        
    else:
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
        tracker.add_result(workflow_id="10", workflow_name="Contextual Compression RAG", metrics=metrics)
        tracker.save_results()
        
        # Calculate execution time
        total_time = time.time() - start_time
        
        # Print unified summary with only essential metrics
        summary_formatter = UnifiedSummaryFormatter("Contextual Compression RAG", 10)
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
