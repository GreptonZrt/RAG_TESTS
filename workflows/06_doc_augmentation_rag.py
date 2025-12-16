#!/usr/bin/env python3
"""
Workflow 06: Document Augmentation RAG

Augments each chunk by generating relevant questions using an LLM.
These questions become part of the searchable content, improving retrieval
by allowing the model to find chunks via question similarity.

Usage:
    python workflows/06_doc_augmentation_rag.py --max 1 --no-eval --num-questions 3
"""

import sys
import os
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
from workflow_parts.generation import generate_response, get_generation_client
from workflow_parts.doc_augmentation import (
    augment_chunks_with_questions,
    augmented_semantic_search
)
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results


def create_generation_fn():
    """
    Creates a function that calls LLM for question generation.
    
    Returns:
        Function that takes (system_prompt, user_prompt) and returns generated text.
    """
    client = get_generation_client()
    
    def gen_fn(system_prompt: str, user_prompt: str) -> dict:
        """Generate text using LLM."""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
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


def create_augmented_retriever(num_questions: int = 5, question_weight: float = 0.5, k: int = 5):
    """
    Creates a retriever function that uses augmented semantic search with questions.
    
    Args:
        num_questions: Number of questions to generate per chunk.
        question_weight: Weight for question similarity (0.0-1.0).
        k: Number of results to retrieve.
    
    Returns:
        Function that retrieves using augmented search.
    """
    # Pre-augment chunks cache
    chunks_cache = {}
    
    def retriever(query, chunks, embeddings, k=None):
        """Retrieve using augmented semantic search with questions."""
        
        # Augment chunks if not already done
        cache_key = hash(tuple(chunks))
        if cache_key not in chunks_cache:
            print("  [Augment] Generating questions for chunks...")
            
            gen_fn = create_generation_fn()
            embed_fn = get_embedding_fn()
            
            augmented_chunks = augment_chunks_with_questions(
                chunks=chunks,
                get_embedding_fn=embed_fn,
                get_generation_fn=gen_fn,
                num_questions=num_questions
            )
            chunks_cache[cache_key] = augmented_chunks
        else:
            augmented_chunks = chunks_cache[cache_key]
        
        # Search using augmented semantic search
        results = augmented_semantic_search(
            query=query,
            augmented_chunks=augmented_chunks,
            get_embedding_fn=get_embedding_fn(),
            k=k or 5,
            question_weight=question_weight
        )
        
        return results
    
    return retriever


def main():
    """Main entry point for workflow 06."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Document Augmentation RAG Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflows/06_doc_augmentation_rag.py --max 3
  python workflows/06_doc_augmentation_rag.py --num-questions 5 --question-weight 0.7 --no-eval
  python workflows/06_doc_augmentation_rag.py --query "What is AI?"
        """
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of queries to process (default: all)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top chunks to retrieve (default: 5)")
    parser.add_argument("--num-questions", type=int, default=5,
                        help="Number of questions to generate per chunk (default: 5)")
    parser.add_argument("--question-weight", type=float, default=0.5,
                        help="Weight for question similarity: 0.0 (text-only) to 1.0 (questions-only) (default: 0.5)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single custom query instead of from validation file")
    parser.add_argument("--all", action="store_true",
                        help="Process all queries in validation file")
    parser.add_argument("--use-ocr", action="store_true",
                        help="Force OCR for PDF extraction")
    
    args = parser.parse_args()
    
    # Validate question weight
    if not (0.0 <= args.question_weight <= 1.0):
        print("[ERROR] --question-weight must be between 0.0 and 1.0")
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
    
    # Create the retriever with augmented search
    retriever = create_augmented_retriever(
        num_questions=args.num_questions,
        question_weight=args.question_weight,
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
        retrieved_chunks = retriever(args.query, chunks, [])
        
        # Generate response
        context_text = "\n".join(retrieved_chunks)
        response = generate_response(args.query, retrieved_chunks)
        
        print(f"Retrieved {len(retrieved_chunks)} chunks (with augmentation)")
        print(f"\nAI Response:\n{response['content']}\n")
        
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
        
        # Track results
        metrics = create_metrics_from_results(results)
        tracker = ResultsTracker()
        tracker.add_result(workflow_id="06", workflow_name="Doc Augmentation RAG", metrics=metrics)
        tracker.save_results()
        
        # Print workflow metrics only
        print(f"\n[Workflow 06] Doc Augmentation RAG")
        print(f"  Overall Score: {metrics.get('overall_score', '-'):.1f}/100")
        print(f"  Valid Response Rate: {metrics.get('valid_response_rate', '-'):.1f}%")
        print(f"  Queries Processed: {metrics.get('queries_processed', '-')}")


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
