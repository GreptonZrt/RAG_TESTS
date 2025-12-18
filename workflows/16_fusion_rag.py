"""
Workflow 16: Fusion RAG

Implements fusion retrieval combining vector-based and keyword-based BM25 search:
- Performs both vector semantic search and BM25 keyword matching
- Normalizes scores from each approach
- Combines them with a weighted formula
- Compares single-method vs fusion retrieval performance
"""

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import discover_documents
from workflow_parts.data_loading import load_validation_data, load_multiple_files
from workflow_parts.chunking import chunk_text_sliding_window
from workflow_parts.embedding import get_embedding_fn
from workflow_parts.generation import generate_response, get_generation_client
from workflow_parts.fusion_retrieval import (
    SimpleVectorStore, create_bm25_index, fusion_retrieval,
    retrieve_vector_only, retrieve_bm25_only
)
from workflow_parts.output_formatter import ConsoleLogger, UnifiedSummaryFormatter
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


def main():
    """Main workflow execution."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Workflow 16: Fusion RAG")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max", type=int, default=None, help="Max queries to process")
    parser.add_argument("--validation-file", default="val_multi.json", 
                       help="Validation file name")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--use-ocr", action="store_true", help="Use OCR for PDFs")
    parser.add_argument("--batch", action="store_true", help="Batch mode (minimal output)")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for vector scores in fusion (0-1)")
    parser.add_argument("--compare-methods", action="store_true",
                       help="Compare vector-only, BM25-only, and fusion approaches")
    
    args = parser.parse_args()
    
    # Initialize logger (batch mode = minimal output)
    logger = ConsoleLogger("Fusion RAG", 16, verbose=True, batch_mode=args.batch)
    
    # Create embedding function
    embedding_fn = get_embedding_fn()
    
    # Discover documents
    documents = discover_documents(Path(args.data_dir))
    if not documents:
        logger.error(f"No documents found in {args.data_dir}")
        return
    
    # Load documents
    file_results = load_multiple_files(documents, use_ocr=args.use_ocr)
    text_content = "\n\n".join(file_results.values())
    
    # Chunk text
    chunks = chunk_text_sliding_window(text_content, chunk_size=1000, overlap=200)
    
    # Create embeddings
    embeddings = [embedding_fn(chunk.strip()) for chunk in chunks]
    
    # Initialize vector store
    vector_store = SimpleVectorStore()
    vector_store.add_items(chunks, embeddings)
    
    # Create BM25 index
    bm25_index = create_bm25_index(chunks)
    
    # Log initialization
    additional_info = {
        "BM25": f"Index created with {len(chunks)} documents",
        "Alpha": f"{args.alpha} (vector weight in fusion)",
    }
    logger.init(len(documents), len(chunks), len(embeddings), additional_info)
    
    # Load validation data
    val_file = Path(args.data_dir) / args.validation_file
    if not val_file.exists():
        val_file = Path(args.data_dir) / "val_multi.json"
    
    if not val_file.exists():
        logger.error(f"Validation file not found")
        return
    
    validation_data = load_validation_data(str(val_file))
    
    # Run RAG queries
    results = []
    num_queries = min(len(validation_data), args.max) if args.max else len(validation_data)
    
    for query_idx, query_data in enumerate(validation_data[:num_queries]):
        query = query_data.get("question") or query_data.get("query")
        
        # Log query
        logger.query(query_idx + 1, num_queries, query)
        
        # Retrieve using different methods
        if args.compare_methods:
            logger.info("Running Vector-Only Retrieval")
            vector_results = retrieve_vector_only(query, vector_store, embedding_fn, k=5)
            logger.retrieval("Vector-Only", vector_results)
            vector_context = [doc["text"] for doc in vector_results]
            vector_response_data = generate_response(query, vector_context)
            vector_response = vector_response_data["content"]
            logger.response(vector_response, "Vector-Only Response")
            
            logger.info("Running BM25-Only Retrieval")
            bm25_results = retrieve_bm25_only(query, chunks, bm25_index, k=5)
            logger.retrieval("BM25-Only", bm25_results)
            bm25_context = [doc["text"] for doc in bm25_results]
            bm25_response_data = generate_response(query, bm25_context)
            bm25_response = bm25_response_data["content"]
            logger.response(bm25_response, "BM25-Only Response")
        
        # Retrieve using fusion method
        logger.info(f"Running Fusion Retrieval (alpha={args.alpha})")
        fusion_results = fusion_retrieval(query, chunks, vector_store, bm25_index, 
                                         embedding_fn, k=5, alpha=args.alpha)
        logger.retrieval("Fusion", fusion_results)
        
        # Generate response from fusion results
        fusion_context = [doc["text"] for doc in fusion_results]
        response_data = generate_response(query, fusion_context)
        response = response_data["content"]
        logger.response(response, "AI Response")
        
        # Store result
        result_entry = {
            "query": query,
            "response": response,
            "retrieval_method": "Fusion",
            "alpha": args.alpha,
            "items_count": len(fusion_results),
            "response_length": len(response)
        }
        
        if args.compare_methods:
            result_entry.update({
                "vector_response": vector_response,
                "bm25_response": bm25_response,
                "fusion_response": response
            })
        
        results.append(result_entry)
    
    # Calculate execution time and metrics
    total_time = time.time() - start_time
    metrics = create_metrics_from_results(results) if results else {}
    
    # Track results
    tracker = ResultsTracker()
    tracker.add_result(
        workflow_id="16",
        workflow_name="Fusion RAG",
        metrics=metrics
    )
    tracker.save_results()
    
    # Print unified summary with only essential metrics
    summary_formatter = UnifiedSummaryFormatter("Fusion RAG", 16)
    summary = summary_formatter.format_summary(
        queries_processed=len(results),
        total_time=total_time,
        metrics=metrics
    )
    print(summary)


if __name__ == "__main__":
    main()
