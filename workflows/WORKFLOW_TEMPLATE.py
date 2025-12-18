"""
Standardized Workflow Template

This template shows how to use the unified output formatter
for consistent logging and output across all RAG workflows.

Use this as a reference when creating or updating workflows.
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import discover_documents
from workflow_parts.data_loading import load_validation_data, load_multiple_files
from workflow_parts.chunking import chunk_text_sliding_window
from workflow_parts.embedding import get_embedding_fn
from workflow_parts.generation import generate_response
from workflow_parts.output_formatter import ConsoleLogger
from workflow_parts.results_tracker import ResultsTracker


def main():
    """Main workflow execution with standardized logging."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Standardized RAG Workflow Template")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max", type=int, default=None, help="Max queries to process")
    parser.add_argument("--validation-file", default="val_multi.json", help="Validation file name")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--use-ocr", action="store_true", help="Use OCR for PDFs")
    args = parser.parse_args()
    
    # Initialize logger (customize workflow name and number)
    WORKFLOW_NAME = "Your Workflow Name"
    WORKFLOW_NUMBER = 99
    logger = ConsoleLogger(WORKFLOW_NAME, WORKFLOW_NUMBER, verbose=True)
    
    try:
        # ========== PHASE 1: LOAD AND PREPARE DATA ==========
        documents = discover_documents(Path(args.data_dir))
        if not documents:
            logger.error(f"No documents found in {args.data_dir}")
            return
        
        file_results = load_multiple_files(documents, use_ocr=args.use_ocr)
        text_content = "\n\n".join(file_results.values())
        
        # ========== PHASE 2: CHUNK AND EMBED ==========
        chunks = chunk_text_sliding_window(text_content, chunk_size=1000, overlap=200)
        embedding_fn = get_embedding_fn()
        embeddings = [embedding_fn(chunk.strip()) for chunk in chunks]
        
        # Log initialization
        logger.init(len(documents), len(chunks), len(embeddings))
        
        # ========== PHASE 3: LOAD QUERIES ==========
        val_file = Path(args.data_dir) / args.validation_file
        if not val_file.exists():
            val_file = Path(args.data_dir) / "val_multi.json"
        
        if not val_file.exists():
            logger.error("Validation file not found")
            return
        
        validation_data = load_validation_data(str(val_file))
        num_queries = min(len(validation_data), args.max) if args.max else len(validation_data)
        
        # ========== PHASE 4: PROCESS QUERIES ==========
        results = []
        
        for query_idx, query_data in enumerate(validation_data[:num_queries]):
            query = query_data.get("question") or query_data.get("query")
            
            # Log query start
            logger.query(query_idx + 1, num_queries, query)
            
            # ===== YOUR RETRIEVAL LOGIC HERE =====
            # Example: retrieved_items = your_retrieval_function(query, chunks, embeddings)
            # logger.retrieval("Your Method Name", retrieved_items)
            
            # Generate response
            # response_data = generate_response(query, [doc["text"] for doc in retrieved_items])
            # response = response_data["content"]
            # logger.response(response)
            
            # (This is placeholder - implement your actual RAG logic)
            retrieved_items = []
            response = "Your response here"
            
            results.append({
                "query": query,
                "response": response,
                "items_count": len(retrieved_items),
                "response_length": len(response)
            })
        
        # ========== PHASE 5: COMPLETE AND TRACK ==========
        logger.complete(num_queries)
        
        # Save results if evaluation is enabled
        if not args.no_eval:
            tracker = ResultsTracker()
            tracker.add_result(
                workflow_id=str(WORKFLOW_NUMBER),
                workflow_name=WORKFLOW_NAME,
                metrics={
                    "queries_processed": num_queries,
                    "avg_response_length": sum(r["response_length"] for r in results) / len(results) if results else 0,
                }
            )
            tracker.save_results()
    
    except Exception as e:
        logger.error(str(e), context="Unexpected error during workflow execution")
        raise


if __name__ == "__main__":
    main()
