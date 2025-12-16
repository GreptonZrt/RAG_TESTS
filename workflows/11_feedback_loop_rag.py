"""
Workflow 11: Feedback Loop RAG

Implements retrieval with feedback-based adjustment:
- Tracks relevance of chunks across queries
- Adjusts future retrievals based on feedback history
- Improves ranking of previously positive chunks
"""

import sys
import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import discover_documents, run_rag_from_validation_file
from workflow_parts.data_loading import load_validation_data, load_multiple_files
from workflow_parts.feedback_loop import (
    FeedbackStore, retrieve_with_feedback, simulate_feedback
)
from workflow_parts.generation import generate_response
from workflow_parts.embedding import get_embedding_fn
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results
from workflow_parts.embedding import get_embedding_fn
from workflow_parts.generation import generate_response




def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(description="Workflow 11: Feedback Loop RAG")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max", type=int, default=None, help="Max queries to process")
    parser.add_argument("--validation-file", default="val_multi.json", 
                       help="Validation file name")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--use-ocr", action="store_true", help="Use OCR for PDFs")
    parser.add_argument("--num-feedbacks", type=int, default=1,
                       help="Number of feedback iterations")
    
    args = parser.parse_args()
    
    # Discover documents
    documents = discover_documents(Path(args.data_dir))
    if not documents:
        print(f"No documents found in {args.data_dir}")
        return
    
    # Load documents
    file_results = load_multiple_files(documents, use_ocr=args.use_ocr)
    text_content = "\n\n".join(file_results.values())
    
    # Chunk text
    from workflow_parts.chunking import chunk_text_sliding_window
    chunks = chunk_text_sliding_window(text_content, chunk_size=1000, overlap=200)
    
    # Create embeddings
    embeddings = [get_embedding_fn()(chunk.strip()) for chunk in chunks]
    
    print(f"\n[Feedback Init] Loaded {len(chunks)} chunks")
    
    # Load validation data
    val_file = Path(args.data_dir) / args.validation_file
    if not val_file.exists():
        val_file = Path(args.data_dir) / "val_multi.json"
    
    if not val_file.exists():
        print(f"Error: Validation file not found")
        return
    
    validation_data = load_validation_data(str(val_file))
    
    # Initialize feedback store
    feedback_store = FeedbackStore()
    
    # Create retriever
    def retriever(query, chunks, embeddings, k=None):
        """Retrieve with feedback-based scoring."""
        return retrieve_with_feedback(query, chunks, embeddings, feedback_store, k)
    
    # Run RAG with feedback
    feedback_results = []
    num_queries = min(len(validation_data), args.max) if args.max else len(validation_data)
    
    for query_idx, query_data in enumerate(validation_data[:num_queries]):
        query = query_data.get("question") or query_data.get("query")
        expected_answer = query_data.get("answer", "")
        
        print(f"\n{'='*70}")
        print(f"Query {query_idx + 1}/{num_queries}: {query}")
        print(f"{'='*70}")
        
        # Retrieve chunks
        retrieved_chunks = retriever(query, chunks, embeddings, k=5)
        
        # Generate response
        response_data = generate_response(query, retrieved_chunks)
        response = response_data["content"]
        
        print(f"\nRetrieved Chunks ({len(retrieved_chunks)} total):")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  [{i}] {chunk[:80]}...")
        
        print(f"\nAI Response:\n{response}")
        
        # Simulate feedback iterations
        for fb_iter in range(args.num_feedbacks):
            # Simulate user feedback
            is_good = expected_answer.lower() in response.lower() or len(response) > 100
            feedback = simulate_feedback(query, response, is_good_response=is_good)
            
            # Record feedback
            feedback_store.add_feedback(
                query=query,
                response=response,
                chunks=retrieved_chunks,
                relevance_score=feedback["relevance_score"],
                quality_score=feedback["quality_score"],
                comments=feedback["comments"]
            )
            
            print(f"\n[Feedback Iteration {fb_iter + 1}]")
            print(f"  Relevance: {feedback['relevance_score']:.2f}")
            print(f"  Quality: {feedback['quality_score']:.2f}")
            print(f"  Comments: {feedback['comments']}")
        
        feedback_results.append({
            "query": query,
            "response": response,
            "chunks_count": len(retrieved_chunks),
            "feedback_count": args.num_feedbacks
        })
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed: {len(feedback_results)}/{num_queries} queries successfully")
    print(f"Total feedback entries: {len(feedback_store.feedback_history)}")
    print(f"Chunks with feedback: {len(feedback_store.chunk_feedback_count)}")
    
    # Print top-scored chunks based on feedback
    if feedback_store.chunk_relevance_scores:
        print(f"\nTop chunks by feedback score:")
        chunk_scores = []
        for chunk_key, scores in feedback_store.chunk_relevance_scores.items():
            avg_score = sum(scores) / len(scores)
            chunk_scores.append((chunk_key, avg_score, len(scores)))
        
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (chunk_key, score, count) in enumerate(chunk_scores[:5], 1):
            print(f"  [{i}] Score: {score:.2f}, Feedbacks: {count}")
    
    # Save results to tracker
    metrics = create_metrics_from_results(feedback_results)
    tracker = ResultsTracker()
    tracker.add_result(
        workflow_id="11",
        workflow_name="Feedback Loop RAG",
        metrics=metrics
    )
    tracker.save_results()
    
    # Print workflow metrics only
    print(f"\n[Workflow 11] Feedback Loop RAG")
    print(f"  Overall Score: {metrics.get('overall_score', '-'):.1f}/100")
    print(f"  Valid Response Rate: {metrics.get('valid_response_rate', '-'):.1f}%")
    print(f"  Queries Processed: {metrics.get('queries_processed', '-')}")


if __name__ == "__main__":
    main()
