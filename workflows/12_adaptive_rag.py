"""
Workflow 12: Adaptive RAG

Implements adaptive retrieval strategy:
- Classifies queries by type (Factual, Analytical, Opinion, Contextual)
- Adjusts retrieval strategy and parameters based on query type
- Uses specialized search strategies for different query categories
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import discover_documents
from workflow_parts.data_loading import load_validation_data, load_multiple_files
from workflow_parts.generation import generate_response, get_generation_client
from workflow_parts.adaptive_rag import retrieve_adaptive, classify_query, get_adaptive_parameters
from workflow_parts.embedding import get_embedding_fn
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
    parser = argparse.ArgumentParser(description="Workflow 12: Adaptive RAG")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max", type=int, default=None, help="Max queries to process")
    parser.add_argument("--validation-file", default="val_multi.json", 
                       help="Validation file name")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--use-ocr", action="store_true", help="Use OCR for PDFs")
    
    args = parser.parse_args()
    
    # Create embedding function early to avoid local scope issues
    embedding_fn = get_embedding_fn()
    
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
    embeddings = [embedding_fn(chunk.strip()) for chunk in chunks]

    print(f"\n[Batch Init] Loading documents and creating embeddings...")
    print(f"[READY] {len(chunks)} chunks, {len(embeddings)} embeddings")
    
    # Load validation data
    val_file = Path(args.data_dir) / args.validation_file
    if not val_file.exists():
        val_file = Path(args.data_dir) / "val_multi.json"
    
    if not val_file.exists():
        print(f"Error: Validation file not found")
        return
    
    validation_data = load_validation_data(str(val_file))
    
    # Create generation function
    gen_fn = create_generation_fn()
    
    # Run RAG with adaptive retrieval
    results = []
    num_queries = min(len(validation_data), args.max) if args.max else len(validation_data)
    
    for query_idx, query_data in enumerate(validation_data[:num_queries]):
        query = query_data.get("question") or query_data.get("query")
        
        print(f"\n{'='*70}")
        print(f"Query {query_idx + 1}/{num_queries}: {query}")
        print(f"{'='*70}")
        
        # Classify query and show adaptive parameters
        query_category = classify_query(query, gen_fn)
        params = get_adaptive_parameters(query_category)
        print(f"\nQuery Classification: {query_category}")
        print(f"Strategy: {params['description']}")
        
        # Retrieve chunks with adaptive strategy
        def retriever(q, c, e, k=None):
            """Retrieve using adaptive strategy."""
            return retrieve_adaptive(q, c, e, gen_fn, k)
        
        retrieved_chunks = retriever(query, chunks, embeddings)
        
        # Generate response
        response_data = generate_response(query, retrieved_chunks)
        response = response_data["content"]
        
        print(f"\nRetrieved Chunks ({len(retrieved_chunks)} total):")
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"  [{i}] {chunk[:80]}...")
        
        print(f"\nAI Response:\n{response}")
        
        results.append({
            "query": query,
            "response": response,
            "category": query_category,
            "strategy": params['strategy'],
            "chunks_count": len(retrieved_chunks),
            "response_length": len(response)
        })
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed: {len(results)}/{num_queries} queries successfully")
    
    # Show distribution of query types
    if results:
        categories = {}
        for r in results:
            cat = r['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nQuery Type Distribution:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        # Show average chunks retrieved per category
        print(f"\nAverage chunks retrieved by type:")
        for cat in ["Factual", "Analytical", "Opinion", "Contextual"]:
            cat_results = [r for r in results if r['category'] == cat]
            if cat_results:
                avg_chunks = sum(r['chunks_count'] for r in cat_results) / len(cat_results)
                print(f"  {cat}: {avg_chunks:.1f}")
    
    # Save results to tracker
    metrics = create_metrics_from_results(results)
    tracker = ResultsTracker()
    tracker.add_result(
        workflow_id="12",
        workflow_name="Adaptive RAG",
        metrics=metrics
    )
    tracker.save_results()
    
    # Print workflow metrics only
    print(f"\n[Workflow 12] Adaptive RAG")
    print(f"  Overall Score: {metrics.get('overall_score', '-'):.1f}/100")
    print(f"  Valid Response Rate: {metrics.get('valid_response_rate', '-'):.1f}%")
    print(f"  Queries Processed: {metrics.get('queries_processed', '-')}")


if __name__ == "__main__":
    main()
