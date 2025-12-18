"""
Workflow 13: Self-RAG

Implements self-reflective retrieval-augmented generation:
- Determines if retrieval is necessary
- Assesses relevance of retrieved chunks
- Evaluates support of response by context
- Rates utility of generated response
- Iterates if needed for better quality
"""

import sys
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from workflow_parts.orchestration import discover_documents
from workflow_parts.data_loading import load_validation_data, load_multiple_files
from workflow_parts.self_rag import self_rag_orchestrate
from workflow_parts.output_formatter import UnifiedSummaryFormatter
from workflow_parts.embedding import get_embedding_fn
from workflow_parts.results_tracker import ResultsTracker, create_metrics_from_results


def create_generation_fn():
    """Create LLM generation function for Self-RAG classification."""
    from workflow_parts.generation import get_generation_client
    
    client = get_generation_client()
    
    def gen_fn(system_prompt: str, user_prompt: str) -> dict:
        """Call LLM for Self-RAG decisions."""
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


def create_response_generation_fn():
    """Create response generation function."""
    from workflow_parts.generation import generate_response
    
    def response_gen(system_prompt, user_prompt):
        """Generate response with LLM."""
        # Extract context and question from user prompt
        if "Context:" in user_prompt and "Question:" in user_prompt:
            parts = user_prompt.split("Question:")
            context = parts[0].replace("Context:", "").strip()
            question = parts[1].strip()
            
            response_obj = generate_response(question, [context])
            return {
                "content": response_obj.get("content", ""),
                "response_obj": response_obj.get("response_obj")
            }
        
        return {"content": "Unable to generate response", "response_obj": None}
    
    return response_gen


def main():
    """Main workflow execution."""
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Workflow 13: Self-RAG")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--max", type=int, default=None, help="Max queries to process")
    parser.add_argument("--validation-file", default="val_multi.json", 
                       help="Validation file name")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--use-ocr", action="store_true", help="Use OCR for PDFs")
    parser.add_argument("--batch", action="store_true", help="Batch mode (minimal output)")
    
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
    
    # Create generation functions
    gen_fn = create_generation_fn()
    response_gen_fn = create_response_generation_fn()
    
    # Run Self-RAG with reflection
    results = []
    num_queries = min(len(validation_data), args.max) if args.max else len(validation_data)
    
    for query_idx, query_data in enumerate(validation_data[:num_queries]):
        query = query_data.get("question") or query_data.get("query")
        
        print(f"\n{'='*70}")
        print(f"Query {query_idx + 1}/{num_queries}: {query}")
        print(f"{'='*70}")
        
        # Run Self-RAG orchestration
        result = self_rag_orchestrate(query, chunks, embeddings, gen_fn, response_gen_fn)
        
        # Display results
        print(f"\n[Reflection]")
        print(f"  Retrieval needed: {result['retrieval_needed']}")
        if result['metadata']:
            print(f"  Iterations: {result['metadata'].get('iterations', 0)}")
            print(f"  Initial chunks: {result['metadata'].get('initial_chunks_count', 0)}")
            print(f"  Filtered chunks: {result['metadata'].get('filtered_chunks_count', 0)}")
        print(f"  Support level: {result['support_level']}")
        print(f"  Utility rating: {result['utility_rating']}/5")
        
        if result['retrieved_chunks']:
            print(f"\nRetrieved Chunks ({len(result['retrieved_chunks'])} total):")
            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                print(f"  [{i}] {chunk[:80]}...")
        
        print(f"\nAI Response:\n{result['response']}")
        
        results.append({
            "query": query,
            "response": result['response'],
            "retrieval_needed": result['retrieval_needed'],
            "chunks_count": len(result['retrieved_chunks']),
            "support_level": result['support_level'],
            "utility_rating": result['utility_rating'],
            "iterations": result['metadata'].get('iterations', 0) if result['metadata'] else 0
        })
    
    # Save results to tracker and calculate metrics
    metrics = create_metrics_from_results(results)
    tracker = ResultsTracker()
    tracker.add_result(
        workflow_id="13",
        workflow_name="Self-RAG",
        metrics=metrics
    )
    tracker.save_results()
    
    # Calculate execution time
    total_time = time.time() - start_time
    
    # Print unified summary with only essential metrics
    summary_formatter = UnifiedSummaryFormatter("Self-RAG", 13)
    summary = summary_formatter.format_summary(
        queries_processed=len(results),
        total_time=total_time,
        metrics=metrics
    )
    print(summary)


if __name__ == "__main__":
    main()
