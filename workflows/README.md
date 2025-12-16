"""
Workflows - RAG Workflow Entry Points

This directory contains entry points for running different RAG workflows.
Each workflow orchestrates a complete RAG pipeline using components from workflow_parts/.

Key Design Principle:
====================

All workflows are simple entry points (~100 lines each) that:
1. Parse command-line arguments
2. Select appropriate chunker, retriever, and optional reranker functions
3. Call generic orchestration functions from workflow_parts.orchestration

This makes adding new workflows trivial - just swap the functions!

Workflows:
==========

01_simple_rag.py
├─ Entry point for: Basic RAG (sliding window chunking + semantic search)
├─ Orchestrator: run_rag_from_validation_file() with chunk_text_sliding_window, semantic_search
├─ Usage: python workflows/01_simple_rag.py --all
└─ Functions used: chunker=chunk_text_sliding_window, retriever=semantic_search

02_semantic_chunking.py [Future]
├─ Entry point for: Semantic Chunking RAG
├─ Orchestrator: run_rag_from_validation_file() with chunk_text_semantic, semantic_search
├─ Usage: python workflows/02_semantic_chunking.py
└─ Functions used: chunker=chunk_text_semantic, retriever=semantic_search

08_reranker.py [Future]
├─ Entry point for: Reranked RAG
├─ Orchestrator: run_rag_from_validation_file() with additional reranker
├─ Usage: python workflows/08_reranker.py
└─ Functions used: chunker=chunk_text_sliding_window, retriever=semantic_search, reranker=rerank_results

Generic Orchestration Functions:
================================

From workflow_parts.orchestration:

run_rag_pipeline()
├─ Single query execution
├─ Usage: result = run_rag_pipeline(
│            file_paths, query, 
│            chunker=func1, retriever=func2, reranker=optional_func3,
│            chunk_size=1000, overlap=200, k=5
│        )
└─ Returns: Dict with query, retrieved_chunks, ai_response, evaluation

run_rag_batch()
├─ Multiple queries with shared embeddings (faster)
├─ Usage: results = run_rag_batch(
│            file_paths, queries=[...],
│            chunker=func1, retriever=func2,
│            chunk_size=1000, k=5
│        )
└─ Returns: List[Dict] with results

run_rag_from_validation_file()
├─ Load queries from JSON file and run
├─ Usage: results = run_rag_from_validation_file(
│            file_paths, val_file="data/val.json",
│            chunker=func1, retriever=func2,
│            max_queries=5
│        )
└─ Returns: List[Dict] with results

print_results()
├─ Pretty-print result dictionaries
├─ Usage: print_results(results)
└─ Displays: queries, chunks, responses, evaluations, summary statistics

New Architecture (Simplified):
==============================

Instead of inheritance-based workflow classes, use function composition:

Orchestration functions handle all RAG logic:
├─ run_rag_pipeline() - single query
├─ run_rag_batch() - multiple queries (shared embeddings)
└─ run_rag_from_validation_file() - from JSON file

Each entry point just swaps functions:
├─ chunker: chunk_text_sliding_window, chunk_text_semantic, etc.
├─ retriever: semantic_search, reranker, contextual_compression, etc.
└─ reranker: optional additional processing

Example: 01_simple_rag.py
=========================

```python
from workflow_parts.orchestration import run_rag_from_validation_file, print_results
from workflow_parts.chunking import chunk_text_sliding_window
from workflow_parts.retrieval import semantic_search

def main():
    args = parse_args()
    
    # Just call the generic orchestrator with your functions!
    results = run_rag_from_validation_file(
        file_paths=args.files,
        val_file=args.val,
        chunker=chunk_text_sliding_window,
        retriever=semantic_search,
        k=args.k,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        max_queries=args.max
    )
    
    print_results(results)
```

Creating Workflow 02 (Semantic Chunking):
==========================================

```python
from workflow_parts.orchestration import run_rag_from_validation_file, print_results
from workflow_parts.chunking import chunk_text_semantic  # Different chunker!
from workflow_parts.retrieval import semantic_search

def main():
    args = parse_args()
    
    results = run_rag_from_validation_file(
        file_paths=args.files,
        val_file=args.val,
        chunker=chunk_text_semantic,  # ← Only change this
        retriever=semantic_search,
        k=args.k,
        chunk_size=args.chunk_size,
        max_queries=args.max
    )
    
    print_results(results)
```

Creating Workflow 08 (Reranker):
================================

```python
from workflow_parts.orchestration import run_rag_from_validation_file, print_results
from workflow_parts.chunking import chunk_text_sliding_window
from workflow_parts.retrieval import semantic_search
from workflow_parts.reranking import rerank_results  # Add reranker!

def main():
    args = parse_args()
    
    results = run_rag_from_validation_file(
        file_paths=args.files,
        val_file=args.val,
        chunker=chunk_text_sliding_window,
        retriever=semantic_search,
        reranker=rerank_results,  # ← Add reranking step
        k=args.k,
        max_queries=args.max
    )
    
    print_results(results)
```

Benefits of This Approach:
==========================

✅ Simple - each workflow is ~50 lines, just function calls
✅ DRY - no code duplication, all logic in orchestration.py
✅ Composable - mix and match functions easily
✅ Testable - orchestration functions are pure and parameterized
✅ Extensible - add new workflows by choosing different functions
✅ Maintainable - changes to orchestration apply to all workflows

Configuration:
===============

Environment variables (highest precedence):
├─ EMBEDDING_DEPLOYMENT: Override embedding model
└─ CHAT_DEPLOYMENT: Override chat model

CLI arguments (medium precedence):
├─ --chunk-size N: Chunk size in characters
├─ --chunk-overlap N: Chunk overlap in characters
├─ --k N: Retrieve top N chunks
└─ --max N: Process first N queries

Defaults in orchestration.py (lowest precedence):
├─ chunk_size: 1000
├─ overlap: 200
└─ k: 5

3. Environment variables (.env):
   EMBEDDING_DEPLOYMENT=text-embedding-3-large
   CHAT_DEPLOYMENT=gpt-4o

4. Hardcoded defaults in components:
   chunk_size = 1000
   embedding_model = "text-embedding-3-large"


State Management:
=================

Workflows cache expensive operations:

workflow = SimpleRAGWorkflow()

# First call: Extract PDF + Chunk + Embed (slow - API calls)
result1 = workflow.run("Query 1")

# Second call: Reuse cached chunks and embeddings (fast)
result2 = workflow.run("Query 2")

# Access cached data:
print(len(workflow.chunks))         # Already chunked
print(len(workflow.embeddings))     # Already embedded

# Reset cache if needed:
workflow.chunks = None
workflow.embeddings = None
workflow.run("Query 3")  # Will re-chunk and re-embed


Next Steps:
===========

1. SimpleRAGWorkflow is implemented and tested
2. Use it as a base class for other workflows
3. Override specific methods as needed:
   - _initialize_chunks() for chunking variants
   - run() for retrieval/generation variants
   - _initialize_embeddings() for embedding provider changes
4. Create corresponding CLI entry points
5. Update this README as new workflows are added
"""
