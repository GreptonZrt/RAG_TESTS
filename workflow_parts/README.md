"""
Workflow Parts - Reusable RAG Components

This directory contains modular building blocks for RAG workflows.
Each file contains functions for a specific step in the RAG pipeline.

Components:
- data_loading.py: PDF extraction and validation data loading
- chunking.py: Text segmentation strategies
- embedding.py: Vector embedding creation with retry logic
- retrieval.py: Semantic search and chunk retrieval
- generation.py: LLM response generation
- evaluation.py: Response evaluation against ground truth

Architecture:
==============

Each component is independent and can be imported by any workflow.
For example, if you want to create a semantic chunking variant:

    from workflow_parts.data_loading import extract_text_from_pdf
    from workflow_parts.chunking import chunk_text_semantic  # (future)
    from workflow_parts.embedding import create_embeddings
    from workflow_parts.retrieval import semantic_search

Usage Example:
===============

# In workflows/semantic_chunking_rag_workflow.py
from workflow_parts.data_loading import extract_text_from_pdf
from workflow_parts.chunking import chunk_text_semantic
from workflow_parts.embedding import create_embeddings

class SemanticChunkingRAGWorkflow(SimpleRAGWorkflow):
    def _initialize_chunks(self):
        '''Override chunking strategy'''
        self.text = extract_text_from_pdf(self.pdf_path)
        self.chunks = chunk_text_semantic(self.text)  # Different chunking!

Extension Pattern:
===================

To add a new component or variant:

1. Add a new function to the appropriate file:
   - If it's a different chunking strategy: add to chunking.py
   - If it's a new embedding provider: add to embedding.py
   - etc.

2. Name variants clearly:
   - chunk_text_sliding_window()
   - chunk_text_semantic()
   - chunk_text_by_sentence()
   - create_embeddings_local()  # for local models
   - create_embeddings_api()    # for cloud APIs

3. Keep function signatures consistent when possible
   - Take same inputs
   - Return same types
   - Makes swapping strategies easy in workflows

File Organization:
===================

data_loading.py
├─ extract_text_from_pdf(pdf_path) → str
├─ load_validation_data(val_file) → List[Dict]
└─ extract_queries_from_validation_data() → List[str]

chunking.py
├─ chunk_text_sliding_window(text, chunk_size, overlap) → List[str]
├─ chunk_text_semantic(text, model) → List[str]  # Future
└─ chunk_text_by_sentence(text, language) → List[str]  # Future

embedding.py
├─ get_embedding_client() → Client
└─ create_embeddings(texts, deployment_name) → List[EmbeddingItem]

retrieval.py
├─ cosine_similarity(vec1, vec2) → float
└─ semantic_search(query, chunks, embeddings, k) → List[str]

generation.py
├─ get_generation_client() → Client
└─ generate_response(query, context_chunks, ...) → Dict

evaluation.py
└─ evaluate_response(query, ai_response, ideal_answer) → Dict
"""
