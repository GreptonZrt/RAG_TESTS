# RAG Exploration Framework - AI Agent Instructions

## Project Overview
This is an **exploratory RAG (Retrieval-Augmented Generation) research repository** containing 20 numbered Jupyter notebooks that progressively explore advanced RAG techniques and variations. It's not a production application—it's a learning/experimentation workspace for evaluating different RAG architectures and improvements.

**Key characteristic**: Each notebook (01-20) implements a distinct RAG variant or improvement technique in isolation, making this a "notebook per technique" pattern rather than a unified codebase.

## Core Architecture

### The RAG Pipeline (5-Step Foundation)
All notebooks build on this fundamental pattern (see [01_simple_rag.ipynb](01_simple_rag.ipynb)):

1. **Data Ingestion** → Extract text from PDF using PyMuPDF (`fitz`)
2. **Chunking** → Split text into overlapping segments (default: 1000 chars, 200 char overlap)
3. **Embedding** → Convert chunks to vectors using OpenAI/Azure embedding models
4. **Semantic Search** → Find top-k relevant chunks via cosine similarity
5. **Generation** → Use LLM to generate response from retrieved context

### Reusable Modules (`rag_code/`)
Core utilities shared across notebooks—**modify here to affect all notebooks**:

- [config.py](rag_code/config.py): **Client initialization** — handles Azure OpenAI vs OpenAI fallback. Environment variables: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `API_VERSION`, `OPENAI_API_KEY`, `EMBEDDING_DEPLOYMENT`, `CHAT_DEPLOYMENT`
- [chunking.py](rag_code/chunking.py): Simple sliding-window chunking (`chunk_text()`)
- [embeddings.py](rag_code/embeddings.py): **API call wrapper** with retry logic, batch processing, model configuration
- [search.py](rag_code/search.py): Cosine similarity search over embeddings
- [generation.py](rag_code/generation.py): LLM response generation via chat completions
- [pdf_utils.py](rag_code/pdf_utils.py): PDF text extraction (PyMuPDF)
- [run_rag.py](rag_code/run_rag.py): End-to-end demo tying all steps together

## Notebook Progression (Technique Variants)

**Foundation (01-07)**: Basic RAG → semantic chunking → chunk sizing → context enrichment → chunk headers → doc augmentation → query transformation  
**Retrieval Enhancement (08-10)**: Reranking → RSE (Reverse Search Engineering) → contextual compression  
**Advanced Patterns (11-20)**: Feedback loops → adaptive routing → self-RAG → proposition chunking → multimodal → graph RAG → hierarchy → HyDE → CRAG

**Pattern**: Each notebook is self-contained; modifications to one notebook don't require changes elsewhere (except shared [rag_code/](rag_code/) modules).

## Critical Developer Workflows

### Environment Setup
```bash
pip install -r requirements.txt
# Create .env with: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, API_VERSION 
# OR: OPENAI_API_KEY (falls back if Azure not configured)
```

### Running Notebooks
- Open notebook in VS Code → select Python kernel → run cells sequentially
- Notebooks expect `data/val.json` (validation queries) and PDF files in `data/`
- Cells are independent; run in order but can re-run specific cells for iteration

### Testing Infrastructure
- [list_deployments.py](list_deployments.py): Azure deployment validation utility
- [data/val.json](data/val.json) & [data/val_rl.json](data/val_rl.json): Query validation datasets
- No formal test suite; evaluation is manual in notebooks

## Key Patterns & Conventions

### Configuration Hierarchy
1. **Hardcoded defaults** in [config.py](rag_code/config.py): `text-embedding-3-large`, `gpt-4o`
2. **Environment variables override** (see [embeddings.py](rag_code/embeddings.py#L34) for `EMBEDDING_DEPLOYMENT`)
3. **Notebook-local overrides**: Each notebook can redefine deployment names at the top (see [01_simple_rag.ipynb](01_simple_rag.ipynb) cell 1)

### Error Handling Pattern
[embeddings.py](rag_code/embeddings.py) exemplifies the retry + diagnostic approach:
- Exponential backoff for API failures
- Specific error hints (e.g., "Deployment not found" → check Azure Portal)
- Print debugging is primary; no logging framework

### Data Structures
- `EmbeddingItem` dataclass: `{embedding: List[float]}`
- Search returns: `List[str]` (text chunks, not scored tuples)
- No ORM or schemas; plain JSON for validation data

## Integration Points & Dependencies

### External APIs
- **OpenAI/Azure**: Embeddings (`text-embedding-3-large`) and chat (`gpt-4o`)
- Fallback chain: Try Azure → fallback to OpenAI → raise if neither configured
- Batch processing in embeddings (32-item default batches with retry)

### File I/O
- PDFs: [data/](data/) directory (e.g., `AI_Information.pdf`)
- Validation queries: [data/val.json](data/val.json) (list of `{question, ...}` dicts)
- .env file for secrets (not in repo)

### Key Dependencies
- **faiss-cpu**, **scikit-learn**, **networkx**: Graph/vector operations
- **transformers**, **accelerate**: Hugging Face models for advanced techniques
- **rdflib**, **pyvis**: RDF/graph visualization for graph RAG notebooks

## When Modifying Core Code

1. **Change chunking strategy** → Edit [chunking.py](rag_code/chunking.py), all notebooks auto-inherit
2. **Add API wrapper feature** → Extend [embeddings.py](rag_code/embeddings.py) or [generation.py](rag_code/generation.py)
3. **Swap embedding model** → Set `EMBEDDING_DEPLOYMENT` env var or override in notebook
4. **New RAG variant** → Create new notebook (`21_variant.ipynb`); import from [rag_code/](rag_code/)

## Common Debugging Steps

- **"Deployment not found"** → Check Azure Portal deployments match `EMBEDDING_DEPLOYMENT` / `CHAT_DEPLOYMENT`
- **DNS/TCP connect errors** → [run_rag.py](rag_code/run_rag.py) has built-in network diagnostics
- **No results from semantic search** → Verify embeddings API succeeded; check chunk count vs query embedding
- **Cells execute but produce no output** → Add `print()` statements; notebooks default to quiet execution
