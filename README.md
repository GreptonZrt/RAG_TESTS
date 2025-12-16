# RAG Workflow Framework

## Gyors Start

```bash
# Telepítés
pip install -r requirements.txt

# 01 - Simple RAG futtatása (alapértelmezett: első query)
python workflows/01_simple_rag.py

# Összes query feldolgozása
python workflows/01_simple_rag.py --all

# Egyedi query
python workflows/01_simple_rag.py --query "What is AI?"

# Első 5 query
python workflows/01_simple_rag.py --max 5
```

## Struktúra

```
workflow_parts/              - Reusable components
├─ data_loading.py
├─ chunking.py
├─ embedding.py
├─ retrieval.py
├─ generation.py
└─ evaluation.py

workflows/                   - Workflow implementations
├─ simple_rag_workflow.py    - Base SimpleRAGWorkflow class
├─ 01_simple_rag.py          - 01 workflow CLI
├─ 02_semantic_chunking.py   - 02 workflow CLI (future)
└─ ...

data/
├─ AI_Information.pdf
├─ val.json
└─ val_rl.json
```

## Workflow-k

| # | Notebook | Workflow | CLI | Stratégia |
|---|----------|----------|-----|-----------|
| 01 | simple_rag.ipynb | simple_rag_workflow.py | workflows/01_simple_rag.py | Sliding window chunking |
| 02 | semantic_chunking.ipynb | semantic_chunking_rag_workflow.py | workflows/02_semantic_chunking.py | Semantic chunking |
| ... | ... | ... | ... | ... |

## Komponensek

### workflow_parts/

- **data_loading.py** - PDF extraction, validation data
- **chunking.py** - Text segmentation strategies
- **embedding.py** - Embedding API (Azure/OpenAI)
- **retrieval.py** - Semantic search
- **generation.py** - LLM response generation
- **evaluation.py** - Response evaluation

Lásd: `workflow_parts/README.md`

## Workflow Development

Új workflow létrehozása:

```python
# workflows/02_semantic_chunking_rag_workflow.py
from workflows.simple_rag_workflow import SimpleRAGWorkflow

class SemanticChunkingRAGWorkflow(SimpleRAGWorkflow):
    def _initialize_chunks(self):
        self.text = extract_text_from_pdf(self.pdf_path)
        self.chunks = chunk_text_semantic(self.text)
```

Lásd: `workflows/README.md`

## Dokumentáció

- **workflow_parts/README.md** - Komponensek dokumentációja
- **workflows/README.md** - Workflow fejlesztési útmutató
