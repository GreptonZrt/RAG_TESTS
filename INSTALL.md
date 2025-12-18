# 🚀 Installation & Setup

> Rövid útmutató - 5 perc alatt kész

---

## ✅ Előfeltételek

- Python 3.10+
- Git

---

## 📋 Telepítés

### 1. Clone

```bash
git clone <repository-url>
cd RAG_tests
```

### 2. Virtual Environment (opcionális, de ajánlott)

```bash
# Létrehozás
python -m venv venv

# Aktiválás
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate
```

### 3. Függőségek

```bash
pip install -r requirements.txt
```

**Idő**: 5-10 perc

### 4. API Konfiguráció

Másolj egy `.env` fájlt `.env.example`-ből:

```bash
cp .env.example .env
```

Szerkeszd a `.env` fájlt és add meg az alábbiak közül az egyiket:

**Azure OpenAI (ajánlott):**
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
API_VERSION=2024-12-01
EMBEDDING_DEPLOYMENT=text-embedding-3-large
CHAT_DEPLOYMENT=gpt-4o
```

**VAGY OpenAI (fallback):**
```env
OPENAI_API_KEY=sk-your-key
```

### 5. Teszt

```bash
# Egy workflow tesztelése
python workflows/01_simple_rag.py --max 1
```

**Elvárt kimenet**: Egy válasz a kérdésre + WORKFLOW SUMMARY

---

## 🧪 Összes Workflow Futtatása

```bash
# Összes workflow (10 query mindegyik)
python run_all_workflows_batch.py

# Csak 3 query-vel (gyorsabb)
python run_all_workflows_batch.py --max 3

# Specifikus workflow-k
python run_all_workflows_batch.py --run "01,02,04" --max 1
```

---

## 🐛 Gyakori Problémák

| Hiba | Megoldás |
|------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `No module named 'sklearn'` | `pip install scikit-learn` |
| `AZURE_OPENAI_ENDPOINT not set` | Szerkeszd a `.env` fájlt |
| `AuthenticationError` | Ellenőrizd az API kulcsot |

---

## 📚 Workflow-k

13 RAG workflow érhető el:

```bash
python workflows/01_simple_rag.py
python workflows/02_semantic_chunking.py
python workflows/04_context_enriched_rag.py
python workflows/05_contextual_chunk_headers_rag.py
python workflows/06_doc_augmentation_rag.py
python workflows/08_reranker.py
python workflows/10_contextual_compression.py
python workflows/11_feedback_loop_rag.py
python workflows/12_adaptive_rag.py
python workflows/13_self_rag.py
python workflows/14_proposition_chunking_rag.py
python workflows/16_fusion_rag.py
python workflows/19_hyde_rag.py
```

Mindegyik támogatja a `--max N` flag-et a query-k számához.

---

## 📊 Eredmények

```bash
# Eredmények megtekintése
python print_results.py

# CSV fájl: workflow_results.csv
