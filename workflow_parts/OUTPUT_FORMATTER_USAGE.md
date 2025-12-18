## Standardized Output Formatting

The `output_formatter.py` module provides standardized logging and output formatting for all RAG workflows. This ensures consistent formatting regardless of the underlying RAG strategy.

### Usage

#### Basic Logger Setup

```python
from workflow_parts.output_formatter import ConsoleLogger

# Initialize logger for your workflow
logger = ConsoleLogger("Your Workflow Name", workflow_number=16, verbose=True)
```

#### Logging Initialization

Log workflow initialization with documents, chunks, and embeddings:

```python
logger.init(
    document_count=5,
    chunk_count=1000,
    embedding_count=1000,
    additional_info={
        "Method": "Vector + BM25 Fusion",
        "Alpha": "0.5"
    }
)
```

Output:
```
======================================================================
Workflow 16: Your Workflow Name
======================================================================

[Init] Starting workflow initialization at 14:23:45
[Documents] Loaded 5 document(s)
[Chunks] Created 1000 chunk(s)
[Embeddings] Generated 1000 embedding(s)
[Method] Vector + BM25 Fusion
[Alpha] 0.5
[READY] Workflow ready to process queries
```

#### Logging Queries

Log query headers before processing:

```python
logger.query(query_num=1, total_queries=5, query_text="Your question here")
```

Output:
```
======================================================================
Query 1/5
======================================================================

Your question here
```

#### Logging Retrieval Results

Log retrieved items with scores:

```python
logger.retrieval("Vector-Only", retrieved_items)
```

Where `retrieved_items` is a list of dicts with:
- `"text"`: The retrieved text
- `"similarity"`: Vector similarity score (optional)
- `"bm25_score"`: BM25 score (optional)
- `"combined_score"`: Combined score (optional)

Output:
```
[Retrieval: Vector-Only] (5 items)
----------------------------------------

  [1] First retrieved document text... (sim: 0.821)
  [2] Second retrieved document text... (sim: 0.754)
  [3] Third retrieved document text... (sim: 0.712)
  [4] Fourth retrieved document text... (sim: 0.689)
  [5] Fifth retrieved document text... (sim: 0.671)
```

#### Logging Responses

Log LLM responses:

```python
logger.response(response="Your generated response...", method="Fusion Response")
```

Output:
```
[Fusion Response]
----------------------------------------
Your generated response...
```

#### Logging Metrics

Log performance metrics:

```python
logger.metrics({
    "items_retrieved": 5,
    "avg_score": 0.754,
    "response_length": 250
})
```

Output:
```
[Metrics]
----------------------------------------
  items_retrieved: 5
  avg_score: 0.754
  response_length: 250
```

#### Logging Completion

Log workflow completion:

```python
logger.complete(queries_processed=5, total_time=45.2)
```

Output:
```
======================================================================
Workflow Completion
======================================================================

[Completed] Processed 5 queries
[Time] Total execution time: 45.20s
[Speed] Average 9.04s per query
[Timestamp] 2025-12-18 14:25:30
```

#### Error and Warning Logging

```python
logger.error("Something went wrong", context="During retrieval")
logger.warning("This might cause issues later")
logger.info("Processing completed successfully", tag="SUCCESS")
```

### Implementation in Workflows

#### Template Structure

```python
from workflow_parts.output_formatter import ConsoleLogger

def main():
    logger = ConsoleLogger("Workflow Name", WORKFLOW_NUMBER, verbose=True)
    
    try:
        # Load and prepare data
        documents = discover_documents(...)
        chunks = chunk_text(...)
        embeddings = [embed(c) for c in chunks]
        
        # Log initialization
        logger.init(len(documents), len(chunks), len(embeddings))
        
        # Process queries
        for i, query in enumerate(queries):
            logger.query(i+1, len(queries), query)
            
            # Retrieval
            items = retrieve(query, chunks)
            logger.retrieval("Your Method", items)
            
            # Generation
            response = generate(query, items)
            logger.response(response)
        
        # Log completion
        logger.complete(len(queries))
        
    except Exception as e:
        logger.error(str(e))
        raise
```

### Benefits

1. **Consistency**: All workflows output in the same format
2. **Readability**: Clear, structured output with headers and separators
3. **Flexibility**: Easy to customize for different retrieval methods
4. **Scalability**: Easily add new logging types without affecting existing code
5. **Maintainability**: Centralized formatter makes changes affect all workflows

### Key Methods

| Method | Purpose | Parameters |
|--------|---------|-----------|
| `init()` | Log initialization | docs, chunks, embeddings, info |
| `query()` | Log query | num, total, text |
| `retrieval()` | Log retrieved items | method, items |
| `response()` | Log response | response, method |
| `metrics()` | Log metrics | metrics dict |
| `complete()` | Log completion | queries_count, time |
| `error()` | Log error | message, context |
| `warning()` | Log warning | message |
| `info()` | Log info | message, tag |

### Migration Guide

To update an existing workflow to use the formatter:

1. **Add import**:
   ```python
   from workflow_parts.output_formatter import ConsoleLogger
   ```

2. **Initialize logger**:
   ```python
   logger = ConsoleLogger("Your Name", WORKFLOW_NUM)
   ```

3. **Replace print statements**:
   - `print(...)` → `logger.info(...)`
   - `print(f"Error: {e}")` → `logger.error(str(e))`

4. **Use structured methods**:
   - Replace retrieval output with `logger.retrieval(...)`
   - Replace response output with `logger.response(...)`
   - Replace metrics output with `logger.metrics(...)`

### Example: Complete Workflow

See [WORKFLOW_TEMPLATE.py](../workflows/WORKFLOW_TEMPLATE.py) for a complete example using the standardized formatter.
