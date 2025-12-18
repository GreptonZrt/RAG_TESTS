## Batch Mode - Standardized Output Formatting

All workflows now support a `--batch` flag for minimal output mode, perfect for automated testing and CI/CD environments.

### Running Workflows

#### Normal Mode (Interactive - Default)
Run with formatted output, detailed logs for each query:

```bash
# Single workflow - beautiful formatted output
python workflows/16_fusion_rag.py --max 1

# Output shows:
# - Initialization (docs, chunks, embeddings)
# - Query header
# - Retrieved items with scores
# - Generated response
# - Metrics
# - Completion summary
```

#### Batch Mode (Testing - Minimal Output)
Run with minimal output, only shows pass/fail:

```bash
# Single workflow - csendes futás
python workflows/16_fusion_rag.py --max 1 --batch --no-eval

# Minimal output - only essential info
```

#### All Tests at Once
Run all workflows with automated testing framework:

```bash
# Runs ALL workflows in batch mode with --max 1 --no-eval
python test_all_workflows.py

# Output example:
# Testing 01_simple_rag... [OK]
# Testing 02_semantic_chunking... [OK]
# Testing 14_proposition_chunking_rag... [OK]
# Testing 16_fusion_rag... [OK]
# ...
```

### Batch Mode Behavior

When `--batch` flag is used:

1. **Suppressed Output**: No query-by-query output during processing
2. **Errors Still Shown**: If a workflow fails, errors are displayed
3. **Fast Execution**: Minimal console I/O overhead
4. **Final Summary**: Results still saved to CSV tracking file

### Use Cases

| Use Case | Command | Output |
|----------|---------|--------|
| **Development** | `python workflows/XX_name.py --max 3` | Detailed, formatted |
| **Single Test** | `python workflows/XX_name.py --max 1 --batch --no-eval` | Minimal |
| **CI/CD Testing** | `python test_all_workflows.py` | Status line per workflow |
| **Full Evaluation** | `python workflows/XX_name.py --all` | Detailed, all queries |

### Implementation Details

#### New Workflows (14, 16 with ConsoleLogger)
These workflows fully respect batch mode:

```python
# They use the new ConsoleLogger with batch_mode parameter
logger = ConsoleLogger("Workflow Name", 16, batch_mode=args.batch)

# In batch_mode=True:
# - logger.query() doesn't print
# - logger.retrieval() doesn't print
# - logger.response() doesn't print
# - Only logger.init() and logger.complete() print
```

#### Existing Workflows (01-13, 19)
These workflows have the `--batch` flag but may still produce some output.
They're handled by `test_all_workflows.py` using `capture_output=True`.

### Workflow Comparison

| Aspect | Normal Mode | Batch Mode |
|--------|------------|-----------|
| **Output** | Detailed, formatted | Minimal |
| **Use Case** | Development, debugging | Testing, CI/CD |
| **Speed** | Slower (I/O) | Faster |
| **Results Tracked** | Yes | Yes |
| **Errors Shown** | Yes | Yes if failure |

### Example Workflow Run

**Normal Mode:**
```
======================================================================
Workflow 16: Fusion RAG
======================================================================

[Init] Starting workflow initialization at 14:23:45
[Documents] Loaded 1 document(s)
[Chunks] Created 1000 chunk(s)
[Embeddings] Generated 1000 embedding(s)
[BM25] Index created with 1000 documents
[READY] Workflow ready to process queries

======================================================================
Query 1/1
======================================================================

Your question here

[Retrieval: Fusion] (5 items)
----------------------------------------

  [1] Retrieved document 1... (combined: 0.854)
  [2] Retrieved document 2... (combined: 0.721)
  [3] Retrieved document 3... (combined: 0.654)
  [4] Retrieved document 4... (combined: 0.601)
  [5] Retrieved document 5... (combined: 0.547)

[Fusion Response]
----------------------------------------
Generated response text...

======================================================================
Workflow Completion
======================================================================

[Completed] Processed 1 queries
[Time] Total execution time: 12.34s
[Speed] Average 12.34s per query
[Timestamp] 2025-12-18 14:25:30
```

**Batch Mode:**
```
(Minimal output - just execution status)
```

### Adding Batch Mode to New Workflows

When creating a new workflow that uses `ConsoleLogger`:

```python
from workflow_parts.output_formatter import ConsoleLogger

def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument("--batch", action="store_true", help="Batch mode (minimal output)")
    args = parser.parse_args()
    
    # Pass batch_mode to logger
    logger = ConsoleLogger("Your Workflow", 20, batch_mode=args.batch)
    
    # Rest of your workflow...
```

That's it! The logger automatically handles batch mode.

### Results Tracking

Regardless of batch mode, all workflows track results:

```bash
# Results are saved to workflow_results.csv
# View them with:
python print_results.py
```

### Summary

- **`--batch`**: Use for automated testing - minimal output
- **Without `--batch`**: Use for interactive/development - detailed output
- **`test_all_workflows.py`**: Automatically uses batch mode for all workflows
- **Results tracking**: Works in both modes

Enjoy clean, organized workflow execution! 🎯
