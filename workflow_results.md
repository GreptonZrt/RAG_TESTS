# RAG Workflow Evaluation Results

**Generated**: 2025-12-16 10:29:56

## Summary

- **Total Workflows**: 5
- **Last Update**: 2025-12-16T10:29:56.324224

## Detailed Results

| Workflow | ID | Queries | Avg Chunks | Avg Response | Avg Utility | Iterations | Timestamp |
|----------|----|---------|-----------|--------------|-----------|---------| ---|
| Feedback Loop RAG | 11 | 2 | 5.0 | 0 | - | - | 10:25:11 |
| Adaptive RAG | 12 | 1 | 3.0 | 97 | - | - | 10:25:57 |
| Self-RAG | 13 | 1 | 2.0 | 0 | 5.00 | 2.0 | 10:26:46 |
| Proposition Chunking RAG | 14 | 1 | 0.0 | 97 | - | - | 10:28:53 |
| HyDE RAG | 19 | 1 | 2.0 | 85 | - | - | 10:29:16 |


## Workflow Descriptions

### [11] Feedback Loop RAG
Tracks relevance feedback on retrieved chunks and adjusts future retrieval scores based on feedback history.

### [12] Adaptive RAG
Classifies queries by type (Factual, Analytical, Opinion, Contextual) and adjusts retrieval strategy accordingly.

### [13] Self-RAG
Implements self-reflective RAG with iterative refinement and relevance filtering.

### [14] Proposition Chunking RAG
Decomposes chunks into atomic propositions for fine-grained retrieval at proposition level.

### [19] HyDE RAG
Generates hypothetical documents to improve semantic matching using hypothetical embeddings.

## Notes

- Results are automatically saved to `workflow_results.csv`
- Each run overwrites previous results with fresh metrics
- Metrics are calculated from all processed queries
- Timestamps show when each workflow was executed

---
*Report generated: 2025-12-16T10:29:56.324224*
