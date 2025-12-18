"""
Unified Output Formatter - Standardized logging and output for all RAG workflows

Provides consistent formatting for:
- Workflow initialization logs
- Query processing logs
- Retrieval results display
- Response generation
- Performance metrics
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class WorkflowFormatter:
    """Standardizes output formatting across all RAG workflows."""
    
    # Color codes for terminal output (if supported)
    COLORS = {
        'HEADER': '\033[95m',
        'OKBLUE': '\033[94m',
        'OKCYAN': '\033[96m',
        'OKGREEN': '\033[92m',
        'WARNING': '\033[93m',
        'FAIL': '\033[91m',
        'ENDC': '\033[0m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
    }
    
    def __init__(self, workflow_name: str, workflow_number: int):
        """
        Initialize formatter for a specific workflow.
        
        Args:
            workflow_name: Human-readable workflow name (e.g., "Fusion RAG")
            workflow_number: Workflow number (e.g., 16)
        """
        self.workflow_name = workflow_name
        self.workflow_number = workflow_number
        self.start_time = datetime.now()
    
    @staticmethod
    def _section_header(title: str, width: int = 70) -> str:
        """Create a formatted section header."""
        return f"\n{'='*width}\n{title.center(width)}\n{'='*width}\n"
    
    @staticmethod
    def _subsection_header(title: str) -> str:
        """Create a formatted subsection header."""
        return f"\n[{title}]\n{'-'*len(title)-2}"
    
    def init_log(self, document_count: int, chunk_count: int, 
                embedding_count: int, additional_info: Dict[str, Any] = None) -> str:
        """
        Generate initialization log message.
        
        Args:
            document_count: Number of documents loaded
            chunk_count: Number of chunks created
            embedding_count: Number of embeddings generated
            additional_info: Optional additional initialization info
            
        Returns:
            Formatted initialization message
        """
        msg = self._section_header(f"Workflow {self.workflow_number}: {self.workflow_name}")
        msg += f"\n[Init] Starting workflow initialization at {datetime.now().strftime('%H:%M:%S')}"
        msg += f"\n[Documents] Loaded {document_count} document(s)"
        msg += f"\n[Chunks] Created {chunk_count} chunk(s)"
        msg += f"\n[Embeddings] Generated {embedding_count} embedding(s)"
        
        if additional_info:
            for key, value in additional_info.items():
                msg += f"\n[{key}] {value}"
        
        msg += f"\n[READY] Workflow ready to process queries\n"
        return msg
    
    def query_header(self, query_num: int, total_queries: int, query_text: str) -> str:
        """
        Generate query header for processing logs.
        
        Args:
            query_num: Current query number
            total_queries: Total number of queries
            query_text: The query text
            
        Returns:
            Formatted query header
        """
        msg = f"\n{'='*70}"
        msg += f"\nQuery {query_num}/{total_queries}"
        msg += f"\n{'='*70}"
        msg += f"\n\n{query_text}\n"
        return msg
    
    def retrieval_header(self, method: str, count: int = None) -> str:
        """
        Generate retrieval method header.
        
        Args:
            method: Retrieval method name (e.g., "Vector-Only", "Fusion")
            count: Optional count of retrieved items
            
        Returns:
            Formatted retrieval header
        """
        msg = f"\n[Retrieval: {method}]"
        if count is not None:
            msg += f" ({count} items)"
        msg += "\n" + "-"*40
        return msg
    
    def retrieval_items(self, items: List[Dict[str, Any]], max_preview: int = 80) -> str:
        """
        Format retrieved items for display.
        
        Args:
            items: List of retrieved items with metadata
            max_preview: Maximum characters to show per item
            
        Returns:
            Formatted retrieval items
        """
        msg = ""
        for i, item in enumerate(items, 1):
            text = item.get("text", "")[:max_preview]
            msg += f"\n  [{i}] {text}..."
            
            # Add scores if available
            scores = []
            if "similarity" in item:
                scores.append(f"sim: {item['similarity']:.3f}")
            if "bm25_score" in item:
                scores.append(f"bm25: {item['bm25_score']:.3f}")
            if "combined_score" in item:
                scores.append(f"combined: {item['combined_score']:.3f}")
            if "score" in item:
                scores.append(f"score: {item['score']:.3f}")
            
            if scores:
                msg += f" ({', '.join(scores)})"
        
        return msg
    
    def response_section(self, response: str, method: str = "Response", 
                        max_chars: int = None) -> str:
        """
        Format response for display.
        
        Args:
            response: The response text
            method: Method name (e.g., "Vector Response", "Fusion Response")
            max_chars: Optional max characters to display
            
        Returns:
            Formatted response section
        """
        display_response = response
        if max_chars and len(response) > max_chars:
            display_response = response[:max_chars] + f"\n... (truncated, {len(response)} total chars)"
        
        msg = f"\n[{method}]\n{'-'*40}\n{display_response}"
        return msg
    
    def metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics for display.
        
        Args:
            metrics: Dictionary of metrics to display
            
        Returns:
            Formatted metrics summary
        """
        msg = "\n[Metrics]\n" + "-"*40
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f"\n  {key}: {value:.3f}"
            else:
                msg += f"\n  {key}: {value}"
        return msg
    
    def completion_log(self, queries_processed: int, total_time: float = None) -> str:
        """
        Generate workflow completion log.
        
        Args:
            queries_processed: Number of queries processed
            total_time: Optional total execution time in seconds
            
        Returns:
            Formatted completion message
        """
        if total_time is None:
            total_time = (datetime.now() - self.start_time).total_seconds()
        
        msg = self._section_header("Workflow Completion")
        msg += f"\n[Completed] Processed {queries_processed} queries"
        msg += f"\n[Time] Total execution time: {total_time:.2f}s"
        if queries_processed > 0:
            msg += f"\n[Speed] Average {total_time/queries_processed:.2f}s per query"
        msg += f"\n[Timestamp] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        return msg
    
    def comparison_table(self, methods_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Format comparison table for different retrieval methods.
        
        Args:
            methods_results: Dictionary mapping method names to their results
            
        Returns:
            Formatted comparison table
        """
        msg = "\n[Comparison: Retrieval Methods]\n" + "-"*60
        msg += "\n{:<20} {:<15} {:<15} {:<10}".format("Method", "Items", "Avg Score", "Response Len")
        msg += "\n" + "-"*60
        
        for method, results in methods_results.items():
            items = results.get("items_count", 0)
            avg_score = results.get("avg_score", 0)
            resp_len = len(results.get("response", ""))
            msg += "\n{:<20} {:<15} {:<15.3f} {:<10}".format(
                method, items, avg_score, resp_len
            )
        
        return msg
    
    def error_log(self, error_msg: str, context: str = "") -> str:
        """
        Format error message.
        
        Args:
            error_msg: Error message
            context: Optional context about where error occurred
            
        Returns:
            Formatted error message
        """
        msg = f"\n[ERROR] {error_msg}"
        if context:
            msg += f"\n[Context] {context}"
        return msg
    
    def warning_log(self, warning_msg: str) -> str:
        """
        Format warning message.
        
        Args:
            warning_msg: Warning message
            
        Returns:
            Formatted warning message
        """
        return f"\n[WARNING] {warning_msg}"
    
    def info_log(self, info_msg: str, tag: str = "INFO") -> str:
        """
        Format info message.
        
        Args:
            info_msg: Info message
            tag: Optional tag name
            
        Returns:
            Formatted info message
        """
        return f"\n[{tag}] {info_msg}"


class ConsoleLogger:
    """Simple console logger that uses WorkflowFormatter for consistent output."""
    
    def __init__(self, workflow_name: str, workflow_number: int, 
                 verbose: bool = True, batch_mode: bool = False,
                 log_file: Optional[str] = None):
        """
        Initialize console logger.
        
        Args:
            workflow_name: Workflow name
            workflow_number: Workflow number
            verbose: Whether to print to console
            batch_mode: If True, minimal output (only init and completion)
            log_file: Optional file path to log to
        """
        self.formatter = WorkflowFormatter(workflow_name, workflow_number)
        self.verbose = verbose and not batch_mode  # Suppress output in batch mode
        self.batch_mode = batch_mode
        self.log_file = log_file
        self.logs = []
    
    def log(self, message: str):
        """Log a message to console and optionally to file."""
        if self.verbose:
            print(message)
        self.logs.append(message)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
            except Exception as e:
                print(f"[WARNING] Could not write to log file: {e}")
    
    def init(self, document_count: int, chunk_count: int, 
            embedding_count: int, additional_info: Dict = None):
        """Log workflow initialization."""
        msg = self.formatter.init_log(document_count, chunk_count, 
                                      embedding_count, additional_info)
        self.log(msg)
    
    def query(self, query_num: int, total_queries: int, query_text: str):
        """Log query header."""
        msg = self.formatter.query_header(query_num, total_queries, query_text)
        self.log(msg)
    
    def retrieval(self, method: str, items: List[Dict], count: int = None):
        """Log retrieval results."""
        msg = self.formatter.retrieval_header(method, count or len(items))
        msg += self.formatter.retrieval_items(items)
        self.log(msg)
    
    def response(self, response: str, method: str = "Response"):
        """Log response."""
        msg = self.formatter.response_section(response, method)
        self.log(msg)
    
    def metrics(self, metrics: Dict):
        """Log metrics."""
        msg = self.formatter.metrics_summary(metrics)
        self.log(msg)
    
    def complete(self, queries_processed: int, total_time: float = None):
        """Log workflow completion."""
        msg = self.formatter.completion_log(queries_processed, total_time)
        self.log(msg)
    
    def error(self, error_msg: str, context: str = ""):
        """Log error."""
        msg = self.formatter.error_log(error_msg, context)
        self.log(msg)
    
    def warning(self, warning_msg: str):
        """Log warning."""
        msg = self.formatter.warning_log(warning_msg)
        self.log(msg)
    
    def info(self, info_msg: str, tag: str = "INFO"):
        """Log info."""
        msg = self.formatter.info_log(info_msg, tag)
        self.log(msg)


class UnifiedSummaryFormatter:
    """Standardized summary formatting for workflow completion."""
    
    def __init__(self, workflow_name: str, workflow_number: int):
        """
        Initialize unified summary formatter.
        
        Args:
            workflow_name: Name of the workflow
            workflow_number: Workflow number
        """
        self.workflow_name = workflow_name
        self.workflow_number = workflow_number
    
    def format_summary(self, 
                      queries_processed: int,
                      total_time: float = None,
                      metrics: Dict[str, Any] = None,
                      custom_sections: Dict[str, str] = None) -> str:
        """
        Format unified workflow summary with essential metrics only.
        
        Shows: Overall Score, Workflow ID/Name, Query Count, Valid Response Rate, Execution Time
        
        Args:
            queries_processed: Number of queries processed
            total_time: Total execution time in seconds
            metrics: Dictionary of metrics to display (must contain 'overall_score' and 'valid_response_rate')
            custom_sections: Optional dict of custom sections (ignored in unified format)
            
        Returns:
            Formatted summary string
        """
        summary = "\n" + "="*70 + "\n"
        summary += "WORKFLOW SUMMARY\n"
        summary += "="*70 + "\n\n"
        
        # Mandatory core information
        summary += f"Workflow: [{self.workflow_number}] {self.workflow_name}\n"
        
        # Overall Score (mandatory)
        if metrics and 'overall_score' in metrics:
            summary += f"Overall Score: {metrics['overall_score']:.1f}/100\n"
        
        # Query statistics (mandatory)
        summary += f"Queries Processed: {queries_processed}\n"
        
        # Valid response rate (mandatory)
        if metrics and 'valid_response_rate' in metrics:
            summary += f"Valid Response Rate: {metrics['valid_response_rate']:.1f}%\n"
        
        # Execution time (mandatory)
        if total_time is not None:
            summary += f"Execution Time: {total_time:.2f}s\n"
        
        summary += "\n" + "="*70 + "\n"
        
        return summary
    
    def format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics as a simple table.
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted metrics table
        """
        table = "Metrics:\n" + "-"*40 + "\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                table += f"  {key:<30} {value:>8.2f}\n"
            else:
                table += f"  {key:<30} {str(value):>8}\n"
        return table
    
    def log_summary(self, logger, 
                   queries_processed: int,
                   total_time: float = None,
                   metrics: Dict[str, Any] = None,
                   custom_sections: Dict[str, str] = None):
        """
        Log summary using a ConsoleLogger instance.
        
        Args:
            logger: ConsoleLogger instance
            queries_processed: Number of queries processed
            total_time: Total execution time
            metrics: Metrics dictionary
            custom_sections: Custom sections for workflow-specific info
        """
        summary = self.format_summary(
            queries_processed,
            total_time,
            metrics,
            custom_sections
        )
        logger.log(summary)
