"""
01 Simple RAG Workflow

This is the basic RAG workflow that implements all fundamental RAG steps:
1. Data Loading - Extract text from PDF/DOCX files (with OCR support)
2. Chunking - Split text into overlapping chunks
3. Embedding - Create vector embeddings for chunks
4. Retrieval - Semantic search for relevant chunks
5. Generation - Generate response using LLM
6. Evaluation - Evaluate response quality (optional)

Supports:
  - Single or multiple input files (PDF, DOCX)
  - Image-based PDFs with OCR
  - Automatic file type detection
"""

import os
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv

from workflow_parts.data_loading import (
    extract_text_from_pdf,
    extract_text_from_file,
    load_multiple_files,
    combine_documents,
    load_validation_data,
    extract_queries_from_validation_data
)
from workflow_parts.chunking import chunk_text_sliding_window
from workflow_parts.embedding import create_embeddings
from workflow_parts.retrieval import semantic_search
from workflow_parts.generation import generate_response
from workflow_parts.evaluation import evaluate_response

# Load environment variables
load_dotenv()


class SimpleRAGWorkflow:
    """
    Simple RAG workflow orchestrator.
    
    Processes queries through the complete RAG pipeline:
    PDF → Chunks → Embeddings → Search → Generation → Evaluation
    """
    
    def __init__(
        self,
        pdf_path: str = None,
        file_paths: Union[str, List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_deployment: str = None,
        chat_deployment: str = None,
        use_ocr: bool = False
    ):
        """
        Initialize the Simple RAG workflow.
        
        Args:
            pdf_path: DEPRECATED - use file_paths instead. Path to single PDF file
            file_paths: Single file path or list of file paths (PDF, DOCX)
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            embedding_deployment: Override embedding model name
            chat_deployment: Override chat model name
            use_ocr: Enable OCR for image-based PDFs
        """
        # Support legacy pdf_path parameter
        if pdf_path and not file_paths:
            file_paths = pdf_path
        elif not file_paths:
            file_paths = "data/AI_Information.pdf"
        
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ocr = use_ocr
        
        # Set deployment names
        if embedding_deployment:
            os.environ["EMBEDDING_DEPLOYMENT"] = embedding_deployment
        if chat_deployment:
            os.environ["CHAT_DEPLOYMENT"] = chat_deployment
        
        # Internal state (will be cached)
        self.text: Optional[str] = None
        self.chunks: Optional[List[str]] = None
        self.embeddings = None
        self.validation_data: Optional[List[Dict]] = None
    
    def _initialize_data(self):
        """Load and prepare data from multiple files (cached)."""
        if self.text is None:
            print(f"[Step 1] Loading documents...")
            
            # Load multiple files
            file_results = load_multiple_files(self.file_paths, use_ocr=self.use_ocr)
            
            # Combine documents
            self.text = combine_documents(file_results)
            print(f"  → Loaded {len(self.text)} total characters from {len(self.file_paths)} file(s)")
    
    def _initialize_chunks(self):
        """Create chunks from text (cached)."""
        if self.chunks is None:
            self._initialize_data()
            print(f"\n[Step 2] Chunking text (size={self.chunk_size}, overlap={self.chunk_overlap})")
            self.chunks = chunk_text_sliding_window(
                self.text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap
            )
            print(f"  → Created {len(self.chunks)} chunks")
    
    def _initialize_embeddings(self):
        """Create embeddings for chunks (cached)."""
        if self.embeddings is None:
            self._initialize_chunks()
            print(f"\n[Step 3] Creating embeddings for {len(self.chunks)} chunks...")
            self.embeddings = create_embeddings(self.chunks)
            print(f"  → Created {len(self.embeddings)} embeddings")
    
    def run(
        self,
        query: str,
        k: int = 5,
        evaluate: bool = True,
        ideal_answer: Optional[str] = None
    ) -> Dict:
        """
        Run the RAG pipeline for a single query.
        
        Args:
            query: The user query
            k: Number of chunks to retrieve
            evaluate: Whether to evaluate the response
            ideal_answer: Ground truth answer for evaluation
            
        Returns:
            Dict containing:
            - query: The input query
            - retrieved_chunks: Retrieved context chunks
            - ai_response: Generated response
            - evaluation: Evaluation results (if evaluate=True)
        """
        print(f"\n{'='*70}")
        print(f"Processing Query: {query}")
        print(f"{'='*70}")
        
        # Ensure embeddings are ready
        self._initialize_embeddings()
        
        # Step 4: Retrieval
        print(f"\n[Step 4] Retrieving top-{k} chunks...")
        retrieved_chunks = semantic_search(query, self.chunks, self.embeddings, k=k)
        print(f"  → Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 5: Generation
        print(f"\n[Step 5] Generating response...")
        gen_result = generate_response(query, retrieved_chunks)
        ai_response = gen_result["content"]
        print(f"  → Generated response ({len(ai_response)} characters)")
        
        result = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "ai_response": ai_response,
        }
        
        # Step 6: Evaluation (optional)
        if evaluate:
            print(f"\n[Step 6] Evaluating response...")
            eval_result = evaluate_response(query, ai_response, ideal_answer)
            result["evaluation"] = eval_result
            print(f"  → Evaluation score: {eval_result.get('score', 'N/A')}")
        
        return result
    
    def run_batch(
        self,
        queries: List[str],
        k: int = 5,
        evaluate: bool = True,
        ideal_answers: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Run the RAG pipeline for multiple queries.
        
        Args:
            queries: List of queries to process
            k: Number of chunks to retrieve
            evaluate: Whether to evaluate responses
            ideal_answers: Dict mapping queries to ideal answers
            
        Returns:
            List of result dictionaries
        """
        ideal_answers = ideal_answers or {}
        results = []
        
        for i, query in enumerate(queries):
            try:
                ideal = ideal_answers.get(query)
                result = self.run(query, k=k, evaluate=evaluate, ideal_answer=ideal)
                results.append(result)
            except Exception as e:
                print(f"ERROR processing query: {e}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return results
    
    def run_from_file(
        self,
        val_file: str = "data/val_multi.json",
        k: int = 5,
        evaluate: bool = True,
        max_queries: Optional[int] = None
    ) -> List[Dict]:
        """
        Run RAG pipeline using queries from a validation file.
        
        Args:
            val_file: Path to JSON validation file
            k: Number of chunks to retrieve
            evaluate: Whether to evaluate responses
            max_queries: Maximum number of queries to process (None = all)
            
        Returns:
            List of result dictionaries
        """
        print(f"Loading validation data from: {val_file}")
        self.validation_data = load_validation_data(val_file)
        
        # Limit queries if specified
        data = self.validation_data[:max_queries] if max_queries else self.validation_data
        
        # Extract queries and ideal answers
        queries = extract_queries_from_validation_data(data)
        ideal_answers = {item['question']: item.get('ideal_answer', '') for item in data}
        
        print(f"Processing {len(queries)} queries from validation file\n")
        
        results = self.run_batch(queries, k=k, evaluate=evaluate, ideal_answers=ideal_answers)
        
        # Attach ideal answers to results
        for result in results:
            if 'error' not in result:
                result['ideal_answer'] = ideal_answers.get(result['query'], '')
        
        return results
    
    def print_result(self, result: Dict) -> None:
        """Pretty-print a single result."""
        print(f"\n{'-'*70}")
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"{'-'*70}")
        
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        # Print retrieved chunks
        print(f"\nRetrieved Chunks ({len(result.get('retrieved_chunks', []))} total):")
        for i, chunk in enumerate(result.get('retrieved_chunks', [])):
            preview = chunk[:150].replace('\n', ' ') + "..." if len(chunk) > 150 else chunk
            print(f"  [{i+1}] {preview}")
        
        # Print AI response
        print(f"\nAI Response:")
        print(result.get('ai_response', 'N/A'))
        
        # Print ideal answer if available
        if result.get('ideal_answer'):
            print(f"\nIdeal Answer:")
            print(result['ideal_answer'])
        
        # Print evaluation if available
        if 'evaluation' in result:
            eval_result = result['evaluation']
            score = eval_result.get('score', 'N/A')
            reasoning = eval_result.get('reasoning', '')
            print(f"\nEvaluation Score: {score}")
            if reasoning:
                print(f"Reasoning: {reasoning}")
    
    def print_results(self, results: List[Dict]) -> None:
        """Pretty-print all results."""
        for result in results:
            self.print_result(result)
        
        # Summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        successful = sum(1 for r in results if 'error' not in r)
        print(f"Processed: {successful}/{len(results)} queries successfully")
        
        # Average evaluation score
        scores = [r['evaluation'].get('score') 
                  for r in results 
                  if 'evaluation' in r and r['evaluation'].get('score') is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"Average Evaluation Score: {avg_score:.2f}")


if __name__ == "__main__":
    # Example usage
    workflow = SimpleRAGWorkflow()
    
    # Run from validation file
    results = workflow.run_from_file("data/val_multi.json", max_queries=1)
    workflow.print_results(results)
