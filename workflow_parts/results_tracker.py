"""
Results Tracker - Workflow Performance Metrics

Tracks and stores evaluation metrics for all RAG workflows in a CSV table.
Supports overwriting previous results on each run.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from datetime import datetime
from typing import Dict, List, Any, Optional


class ResultsTracker:
    """Tracks and persists workflow evaluation results."""
    
    def __init__(self, results_file: str = "workflow_results.csv"):
        """
        Initialize results tracker.
        
        Args:
            results_file: Path to CSV file for storing results
        """
        self.results_file = Path(results_file)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.load_existing_results()
    
    def load_existing_results(self):
        """Load existing results from CSV file if it exists."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    # Use semicolon (;) as delimiter for Windows/Hungarian locale compatibility
                    reader = csv.DictReader(f, delimiter=';')
                    for row in reader:
                        workflow_id = row.get('workflow_id')
                        if workflow_id:
                            self.results[workflow_id] = dict(row)
            except Exception as e:
                print(f"  [WARNING] Could not load existing results: {e}")
    
    def add_result(self, 
                   workflow_id: str,
                   workflow_name: str,
                   metrics: Dict[str, Any]):
        """
        Add or update result for a workflow.
        
        Args:
            workflow_id: Workflow identifier (11, 12, 13, 14, 19)
            workflow_name: Human-readable workflow name
            metrics: Dictionary of metrics to store
        """
        result_entry = {
            'workflow_id': workflow_id,
            'workflow_name': workflow_name,
            'timestamp': datetime.now().isoformat(),
            'last_execution': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **metrics
        }
        self.results[workflow_id] = result_entry
    
    def save_results(self):
        """Save all results to CSV file, overwriting previous data."""
        if not self.results:
            print("  [WARNING] No results to save")
            return
        
        # Define logical column order (primary metrics first, then supplementary)
        fieldnames_ordered = [
            'workflow_id',
            'workflow_name',
            'overall_score',
            'valid_response_rate',
            'avg_answer_similarity',
            'avg_chunks_retrieved',
            'avg_response_length',
            'queries_processed',
            'last_execution',
            'timestamp'
        ]
        
        # Add any additional fields not in the ordered list
        all_fields = set()
        for result in self.results.values():
            all_fields.update(result.keys())
        
        additional_fields = sorted([f for f in all_fields if f not in fieldnames_ordered])
        fieldnames = fieldnames_ordered + additional_fields
        
        try:
            with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
                # Use semicolon (;) as delimiter for Windows/Hungarian locale compatibility
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                writer.writeheader()
                
                # Sort by workflow_id for consistent ordering
                sorted_results = sorted(self.results.items(), 
                                      key=lambda x: str(x[0]))
                
                for workflow_id, result in sorted_results:
                    writer.writerow(result)
            
            print(f"  [SAVED] Results saved to {self.results_file}")
        except Exception as e:
            print(f"  [ERROR] Failed to save results: {e}")
    
    def get_summary(self) -> str:
        """Generate a summary of all results."""
        if not self.results:
            return "No results available"
        
        summary = "\n" + "="*70 + "\n"
        summary += "WORKFLOW COMPARISON SUMMARY\n"
        summary += "="*70 + "\n\n"
        
        # Sort by workflow_id
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: str(x[0]))
        
        for workflow_id, result in sorted_results:
            summary += f"[{result.get('workflow_id')}] {result.get('workflow_name')}\n"
            summary += f"  Timestamp: {result.get('timestamp', 'N/A')}\n"
            
            # Display relevant metrics
            metric_keys = [k for k in result.keys() 
                         if k not in {'workflow_id', 'workflow_name', 'timestamp'}]
            
            for key in sorted(metric_keys):
                value = result.get(key, 'N/A')
                # Format numeric values nicely
                if isinstance(value, str):
                    try:
                        num_val = float(value)
                        if '.' in str(num_val):
                            summary += f"  {key}: {num_val:.2f}\n"
                        else:
                            summary += f"  {key}: {int(num_val)}\n"
                    except (ValueError, TypeError):
                        summary += f"  {key}: {value}\n"
                else:
                    summary += f"  {key}: {value}\n"
            
            summary += "\n"
        
        summary += "="*70 + "\n"
        return summary
    
    def export_json(self, json_file: Optional[str] = None) -> str:
        """
        Export results as JSON.
        
        Args:
            json_file: Path to JSON file (default: workflow_results.json)
            
        Returns:
            Path to exported JSON file
        """
        if json_file is None:
            json_file = self.results_file.stem + ".json"
        
        json_path = Path(json_file)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"  [SAVED] Results exported to {json_path}")
            return str(json_path)
        except Exception as e:
            print(f"  [ERROR] Failed to export JSON: {e}")
            return ""


def create_metrics_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create metrics dictionary from workflow results.
    
    Args:
        results: List of query results from workflow
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not results:
        return {}
    
    # Count valid responses (those that actually answer the question)
    valid_responses = 0
    total_response_length = 0
    similarity_scores = []
    
    for r in results:
        # Support multiple response field names from different workflow types
        response = r.get('response', '') or r.get('ai_response', '') or r.get('content', '')
        if response and _is_valid_response(response):
            valid_responses += 1
        total_response_length += len(response)
        
        # Calculate answer similarity if ideal_answer is available
        ideal_answer = r.get('ideal_answer', '')
        if ideal_answer and response:
            similarity = _calculate_answer_similarity(response, ideal_answer)
            similarity_scores.append(similarity)
    
    # Calculate chunks_count - from 'chunks_count' field or from 'retrieved_chunks' list
    chunks_counts = []
    for r in results:
        if 'chunks_count' in r:
            chunks_counts.append(r['chunks_count'])
        elif 'retrieved_chunks' in r:
            chunks_counts.append(len(r['retrieved_chunks']))
        else:
            chunks_counts.append(0)
    
    metrics = {
        'queries_processed': len(results),
        'valid_response_rate': (valid_responses / len(results) * 100) if results else 0,
        'avg_chunks_retrieved': sum(chunks_counts) / len(results) if results else 0,
        'avg_response_length': total_response_length / len(results) if results else 0,
        'avg_answer_similarity': sum(similarity_scores) / len(similarity_scores) if similarity_scores else 50.0,
    }
    
    # Calculate additional metrics if available
    if 'category' in results[0]:
        categories = {}
        for r in results:
            cat = r.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in categories.items():
            metrics[f'category_{cat}'] = count
    
    if 'utility_rating' in results[0]:
        ratings = [r.get('utility_rating', 0) for r in results if 'utility_rating' in r]
        if ratings:
            metrics['avg_utility_rating'] = sum(ratings) / len(ratings)
    
    if 'iterations' in results[0]:
        iterations = [r.get('iterations', 0) for r in results if 'iterations' in r]
        if iterations:
            metrics['avg_iterations'] = sum(iterations) / len(iterations)
    
    # Calculate overall score (0-100) - NEW: with answer similarity as primary factor
    overall_score = _calculate_overall_score(
        valid_response_rate=metrics['valid_response_rate'],
        avg_chunks=metrics['avg_chunks_retrieved'],
        avg_response_len=metrics['avg_response_length'],
        avg_utility=metrics.get('avg_utility_rating', 3.0),
        avg_answer_similarity=metrics['avg_answer_similarity']
    )
    metrics['overall_score'] = overall_score
    
    return metrics


def _is_valid_response(response: str) -> bool:
    """
    Check if a response is valid (actually answers the question).
    
    Args:
        response: The response text
        
    Returns:
        True if response is valid, False if it's a "I don't know" response
    """
    if not response or len(response.strip()) < 10:
        return False
    
    # Check for common "I don't know" phrases
    no_answer_phrases = [
        "i do not have enough",
        "i don't have enough",
        "cannot be derived",
        "cannot derive",
        "no information",
        "not enough information",
        "unable to provide",
        "cannot determine",
        "i cannot",
        "i am unable",
    ]
    
    response_lower = response.lower()
    for phrase in no_answer_phrases:
        if phrase in response_lower:
            return False
    
    return True


def _calculate_answer_similarity(generated_answer: str, ideal_answer: str) -> float:
    """
    Calculate similarity between generated and ideal answer using LLM evaluation.
    
    Uses Claude/GPT to assess how well the generated answer matches the ideal answer
    on a 0-100 scale based on content quality and relevance.
    
    Args:
        generated_answer: The AI-generated response
        ideal_answer: The ideal/expected answer
        
    Returns:
        Similarity score 0-100
    """
    try:
        from openai import AzureOpenAI, OpenAI
        import re
        
        # Get appropriate client
        if (os.getenv("AZURE_OPENAI_ENDPOINT") and 
            os.getenv("AZURE_OPENAI_API_KEY") and 
            os.getenv("API_VERSION")):
            client = AzureOpenAI(
                api_version=os.getenv("API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
            deployment = os.getenv("CHAT_DEPLOYMENT", "gpt-4o")
        elif os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            deployment = "gpt-4o"
        else:
            # Fallback: simple string similarity if no API available
            return _simple_string_similarity(generated_answer, ideal_answer)
        
        # Create evaluation prompt
        prompt = f"""Evaluate how well the generated answer matches the ideal answer.

IDEAL ANSWER:
{ideal_answer}

GENERATED ANSWER:
{generated_answer}

Rate the similarity on a 0-100 scale, where:
- 0-20: Completely wrong or irrelevant
- 21-40: Partially relevant but missing key information
- 41-60: Mostly correct but with some gaps or inaccuracies
- 61-80: Very good match with minor differences
- 81-100: Excellent match, ideal answer well covered

Respond with ONLY a single number (0-100), no explanation."""

        # Call LLM
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are an answer quality evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10,
        )
        
        # Extract score from response
        response_text = response.choices[0].message.content.strip()
        score = float(re.sub(r'[^0-9]', '', response_text)[:3]) if response_text else 50.0
        return min(100, max(0, score))
        
    except Exception as e:
        print(f"  [WARNING] Answer similarity calculation failed: {e}")
        # Fallback to simple similarity
        return _simple_string_similarity(generated_answer, ideal_answer)


def _simple_string_similarity(text1: str, text2: str) -> float:
    """
    Simple string similarity fallback using character overlap.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score 0-100
    """
    # Normalize texts
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    
    if not t1 or not t2:
        return 0.0
    
    intersection = len(t1.intersection(t2))
    union = len(t1.union(t2))
    
    return (intersection / union * 100) if union > 0 else 0.0


def _calculate_overall_score(valid_response_rate: float,
                            avg_chunks: float,
                            avg_response_len: float,
                            avg_utility: float,
                            avg_answer_similarity: float = 50.0) -> float:
    """
    Calculate an overall quality score (0-100).
    
    Weights:
    - Answer similarity: 50% (MOST IMPORTANT - relevance to ideal answer)
    - Valid response rate: 10% (not a "I don't know" response)
    - Chunks retrieved: 10% (optimal 3-5 chunks)
    - Response length: 10% (optimal 80-150 chars)
    - Utility rating: 10% (user satisfaction)
    
    Args:
        valid_response_rate: Percentage of valid responses (0-100)
        avg_chunks: Average number of chunks retrieved
        avg_response_len: Average response length
        avg_utility: Average utility rating (1-5)
        avg_answer_similarity: Average answer similarity to ideal (0-100)
        
    Returns:
        Overall score 0-100
    """
    # Answer similarity is most important (50%) - how well it matches the ideal answer
    score_similarity = avg_answer_similarity * 0.5
    
    # Valid response rate (10%) - not a "I don't know"
    score_response = valid_response_rate * 0.1
    
    # Chunks retrieved (optimal is 3-5, 10%)
    chunks_score = min(100, (avg_chunks / 5.0 * 100)) * 0.1 if avg_chunks > 0 else 0
    
    # Response length (optimal is 80-150 chars, 10%)
    if 80 <= avg_response_len <= 150:
        length_score = 100 * 0.1
    elif avg_response_len > 0:
        length_score = (min(avg_response_len, 150) / 150.0 * 100) * 0.1
    else:
        length_score = 0
    
    # Utility rating if available (10%)
    utility_score = (avg_utility / 5.0 * 100) * 0.1 if avg_utility > 0 else 0
    
    total = score_similarity + score_response + chunks_score + length_score + utility_score
    return min(100, max(0, total))



# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = ResultsTracker("workflow_results.csv")
    
    # Add sample results
    tracker.add_result(
        workflow_id="11",
        workflow_name="Feedback Loop RAG",
        metrics={
            "queries_processed": 5,
            "avg_chunks_retrieved": 5.2,
            "avg_response_length": 245,
            "feedback_iterations": 2.4
        }
    )
    
    tracker.add_result(
        workflow_id="12",
        workflow_name="Adaptive RAG",
        metrics={
            "queries_processed": 5,
            "avg_chunks_retrieved": 4.6,
            "avg_response_length": 268,
            "factual_queries": 2,
            "analytical_queries": 1,
            "opinion_queries": 1,
            "contextual_queries": 1
        }
    )
    
    # Save and display
    tracker.save_results()
    print(tracker.get_summary())
