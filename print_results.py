#!/usr/bin/env python3
"""
Results Printer - Display workflow results in nice ASCII table format
"""

import csv
from pathlib import Path
from datetime import datetime


def print_results_table(csv_file: str = "workflow_results.csv"):
    """
    Print workflow results as a beautiful ASCII table.
    
    Args:
        csv_file: Path to CSV file with results
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"❌ {csv_file} not found")
        return
    
    # Read CSV
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Use semicolon (;) as delimiter for Windows/Hungarian locale compatibility
        reader = csv.DictReader(f, delimiter=';')
        results = list(reader)
    
    if not results:
        print("No results available")
        return
    
    # Print header
    print("\n" + "="*100)
    print("RAG WORKFLOW EVALUATION RESULTS".center(100))
    print("="*100)
    
    # Print summary
    print(f"\nTotal Workflows: {len(results)}")
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Sort by overall_score if available
    try:
        results_sorted = sorted(results, 
                               key=lambda x: float(x.get('overall_score', 0) or 0), 
                               reverse=True)
    except:
        results_sorted = results
    
    # Main ranking table with overall_score
    print("-"*120)
    print(f"{'Rank':<5} | {'ID':<5} | {'Workflow':<25} | {'Overall Score':<15} | {'Valid %':<10} | {'Queries':<8} | {'Chunks':<8} | {'Utility':<8}")
    print("-"*120)
    
    for idx, result in enumerate(results_sorted, 1):
        wf_id = result.get('workflow_id', 'N/A')
        wf_name = result.get('workflow_name', 'Unknown')[:25]
        overall_score = result.get('overall_score', '-')
        valid_rate = result.get('valid_response_rate', '-')
        queries = result.get('queries_processed', 'N/A')
        avg_chunks = result.get('avg_chunks_retrieved', '-')
        avg_utility = result.get('avg_utility_rating', '-')
        last_exec = result.get('last_execution', '-')
        
        # Format numbers
        try:
            overall_score = f"{float(overall_score):.1f}/100" if overall_score and overall_score != '-' else '-'
        except:
            overall_score = '-'
        
        try:
            valid_rate = f"{float(valid_rate):.0f}%" if valid_rate and valid_rate != '-' else '-'
        except:
            valid_rate = '-'
        
        try:
            avg_chunks = f"{float(avg_chunks):.1f}" if avg_chunks and avg_chunks != '-' else '-'
        except:
            avg_chunks = '-'
        
        try:
            avg_utility = f"{float(avg_utility):.2f}" if avg_utility and avg_utility != '-' else '-'
        except:
            avg_utility = '-'
        
        # Medal/ranking emoji
        medal = "🥇" if idx == 1 else ("🥈" if idx == 2 else ("🥉" if idx == 3 else f"{idx}. "))
        
        print(f"{medal:<5} | {wf_id:<5} | {wf_name:<25} | {overall_score:<15} | {valid_rate:<10} | {str(queries):<8} | {avg_chunks:<8} | {avg_utility:<8}")
    
    print("-"*120)
    
    # Workflow Statistics
    print("\n📊 WORKFLOW STATISTICS:\n" + "-"*120)
    
    for result in results_sorted:
        wf_id = result.get('workflow_id', 'N/A')
        wf_name = result.get('workflow_name', 'Unknown')
        overall_score = result.get('overall_score', '-')
        valid_rate = result.get('valid_response_rate', '-')
        last_exec = result.get('last_execution', '-')
        queries = result.get('queries_processed', 'N/A')
        
        try:
            overall_score = f"{float(overall_score):.1f}" if overall_score and overall_score != '-' else '-'
        except:
            overall_score = '-'
        
        try:
            valid_rate = f"{float(valid_rate):.1f}%" if valid_rate and valid_rate != '-' else '-'
        except:
            valid_rate = '-'
        
        print(f"\n[Workflow {wf_id}] {wf_name}")
        print(f"  Overall Score: {overall_score}/100")
        print(f"  Valid Response Rate: {valid_rate}")
        print(f"  Queries Processed: {queries}")
        print(f"  Last Execution: {last_exec}")
    
    print("\n" + "="*120 + "\n")


def print_comparison(csv_file: str = "workflow_results.csv"):
    """
    Print a side-by-side comparison of key metrics.
    
    Args:
        csv_file: Path to CSV file with results
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"❌ {csv_file} not found")
        return
    
    # Read CSV
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Use semicolon (;) as delimiter for Windows/Hungarian locale compatibility
        reader = csv.DictReader(f, delimiter=';')
        results = list(reader)
    
    if not results:
        return
    
    print("\nCOMPARISON: Average Chunks Retrieved\n" + "-"*50)
    
    # Find max for scaling
    max_chunks = 0
    for result in results:
        try:
            chunks = float(result.get('avg_chunks_retrieved', 0) or 0)
            if chunks > max_chunks:
                max_chunks = chunks
        except:
            pass
    
    if max_chunks == 0:
        max_chunks = 1
    
    # Print bars
    for result in results:
        wf_id = result.get('workflow_id', 'N/A')
        wf_name = result.get('workflow_name', 'Unknown')[:20]
        
        try:
            chunks = float(result.get('avg_chunks_retrieved', 0) or 0)
            bar_length = int((chunks / max_chunks) * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"[{wf_id}] {wf_name:<20} | {bar} | {chunks:.1f}")
        except:
            pass
    
    print("-"*50 + "\n")


if __name__ == "__main__":
    print_results_table()
    print_comparison()
