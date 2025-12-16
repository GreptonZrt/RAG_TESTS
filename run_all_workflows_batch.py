#!/usr/bin/env python3
"""
Batch runner for all RAG workflows with centralized results comparison

Runs all 12 workflows sequentially and displays a comprehensive comparison at the end.
"""
import subprocess
import sys
from pathlib import Path

workflows = [
    ("01_simple_rag", "01"),
    ("02_semantic_chunking", "02"),
    ("04_context_enriched_rag", "04"),
    ("05_contextual_chunk_headers_rag", "05"),
    ("06_doc_augmentation_rag", "06"),
    ("08_reranker", "08"),
    ("10_contextual_compression", "10"),
    ("11_feedback_loop_rag", "11"),
    ("12_adaptive_rag", "12"),
    ("13_self_rag", "13"),
    ("14_proposition_chunking_rag", "14"),
    ("19_hyde_rag", "19"),
]

def run_workflow(workflow_name, num_queries=3):
    """Run a single workflow with specified number of queries."""
    try:
        result = subprocess.run(
            [sys.executable, f"workflows/{workflow_name}.py", "--max", str(num_queries), "--no-eval"],
            capture_output=True,
            timeout=120,
            cwd=Path(__file__).parent,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch run all RAG workflows")
    parser.add_argument("--max", type=int, default=3, help="Max queries per workflow (default: 3)")
    parser.add_argument("--silent", action="store_true", help="Suppress individual workflow output")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BATCH RUNNER - All RAG Workflows".center(80))
    print("="*80)
    print(f"\nRunning {len(workflows)} workflows with {args.max} queries each...\n")
    
    results = {}
    for workflow_name, workflow_id in workflows:
        status_str = f"[{workflow_id}] {workflow_name}"
        print(f"Running {status_str}...", end=" ", flush=True)
        
        success, stdout, stderr = run_workflow(workflow_name, args.max)
        
        if success:
            print("[OK]")
            results[workflow_id] = "OK"
            if not args.silent:
                # Extract and print the workflow metrics line
                for line in stdout.split('\n'):
                    if f"[Workflow {workflow_id}]" in line or "Overall Score:" in line or "Valid Response Rate:" in line or "Queries Processed:" in line:
                        print(f"  {line}")
        else:
            print("[FAILED]")
            results[workflow_id] = "FAILED"
            if stderr:
                print(f"  Error: {stderr[:100]}")
    
    # Print comprehensive comparison
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS COMPARISON".center(80))
    print("="*80)
    
    subprocess.run([sys.executable, "print_results.py"], cwd=Path(__file__).parent)
    
    # Summary
    print("\n" + "="*80)
    print("BATCH RUN SUMMARY")
    print("="*80)
    ok_count = sum(1 for v in results.values() if v == "OK")
    failed_count = sum(1 for v in results.values() if v == "FAILED")
    print(f"Completed: {ok_count}/{len(workflows)}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
        failed_wfs = [wf_id for wf_id, status in results.items() if status == "FAILED"]
        for wf_id in failed_wfs:
            print(f"  - {wf_id}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
