#!/usr/bin/env python3
"""
Batch runner for all RAG workflows with centralized results comparison

Runs all 13 workflows sequentially with val_multi.json validation data.
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
    ("16_fusion_rag", "16"),
    ("19_hyde_rag", "19"),
]

def run_workflow(workflow_name, num_queries=999):
    """Run a single workflow with specified number of queries."""
    try:
        import os
        env = os.environ.copy()
        # Use the venv python directly
        venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
        if not venv_python.exists():
            venv_python = sys.executable
        
        result = subprocess.run(
            [str(venv_python), f"workflows/{workflow_name}.py", "--max", str(num_queries), "--no-eval"],
            capture_output=True,
            timeout=300,
            cwd=Path(__file__).parent,
            text=True,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch run all RAG workflows")
    parser.add_argument("--max", type=int, default=10, help="Max queries per workflow (default: 10)")
    parser.add_argument("--run", type=str, default=None, help="Specific workflows to run (comma-separated IDs, e.g., '01,04,12')")
    parser.add_argument("--silent", action="store_true", help="Suppress individual workflow output")
    args = parser.parse_args()
    
    # Filter workflows if --run is specified
    workflows_to_run = workflows
    if args.run:
        requested_ids = [id.strip() for id in args.run.split(",")]
        workflows_to_run = [wf for wf in workflows if wf[1] in requested_ids]
        if not workflows_to_run:
            available_ids = ", ".join([wf[1] for wf in workflows])
            print(f"\nError: No workflows found matching IDs: {requested_ids}")
            print(f"Available workflow IDs: {available_ids}\n")
            sys.exit(1)
    
    print("\n" + "="*80)
    print("BATCH RUNNER - All RAG Workflows".center(80))
    print("="*80)
    print(f"\nRunning {len(workflows_to_run)} workflows with {args.max} queries each (val_multi.json)...\n")
    
    results = {}
    for workflow_name, workflow_id in workflows_to_run:
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
            # Save full error to file for debugging
            if stderr:
                error_file = Path(__file__).parent / f"error_debug_{workflow_id}.txt"
                error_file.write_text(stderr)
                # Print first 500 chars
                error_preview = stderr[:500] if len(stderr) > 500 else stderr
                print(f"  Error (first 500 chars): {error_preview}")
                print(f"  (Full error saved to: {error_file.name})")
                if "TIMEOUT" in stderr:
                    print(f"  (Workflow took too long, try increasing timeout)")
            if stdout:
                # Show last few lines of stdout if available
                stdout_lines = stdout.split('\n')
                last_lines = [l for l in stdout_lines if l.strip()][-3:]
                if last_lines:
                    print(f"  Last output: {last_lines[-1][:100]}")
    
    # Print comprehensive comparison (only if any succeeded)
    if any(v == "OK" for v in results.values()):
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS COMPARISON".center(80))
        print("="*80)
        
        try:
            subprocess.run([sys.executable, "print_results.py"], cwd=Path(__file__).parent)
        except Exception as e:
            print(f"Note: Could not display results comparison: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("BATCH RUN SUMMARY")
    print("="*80)
    ok_count = sum(1 for v in results.values() if v == "OK")
    failed_count = sum(1 for v in results.values() if v == "FAILED")
    print(f"Completed: {ok_count}/{len(workflows_to_run)}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
        failed_wfs = [wf_id for wf_id, status in results.items() if status == "FAILED"]
        for wf_id in failed_wfs:
            print(f"  - {wf_id}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
