#!/usr/bin/env python3
"""Quick test runner for all integrated workflows"""
import subprocess
import sys
from pathlib import Path

workflows = ["01_simple_rag", "02_semantic_chunking", "04_context_enriched_rag", 
             "05_contextual_chunk_headers_rag", "06_doc_augmentation_rag", 
             "08_reranker", "10_contextual_compression",
             "11_feedback_loop_rag", "12_adaptive_rag", "13_self_rag",
             "14_proposition_chunking_rag", "16_fusion_rag", "19_hyde_rag"]

print("\n" + "="*70)
print("TESTING ALL WORKFLOWS FOR RESULTS TRACKING")
print("="*70 + "\n")

failed = []
for wf in workflows:
    print(f"Testing {wf}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, f"workflows/{wf}.py", "--max", "1", "--no-eval", "--batch"],
            capture_output=True,
            timeout=60,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            print("[OK]")
        else:
            print("[FAILED]")
            failed.append(wf)
    except subprocess.TimeoutExpired:
        print("[TIMEOUT]")
        failed.append(wf)
    except Exception as e:
        print(f"[ERROR] {e}")
        failed.append(wf)

print("\n" + "="*70)
if failed:
    print(f"FAILED WORKFLOWS: {len(failed)}")
    for wf in failed:
        print(f"  - {wf}")
else:
    print("ALL WORKFLOWS TESTED SUCCESSFULLY!")
print("="*70 + "\n")

# Print results summary
print("Running results summary...")
subprocess.run([sys.executable, "print_results.py"], cwd=Path(__file__).parent)
