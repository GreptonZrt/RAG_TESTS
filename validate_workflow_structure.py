#!/usr/bin/env python3
"""
Validate the new workflow structure without requiring installed packages.

This checks:
1. File structure and organization
2. Python syntax of all files
3. Import paths (without executing API calls)
"""

import os
import ast
import sys
from pathlib import Path

# Fix for Windows console encoding issues
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_file_structure():
    """Verify directory and file structure."""
    print("=" * 70)
    print("FILE STRUCTURE CHECK")
    print("=" * 70)
    
    required_dirs = [
        "workflow_parts",
        "workflows",
    ]
    
    required_files = {
        "workflow_parts": [
            "__init__.py",
            "data_loading.py",
            "chunking.py",
            "embedding.py",
            "retrieval.py",
            "generation.py",
            "evaluation.py",
            "README.md",
        ],
        "workflows": [
            "__init__.py",
            "simple_rag_workflow.py",
            "README.md",
        ],
        "": [
            "01_simple_rag_cli.py",
        ]
    }
    
    all_ok = True
    
    # Check directories
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"✓ Directory: {dir_name}/")
        else:
            print(f"✗ Directory: {dir_name}/ - MISSING")
            all_ok = False
    
    # Check files
    for dir_path, files in required_files.items():
        for file_name in files:
            full_path = Path(dir_path) / file_name if dir_path else Path(file_name)
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"✓ File: {full_path} ({size} bytes)")
            else:
                print(f"✗ File: {full_path} - MISSING")
                all_ok = False
    
    return all_ok


def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_python_syntax():
    """Check syntax of all Python files."""
    print("\n" + "=" * 70)
    print("PYTHON SYNTAX CHECK")
    print("=" * 70)
    
    python_files = list(Path("workflow_parts").glob("*.py")) + \
                   list(Path("workflows").glob("*.py")) + \
                   [Path("01_simple_rag_cli.py")]
    
    all_ok = True
    
    for file_path in sorted(python_files):
        ok, error = check_syntax(file_path)
        if ok:
            print(f"✓ Syntax: {file_path}")
        else:
            print(f"✗ Syntax: {file_path}")
            print(f"  Error: {error}")
            all_ok = False
    
    return all_ok


def check_imports():
    """Check import statements in key files."""
    print("\n" + "=" * 70)
    print("IMPORT PATHS CHECK")
    print("=" * 70)
    
    files_to_check = [
        "workflows/simple_rag_workflow.py",
        "01_simple_rag_cli.py",
    ]
    
    all_ok = True
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or "(relative)"
                    imports.append(f"from {module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
            
            print(f"\n{file_path}:")
            for imp in imports:
                print(f"  {imp}")
        
        except Exception as e:
            print(f"✗ {file_path}: {e}")
            all_ok = False
    
    return all_ok


def print_structure_overview():
    """Print an overview of the new structure."""
    print("\n" + "=" * 70)
    print("STRUCTURE OVERVIEW")
    print("=" * 70)
    
    overview = """
workflow_parts/          [Reusable RAG building blocks]
├─ __init__.py          
├─ data_loading.py      ← PDF extraction, validation data loading
├─ chunking.py          ← Text segmentation strategies
├─ embedding.py         ← Vector embedding creation
├─ retrieval.py         ← Semantic search
├─ generation.py        ← LLM response generation
├─ evaluation.py        ← Response evaluation
└─ README.md            ← Component documentation

workflows/              [RAG workflow orchestrators]
├─ __init__.py
├─ simple_rag_workflow.py   ← SimpleRAGWorkflow class
└─ README.md               ← Workflow development guide

Root Level:
├─ 01_simple_rag_cli.py    ← CLI entry point for Simple RAG

Data Flow:
==========
01_simple_rag_cli.py
    ↓
SimpleRAGWorkflow (workflows/simple_rag_workflow.py)
    ↓
workflow_parts/ components
    ├─ data_loading.extract_text_from_pdf()
    ├─ chunking.chunk_text_sliding_window()
    ├─ embedding.create_embeddings()
    ├─ retrieval.semantic_search()
    ├─ generation.generate_response()
    └─ evaluation.evaluate_response()

Next Steps:
===========
1. Ensure all packages are installed: pip install -r requirements.txt
2. Configure credentials in .env
3. Run: python 01_simple_rag_cli.py
4. For other workflows, inherit from SimpleRAGWorkflow
"""
    print(overview)


def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print("NEW WORKFLOW STRUCTURE VALIDATION")
    print("=" * 70 + "\n")
    
    results = {
        "File Structure": check_file_structure(),
        "Python Syntax": check_python_syntax(),
        "Import Paths": check_imports(),
    }
    
    print_structure_overview()
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✓ ALL CHECKS PASSED!")
        print("\nNext: Install dependencies and run")
        print("  pip install -r requirements.txt")
        print("  python 01_simple_rag_cli.py")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
