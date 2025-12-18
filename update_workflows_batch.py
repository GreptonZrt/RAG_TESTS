"""
Batch Mode Updater Script

This script adds --batch flag support to all workflows that use argparse.
Workflows with --batch flag will run in minimal output mode during test_all_workflows.py runs.
"""

import re
from pathlib import Path


def add_batch_flag_to_workflow(filepath):
    """
    Add --batch flag to a workflow's argument parser.
    
    Returns True if successful, False if already has it or error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already has --batch
        if '--batch' in content:
            print(f"  [SKIP] Already has --batch: {filepath.name}")
            return False
        
        # Check if it has argparse
        if 'argparse' not in content or 'add_argument' not in content:
            print(f"  [SKIP] No argparse found: {filepath.name}")
            return False
        
        # Find the last add_argument line and add --batch after it
        # Pattern: parser.add_argument(...) - we want to add after the last one before args = parser.parse_args()
        
        # Find all add_argument calls
        add_arg_pattern = r'(parser\.add_argument\([^)]+\))'
        matches = list(re.finditer(add_arg_pattern, content, re.DOTALL))
        
        if not matches:
            print(f"  [SKIP] No add_argument found: {filepath.name}")
            return False
        
        # Get the position after the last add_argument
        last_match = matches[-1]
        insert_pos = last_match.end()
        
        # Add new line with --batch argument
        batch_arg = '\n    parser.add_argument("--batch", action="store_true", help="Batch mode (minimal output)")'
        
        new_content = content[:insert_pos] + batch_arg + content[insert_pos:]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  [UPDATED] Added --batch flag: {filepath.name}")
        return True
    
    except Exception as e:
        print(f"  [ERROR] {filepath.name}: {e}")
        return False


def main():
    """Update all workflow files to support --batch flag."""
    workflows_dir = Path("workflows")
    
    if not workflows_dir.exists():
        print(f"Error: {workflows_dir} not found")
        return
    
    # Get all workflow Python files
    workflow_files = sorted([f for f in workflows_dir.glob("*.py") 
                            if f.name.startswith(tuple("0123456789"))
                            and not f.name.endswith("_template.py")
                            and f.name.endswith(".py")])
    
    print(f"\nUpdating {len(workflow_files)} workflow files...\n")
    
    updated = 0
    for wf in workflow_files:
        if add_batch_flag_to_workflow(wf):
            updated += 1
    
    print(f"\n✓ Updated {updated} workflow(s) with --batch flag support")
    print("\nNote: Workflows now support --batch flag for minimal output mode.")
    print("Use: python workflows/XX_name.py --batch --max 1 --no-eval")


if __name__ == "__main__":
    main()
