#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from workflow_parts.orchestration import discover_documents

files = discover_documents("data")
print(f"\nDiscovered {len(files)} document(s) in data/ directory:\n")
for f in files:
    print(f"  ✓ {Path(f).name}")
print()
