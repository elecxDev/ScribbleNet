"""
ScribbleNet - Structure Validation Script
Validates and optionally fixes the project directory structure.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.file_manager import (
    clean_project,
    detect_redundant_files,
    validate_structure,
)
from utils.logger import get_logger

logger = get_logger("scribblenet.validate")


def run_validation(fix: bool = False) -> None:
    """
    Run full project structure validation.

    Args:
        fix: If True, attempt to fix missing directories.
    """
    print("\n" + "=" * 50)
    print("  ScribbleNet - Structure Validation")
    print("=" * 50)

    # Validate structure
    issues = validate_structure(fix=fix)

    if issues:
        print(f"\n  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
        if fix:
            print("\n  Auto-fix applied for missing directories.")
    else:
        print("\n  All checks passed! Project structure is valid.")

    # Check for redundant files
    redundant = detect_redundant_files()
    if redundant:
        print(f"\n  Found {len(redundant)} redundant file(s):")
        for r in redundant:
            print(f"    - {r.name}")
    else:
        print("\n  No redundant files detected.")

    print("=" * 50)


if __name__ == "__main__":
    fix_mode = "--fix" in sys.argv
    run_validation(fix=fix_mode)
