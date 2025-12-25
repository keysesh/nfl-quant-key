#!/usr/bin/env python3
"""
Validate No Hardcoded Version Numbers

This script checks for hardcoded version references outside of configs/model_config.py.
Run this before commits to ensure version-agnostic architecture is maintained.

Usage:
    python scripts/test/validate_no_hardcoded_versions.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Files allowed to contain version references
ALLOWED_FILES = [
    'configs/model_config.py',  # Single source of truth
    'configs/__init__.py',      # Re-exports backward-compat aliases
    'CHANGELOG.md',             # Historical changelog
    'scripts/test/validate_no_hardcoded_versions.py',  # This script
]

# Directories to skip
SKIP_DIRS = [
    '_archive',
    '__pycache__',
    '.venv',
    '.git',
    '.claude',      # Claude memory files
    'node_modules',
    'data/models',  # Model files may contain version metadata
    'data/backtest',
    'reports',
    'docs/',        # Documentation with historical references
]

# Files to skip entirely (documentation)
SKIP_FILES = [
    'README.md',
    'CLAUDE.md',    # Project documentation
    'CODEBASE_MODERNIZATION_REPORT.md',
    'FEATURE_IMPLEMENTATION_PLAN.md',
    'TIER_1_2_',   # Any tier integration docs
    '_COMPLETE.md',
    '_GUIDE.md',
]

# Patterns to detect hardcoded versions (only problematic code patterns)
VERSION_PATTERNS = [
    r'train_v\d+',        # train_v18, train_v19 (scripts)
]


def should_skip(path: Path) -> bool:
    """Check if path should be skipped."""
    path_str = str(path)
    for skip in SKIP_DIRS:
        if skip in path_str:
            return True
    # Also skip documentation files
    for skip_file in SKIP_FILES:
        if skip_file in path_str:
            return True
    return False


def is_allowed_file(path: Path) -> bool:
    """Check if file is allowed to contain version refs."""
    rel_path = str(path.relative_to(PROJECT_ROOT))
    for allowed in ALLOWED_FILES:
        if allowed in rel_path:
            return True
    return False


def check_file(path: Path) -> list:
    """Check a file for hardcoded versions."""
    violations = []

    try:
        content = path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Skip comments and docstrings (historical context is OK)
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue

            for pattern in VERSION_PATTERNS:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Skip if it's a reference to config import
                    if 'from configs.model_config import' in line:
                        continue
                    if 'from configs import' in line:
                        continue
                    if 'configs/model_config.py' in line:
                        continue
                    # Skip backward compatibility aliases
                    if 'backward compat' in line.lower():
                        continue
                    if 'legacy' in line.lower():
                        continue
                    if 'alias' in line.lower():
                        continue
                    # Skip string mentions in logs/messages
                    if 'logger.' in line or 'print(' in line or 'typer.echo' in line:
                        continue

                    violations.append({
                        'file': str(path.relative_to(PROJECT_ROOT)),
                        'line': i,
                        'match': match,
                        'content': line.strip()[:80]
                    })
    except Exception as e:
        print(f"Error reading {path}: {e}")

    return violations


def main():
    """Main validation function."""
    print("=" * 70)
    print("VALIDATING NO HARDCODED VERSION NUMBERS")
    print("=" * 70)

    all_violations = []
    files_checked = 0

    # Check Python files
    for pattern in ['**/*.py', '**/*.md']:
        for path in PROJECT_ROOT.glob(pattern):
            if should_skip(path):
                continue
            if is_allowed_file(path):
                continue
            if not path.is_file():
                continue

            files_checked += 1
            violations = check_file(path)
            all_violations.extend(violations)

    print(f"\nFiles checked: {files_checked}")
    print(f"Violations found: {len(all_violations)}")

    if all_violations:
        print("\n" + "=" * 70)
        print("VIOLATIONS:")
        print("=" * 70)

        # Group by file
        by_file = {}
        for v in all_violations:
            if v['file'] not in by_file:
                by_file[v['file']] = []
            by_file[v['file']].append(v)

        for file, violations in sorted(by_file.items()):
            print(f"\n{file}:")
            for v in violations:
                print(f"  Line {v['line']}: {v['match']}")
                print(f"    {v['content']}")

        print("\n" + "=" * 70)
        print("FAILED: Found hardcoded version references")
        print("Fix: Use configs/model_config.py imports instead")
        print("=" * 70)
        return 1
    else:
        print("\n" + "=" * 70)
        print("PASSED: No hardcoded version references found")
        print("=" * 70)
        return 0


if __name__ == '__main__':
    sys.exit(main())
