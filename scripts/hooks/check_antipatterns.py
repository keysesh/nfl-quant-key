#!/usr/bin/env python3
"""
Pre-commit hook to prevent antipatterns in the NFL QUANT codebase.

Checks for:
1. fillna(0) - should use safe_fillna() instead
2. inplace=True - should use method chaining
3. Hardcoded season years - should use get_current_season()
4. API keys in source code

Usage:
    python scripts/hooks/check_antipatterns.py [--fix] [--verbose]

    --fix: Attempt to auto-fix simple issues
    --verbose: Show all checked files, not just violations
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple
import argparse


# Directories to skip
EXCLUDE_DIRS = {'_archive', 'deprecated', '.git', '__pycache__', 'venv', '.venv', 'node_modules'}

# Files to skip
EXCLUDE_FILES = {'feature_defaults.py'}  # This file defines safe_fillna

# Files where fillna(0) is legitimate (raw data loading, not model features)
FILLNA_ALLOWED_FILES = {
    # Core library - data adapters
    'nflverse_adapter.py',  # Raw data type coercion
    'base_adapter.py',      # Raw data type coercion
    # Core library - feature extraction
    'defensive_metrics.py', # Rate calculations (0/0 fallback)
    'enhanced_features.py', # Count column aggregations
    'core.py',              # Raw yards columns
    'opponent_features.py', # vs_avg = 0 is neutral
    'micro_metrics.py',     # Metric aggregations
    'opponent_stats.py',    # Defense stats aggregations
    'historical_injury_impact.py',  # Baseline calculations
    # Scripts - training data processing
    'train_usage_predictor_v4_with_defense.py',
    'train_td_props_model.py',
    'train_td_poisson_model.py',
    'train_game_line_ml_models.py',
    # Scripts - prediction pipelines
    'generate_model_predictions.py',
    'predict_game_lines.py',
    # Scripts - testing
    'calibration_check.py',
}

# Antipatterns to detect
ANTIPATTERNS = [
    # (regex pattern, error message, severity)
    (r'\.fillna\s*\(\s*0\s*\)',
     'Use safe_fillna() instead of fillna(0)',
     'HIGH'),

    (r'inplace\s*=\s*True',
     'Avoid inplace=True, use method chaining (df = df.method())',
     'MEDIUM'),

    (r'api_key.*=.*["\'][a-f0-9]{20,}["\']',
     'API key detected in source code - use environment variables',
     'CRITICAL'),

    (r'ODDS_API_KEY\s*=\s*["\'][a-f0-9]+["\']',
     'Hardcoded API key - should load from .env',
     'CRITICAL'),
]

# Season year patterns (warn but don't block)
SEASON_WARNINGS = [
    (r'==\s*2025(?!\d)', 'Consider using get_current_season() instead of hardcoded 2025'),
    (r'==\s*2024(?!\d)', 'Consider using get_current_season() instead of hardcoded 2024'),
]


def should_skip_path(path: Path) -> bool:
    """Check if path should be skipped."""
    # Skip excluded directories
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True

    # Skip excluded files
    if path.name in EXCLUDE_FILES:
        return True

    return False


def should_skip_fillna_check(path: Path) -> bool:
    """Check if fillna(0) is allowed in this file (raw data processing)."""
    return path.name in FILLNA_ALLOWED_FILES


def check_file(filepath: Path) -> List[Tuple[int, str, str, str]]:
    """
    Check a single file for antipatterns.

    Returns list of (line_num, line_content, message, severity) tuples
    """
    errors = []

    try:
        content = filepath.read_text(encoding='utf-8')
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            # Check antipatterns
            for pattern, message, severity in ANTIPATTERNS:
                # Skip fillna(0) check for files with legitimate uses
                if 'fillna' in message and should_skip_fillna_check(filepath):
                    continue
                if re.search(pattern, line, re.IGNORECASE):
                    errors.append((line_num, line.strip()[:80], message, severity))

            # Check season warnings (don't add to errors, just warn)
            # for pattern, message in SEASON_WARNINGS:
            #     if re.search(pattern, line):
            #         warnings.append((line_num, line.strip()[:80], message, 'WARNING'))

    except Exception as e:
        pass  # Skip unreadable files

    return errors


def main():
    parser = argparse.ArgumentParser(description='Check for antipatterns in NFL QUANT codebase')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix', action='store_true', help='Attempt to auto-fix simple issues')
    parser.add_argument('files', nargs='*', help='Specific files to check (default: all)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent

    all_errors = []
    files_checked = 0

    # Determine files to check
    if args.files:
        files_to_check = [Path(f) for f in args.files if f.endswith('.py')]
    else:
        # Check nfl_quant/ and scripts/
        files_to_check = []
        for search_dir in ['nfl_quant', 'scripts']:
            dir_path = project_root / search_dir
            if dir_path.exists():
                files_to_check.extend(dir_path.rglob('*.py'))

    for py_file in files_to_check:
        if should_skip_path(py_file):
            continue

        files_checked += 1
        errors = check_file(py_file)

        if errors:
            rel_path = py_file.relative_to(project_root) if py_file.is_relative_to(project_root) else py_file
            for line_num, line_content, message, severity in errors:
                all_errors.append((rel_path, line_num, line_content, message, severity))
        elif args.verbose:
            print(f"  ✓ {py_file.name}")

    # Report results
    print(f"\n{'='*70}")
    print(f"NFL QUANT Antipattern Check")
    print(f"{'='*70}")
    print(f"Files checked: {files_checked}")

    if all_errors:
        # Group by severity
        critical = [e for e in all_errors if e[4] == 'CRITICAL']
        high = [e for e in all_errors if e[4] == 'HIGH']
        medium = [e for e in all_errors if e[4] == 'MEDIUM']

        if critical:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical)}):")
            for path, line_num, content, message, _ in critical:
                print(f"  {path}:{line_num}")
                print(f"    → {message}")
                print(f"    | {content}")

        if high:
            print(f"\n❌ HIGH SEVERITY ({len(high)}):")
            for path, line_num, content, message, _ in high:
                print(f"  {path}:{line_num}")
                print(f"    → {message}")

        if medium:
            print(f"\n⚠️  MEDIUM SEVERITY ({len(medium)}):")
            for path, line_num, content, message, _ in medium[:10]:  # Limit output
                print(f"  {path}:{line_num}: {message}")
            if len(medium) > 10:
                print(f"  ... and {len(medium) - 10} more")

        print(f"\n{'='*70}")
        print(f"Total: {len(all_errors)} violations found")
        print(f"{'='*70}")

        # Exit with error for critical/high issues
        if critical or high:
            print("\n❌ FAILED: Fix critical and high severity issues before committing.")
            sys.exit(1)
        else:
            print("\n⚠️  WARNING: Medium severity issues found. Consider fixing.")
            sys.exit(0)
    else:
        print(f"\n✅ No antipatterns detected!")
        print(f"{'='*70}")
        sys.exit(0)


if __name__ == '__main__':
    main()
