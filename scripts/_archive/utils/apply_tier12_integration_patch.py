#!/usr/bin/env python3
"""
Apply TIER 1 & TIER 2 Integration Patch

This script applies the TIER 1 & 2 feature integration to generate_model_predictions.py.

Usage:
    python scripts/utils/apply_tier12_integration_patch.py [--dry-run] [--backup]

Options:
    --dry-run: Show what would be changed without making modifications
    --backup: Create backup of original file before modifying
"""

import sys
import shutil
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
PREDICTIONS_FILE = PROJECT_ROOT / 'scripts/predict/generate_model_predictions.py'
PATCH_FILE = PROJECT_ROOT / 'scripts/predict/generate_model_predictions_tier12_PATCH.py'
BACKUP_FILE = PROJECT_ROOT / 'scripts/predict/generate_model_predictions_BACKUP.py'


def create_backup():
    """Create backup of original file."""
    if PREDICTIONS_FILE.exists():
        shutil.copy2(PREDICTIONS_FILE, BACKUP_FILE)
        print(f"✓ Created backup: {BACKUP_FILE}")
    else:
        print(f"✗ Original file not found: {PREDICTIONS_FILE}")
        sys.exit(1)


def add_tier12_import(content: str) -> str:
    """Add TIER 1 & 2 import statement."""
    import_line = "from nfl_quant.features.tier1_2_integration import extract_all_tier1_2_features"

    # Check if already imported
    if import_line in content:
        print("  ✓ TIER 1 & 2 import already present")
        return content

    # Find import section (after other nfl_quant imports)
    lines = content.split('\n')
    insert_idx = None

    for i, line in enumerate(lines):
        if line.startswith('from nfl_quant.') or line.startswith('import nfl_quant.'):
            insert_idx = i + 1

    if insert_idx is None:
        # Fallback: insert after last import
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1

    if insert_idx is not None:
        lines.insert(insert_idx, import_line)
        print(f"  ✓ Added import at line {insert_idx}")
        return '\n'.join(lines)
    else:
        print("  ✗ Could not find import section")
        return content


def add_cli_flags(content: str) -> str:
    """Add CLI flags for TIER 1 & 2 features."""
    # Find argparse section
    if '--use-tier12' in content:
        print("  ✓ CLI flags already present")
        return content

    flags_to_add = """
    # TIER 1 & TIER 2 feature flags
    parser.add_argument('--use-tier12', action='store_true',
                        help='Enable TIER 1 & 2 enhanced features (EWMA, regime, game script, NGS, EPA)')
    parser.add_argument('--no-ewma', action='store_true',
                        help='Disable EWMA weighting (use simple mean)')
    parser.add_argument('--no-regime', action='store_true',
                        help='Disable regime detection features')
    parser.add_argument('--no-game-script', action='store_true',
                        help='Disable game script features')
    parser.add_argument('--no-ngs', action='store_true',
                        help='Disable NGS advanced metrics')
    parser.add_argument('--no-situational-epa', action='store_true',
                        help='Disable situational EPA features')
"""

    # Find last parser.add_argument
    lines = content.split('\n')
    insert_idx = None

    for i in range(len(lines) - 1, -1, -1):
        if 'parser.add_argument' in lines[i]:
            # Find end of this argument (next line that doesn't start with whitespace + help=)
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith('help=') or
                                     lines[j].strip().startswith('default=') or
                                     lines[j].strip().startswith('type=') or
                                     lines[j].strip().startswith('action=')):
                j += 1
            insert_idx = j
            break

    if insert_idx is not None:
        lines.insert(insert_idx, flags_to_add)
        print(f"  ✓ Added CLI flags at line {insert_idx}")
        return '\n'.join(lines)
    else:
        print("  ✗ Could not find argparse section")
        return content


def replace_load_trailing_stats_function(content: str) -> str:
    """Replace load_trailing_stats() with TIER 1 & 2 enhanced version."""
    # Read patch file to get new function
    if not PATCH_FILE.exists():
        print(f"  ✗ Patch file not found: {PATCH_FILE}")
        return content

    with open(PATCH_FILE, 'r') as f:
        patch_content = f.read()

    # Extract the enhanced function from patch file
    func_start = patch_content.find('def load_trailing_stats_tier12(')
    if func_start == -1:
        print("  ✗ Could not find load_trailing_stats_tier12 in patch file")
        return content

    # Find end of function (next def at same indentation level)
    func_end = patch_content.find('\n# ============================================================================', func_start)
    if func_end == -1:
        func_end = len(patch_content)

    new_function = patch_content[func_start:func_end].strip()

    # Find and replace old function in content
    old_func_start = content.find('def load_trailing_stats(')
    if old_func_start == -1:
        print("  ✗ Could not find load_trailing_stats() in original file")
        return content

    # Find end of old function (next def at same indentation level)
    old_func_end = content.find('\n\ndef ', old_func_start + 1)
    if old_func_end == -1:
        print("  ✗ Could not find end of load_trailing_stats()")
        return content

    # Replace
    modified = content[:old_func_start] + new_function + content[old_func_end:]
    print("  ✓ Replaced load_trailing_stats() with TIER 1 & 2 enhanced version")

    return modified


def update_function_calls(content: str) -> str:
    """Update function calls to use new parameters."""
    # Find calls to load_trailing_stats
    old_pattern = "load_trailing_stats("
    new_pattern = "load_trailing_stats_tier12("

    if old_pattern in content:
        count = content.count(old_pattern)
        content = content.replace(old_pattern, new_pattern)
        print(f"  ✓ Updated {count} function call(s) to load_trailing_stats_tier12()")

        # Add parameters to function calls
        # Look for pattern: load_trailing_stats_tier12(current_week=..., current_season=...)
        # and add TIER 1 & 2 parameters

        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'load_trailing_stats_tier12(' in line and 'use_tier12' not in line:
                # Find the closing parenthesis
                if ')' in line:
                    # Single-line call
                    lines[i] = line.replace(
                        ')',
                        ',\n        use_tier12=args.use_tier12 if hasattr(args, "use_tier12") else False,\n'
                        '        use_ewma=not (args.no_ewma if hasattr(args, "no_ewma") else False),\n'
                        '        use_regime=not (args.no_regime if hasattr(args, "no_regime") else False),\n'
                        '        use_game_script=not (args.no_game_script if hasattr(args, "no_game_script") else False),\n'
                        '        use_ngs=not (args.no_ngs if hasattr(args, "no_ngs") else False),\n'
                        '        use_situational_epa=not (args.no_situational_epa if hasattr(args, "no_situational_epa") else False)\n'
                        '    )'
                    )

        content = '\n'.join(lines)
        print("  ✓ Added TIER 1 & 2 parameters to function calls")

    return content


def apply_patch(dry_run=False, create_backup_flag=True):
    """Apply the complete TIER 1 & 2 integration patch."""
    print(f"\n{'='*80}")
    print("TIER 1 & TIER 2 INTEGRATION PATCH")
    print(f"{'='*80}\n")

    if not PREDICTIONS_FILE.exists():
        print(f"✗ File not found: {PREDICTIONS_FILE}")
        return False

    # Create backup
    if create_backup_flag and not dry_run:
        create_backup()

    # Read original file
    with open(PREDICTIONS_FILE, 'r') as f:
        content = f.read()

    print("\nApplying patches:")
    print("-" * 80)

    # Apply patches
    print("\n1. Adding TIER 1 & 2 imports...")
    content = add_tier12_import(content)

    print("\n2. Adding CLI flags...")
    content = add_cli_flags(content)

    print("\n3. Replacing load_trailing_stats() function...")
    content = replace_load_trailing_stats_function(content)

    print("\n4. Updating function calls...")
    content = update_function_calls(content)

    if dry_run:
        print(f"\n{'='*80}")
        print("DRY RUN - No changes made")
        print(f"{'='*80}")
        return True
    else:
        # Write modified file
        with open(PREDICTIONS_FILE, 'w') as f:
            f.write(content)

        print(f"\n{'='*80}")
        print("✅ PATCH APPLIED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nModified file: {PREDICTIONS_FILE}")
        if create_backup_flag:
            print(f"Backup saved: {BACKUP_FILE}")

        print("\n📋 Next steps:")
        print("1. Test with: python scripts/predict/generate_model_predictions.py --week 11 --use-tier12")
        print("2. Compare predictions with/without --use-tier12 flag")
        print("3. Retrain models with new features (see scripts/train/)")
        print("4. Validate with temporal cross-validation")

        return True


def main():
    parser = argparse.ArgumentParser(description='Apply TIER 1 & 2 integration patch')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    parser.add_argument('--backup', action='store_true', default=True, help='Create backup (default: True)')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')

    args = parser.parse_args()

    create_backup_flag = args.backup and not args.no_backup

    success = apply_patch(dry_run=args.dry_run, create_backup_flag=create_backup_flag)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
