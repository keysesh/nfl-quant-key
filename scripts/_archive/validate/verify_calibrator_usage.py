#!/usr/bin/env python3
"""
Verify Calibrator Usage Across Codebase
========================================

This script verifies that calibrators are being used correctly:
1. Player props SHOULD use calibrator (trained on player props)
2. Game lines SHOULD NOT use calibrator (simulation-based, already calibrated)
3. Calibrator file exists and is loaded correctly
"""

import sys
from pathlib import Path
import ast
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_file_for_calibrator_usage(file_path: Path):
    """Check a file for calibrator usage patterns."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except Exception as e:
        return {'error': str(e)}

    issues = []
    warnings = []
    info = []

    # Check if file handles game lines
    has_game_lines = False
    if 'game_line' in content.lower() or 'game_moneyline' in content or 'game_spread' in content:
        has_game_lines = True

    # Check if file handles player props
    has_player_props = False
    if 'player_prop' in content.lower() or 'player_reception' in content or 'player_rush' in content:
        has_player_props = True

    # Check for calibrator.transform calls
    calibrator_calls = re.findall(r'calibrator\.transform\([^)]+\)', content)

    # Check for game line + calibrator usage (BAD)
    if has_game_lines and calibrator_calls:
        # Check if calibrator is applied to game lines
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'game' in line.lower() and 'calibrator' in line.lower():
                # Check context around this line
                context = '\n'.join(lines[max(0, i-5):min(len(lines), i+5)])
                if 'moneyline' in context or 'spread' in context or 'total' in context:
                    if 'calibrator.transform' in context:
                        issues.append(f"Line {i+1}: Calibrator applied to game line (should NOT be calibrated)")

    # Check for player props without calibrator usage (WARNING, not error)
    if has_player_props:
        # Check if calibrator is available but not used
        if 'calibrator' in content.lower() and 'calibrator.transform' not in content:
            warnings.append(f"Calibrator available but not applied to player props")

    # Check if calibrator is loaded
    if 'NFLProbabilityCalibrator' in content or 'calibrator' in content.lower():
        if 'calibrator.load' in content or 'calibrator_path' in content:
            info.append("Calibrator loading code present")

    return {
        'has_game_lines': has_game_lines,
        'has_player_props': has_player_props,
        'calibrator_calls': len(calibrator_calls),
        'issues': issues,
        'warnings': warnings,
        'info': info
    }

def main():
    print("=" * 80)
    print("CALIBRATOR USAGE VERIFICATION")
    print("=" * 80)
    print()

    # Check calibrator file exists
    calibrator_path = Path('configs/calibrator.json')
    if calibrator_path.exists():
        size = calibrator_path.stat().st_size
        import os
        import datetime
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(calibrator_path))
        print(f"✅ Calibrator file exists: {calibrator_path}")
        print(f"   Size: {size} bytes")
        print(f"   Last modified: {mtime}")
    else:
        print(f"⚠️  Calibrator file NOT found: {calibrator_path}")
        print(f"   Run: python scripts/train/retrain_calibrator_full_history.py")

    print()

    # Check key files
    files_to_check = [
        Path('scripts/predict/generate_current_week_recommendations.py'),
        Path('scripts/predict/generate_model_predictions.py'),
        Path('scripts/utils/merge_framework_recommendations_fixed.py'),
        Path('nfl_quant/simulation/player_simulator.py'),
    ]

    all_issues = []
    all_warnings = []

    for file_path in files_to_check:
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue

        print(f"Checking: {file_path}")
        result = check_file_for_calibrator_usage(file_path)

        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            continue

        if result['has_game_lines']:
            print(f"   📊 Handles game lines: YES")
        if result['has_player_props']:
            print(f"   👤 Handles player props: YES")
        if result['calibrator_calls'] > 0:
            print(f"   🔧 Calibrator calls: {result['calibrator_calls']}")

        if result['issues']:
            print(f"   ❌ ISSUES:")
            for issue in result['issues']:
                print(f"      - {issue}")
                all_issues.append(f"{file_path}: {issue}")

        if result['warnings']:
            print(f"   ⚠️  WARNINGS:")
            for warning in result['warnings']:
                print(f"      - {warning}")
                all_warnings.append(f"{file_path}: {warning}")

        if result['info']:
            for info_item in result['info']:
                print(f"   ℹ️  {info_item}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_issues:
        print(f"❌ Found {len(all_issues)} ISSUES:")
        for issue in all_issues:
            print(f"   - {issue}")
        print()
        print("RECOMMENDATION: Fix these issues - game lines should NOT use calibrator")
    else:
        print("✅ No issues found - calibrator usage looks correct")

    if all_warnings:
        print(f"⚠️  Found {len(all_warnings)} warnings (non-critical)")

    print()
    print("EXPECTED BEHAVIOR:")
    print("  ✅ Player props: SHOULD use calibrator (trained on player props)")
    print("  ✅ Game lines: SHOULD NOT use calibrator (simulation-based)")
    print("  ✅ Calibrator file: Should exist and be up to date")

if __name__ == '__main__':
    main()





























