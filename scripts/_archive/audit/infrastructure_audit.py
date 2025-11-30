#!/usr/bin/env python3
"""
Infrastructure Audit: Find What's Built But Not Integrated

This script audits the codebase to identify:
1. Infrastructure that exists but isn't being used
2. Features that are partially implemented
3. Missing integrations that could improve predictions
"""

import re
from pathlib import Path


def find_unused_classes():
    """Find classes that are defined but never imported."""
    print("=" * 80)
    print("🔍 AUDITING INFRASTRUCTURE: What's Built But Not Integrated")
    print("=" * 80)
    print()

    issues = []

    # Check defensive stats
    print("1. DEFENSIVE STATS INTEGRATION")
    print("-" * 80)

    defensive_extractor_path = Path('nfl_quant/features/defensive_metrics.py')
    if defensive_extractor_path.exists():
        print("   ✅ DefensiveMetricsExtractor class exists")

        # Check if it's imported anywhere
        import_count = 0
        for py_file in Path('scripts').rglob('*.py'):
            try:
                content = py_file.read_text()
                has_extractor = 'DefensiveMetricsExtractor' in content
                has_metrics = 'defensive_metrics' in content
                if has_extractor or has_metrics:
                    import_count += 1
            except Exception:
                pass

        if import_count == 0:
            print("   ❌ NOT imported in any script")
            issues.append({
                'component': 'DefensiveMetricsExtractor',
                'status': 'Not integrated',
                'impact': 'High - All opponents treated as average',
                'file': str(defensive_extractor_path)
            })
        else:
            print(f"   ✅ Found in {import_count} files")
    else:
        print("   ❌ DefensiveMetricsExtractor not found")

    print()

    # Check anytime TD
    print("2. ANYTIME TD INTEGRATION")
    print("-" * 80)

    anytime_td_path = Path('scripts/predict/add_anytime_td_props.py')
    if anytime_td_path.exists():
        print("   ✅ Anytime TD script exists")

        # Check if it's called in pipeline
        pipeline_files = [
            'scripts/run_complete_pipeline.py',
            'scripts/predict/generate_current_week_recommendations.py'
        ]

        integrated = False
        for pipeline_file in pipeline_files:
            if Path(pipeline_file).exists():
                content = Path(pipeline_file).read_text()
                if 'add_anytime_td' in content or 'anytime_td' in content:
                    integrated = True
                    break

        if not integrated:
            print("   ❌ NOT integrated into pipeline")
            issues.append({
                'component': 'Anytime TD Props',
                'status': 'Not integrated',
                'impact': 'Medium - Missing high-volume market',
                'file': str(anytime_td_path)
            })
        else:
            print("   ✅ Integrated into pipeline")
    else:
        print("   ❌ Anytime TD script not found")

    print()

    # Check weather adjustments
    print("3. WEATHER ADJUSTMENTS")
    print("-" * 80)

    weather_mentions = 0
    for py_file in Path('scripts').rglob('*.py'):
        try:
            content = py_file.read_text()
            pattern = r'weather|temperature|wind|precipitation'
            if re.search(pattern, content, re.IGNORECASE):
                weather_mentions += 1
        except Exception:
            pass

    if weather_mentions == 0:
        print("   ❌ Weather adjustments not found")
        issues.append({
            'component': 'Weather Adjustments',
            'status': 'Not implemented',
            'impact': 'Low-Medium - Affects outdoor games',
            'file': 'N/A'
        })
    else:
        msg = (
            f"   ⚠️  Found {weather_mentions} mentions "
            f"(may not be integrated)"
        )
        print(msg)

    print()

    # Check game script integration
    print("4. GAME SCRIPT INTEGRATION")
    print("-" * 80)

    # Check if game script is hardcoded
    game_script_hardcoded = False
    for py_file in Path('scripts').rglob('*.py'):
        try:
            content = py_file.read_text()
            has_script = 'projected_game_script' in content
            has_defaults = '28.0' in content or '4.0' in content
            if has_script and has_defaults:
                game_script_hardcoded = True
                break
        except Exception:
            pass

    if game_script_hardcoded:
        print("   ⚠️  Game script appears hardcoded (not from simulation)")
        issues.append({
            'component': 'Game Script',
            'status': 'Hardcoded defaults',
            'impact': 'Medium - Should come from game simulation',
            'file': 'Multiple'
        })
    else:
        print("   ✅ Game script integration appears OK")

    print()

    # Check parlay integration
    print("5. PARLAY INTEGRATION")
    print("-" * 80)

    parlay_files = list(Path('nfl_quant/parlay').rglob('*.py'))
    if parlay_files:
        print(f"   ✅ Found {len(parlay_files)} parlay files")

        # Check if used in pipeline
        integrated = False
        pipeline_files = [
            'scripts/run_complete_pipeline.py',
            'scripts/optimize/optimize_bet_portfolio.py'
        ]
        for pipeline_file in pipeline_files:
            if Path(pipeline_file).exists():
                content = Path(pipeline_file).read_text()
                if 'parlay' in content.lower():
                    integrated = True
                    break

        if integrated:
            print("   ✅ Integrated into optimization")
        else:
            print("   ⚠️  May not be fully integrated")
    else:
        print("   ❌ Parlay files not found")

    print()

    # Summary
    print("=" * 80)
    print("📊 SUMMARY OF ISSUES")
    print("=" * 80)
    print()

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue['component']}")
            print(f"   Status: {issue['status']}")
            print(f"   Impact: {issue['impact']}")
            print(f"   File: {issue['file']}")
            print()
    else:
        print("✅ No major integration issues found!")

    return issues


if __name__ == '__main__':
    issues = find_unused_classes()

    print("=" * 80)
    print("🎯 RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Priority fixes:")
    print("1. Integrate defensive stats (HIGH impact)")
    print("2. Integrate anytime TD props (MEDIUM impact)")
    print("3. Integrate weather adjustments (LOW-MEDIUM impact)")
    print("4. Fix game script defaults (MEDIUM impact)")
    print()
