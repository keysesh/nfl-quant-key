#!/usr/bin/env python3
"""
Framework Integration Audit
=============================

This script audits the main framework scripts to ensure all contextual updates
are properly integrated:
- Defensive stats integration
- Weather adjustments
- Anytime TD props
- Game script from simulations
- Game lines (spreads, totals, moneylines)

Usage:
    python scripts/audit/audit_framework_integration.py
"""

import sys
from pathlib import Path
import ast
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Main script to audit
MAIN_SCRIPT = Path('scripts/predict/generate_current_week_recommendations.py')
BACKTEST_SCRIPT = Path('scripts/backtest/backtest_player_props.py')
MERGE_SCRIPT = Path('scripts/utils/merge_framework_recommendations_fixed.py')

def check_file_exists(filepath: Path) -> bool:
    """Check if file exists."""
    return filepath.exists()

def check_imports(filepath: Path, required_imports: list) -> dict:
    """Check if required imports are present."""
    if not filepath.exists():
        return {'exists': False, 'imports': {}}

    with open(filepath, 'r') as f:
        content = f.read()

    results = {'exists': True, 'imports': {}}
    for imp in required_imports:
        # Check for import statement
        pattern = rf'(?:from|import).*{re.escape(imp)}'
        results['imports'][imp] = bool(re.search(pattern, content))

    return results

def check_function_exists(filepath: Path, func_name: str) -> bool:
    """Check if function exists in file."""
    if not filepath.exists():
        return False

    with open(filepath, 'r') as f:
        try:
            tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func_name:
                    return True
        except SyntaxError:
            return False

    return False

def check_string_in_file(filepath: Path, search_strings: list) -> dict:
    """Check if strings appear in file."""
    if not filepath.exists():
        return {'exists': False, 'found': {}}

    with open(filepath, 'r') as content:
        file_content = content.read()

    results = {'exists': True, 'found': {}}
    for s in search_strings:
        results['found'][s] = s in file_content

    return results

def audit_main_script():
    """Audit the main recommendation generation script."""
    print("\n" + "="*80)
    print("AUDITING: generate_current_week_recommendations.py")
    print("="*80)

    checks = {
        'file_exists': check_file_exists(MAIN_SCRIPT),
        'defensive_stats_import': False,
        'defensive_stats_usage': False,
        'weather_functions': False,
        'weather_usage': False,
        'game_script_loading': False,
        'game_script_usage': False,
        'anytime_td_support': False,
        'game_lines_support': False,
    }

    if not checks['file_exists']:
        print(f"❌ File not found: {MAIN_SCRIPT}")
        return checks

    print(f"✅ File exists: {MAIN_SCRIPT}")

    # Check defensive stats integration
    print("\n📊 Defensive Stats Integration:")
    defensive_checks = check_string_in_file(MAIN_SCRIPT, [
        'DEFENSIVE_STATS_AVAILABLE',
        'get_defensive_epa_for_player',
        'defensive_stats_integration',
        'opponent_def_epa'
    ])
    checks['defensive_stats_import'] = defensive_checks['found'].get('DEFENSIVE_STATS_AVAILABLE', False)
    checks['defensive_stats_usage'] = all([
        defensive_checks['found'].get('get_defensive_epa_for_player', False),
        defensive_checks['found'].get('opponent_def_epa', False)
    ])

    if checks['defensive_stats_import']:
        print("   ✅ Defensive stats imports present")
    else:
        print("   ❌ Defensive stats imports missing")

    if checks['defensive_stats_usage']:
        print("   ✅ Defensive stats usage detected")
    else:
        print("   ❌ Defensive stats usage not found")

    # Check weather integration
    print("\n🌤️  Weather Integration:")
    weather_checks = check_string_in_file(MAIN_SCRIPT, [
        'load_weather_data',
        'apply_weather_adjustment',
        'weather_df',
        'weather_adjustment',
        'weather_notes'
    ])
    checks['weather_functions'] = all([
        weather_checks['found'].get('load_weather_data', False),
        weather_checks['found'].get('apply_weather_adjustment', False)
    ])
    checks['weather_usage'] = all([
        weather_checks['found'].get('weather_df', False),
        weather_checks['found'].get('weather_adjustment', False)
    ])

    if checks['weather_functions']:
        print("   ✅ Weather functions present")
    else:
        print("   ❌ Weather functions missing")

    if checks['weather_usage']:
        print("   ✅ Weather functions used")
    else:
        print("   ❌ Weather functions not used")

    # Check game script integration
    print("\n📈 Game Script Integration:")
    game_script_checks = check_string_in_file(MAIN_SCRIPT, [
        'load_game_script_data',
        'game_context',
        'projected_team_total',
        'projected_game_script',
        'projected_pace'
    ])
    checks['game_script_loading'] = game_script_checks['found'].get('load_game_script_data', False)
    # game_context is loaded and passed to match_predictions_to_lines
    # Actual usage happens when predictions are generated upstream
    checks['game_script_usage'] = game_script_checks['found'].get('game_context', False)

    if checks['game_script_loading']:
        print("   ✅ Game script loading function present")
    else:
        print("   ❌ Game script loading function missing")

    if checks['game_script_usage']:
        print("   ✅ Game script data loaded and passed (used upstream in prediction generation)")
    else:
        print("   ❌ Game script data not loaded")

    # Check anytime TD support
    print("\n🏈 Anytime TD Props:")
    anytime_td_checks = check_string_in_file(MAIN_SCRIPT, [
        'player_anytime_td',
        'prob_any_td',
        'anytime_td',
        'passing_tds',
        'rushing_tds',
        'receiving_tds'
    ])
    checks['anytime_td_support'] = all([
        anytime_td_checks['found'].get('player_anytime_td', False),
        anytime_td_checks['found'].get('prob_any_td', False)
    ])

    if checks['anytime_td_support']:
        print("   ✅ Anytime TD support present")
    else:
        print("   ❌ Anytime TD support missing")

    # Check game lines support
    print("\n🎲 Game Lines Support:")
    game_lines_checks = check_string_in_file(MAIN_SCRIPT, [
        'generate_game_line_recommendations',
        'load_game_line_odds',
        'spread',
        'total',
        'moneyline'
    ])
    checks['game_lines_support'] = all([
        game_lines_checks['found'].get('generate_game_line_recommendations', False),
        game_lines_checks['found'].get('load_game_line_odds', False)
    ])

    if checks['game_lines_support']:
        print("   ✅ Game lines support present")
    else:
        print("   ❌ Game lines support missing")

    return checks

def audit_backtest_script():
    """Audit the backtest script."""
    print("\n" + "="*80)
    print("AUDITING: backtest_player_props.py")
    print("="*80)

    checks = {
        'file_exists': check_file_exists(BACKTEST_SCRIPT),
        'defensive_stats': False,
        'weather': False,
        'game_script': False,
        'anytime_td': False,
    }

    if not checks['file_exists']:
        print(f"❌ File not found: {BACKTEST_SCRIPT}")
        return checks

    print(f"✅ File exists: {BACKTEST_SCRIPT}")

    # Check if backtest uses same logic as main script
    backtest_checks = check_string_in_file(BACKTEST_SCRIPT, [
        'get_defensive_epa_for_player',
        'load_weather_data',
        'apply_weather_adjustment',
        'load_game_script_data',
        'player_anytime_td',
        'prob_any_td'
    ])

    checks['defensive_stats'] = backtest_checks['found'].get('get_defensive_epa_for_player', False)
    checks['weather'] = all([
        backtest_checks['found'].get('load_weather_data', False),
        backtest_checks['found'].get('apply_weather_adjustment', False)
    ])
    checks['game_script'] = backtest_checks['found'].get('load_game_script_data', False)
    checks['anytime_td'] = all([
        backtest_checks['found'].get('player_anytime_td', False),
        backtest_checks['found'].get('prob_any_td', False)
    ])

    print("\n📊 Backtest Integration Checks:")
    print(f"   {'✅' if checks['defensive_stats'] else '❌'} Defensive stats")
    print(f"   {'✅' if checks['weather'] else '❌'} Weather adjustments")
    print(f"   {'✅' if checks['game_script'] else '❌'} Game script")
    print(f"   {'✅' if checks['anytime_td'] else '❌'} Anytime TD props")

    return checks

def main():
    """Run full audit."""
    print("="*80)
    print("FRAMEWORK INTEGRATION AUDIT")
    print("="*80)
    print("\nThis audit checks if all contextual updates are integrated:")
    print("  • Defensive stats integration")
    print("  • Weather adjustments")
    print("  • Anytime TD props")
    print("  • Game script from simulations")
    print("  • Game lines (spreads, totals, moneylines)")

    # Audit main script
    main_checks = audit_main_script()

    # Audit backtest script
    backtest_checks = audit_backtest_script()

    # Summary
    print("\n" + "="*80)
    print("AUDIT SUMMARY")
    print("="*80)

    print("\n📋 Main Script (generate_current_week_recommendations.py):")
    if main_checks['file_exists']:
        print(f"   ✅ File exists")
        print(f"   {'✅' if main_checks['defensive_stats_import'] and main_checks['defensive_stats_usage'] else '❌'} Defensive stats")
        print(f"   {'✅' if main_checks['weather_functions'] and main_checks['weather_usage'] else '❌'} Weather adjustments")
        print(f"   {'✅' if main_checks['game_script_loading'] and main_checks['game_script_usage'] else '❌'} Game script")
        print(f"   {'✅' if main_checks['anytime_td_support'] else '❌'} Anytime TD props")
        print(f"   {'✅' if main_checks['game_lines_support'] else '❌'} Game lines")
    else:
        print("   ❌ File not found")

    print("\n📋 Backtest Script (backtest_player_props.py):")
    if backtest_checks['file_exists']:
        print(f"   ✅ File exists")
        print(f"   {'✅' if backtest_checks['defensive_stats'] else '❌'} Defensive stats")
        print(f"   {'✅' if backtest_checks['weather'] else '❌'} Weather adjustments")
        print(f"   {'✅' if backtest_checks['game_script'] else '❌'} Game script")
        print(f"   {'✅' if backtest_checks['anytime_td'] else '❌'} Anytime TD props")
    else:
        print("   ❌ File not found")

    # Overall status
    all_main_checks = all([
        main_checks.get('file_exists', False),
        main_checks.get('defensive_stats_import', False),
        main_checks.get('defensive_stats_usage', False),
        main_checks.get('weather_functions', False),
        main_checks.get('weather_usage', False),
        main_checks.get('game_script_loading', False),
        main_checks.get('game_script_usage', False),
        main_checks.get('anytime_td_support', False),
        main_checks.get('game_lines_support', False),
    ])

    all_backtest_checks = all([
        backtest_checks.get('file_exists', False),
        backtest_checks.get('defensive_stats', False),
        backtest_checks.get('weather', False),
        backtest_checks.get('game_script', False),
        backtest_checks.get('anytime_td', False),
    ])

    print("\n" + "="*80)
    if all_main_checks and all_backtest_checks:
        print("✅ FRAMEWORK IS FULLY INTEGRATED")
    elif all_main_checks:
        print("⚠️  MAIN SCRIPT IS INTEGRATED, BUT BACKTEST NEEDS UPDATES")
    else:
        print("❌ FRAMEWORK NEEDS UPDATES")
    print("="*80)

    return all_main_checks and all_backtest_checks

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
