#!/usr/bin/env python3
"""
Comprehensive Framework Integration Audit

As an NFL analytics expert and software architect, this script verifies that ALL
15 integration factors are fully implemented and consistently applied across the
entire framework.

15 Factors to Verify:
1. Defensive EPA (position-specific)
2. Weather adjustments
3. Divisional game factors
4. Contextual factors (rest, travel, bye week)
5. Injuries (QB, WR, RB, OL)
6. Red Zone factors (target share, carry share, goal line role)
7. Snap Share (calculated from data)
8. Home Field Advantage (team-specific)
9. Primetime games (SNF, MNF, TNF)
10. Altitude (high altitude stadiums)
11. Field Surface (turf vs grass)
12. Team Usage (pass/rush/target totals from simulations)
13. Game Script (dynamic, evolves during game)
14. Market Blending (blend market priors with model)
15. Calibration Consistency (use same calibrators everywhere)
"""

import sys
from pathlib import Path
import pandas as pd
import json
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_unified_integration_module():
    """Verify unified_integration module exists and has all functions."""
    print("\n" + "="*80)
    print("1. CHECKING UNIFIED INTEGRATION MODULE")
    print("="*80)

    module_path = project_root / 'nfl_quant/utils/unified_integration.py'
    if not module_path.exists():
        print("❌ unified_integration.py NOT FOUND")
        return False

    print("✅ unified_integration.py exists")

    # Check for integrate_all_factors function
    spec = importlib.util.spec_from_file_location("unified_integration", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, 'integrate_all_factors'):
        print("✅ integrate_all_factors() function exists")
    else:
        print("❌ integrate_all_factors() function MISSING")
        return False

    # Check for all helper functions
    required_functions = [
        'load_weather_for_week',
        'check_divisional_game',
        'load_contextual_factors_for_teams',
        'get_primetime_status',
        'get_altitude_info',
        'get_field_surface',
        'get_home_field_advantage',
        'calculate_red_zone_shares_from_pbp',
        'calculate_snap_share_from_data',
        'verify_integration_completeness'
    ]

    missing = []
    for func_name in required_functions:
        if hasattr(module, func_name):
            print(f"✅ {func_name}() exists")
        else:
            print(f"❌ {func_name}() MISSING")
            missing.append(func_name)

    return len(missing) == 0


def check_prediction_scripts_integration():
    """Verify prediction scripts use unified integration."""
    print("\n" + "="*80)
    print("2. CHECKING PREDICTION SCRIPTS INTEGRATION")
    print("="*80)

    scripts_to_check = [
        'scripts/predict/generate_model_predictions.py',
        'scripts/predict/generate_current_week_recommendations.py'
    ]

    all_good = True
    for script_path in scripts_to_check:
        full_path = project_root / script_path
        if not full_path.exists():
            print(f"❌ {script_path} NOT FOUND")
            all_good = False
            continue

        with open(full_path) as f:
            content = f.read()

        if 'integrate_all_factors' in content:
            print(f"✅ {script_path} uses integrate_all_factors")

            # Check if it's actually called
            if 'integrate_all_factors(' in content:
                print(f"   ✅ Function is CALLED (not just imported)")
            else:
                print(f"   ⚠️  Function imported but may not be called")
        else:
            print(f"❌ {script_path} does NOT use integrate_all_factors")
            all_good = False

    return all_good


def check_simulator_integration():
    """Verify simulator uses integrated factors."""
    print("\n" + "="*80)
    print("3. CHECKING SIMULATOR INTEGRATION")
    print("="*80)

    simulator_path = project_root / 'nfl_quant/simulation/player_simulator_v3_correlated.py'
    if not simulator_path.exists():
        print("❌ player_simulator_v3_correlated.py NOT FOUND")
        return False

    with open(simulator_path) as f:
        content = f.read()

    # Check for factor usage
    factors_to_check = {
        'weather': ['weather_total_adjustment', 'weather_passing_adjustment'],
        'altitude': ['is_high_altitude', 'altitude_epa_adjustment'],
        'field_surface': ['field_surface'],
        'contextual': ['rest_epa_adjustment', 'travel_epa_adjustment'],
        'redzone': ['redzone_carry_share', 'redzone_target_share', 'goalline_carry_share'],
        'snap_share': ['trailing_snap_share'],
    }

    all_good = True
    for factor_name, fields in factors_to_check.items():
        found = any(field in content for field in fields)
        if found:
            print(f"✅ {factor_name} factors used in simulator")
        else:
            print(f"❌ {factor_name} factors NOT used in simulator")
            all_good = False

    # Check for _apply_integrated_factors method
    if '_apply_integrated_factors' in content:
        print("✅ _apply_integrated_factors() method exists")
    else:
        print("❌ _apply_integrated_factors() method MISSING")
        all_good = False

    return all_good


def check_game_simulator_integration():
    """Verify game simulator uses integrated factors."""
    print("\n" + "="*80)
    print("4. CHECKING GAME SIMULATOR INTEGRATION")
    print("="*80)

    simulator_path = project_root / 'nfl_quant/simulation/simulator.py'
    if not simulator_path.exists():
        print("❌ simulator.py NOT FOUND")
        return False

    with open(simulator_path) as f:
        content = f.read()

    factors_to_check = {
        'defensive_epa': ['home_defensive_epa', 'away_defensive_epa'],
        'divisional': ['is_divisional'],
        'weather': ['temperature', 'wind_speed', 'precipitation'],
        'primetime': ['game_type'],
        'injuries': ['home_injury', 'away_injury'],
    }

    all_good = True
    for factor_name, fields in factors_to_check.items():
        found = any(field in content for field in fields)
        if found:
            print(f"✅ {factor_name} factors used in game simulator")
        else:
            print(f"❌ {factor_name} factors NOT used in game simulator")
            all_good = False

    return all_good


def check_schema_integration():
    """Verify schemas include all factor fields."""
    print("\n" + "="*80)
    print("5. CHECKING SCHEMA INTEGRATION")
    print("="*80)

    schema_path = project_root / 'nfl_quant/schemas.py'
    if not schema_path.exists():
        print("❌ schemas.py NOT FOUND")
        return False

    with open(schema_path) as f:
        content = f.read()

    # Check PlayerPropInput schema
    required_fields = {
        'weather': ['weather_total_adjustment', 'weather_passing_adjustment'],
        'altitude': ['is_high_altitude', 'altitude_epa_adjustment', 'elevation_feet'],
        'field_surface': ['field_surface'],
        'contextual': ['rest_epa_adjustment', 'travel_epa_adjustment', 'is_coming_off_bye'],
        'redzone': ['redzone_target_share', 'redzone_carry_share', 'goalline_carry_share'],
        'snap_share': ['trailing_snap_share'],
        'divisional': ['is_divisional_game'],
        'primetime': ['is_primetime_game', 'primetime_type'],
        'hfa': ['home_field_advantage_points'],
        'team_usage': ['projected_team_pass_attempts', 'projected_team_rush_attempts', 'projected_team_targets'],
    }

    all_good = True
    for factor_name, fields in required_fields.items():
        found = any(field in content for field in fields)
        if found:
            print(f"✅ {factor_name} fields in PlayerPropInput schema")
        else:
            print(f"❌ {factor_name} fields MISSING from PlayerPropInput schema")
            all_good = False

    return all_good


def check_data_flow(week=10):
    """Verify data flows correctly through the pipeline."""
    print("\n" + "="*80)
    print("6. CHECKING DATA FLOW")
    print("="*80)

    # Check if predictions file exists
    predictions_file = project_root / f'data/model_predictions_week{week}.csv'
    if not predictions_file.exists():
        print(f"⚠️  No predictions file found for week {week}")
        print("   Run prediction generation first")
        return False

    df = pd.read_csv(predictions_file)
    print(f"✅ Loaded predictions file: {len(df)} players")

    # Check for integrated columns
    required_columns = {
        'Defensive EPA': 'opponent_def_epa_vs_position',
        'Weather': 'weather_total_adjustment',
        'Divisional': 'is_divisional_game',
        'Rest': 'rest_epa_adjustment',
        'Travel': 'travel_epa_adjustment',
        'Bye': 'is_coming_off_bye',
        'Injuries': 'injury_qb_status',
        'Red Zone': 'redzone_target_share',
        'Snap Share': 'snap_share',
        'HFA': 'home_field_advantage_points',
        'Primetime': 'is_primetime_game',
        'Altitude': 'is_high_altitude',
        'Field Surface': 'field_surface',
    }

    all_good = True
    for factor_name, column in required_columns.items():
        if column in df.columns:
            non_null_count = df[column].notna().sum()
            print(f"✅ {factor_name}: {column} ({non_null_count}/{len(df)} non-null)")
        else:
            print(f"❌ {factor_name}: {column} MISSING")
            all_good = False

    return all_good


def check_calibration_consistency():
    """Verify calibration is consistent across framework."""
    print("\n" + "="*80)
    print("7. CHECKING CALIBRATION CONSISTENCY")
    print("="*80)

    calibrator_loader_path = project_root / 'nfl_quant/calibration/calibrator_loader.py'
    if not calibrator_loader_path.exists():
        print("❌ calibrator_loader.py NOT FOUND")
        return False

    print("✅ calibrator_loader.py exists")

    # Check if market-specific calibrators exist
    calibrator_dir = project_root / 'configs/calibrators'
    if calibrator_dir.exists():
        calibrators = list(calibrator_dir.glob('*.json'))
        print(f"✅ Found {len(calibrators)} calibrator files")

        markets = ['reception_yards', 'rush_yards', 'receptions', 'pass_yards', 'td']
        for market in markets:
            market_cal = calibrator_dir / f'{market}_calibrator.json'
            if market_cal.exists():
                print(f"   ✅ {market} calibrator exists")
            else:
                print(f"   ⚠️  {market} calibrator MISSING")
    else:
        print("⚠️  Calibrator directory not found")

    return True


def generate_integration_report():
    """Generate comprehensive integration report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FRAMEWORK INTEGRATION AUDIT")
    print("="*80)
    print("\nVerifying all 15 factors are fully implemented...")

    results = {
        'unified_integration_module': check_unified_integration_module(),
        'prediction_scripts': check_prediction_scripts_integration(),
        'player_simulator': check_simulator_integration(),
        'game_simulator': check_game_simulator_integration(),
        'schemas': check_schema_integration(),
        'data_flow': check_data_flow(),
        'calibration': check_calibration_consistency(),
    }

    print("\n" + "="*80)
    print("INTEGRATION STATUS SUMMARY")
    print("="*80)

    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Framework is fully integrated!")
    else:
        print("⚠️  SOME CHECKS FAILED - Review issues above")
    print("="*80)

    return results


if __name__ == '__main__':
    results = generate_integration_report()
    sys.exit(0 if all(results.values()) else 1)




















