#!/usr/bin/env python3
"""
Comprehensive Integration Verification Script

Verifies that ALL 15+ factors are integrated consistently across the entire framework.

Checks:
1. Defensive EPA
2. Weather
3. Divisional
4. Contextual (Rest/Travel/Bye)
5. Injuries
6. Red Zone Factors
7. Snap Share
8. Home Field Advantage
9. Primetime
10. Altitude
11. Field Surface
12. Team Usage
13. Game Script
14. Market Blending
15. Calibration Consistency

Usage:
    python scripts/testing/verify_integration_completeness.py --week 10
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.unified_integration import verify_integration_completeness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_prediction_file(week: int) -> dict:
    """Check if prediction file has all factors integrated."""
    pred_file = Path(f'data/model_predictions_week{week}.csv')

    if not pred_file.exists():
        return {
            'file_exists': False,
            'factors': {}
        }

    try:
        df = pd.read_csv(pred_file)
        factors = verify_integration_completeness(df)
        return {
            'file_exists': True,
            'num_players': len(df),
            'factors': factors
        }
    except Exception as e:
        return {
            'file_exists': True,
            'error': str(e),
            'factors': {}
        }


def check_recommendation_generation() -> dict:
    """Check if recommendation generation script uses unified integration."""
    script_path = Path('scripts/predict/generate_current_week_recommendations.py')

    if not script_path.exists():
        return {'script_exists': False}

    try:
        content = script_path.read_text()
        uses_unified = 'integrate_all_factors' in content
        uses_individual = 'integrate_defensive_stats_into_predictions' in content

        return {
            'script_exists': True,
            'uses_unified_integration': uses_unified,
            'uses_individual_integration': uses_individual,
            'has_fallback': uses_individual and uses_unified
        }
    except Exception as e:
        return {
            'script_exists': True,
            'error': str(e)
        }


def check_model_predictions_script() -> dict:
    """Check if model predictions script uses unified integration."""
    script_path = Path('scripts/predict/generate_model_predictions.py')

    if not script_path.exists():
        return {'script_exists': False}

    try:
        content = script_path.read_text()
        uses_unified = 'integrate_all_factors' in content

        return {
            'script_exists': True,
            'uses_unified_integration': uses_unified
        }
    except Exception as e:
        return {
            'script_exists': True,
            'error': str(e)
        }


def check_backtest_scripts() -> dict:
    """Check if backtest scripts use unified integration."""
    backtest_dir = Path('scripts/backtest')

    if not backtest_dir.exists():
        return {'backtest_dir_exists': False}

    scripts = list(backtest_dir.glob('*.py'))
    results = {}

    for script_path in scripts:
        try:
            content = script_path.read_text()
            uses_unified = 'integrate_all_factors' in content
            has_epa_defaults = 'opponent_def_epa_vs_position.*=.*0.0' in content or 'opponent_def_epa.*=.*0.0' in content

            results[script_path.name] = {
                'uses_unified_integration': uses_unified,
                'has_epa_defaults': has_epa_defaults
            }
        except Exception as e:
            results[script_path.name] = {'error': str(e)}

    return {
        'backtest_dir_exists': True,
        'scripts': results
    }


def check_schema_fields() -> dict:
    """Check if PlayerPropInput schema has all required fields."""
    schema_path = Path('nfl_quant/schemas.py')

    if not schema_path.exists():
        return {'schema_exists': False}

    try:
        content = schema_path.read_text()

        required_fields = {
            'redzone_target_share': 'redzone_target_share' in content,
            'redzone_carry_share': 'redzone_carry_share' in content,
            'goalline_carry_share': 'goalline_carry_share' in content,
            'is_divisional_game': 'is_divisional_game' in content,
            'is_primetime_game': 'is_primetime_game' in content,
            'is_high_altitude': 'is_high_altitude' in content,
            'field_surface': 'field_surface' in content,
            'weather_total_adjustment': 'weather_total_adjustment' in content,
            'rest_epa_adjustment': 'rest_epa_adjustment' in content,
            'travel_epa_adjustment': 'travel_epa_adjustment' in content,
            'projected_team_pass_attempts': 'projected_team_pass_attempts' in content,
            'projected_team_rush_attempts': 'projected_team_rush_attempts' in content,
            'projected_team_targets': 'projected_team_targets' in content,
            'injury_qb_status': 'injury_qb_status' in content,
        }

        return {
            'schema_exists': True,
            'fields': required_fields,
            'all_fields_present': all(required_fields.values())
        }
    except Exception as e:
        return {
            'schema_exists': True,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Verify integration completeness across framework"
    )
    parser.add_argument(
        '--week',
        type=int,
        default=10,
        help='Week number to check (default: 10)'
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPREHENSIVE INTEGRATION VERIFICATION")
    print("="*80)
    print()

    # Check 1: Prediction file
    print("1. Checking prediction file...")
    pred_check = check_prediction_file(args.week)
    if pred_check.get('file_exists'):
        print(f"   ✅ Prediction file exists ({pred_check.get('num_players', 0)} players)")
        factors = pred_check.get('factors', {})
        for factor, present in factors.items():
            status = "✅" if present else "❌"
            print(f"   {status} {factor}: {'Present' if present else 'Missing'}")
    else:
        print(f"   ❌ Prediction file not found: data/model_predictions_week{args.week}.csv")
    print()

    # Check 2: Recommendation generation script
    print("2. Checking recommendation generation script...")
    rec_check = check_recommendation_generation()
    if rec_check.get('script_exists'):
        if rec_check.get('uses_unified_integration'):
            print("   ✅ Uses unified_integration module")
        else:
            print("   ❌ Does NOT use unified_integration module")
        if rec_check.get('has_fallback'):
            print("   ⚠️  Has fallback to individual integration (acceptable)")
    else:
        print("   ❌ Script not found")
    print()

    # Check 3: Model predictions script
    print("3. Checking model predictions script...")
    model_check = check_model_predictions_script()
    if model_check.get('script_exists'):
        if model_check.get('uses_unified_integration'):
            print("   ✅ Uses unified_integration module")
        else:
            print("   ❌ Does NOT use unified_integration module")
    else:
        print("   ❌ Script not found")
    print()

    # Check 4: Backtest scripts
    print("4. Checking backtest scripts...")
    backtest_check = check_backtest_scripts()
    if backtest_check.get('backtest_dir_exists'):
        scripts = backtest_check.get('scripts', {})
        for script_name, script_info in scripts.items():
            if script_info.get('uses_unified_integration'):
                print(f"   ✅ {script_name}: Uses unified integration")
            elif script_info.get('has_epa_defaults'):
                print(f"   ❌ {script_name}: Has EPA defaults (should use unified integration)")
            else:
                print(f"   ⚠️  {script_name}: Status unknown")
    else:
        print("   ❌ Backtest directory not found")
    print()

    # Check 5: Schema fields
    print("5. Checking PlayerPropInput schema...")
    schema_check = check_schema_fields()
    if schema_check.get('schema_exists'):
        fields = schema_check.get('fields', {})
        all_present = schema_check.get('all_fields_present', False)
        if all_present:
            print("   ✅ All required fields present in schema")
        else:
            print("   ❌ Missing fields in schema:")
            for field, present in fields.items():
                if not present:
                    print(f"      - {field}")
    else:
        print("   ❌ Schema file not found")
    print()

    # Summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    # Count issues
    issues = []
    if not pred_check.get('file_exists'):
        issues.append("Prediction file missing")
    if not rec_check.get('uses_unified_integration'):
        issues.append("Recommendation script doesn't use unified integration")
    if not model_check.get('uses_unified_integration'):
        issues.append("Model predictions script doesn't use unified integration")
    if not schema_check.get('all_fields_present'):
        issues.append("Schema missing required fields")

    if issues:
        print(f"\n⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\n✅ All checks passed! Integration appears complete.")

    print()


if __name__ == '__main__':
    main()
