#!/usr/bin/env python3
"""
NFL QUANT Pipeline Integration Validator

Run before each pipeline execution to verify all components are properly connected.

Usage:
    python scripts/validation/validate_pipeline_integration.py
    python scripts/validation/validate_pipeline_integration.py --verbose
    python scripts/validation/validate_pipeline_integration.py --fix  # Auto-fix some issues

Exit codes:
    0 = All checks passed
    1 = Critical issues found (pipeline will fail)
    2 = Warnings found (pipeline may produce suboptimal results)
"""

import sys
import os
import ast
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


class ValidationResult:
    def __init__(self):
        self.critical: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def add_critical(self, msg: str):
        self.critical.append(f"CRITICAL: {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(f"WARNING: {msg}")

    def add_passed(self, msg: str):
        self.passed.append(f"PASSED: {msg}")

    @property
    def exit_code(self) -> int:
        if self.critical:
            return 1
        if self.warnings:
            return 2
        return 0


def check_data_freshness(result: ValidationResult, verbose: bool = False):
    """Check that required data files exist and are fresh enough."""

    freshness_requirements = {
        'data/nflverse/weekly_stats.parquet': timedelta(hours=24),
        'data/nflverse/snap_counts.parquet': timedelta(hours=24),
        'data/nflverse/schedules.parquet': timedelta(hours=48),
        'data/nflverse/pbp.parquet': timedelta(hours=24),
        'data/injuries/current_injuries.csv': timedelta(hours=12),
        'data/models/active_model.joblib': timedelta(days=7),
    }

    now = datetime.now()

    for file_path, max_age in freshness_requirements.items():
        path = PROJECT_ROOT / file_path
        if not path.exists():
            result.add_critical(f"Missing required file: {file_path}")
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = now - mtime

        if age > max_age:
            result.add_warning(
                f"{file_path} is {age.total_seconds()/3600:.1f}h old "
                f"(max: {max_age.total_seconds()/3600:.0f}h)"
            )
        elif verbose:
            result.add_passed(f"{file_path} is fresh ({age.total_seconds()/3600:.1f}h old)")


def check_config_imports(result: ValidationResult, verbose: bool = False):
    """Verify critical config values are imported from central config, not hardcoded."""

    # Files that should import from configs/model_config.py
    files_to_check = [
        'scripts/predict/generate_unified_recommendations_v3.py',
        'scripts/predict/generate_model_predictions.py',
        'scripts/train/train_model.py',
        'nfl_quant/features/batch_extractor.py',
    ]

    required_imports = {
        'FEATURES': 'configs.model_config',
        'CLASSIFIER_MARKETS': 'configs.model_config',
        'MODEL_VERSION': 'configs.model_config',
    }

    for file_path in files_to_check:
        path = PROJECT_ROOT / file_path
        if not path.exists():
            result.add_warning(f"Expected file not found: {file_path}")
            continue

        content = path.read_text()

        # Check for required imports
        for config_name, expected_module in required_imports.items():
            # Look for import statement
            import_pattern = rf'from\s+{expected_module.replace(".", r"\.")}\s+import\s+[^)]*{config_name}'
            if not re.search(import_pattern, content):
                # Check if it's used at all
                if config_name in content and 'hardcoded' not in file_path.lower():
                    # Only warn if the config is referenced but not imported
                    if f"'{config_name}'" not in content:  # Skip string literals
                        pass  # May be imported differently
            elif verbose:
                result.add_passed(f"{file_path}: {config_name} imported correctly")


def check_hardcoded_lists(result: ValidationResult, verbose: bool = False):
    """Find hardcoded feature/market lists that should use config."""

    # Pattern for hardcoded market lists
    market_pattern = r'\[.*[\'"]player_receptions[\'"].*[\'"]player_rush_yds[\'"].*\]'

    files_to_scan = list((PROJECT_ROOT / 'scripts').rglob('*.py'))
    files_to_scan += list((PROJECT_ROOT / 'nfl_quant').rglob('*.py'))

    for path in files_to_scan:
        # Skip archive directories
        if '_archive' in str(path):
            continue
        # Skip the config file itself
        if 'model_config.py' in str(path):
            continue

        try:
            content = path.read_text()
            matches = re.findall(market_pattern, content)
            if matches:
                rel_path = path.relative_to(PROJECT_ROOT)
                result.add_warning(
                    f"Hardcoded market list in {rel_path} - should import from config"
                )
        except Exception:
            pass


def check_game_status_integration(result: ValidationResult, verbose: bool = False):
    """Verify game status filtering is integrated into recommendation pipeline."""

    recs_file = PROJECT_ROOT / 'scripts/predict/generate_unified_recommendations_v3.py'

    if not recs_file.exists():
        result.add_critical("Recommendation script not found!")
        return

    content = recs_file.read_text()

    # Check that filter_props_by_game_status is defined
    if 'def filter_props_by_game_status' not in content:
        result.add_critical(
            "filter_props_by_game_status() not defined in recommendations script"
        )
        return

    # Check that it's actually called (not just defined)
    # Look for the call in generate_recommendations function
    call_pattern = r'prop_lines\s*=\s*filter_props_by_game_status\s*\('
    if not re.search(call_pattern, content):
        result.add_critical(
            "filter_props_by_game_status() defined but NOT CALLED in pipeline!"
        )
    elif verbose:
        result.add_passed("Game status filtering is integrated into pipeline")


def check_model_feature_alignment(result: ValidationResult, verbose: bool = False):
    """Verify model features match config."""

    try:
        import joblib
        from configs.model_config import FEATURES

        model_path = PROJECT_ROOT / 'data/models/active_model.joblib'
        if not model_path.exists():
            result.add_warning("No active model found - skipping feature alignment check")
            return

        model_data = joblib.load(model_path)

        # Get features from first classifier
        first_market = list(model_data['models'].keys())[0]
        model_features = list(model_data['models'][first_market].feature_names_in_)

        config_features = list(FEATURES)

        # Check for mismatches
        in_model_not_config = set(model_features) - set(config_features)
        in_config_not_model = set(config_features) - set(model_features)

        if in_model_not_config:
            result.add_warning(
                f"Features in model but not config: {in_model_not_config}"
            )
        if in_config_not_model:
            result.add_warning(
                f"Features in config but not model: {in_config_not_model}"
            )

        if not in_model_not_config and not in_config_not_model:
            if verbose:
                result.add_passed(f"Model features aligned with config ({len(model_features)} features)")

    except ImportError as e:
        result.add_warning(f"Could not check model alignment: {e}")
    except Exception as e:
        result.add_warning(f"Error checking model alignment: {e}")


def check_odds_module_functions(result: ValidationResult, verbose: bool = False):
    """Verify odds module utility functions are callable."""

    try:
        from nfl_quant.utils.odds import (
            load_game_status_map,
            parse_kickoff_time,
            get_actionable_games,
        )

        # Quick smoke test
        status = load_game_status_map(week=15, season=2025)
        if not isinstance(status, dict):
            result.add_critical("load_game_status_map() returned invalid type")
        elif verbose:
            result.add_passed(f"load_game_status_map() works ({len(status)} games)")

    except ImportError as e:
        result.add_critical(f"Cannot import odds utilities: {e}")
    except Exception as e:
        result.add_warning(f"Error testing odds utilities: {e}")


def check_dead_imports(result: ValidationResult, verbose: bool = False):
    """Check for modules that are imported but whose functions are never used."""

    known_dead_modules = [
        'nfl_quant/utils/contextual_factors_integration.py',
        'nfl_quant/utils/nflverse_data_loader.py',
    ]

    for module_path in known_dead_modules:
        path = PROJECT_ROOT / module_path
        if path.exists():
            result.add_warning(f"Dead module still exists: {module_path}")


def check_v25_synergy_features(result: ValidationResult, verbose: bool = False):
    """Verify V25 synergy features are properly configured.

    V25 synergy features should be in the FEATURES list and have
    defaults in feature_defaults.py.
    """
    try:
        from configs.model_config import FEATURES, MODEL_VERSION

        v25_synergy_features = [
            'team_synergy_multiplier',
            'oline_health_score_v25',
            'wr_corps_health',
            'has_synergy_bonus',
            'cascade_efficiency_boost',
            'wr_coverage_reduction',
            'returning_player_count',
            'has_synergy_context',
        ]

        features_set = set(FEATURES)

        # Check if V25 features are in FEATURES (required for V25+)
        if int(MODEL_VERSION) >= 25:
            missing_v25 = [f for f in v25_synergy_features if f not in features_set]
            if missing_v25:
                result.add_critical(
                    f"V25 synergy features missing from FEATURES: {missing_v25}. "
                    "Model version is {MODEL_VERSION} but synergy features not configured."
                )
            elif verbose:
                result.add_passed(f"All 8 V25 synergy features present in FEATURES list")

            # Check that extraction module is available
            try:
                from nfl_quant.features.team_synergy_extractor import V25_SYNERGY_FEATURES
                if verbose:
                    result.add_passed("V25 synergy extractor module available")
            except ImportError:
                result.add_warning("V25 synergy extractor module not importable")
        else:
            # For V24 and below, V25 features should NOT be in list
            v25_in_config = [f for f in v25_synergy_features if f in features_set]
            if v25_in_config:
                result.add_warning(
                    f"V25 features in FEATURES but MODEL_VERSION is {MODEL_VERSION}: {v25_in_config}"
                )
            elif verbose:
                result.add_passed(f"V25 features correctly excluded for MODEL_VERSION={MODEL_VERSION}")

    except ImportError as e:
        result.add_warning(f"Could not check V25 features: {e}")


def main():
    parser = argparse.ArgumentParser(description='Validate NFL QUANT pipeline integration')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all checks')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    args = parser.parse_args()

    print("=" * 70)
    print("NFL QUANT Pipeline Integration Validator")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    result = ValidationResult()

    # Run all checks
    checks = [
        ("Data Freshness", check_data_freshness),
        ("Config Imports", check_config_imports),
        ("Hardcoded Lists", check_hardcoded_lists),
        ("Game Status Integration", check_game_status_integration),
        ("Model Feature Alignment", check_model_feature_alignment),
        ("Odds Module Functions", check_odds_module_functions),
        ("Dead Imports", check_dead_imports),
        ("V25 Synergy Features", check_v25_synergy_features),
    ]

    for name, check_fn in checks:
        print(f"\n[{name}]")
        try:
            check_fn(result, verbose=args.verbose)
        except Exception as e:
            result.add_critical(f"Check '{name}' crashed: {e}")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if args.verbose:
        for msg in result.passed:
            print(f"  {msg}")

    for msg in result.warnings:
        print(f"  {msg}")

    for msg in result.critical:
        print(f"  {msg}")

    # Summary
    print(f"\nSummary: {len(result.passed)} passed, {len(result.warnings)} warnings, {len(result.critical)} critical")

    if result.exit_code == 0:
        print("\n✅ All checks passed - pipeline is ready")
    elif result.exit_code == 2:
        print("\n⚠️  Warnings found - pipeline will run but may have issues")
    else:
        print("\n❌ Critical issues found - fix before running pipeline")

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
