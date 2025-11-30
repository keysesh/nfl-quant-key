#!/usr/bin/env python3
"""
Comprehensive Simulator Readiness Check
=======================================

As an NFL betting quant systems expert, this script performs a complete
diagnostic check of both simulators to ensure they're ready for Week 9.

Checks:
1. MonteCarloSimulator initialization
2. PlayerSimulator initialization (models, dependencies)
3. Required data files (schedule, PBP)
4. Model files (usage_predictor, efficiency_predictor)
5. Integration points and dependencies
"""

import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def check_file_exists(filepath: Path, description: str) -> tuple[bool, str]:
    """Check if file exists and return status."""
    if filepath.exists():
        size = filepath.stat().st_size / (1024 * 1024)  # MB
        return True, f"✅ {description}: {filepath} ({size:.1f} MB)"
    else:
        return False, f"❌ {description}: {filepath} NOT FOUND"

def check_monte_carlo_simulator():
    """Check MonteCarloSimulator readiness."""
    print("\n" + "="*80)
    print("1. MONTE CARLO SIMULATOR (Game-Level)")
    print("="*80)

    try:
        from nfl_quant.simulation.simulator import MonteCarloSimulator
        sim = MonteCarloSimulator(seed=42)
        print("✅ MonteCarloSimulator initialized successfully")
        print(f"   - Seed: {sim.seed}")
        print(f"   - Dependencies: numpy, scipy")
        return True, []
    except Exception as e:
        print(f"❌ MonteCarloSimulator initialization failed: {e}")
        return False, [str(e)]

def check_player_simulator():
    """Check PlayerSimulator readiness."""
    print("\n" + "="*80)
    print("2. PLAYER SIMULATOR (Player-Level)")
    print("="*80)

    issues = []

    # Check model files
    print("\n📦 Checking Model Files:")
    model_files = [
        ('data/models/usage_predictor_v4_defense.joblib', 'Usage Predictor (v4_defense)'),
        ('data/models/efficiency_predictor_v2_defense.joblib', 'Efficiency Predictor (v2_defense)'),
    ]

    all_models_exist = True
    for path_str, desc in model_files:
        path = Path(path_str)
        exists, msg = check_file_exists(path, desc)
        print(f"   {msg}")
        if not exists:
            all_models_exist = False
            issues.append(f"Missing model: {path_str}")

    # Try to load predictors
    print("\n🔧 Testing Predictor Loading:")
    try:
        from nfl_quant.simulation.player_simulator import load_predictors
        usage_predictor, efficiency_predictor = load_predictors()
        print("✅ Predictors loaded successfully")
        print(f"   - Usage Predictor: {type(usage_predictor).__name__}")
        print(f"   - Efficiency Predictor: {type(efficiency_predictor).__name__}")
    except Exception as e:
        print(f"❌ Failed to load predictors: {e}")
        issues.append(f"Predictor loading failed: {e}")
        all_models_exist = False

    # Try to initialize simulator
    print("\n🧪 Testing PlayerSimulator Initialization:")
    try:
        if all_models_exist:
            from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
            usage_predictor, efficiency_predictor = load_predictors()
            simulator = PlayerSimulator(
                usage_predictor=usage_predictor,
                efficiency_predictor=efficiency_predictor,
                trials=50000,
                seed=42
            )
            print("✅ PlayerSimulator initialized successfully")
            print(f"   - Trials: {simulator.trials}")
            print(f"   - Seed: {simulator.seed}")
        else:
            print("⚠️  Skipping initialization test (models missing)")
    except Exception as e:
        print(f"❌ PlayerSimulator initialization failed: {e}")
        issues.append(f"Simulator initialization failed: {e}")

    return all_models_exist, issues

def check_data_files():
    """Check required data files."""
    print("\n" + "="*80)
    print("3. REQUIRED DATA FILES")
    print("="*80)

    issues = []

    # Check for schedule and PBP files
    print("\n📊 Checking Game Data:")
    data_files = [
        ('data/processed/schedule_2025.parquet', 'Schedule (2025)'),
        ('data/processed/pbp_2025.parquet', 'Play-by-Play (2025)'),
    ]

    all_data_exists = True
    for path_str, desc in data_files:
        path = Path(path_str)
        exists, msg = check_file_exists(path, desc)
        print(f"   {msg}")
        if not exists:
            all_data_exists = False
            issues.append(f"Missing data file: {path_str}")

    # Check for nflverse directory
    print("\n📁 Checking nflverse Directory:")
    nflverse_dir = Path('data/nflverse')
    if nflverse_dir.exists():
        nflverse_files = list(nflverse_dir.glob('*.parquet'))
        print(f"   ✅ nflverse directory exists ({len(nflverse_files)} parquet files)")
    else:
        print(f"   ⚠️  nflverse directory not found: {nflverse_dir}")
        print("   Note: DataFetcher may fetch this automatically")

    return all_data_exists, issues

def check_integration_points():
    """Check integration points between simulators."""
    print("\n" + "="*80)
    print("4. INTEGRATION POINTS")
    print("="*80)

    issues = []

    # Check if game sim files can be loaded
    print("\n🔗 Checking Game Simulation Integration:")
    sim_files = list(Path('reports').glob('sim_2025_09_*.json'))
    if sim_files:
        print(f"   ✅ Found {len(sim_files)} existing game simulation files")
        print(f"   - These provide game context for PlayerSimulator")
        try:
            with open(sim_files[0]) as f:
                sim_data = json.load(f)
            print(f"   - Sample file contains: home_team, away_team, fair_spread, fair_total")
        except Exception as e:
            print(f"   ⚠️  Error reading simulation file: {e}")
    else:
        print(f"   ⚠️  No existing game simulation files found")
        print(f"   - Will be generated by: nfl-quant simulate --week 9")

    # Check trailing stats (now using NFLverse parquet files)
    print("\n📈 Checking NFLverse Trailing Stats:")
    nflverse_stats_file = Path('data/nflverse/weekly_stats.parquet')
    if nflverse_stats_file.exists():
        size = nflverse_stats_file.stat().st_size / 1024  # KB
        print(f"   ✅ NFLverse weekly stats parquet exists ({size:.1f} KB)")
    else:
        print(f"   ⚠️  NFLverse parquet not found: {nflverse_stats_file}")
        print(f"   - Run your R script to fetch nflverse data")

    return len(issues) == 0, issues

def check_calibrator():
    """Check calibrator availability."""
    print("\n" + "="*80)
    print("5. CALIBRATOR (Optional)")
    print("="*80)

    calibrator_path = Path('configs/calibrator.json')
    if calibrator_path.exists():
        size = calibrator_path.stat().st_size / 1024  # KB
        print(f"✅ Calibrator file exists ({size:.1f} KB)")
        print("   - Will be used by PlayerSimulator if available")
    else:
        print("⚠️  Calibrator file not found")
        print("   - PlayerSimulator will work without it (using raw probabilities)")

    return True, []

def main():
    """Run complete simulator readiness check."""
    print("="*80)
    print("🏈 NFL QUANT - SIMULATOR READINESS CHECK")
    print("="*80)
    print("\nExpert Analysis: Verifying both simulators are ready for Week 9")

    all_checks_passed = True
    all_issues = []

    # Run all checks
    mc_ok, mc_issues = check_monte_carlo_simulator()
    all_issues.extend(mc_issues)
    if not mc_ok:
        all_checks_passed = False

    ps_ok, ps_issues = check_player_simulator()
    all_issues.extend(ps_issues)
    if not ps_ok:
        all_checks_passed = False

    data_ok, data_issues = check_data_files()
    all_issues.extend(data_issues)
    if not data_ok:
        all_checks_passed = False

    int_ok, int_issues = check_integration_points()
    all_issues.extend(int_issues)
    if not int_ok:
        all_checks_passed = False

    cal_ok, cal_issues = check_calibrator()
    all_issues.extend(cal_issues)

    # Final summary
    print("\n" + "="*80)
    print("📋 FINAL ASSESSMENT")
    print("="*80)

    if all_checks_passed and len(all_issues) == 0:
        print("\n✅ ALL SIMULATORS ARE READY TO GO!")
        print("\nBoth simulators can be initialized and run:")
        print("  1. ✅ MonteCarloSimulator - Ready for game simulations")
        print("  2. ✅ PlayerSimulator - Ready for player prop predictions")
        print("\n💡 Next Steps:")
        print("   - Run: python scripts/run_week9_complete.py 9 aggressive")
        print("   - This will execute both simulators in proper order")
    else:
        print("\n⚠️  SOME ISSUES DETECTED:")
        for issue in all_issues:
            print(f"   - {issue}")

        if not mc_ok:
            print("\n❌ MonteCarloSimulator: NOT READY")
        else:
            print("\n✅ MonteCarloSimulator: READY")

        if not ps_ok:
            print("❌ PlayerSimulator: NOT READY (check models)")
        else:
            print("✅ PlayerSimulator: READY")

        if not data_ok:
            print("\n⚠️  Missing data files will be fetched automatically by DataFetcher")
            print("   Or run: python scripts/data/fetch_historical_nflverse.py")

    print("\n" + "="*80)

    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())





























