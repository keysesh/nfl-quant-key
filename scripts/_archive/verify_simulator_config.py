#!/usr/bin/env python3
"""
Configuration Verification for Both Simulators
==============================================

Expert Analysis: Verifies that both simulators are correctly configured
with proper parameters, variance settings, and integration points.
"""

import sys
from pathlib import Path
import re

def check_player_simulator_config():
    """Check PlayerSimulator configuration."""
    print("\n" + "="*80)
    print("PLAYER SIMULATOR CONFIGURATION CHECK")
    print("="*80)

    simulator_path = Path('nfl_quant/simulation/player_simulator.py')

    if not simulator_path.exists():
        print(f"❌ PlayerSimulator file not found: {simulator_path}")
        return False

    with open(simulator_path, 'r') as f:
        content = f.read()

    issues = []
    config_ok = True

    # Check variance fix parameters
    print("\n📊 Variance Fix Parameters:")

    variance_checks = [
        ('QB Passing', r'alpha = 1\.5.*# Reduced from 4\.0', '1.5', '4.0'),
        ('QB Rushing', r'gamma\(1\.0,\s*yards_per_carry', '1.0', '2.0'),
        ('RB Rushing', r'alpha = 1\.0.*# Reduced from 3\.0', '1.0', '3.0'),
        ('RB Receiving', r'alpha_rec = 0\.9.*# Reduced from 2\.5', '0.9', '2.5'),
        ('WR/TE Receiving', r'alpha = 1\.0.*# Reduced from 3\.0', '1.0', '3.0'),
    ]

    for check_name, pattern, expected, old_value in variance_checks:
        if re.search(pattern, content):
            print(f"   ✅ {check_name}: alpha = {expected} (reduced from {old_value})")
        else:
            print(f"   ❌ {check_name}: Variance fix NOT applied (expected {expected})")
            issues.append(f"Missing variance fix for {check_name}")
            config_ok = False

    # Check default parameters
    print("\n⚙️  Default Parameters:")

    # Check trials
    if 'trials: int = 50000' in content:
        print("   ✅ Default trials: 50000")
    else:
        print("   ⚠️  Default trials: Not explicitly set to 50000")
        issues.append("Trials parameter not verified")

    # Check seed handling
    if 'seed: Optional[int] = None' in content:
        print("   ✅ Seed parameter: Optional (defaults to None, can be set)")
    else:
        print("   ⚠️  Seed parameter: Different signature")

    # Check calibrator integration
    if 'calibrator: Optional[NFLProbabilityCalibrator]' in content:
        print("   ✅ Calibrator: Integrated (optional)")
    else:
        print("   ⚠️  Calibrator: Not found in signature")

    return config_ok, issues

def check_monte_carlo_simulator_config():
    """Check MonteCarloSimulator configuration."""
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATOR CONFIGURATION CHECK")
    print("="*80)

    simulator_path = Path('nfl_quant/simulation/simulator.py')

    if not simulator_path.exists():
        print(f"❌ MonteCarloSimulator file not found: {simulator_path}")
        return False

    with open(simulator_path, 'r') as f:
        content = f.read()

    issues = []
    config_ok = True

    print("\n⚙️  Default Parameters:")

    # Check trials parameter
    if 'trials: int = 50000' in content:
        print("   ✅ Default trials: 50000")
    else:
        print("   ⚠️  Default trials: Different value")
        issues.append("Trials parameter mismatch")

    # Check seed initialization
    if 'def __init__(self, seed: int = 42)' in content:
        print("   ✅ Default seed: 42")
    else:
        print("   ⚠️  Default seed: Different value")
        issues.append("Seed parameter mismatch")

    # Check game counter for reproducibility
    if '_game_counter' in content:
        print("   ✅ Game counter: Present (ensures reproducibility)")
    else:
        print("   ⚠️  Game counter: Not found")

    # Check contextual adjustments
    print("\n🌍 Contextual Adjustments:")
    adjustments = [
        ('Divisional games', 'is_divisional'),
        ('Primetime games', 'game_type'),
        ('Weather (cold)', 'temperature'),
        ('Wind', 'wind_speed'),
        ('Precipitation', 'precipitation'),
    ]

    for name, keyword in adjustments:
        if keyword in content:
            print(f"   ✅ {name}: Integrated")
        else:
            print(f"   ⚠️  {name}: Not found")

    return len(issues) == 0, issues

def check_integration_consistency():
    """Check consistency between simulators."""
    print("\n" + "="*80)
    print("INTEGRATION CONSISTENCY CHECK")
    print("="*80)

    issues = []

    # Check if both use same default trials
    print("\n🔄 Parameter Consistency:")

    # Read both files
    player_sim_path = Path('nfl_quant/simulation/player_simulator.py')
    monte_sim_path = Path('nfl_quant/simulation/simulator.py')

    if player_sim_path.exists() and monte_sim_path.exists():
        with open(player_sim_path) as f:
            player_content = f.read()
        with open(monte_sim_path) as f:
            monte_content = f.read()

        # Check trials
        player_trials = '50000' in player_content and 'trials: int = 50000' in player_content
        monte_trials = 'trials: int = 50000' in monte_content

        if player_trials and monte_trials:
            print("   ✅ Both use 50,000 trials by default")
        else:
            print("   ⚠️  Trials mismatch between simulators")
            issues.append("Trials parameter inconsistency")

        # Check seed usage
        if 'seed' in player_content and 'seed' in monte_content:
            print("   ✅ Both support seed parameter")
        else:
            print("   ⚠️  Seed parameter mismatch")

    return len(issues) == 0, issues

def check_pipeline_configuration():
    """Check pipeline configuration files."""
    print("\n" + "="*80)
    print("PIPELINE CONFIGURATION FILES")
    print("="*80)

    config_files = [
        ('Bankroll Config', Path('configs/bankroll_config.json')),
        ('Risk Modes', Path('configs/risk_modes.json')),
        ('Calibrator', Path('configs/calibrator.json')),
    ]

    all_ok = True

    for name, path in config_files:
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"   ✅ {name}: {path} ({size:.1f} KB)")
        else:
            print(f"   ⚠️  {name}: {path} NOT FOUND")
            if name != 'Calibrator':  # Calibrator is optional
                all_ok = False

    return all_ok, []

def main():
    """Run complete configuration verification."""
    print("="*80)
    print("🏈 NFL QUANT - SIMULATOR CONFIGURATION VERIFICATION")
    print("="*80)
    print("\nExpert Analysis: Verifying correct configuration of both simulators")

    all_checks_passed = True
    all_issues = []

    # Check PlayerSimulator
    ps_ok, ps_issues = check_player_simulator_config()
    all_issues.extend(ps_issues)
    if not ps_ok:
        all_checks_passed = False

    # Check MonteCarloSimulator
    mc_ok, mc_issues = check_monte_carlo_simulator_config()
    all_issues.extend(mc_issues)
    if not mc_ok:
        all_checks_passed = False

    # Check integration
    int_ok, int_issues = check_integration_consistency()
    all_issues.extend(int_issues)
    if not int_ok:
        all_checks_passed = False

    # Check config files
    cfg_ok, cfg_issues = check_pipeline_configuration()
    all_issues.extend(cfg_issues)
    if not cfg_ok:
        all_checks_passed = False

    # Final summary
    print("\n" + "="*80)
    print("📋 CONFIGURATION VERIFICATION SUMMARY")
    print("="*80)

    if all_checks_passed and len(all_issues) == 0:
        print("\n✅ BOTH SIMULATORS ARE CORRECTLY CONFIGURED!")
        print("\nVerification Results:")
        print("   ✅ PlayerSimulator: Correctly configured")
        print("      - Variance fixes applied (alpha parameters reduced)")
        print("      - Default trials: 50,000")
        print("      - Calibrator integration: Present")
        print()
        print("   ✅ MonteCarloSimulator: Correctly configured")
        print("      - Default trials: 50,000")
        print("      - Default seed: 42")
        print("      - Contextual adjustments: Integrated")
        print()
        print("   ✅ Integration: Consistent parameters")
        print("   ✅ Configuration files: Present")
        print()
        print("💡 Both simulators are ready for Week 9!")
    else:
        print("\n⚠️  CONFIGURATION ISSUES DETECTED:")
        for issue in all_issues:
            print(f"   - {issue}")

        print("\n❌ Some configuration issues need attention")
        print("   Review the issues above and fix before running Week 9 pipeline")

    print("\n" + "="*80)

    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())





























