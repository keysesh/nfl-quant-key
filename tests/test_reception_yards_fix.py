#!/usr/bin/env python3
"""
Test that reception yards fix works correctly
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

def test_reception_yards_fix():
    """Test that WR receiving yards are realistic after fix"""

    print("=" * 80)
    print("TESTING RECEPTION YARDS FIX")
    print("=" * 80)
    print()

    # Load simulator
    print("Loading simulator...")
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,
        seed=42
    )
    print("✅ Simulator loaded")
    print()

    # Create test WR input (CeeDee Lamb-like stats)
    wr_input = PlayerPropInput(
        player_id="test_wr",
        player_name="Test WR",
        position="WR",
        team="DAL",
        opponent="PHI",
        week=9,
        projected_team_total=28.0,
        projected_opponent_total=25.0,
        projected_game_script=3.0,
        projected_pace=30.0,
        trailing_snap_share=0.85,  # 85% snaps
        trailing_target_share=0.25,  # 25% of team targets (~10 targets/game)
        trailing_carry_share=None,
        trailing_yards_per_opportunity=8.5,  # ~8.5 yards per target
        trailing_td_rate=0.06,
        opponent_def_epa_vs_position=0.0,
    )

    print("Test WR Profile:")
    print(f"  Name: {wr_input.player_name}")
    print(f"  Position: {wr_input.position}")
    print(f"  Yards per target: {wr_input.trailing_yards_per_opportunity}")
    print(f"  Target share: {wr_input.trailing_target_share}")
    print()

    # Run simulation
    print("Running 10,000 simulations...")
    results = simulator.simulate_player(wr_input)
    print()

    # Check results
    print("=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print()

    if 'receiving_yards' not in results:
        print("❌ ERROR: 'receiving_yards' not in results!")
        return False

    rec_yards = results['receiving_yards']
    targets = results['targets']
    receptions = results['receptions']

    # Calculate statistics
    print("RECEIVING YARDS:")
    print(f"  Mean: {np.mean(rec_yards):.1f} yards")
    print(f"  Median: {np.median(rec_yards):.1f} yards")
    print(f"  Std Dev: {np.std(rec_yards):.1f} yards")
    print(f"  Min: {np.min(rec_yards):.1f} yards")
    print(f"  Max: {np.max(rec_yards):.1f} yards")
    print()

    print("TARGETS:")
    print(f"  Mean: {np.mean(targets):.1f}")
    print()

    print("RECEPTIONS:")
    print(f"  Mean: {np.mean(receptions):.1f}")
    print()

    # Calculate implied yards per target
    implied_ypt = np.mean(rec_yards) / np.mean(targets)
    print(f"IMPLIED YARDS PER TARGET: {implied_ypt:.2f}")
    print(f"EXPECTED YARDS PER TARGET: {wr_input.trailing_yards_per_opportunity:.2f}")
    print()

    # Validation
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    print()

    checks_passed = 0
    checks_total = 0

    # Check 1: Mean yards realistic (60-150 for a WR1)
    checks_total += 1
    if 60 <= np.mean(rec_yards) <= 150:
        print(f"✅ CHECK 1: Mean yards realistic ({np.mean(rec_yards):.1f} in range 60-150)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 1: Mean yards UNREALISTIC ({np.mean(rec_yards):.1f} NOT in range 60-150)")

    # Check 2: Yards per target matches input
    checks_total += 1
    ypt_error = abs(implied_ypt - wr_input.trailing_yards_per_opportunity)
    if ypt_error < 1.0:  # Within 1 yard
        print(f"✅ CHECK 2: YPT matches input (error: {ypt_error:.2f} yards)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 2: YPT doesn't match (error: {ypt_error:.2f} yards, expected < 1.0)")

    # Check 3: Variance reasonable
    checks_total += 1
    std_dev = np.std(rec_yards)
    if 30 <= std_dev <= 80:
        print(f"✅ CHECK 3: Std dev realistic ({std_dev:.1f} in range 30-80)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 3: Std dev off ({std_dev:.1f} NOT in range 30-80)")

    print()
    print(f"Checks passed: {checks_passed}/{checks_total}")
    print()

    if checks_passed == checks_total:
        print("🎉 SUCCESS: Reception yards fix working correctly!")
        return True
    else:
        print("❌ FAILURE: Reception yards still broken")
        return False

if __name__ == "__main__":
    success = test_reception_yards_fix()
    sys.exit(0 if success else 1)
