#!/usr/bin/env python3
"""
Test QB simulation to verify actual passing yards are realistic
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

def test_qb_simulation():
    """Test that QB simulations produce realistic passing yard totals"""

    print("=" * 80)
    print("TESTING QB SIMULATION - VERIFYING REALISTIC PASSING YARDS")
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

    # Create a sample QB input (Patrick Mahomes-like stats)
    qb_input = PlayerPropInput(
        player_id="test_qb",
        player_name="Test QB",
        position="QB",
        team="KC",
        opponent="BUF",
        week=9,
        # Game context
        projected_team_total=28.0,
        projected_opponent_total=25.0,
        projected_game_script=3.0,
        projected_pace=30.0,
        # Usage features
        trailing_snap_share=1.0,  # QBs play every snap
        trailing_target_share=None,  # QBs don't have targets
        trailing_carry_share=0.05,  # Occasional scrambles
        trailing_yards_per_opportunity=7.0,  # ~7 yards per attempt (league avg)
        trailing_td_rate=0.04,  # ~4% TD rate per attempt
        # Defense
        opponent_def_epa_vs_position=0.0,  # Neutral defense
    )

    print("Test QB Profile:")
    print(f"  Name: {qb_input.player_name}")
    print(f"  Position: {qb_input.position}")
    print(f"  Yards per attempt: {qb_input.trailing_yards_per_opportunity}")
    print(f"  TD rate: {qb_input.trailing_td_rate}")
    print()

    # Run simulation
    print("Running 10,000 simulations...")
    results = simulator.simulate_player(qb_input)
    print()

    # Check results
    print("=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print()

    if not results:
        print("❌ ERROR: No results returned!")
        return False

    print(f"Keys returned: {list(results.keys())}")
    print()

    # Check passing yards
    if 'passing_yards' not in results:
        print("❌ ERROR: 'passing_yards' not in results!")
        return False

    passing_yards = results['passing_yards']

    print("PASSING YARDS ANALYSIS:")
    print(f"  Type: {type(passing_yards)}")
    print(f"  Length: {len(passing_yards)}")
    print()

    # Calculate statistics
    mean_yards = np.mean(passing_yards)
    median_yards = np.median(passing_yards)
    std_yards = np.std(passing_yards)
    min_yards = np.min(passing_yards)
    max_yards = np.max(passing_yards)
    p25 = np.percentile(passing_yards, 25)
    p75 = np.percentile(passing_yards, 75)

    print("Statistics:")
    print(f"  Mean: {mean_yards:.1f} yards")
    print(f"  Median: {median_yards:.1f} yards")
    print(f"  Std Dev: {std_yards:.1f} yards")
    print(f"  Min: {min_yards:.1f} yards")
    print(f"  Max: {max_yards:.1f} yards")
    print(f"  25th percentile: {p25:.1f} yards")
    print(f"  75th percentile: {p75:.1f} yards")
    print()

    # Sample some values
    print("Sample values (first 20 simulations):")
    for i in range(min(20, len(passing_yards))):
        print(f"  Sim {i+1}: {passing_yards[i]:.1f} yards")
    print()

    # Validation checks
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    print()

    checks_passed = 0
    checks_total = 0

    # Check 1: Mean should be in realistic range (150-350 yards)
    checks_total += 1
    if 150 <= mean_yards <= 350:
        print(f"✅ CHECK 1: Mean yards realistic ({mean_yards:.1f} in range 150-350)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 1: Mean yards UNREALISTIC ({mean_yards:.1f} NOT in range 150-350)")
        print(f"   Expected: ~200-280 yards for average QB")
    print()

    # Check 2: Std dev should be reasonable (40-100 yards)
    checks_total += 1
    if 40 <= std_yards <= 100:
        print(f"✅ CHECK 2: Std dev realistic ({std_yards:.1f} in range 40-100)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 2: Std dev UNREALISTIC ({std_yards:.1f} NOT in range 40-100)")
        if std_yards < 40:
            print(f"   PROBLEM: Variance too low! Model is overconfident.")
        else:
            print(f"   PROBLEM: Variance too high! Model is too uncertain.")
    print()

    # Check 3: No absurdly low values (< 50 yards is rare but possible)
    checks_total += 1
    very_low = (passing_yards < 50).sum()
    pct_very_low = 100 * very_low / len(passing_yards)
    if pct_very_low < 5:  # Less than 5% should be < 50 yards
        print(f"✅ CHECK 3: Few very low values ({pct_very_low:.1f}% < 50 yards)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 3: Too many very low values ({pct_very_low:.1f}% < 50 yards)")
        print(f"   Expected: < 5% below 50 yards")
    print()

    # Check 4: Some high values (> 350 yards should be possible but rare)
    checks_total += 1
    very_high = (passing_yards > 350).sum()
    pct_very_high = 100 * very_high / len(passing_yards)
    if 1 <= pct_very_high <= 15:  # 1-15% should be > 350 yards
        print(f"✅ CHECK 4: Reasonable high-end values ({pct_very_high:.1f}% > 350 yards)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 4: High-end distribution off ({pct_very_high:.1f}% > 350 yards)")
        print(f"   Expected: 1-15% above 350 yards")
    print()

    # Check 5: Not returning placeholder/dummy values
    checks_total += 1
    if not (np.allclose(passing_yards, passing_yards[0])):
        print(f"✅ CHECK 5: Values vary (not placeholder)")
        checks_passed += 1
    else:
        print(f"❌ CHECK 5: All values identical! ({passing_yards[0]:.1f}) - PLACEHOLDER BUG")
    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()
    print(f"Checks passed: {checks_passed}/{checks_total}")
    print()

    if checks_passed == checks_total:
        print("🎉 SUCCESS: QB simulation producing realistic passing yards!")
        return True
    elif checks_passed >= checks_total - 1:
        print("⚠️  WARNING: QB simulation mostly working but has minor issues")
        return True
    else:
        print("❌ FAILURE: QB simulation NOT producing realistic passing yards!")
        print()
        print("LIKELY ISSUES:")
        if std_yards < 20:
            print("  1. Variance way too low (alpha parameter issue)")
        if mean_yards < 100:
            print("  2. Mean too low (attempts not being multiplied correctly)")
        if np.allclose(passing_yards, passing_yards[0]):
            print("  3. Returning placeholder values (simulation not running)")
        return False

if __name__ == "__main__":
    success = test_qb_simulation()
    sys.exit(0 if success else 1)
