#!/usr/bin/env python3
"""
Test George Kittle's efficiency prediction to understand the 69% reduction.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Enable debug logging for efficiency predictor
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

from nfl_quant.schemas import PlayerPropInput
from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors

def main():
    print("\n" + "=" * 80)
    print("GEORGE KITTLE EFFICIENCY PREDICTION ANALYSIS")
    print("=" * 80)

    # Load models
    print("\n📦 Loading ML models...")
    usage_predictor, efficiency_predictor = load_predictors()
    print("   ✅ Models loaded")

    # Create Kittle's input based on Week 12 data
    kittle_input = PlayerPropInput(
        player_id="kittle_george",
        player_name="George Kittle",
        position="TE",
        team="SF",
        opponent="CAR",
        week=12,
        # Trailing stats (from our trace)
        trailing_snap_share=0.856,  # From predictions
        trailing_target_share=0.205,  # Calculated from SF pass attempts
        trailing_yards_per_target=9.417,  # ✅ THIS IS CALCULATED (Bug #3 fix working)
        trailing_yards_per_opportunity=7.0,  # Generic fallback (should NOT be used)
        trailing_td_rate=0.05,
        trailing_td_rate_pass=0.1667,  # From our trace
        # Opponent defense
        opponent_def_epa_vs_position=0.0686,  # CAR weak vs TE
        # Game context
        projected_team_total=28.0,
        projected_opponent_total=20.0,
        projected_game_script=0.4,  # SF likely to lead
        projected_pace=65.0,
        # Team usage
        projected_team_pass_attempts=35.0,
        projected_team_rush_attempts=25.0,
        projected_team_targets=35.0,
    )

    print(f"\n📊 INPUT FEATURES:")
    print(f"   Player: {kittle_input.player_name} (TE)")
    print(f"   Opponent: {kittle_input.opponent}")
    print(f"   ")
    print(f"   Trailing Yards/Target: {kittle_input.trailing_yards_per_target:.3f} ✅ Position-specific")
    print(f"   Trailing Yards/Opp: {kittle_input.trailing_yards_per_opportunity:.3f} (generic - should NOT use)")
    print(f"   ")
    print(f"   Opponent DEF EPA vs TE: {kittle_input.opponent_def_epa_vs_position:+.4f} (WEAK)")
    print(f"   Game Script: {kittle_input.projected_game_script:+.2f} (likely leading)")

    print("\n" + "=" * 80)
    print("RUNNING SIMULATION - WATCH FOR EFFICIENCY DEBUG LOGS")
    print("=" * 80)

    # Create simulator
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,  # Fewer trials for faster testing
        seed=42,
    )

    # Run simulation (will trigger debug logs showing which field is used)
    result = simulator.simulate_player(kittle_input)

    print("\n" + "=" * 80)
    print("SIMULATION RESULT")
    print("=" * 80)

    print(f"\n✅ FINAL PREDICTION:")
    print(f"   Targets: {result.targets_mean:.1f} ± {result.targets_std:.1f}")
    print(f"   Receiving Yards: {result.receiving_yards_mean:.1f} ± {result.receiving_yards_std:.1f}")
    print(f"   Receptions: {result.receptions_mean:.1f} ± {result.receptions_std:.1f}")
    print(f"   TD Probability: {result.receiving_tds_mean:.3f}")

    # Calculate implied yards per target
    implied_ypt = result.receiving_yards_mean / result.targets_mean if result.targets_mean > 0 else 0

    print(f"\n🔍 ANALYSIS:")
    print(f"   Input Yards/Target: {kittle_input.trailing_yards_per_target:.3f}")
    print(f"   Output Yards/Target: {implied_ypt:.3f}")
    print(f"   Efficiency Change: {((implied_ypt / kittle_input.trailing_yards_per_target) - 1) * 100:+.1f}%")

    if implied_ypt < kittle_input.trailing_yards_per_target * 0.5:
        print(f"\n   ⚠️  WARNING: >50% efficiency reduction detected!")
        print(f"   This suggests the efficiency predictor may be:")
        print(f"   1. Using wrong input field (generic instead of position-specific)")
        print(f"   2. Over-adjusting for opponent defense")
        print(f"   3. Model needs retraining with correct features")

if __name__ == "__main__":
    main()
