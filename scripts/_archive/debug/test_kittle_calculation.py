#!/usr/bin/env python3
"""
Test George Kittle Week 12 prediction calculation.
Traces the complete flow from input features to final prediction.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

from nfl_quant.schemas import PlayerPropInput
from nfl_quant.simulation.player_simulator_v3_correlated import PlayerSimulatorV3

def main():
    print("\n" + "=" * 80)
    print("GEORGE KITTLE WEEK 12 PREDICTION TRACE")
    print("=" * 80)

    # Load actual Week 12 prediction to get input features
    import pandas as pd
    predictions = pd.read_csv('data/model_predictions_week12.csv')
    kittle = predictions[predictions['player_name'] == 'George Kittle'].iloc[0]

    print(f"\n📊 ACTUAL PREDICTION OUTPUT:")
    print(f"   Receiving Yards: {kittle['receiving_yards_mean']:.1f} ± {kittle['receiving_yards_std']:.1f}")
    print(f"   Receptions: {kittle['receptions_mean']:.1f} ± {kittle['receptions_std']:.1f}")
    print(f"   Targets: {kittle['targets_mean']:.1f} ± {kittle['targets_std']:.1f}")
    print(f"   TD Probability: {kittle['receiving_tds_mean']:.3f}")

    # Now load the trailing stats that were used as input
    import json
    with open('data/trailing_stats_week12.json', 'r') as f:
        trailing_stats = json.load(f)

    # Find Kittle's trailing stats
    kittle_key = None
    for key in trailing_stats.keys():
        if 'kittle' in key.lower() and 'SF' in key:
            kittle_key = key
            break

    if not kittle_key:
        print("\n❌ Could not find Kittle in trailing stats")
        return

    kittle_hist = trailing_stats[kittle_key]

    print(f"\n📈 INPUT TRAILING STATS (from trailing_stats_week12.json):")
    print(f"   Snap Share: {kittle_hist.get('trailing_snap_share', 'N/A')}")
    print(f"   Target Share: {kittle_hist.get('trailing_target_share', 'N/A')}")
    print(f"   Yards per Opportunity: {kittle_hist.get('trailing_yards_per_opportunity', 'N/A')}")
    print(f"   Yards per Target: {kittle_hist.get('trailing_yards_per_target', 'N/A')}")
    print(f"   TD Rate: {kittle_hist.get('trailing_td_rate', 'N/A')}")
    print(f"   TD Rate Pass: {kittle_hist.get('trailing_td_rate_pass', 'N/A')}")

    # Create PlayerPropInput
    kittle_input = PlayerPropInput(
        player_id="kittle_george",
        player_name="George Kittle",
        position="TE",
        team="SF",
        opponent="CAR",
        week=12,
        # Trailing stats
        trailing_snap_share=kittle_hist.get('trailing_snap_share', 0.85),
        trailing_target_share=kittle_hist.get('trailing_target_share', 0.20),
        trailing_yards_per_opportunity=kittle_hist.get('trailing_yards_per_opportunity', 7.0),
        trailing_yards_per_target=kittle_hist.get('trailing_yards_per_target'),
        trailing_td_rate=kittle_hist.get('trailing_td_rate', 0.05),
        trailing_td_rate_pass=kittle_hist.get('trailing_td_rate_pass'),
        # Opponent defense (CAR vs TE)
        opponent_def_epa_vs_position=kittle['opponent_def_epa_vs_position'],
        # Game context
        projected_team_total=28.0,
        projected_opponent_total=20.0,
        projected_game_script=0.4,
        projected_pace=65.0,
        # Team usage
        projected_team_pass_attempts=35.0,
        projected_team_rush_attempts=25.0,
        projected_team_targets=35.0,
    )

    print(f"\n🎯 OPPONENT DEFENSE:")
    print(f"   CAR DEF EPA vs TE: {kittle['opponent_def_epa_vs_position']:+.4f}")
    print(f"   Interpretation: {'WEAK (should INCREASE yards)' if kittle['opponent_def_epa_vs_position'] > 0 else 'STRONG (should DECREASE yards)'}")

    print("\n" + "=" * 80)
    print("RUNNING SIMULATION (WATCH FOR EFFICIENCY DEBUG LOGS)")
    print("=" * 80)
    print("\n🔍 Looking for how trailing_yards_per_target is used...\n")

    # Create simulator
    simulator = PlayerSimulatorV3()

    # Run simulation (will trigger debug logs)
    result = simulator.simulate_player(kittle_input)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\n✅ RESULT:")
    print(f"   Receiving Yards: {result.receiving_yards_mean:.1f} ± {result.receiving_yards_std:.1f}")
    print(f"   Receptions: {result.receptions_mean:.1f} ± {result.receptions_std:.1f}")
    print(f"   Targets: {result.targets_mean:.1f} ± {result.targets_std:.1f}")

    # Compare to actual
    yards_diff = result.receiving_yards_mean - kittle['receiving_yards_mean']
    recs_diff = result.receptions_mean - kittle['receptions_mean']

    print(f"\n📊 COMPARISON TO ACTUAL PREDICTION:")
    print(f"   Receiving Yards: {yards_diff:+.1f} yards difference")
    print(f"   Receptions: {recs_diff:+.1f} receptions difference")

    if abs(yards_diff) < 1.0 and abs(recs_diff) < 0.1:
        print("\n✅ Match! Reproduction successful.")
    else:
        print("\n⚠️  Difference detected - may need to adjust input features")

if __name__ == "__main__":
    main()
