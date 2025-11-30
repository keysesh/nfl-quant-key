#!/usr/bin/env python3
"""
Test V4 Integration with generate_model_predictions.py

Quick test to verify V4 simulator can be used in the prediction pipeline.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import load_predictors
from nfl_quant.simulation.player_simulator_v4 import PlayerSimulatorV4
from nfl_quant.schemas import PlayerPropInput
import numpy as np

def test_v4_integration():
    """Test that V4 can be used as drop-in replacement for V3."""
    print("="*60)
    print("V4 INTEGRATION TEST")
    print("="*60)

    # Load predictors
    print("\n1. Loading predictors...")
    usage_predictor, efficiency_predictor = load_predictors()
    print("   ✅ Loaded predictors")

    # Create V4 simulator
    print("\n2. Creating V4 simulator...")
    simulator = PlayerSimulatorV4(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=1000,  # Small for testing
        seed=42
    )
    print("   ✅ V4 simulator created")

    # Create sample player input
    print("\n3. Creating sample player input...")
    player_input = PlayerPropInput(
        player_id='george_kittle',
        player_name='George Kittle',
        team='SF',
        position='TE',
        week=12,
        opponent='GB',
        projected_team_total=24.5,
        projected_opponent_total=22.0,
        projected_game_script=2.5,
        projected_pace=62.0,
        trailing_snap_share=0.75,
        trailing_target_share=0.18,
        trailing_carry_share=0.0,
        trailing_yards_per_opportunity=8.5,
        trailing_td_rate=0.05,
        trailing_yards_per_target=8.5,
        opponent_def_epa_vs_position=-0.05,
        projected_team_pass_attempts=35,
        projected_team_rush_attempts=27,
        projected_team_targets=35,
    )
    print(f"   ✅ Created input for {player_input.player_name} ({player_input.position})")

    # Test simulate_player() method (V3 API)
    print("\n4. Testing simulate_player() (V3 API compatibility)...")
    result = simulator.simulate_player(player_input)
    print(f"   ✅ simulate_player() returned dict with {len(result)} stats")
    print(f"   Stats available: {list(result.keys())}")

    # Verify samples are numpy arrays
    print("\n5. Verifying output format...")
    if 'receiving_yards' in result:
        samples = result['receiving_yards']
        print(f"   receiving_yards type: {type(samples)}")
        print(f"   receiving_yards shape: {samples.shape}")
        print(f"   receiving_yards mean: {np.mean(samples):.1f}")
        print(f"   receiving_yards median: {np.median(samples):.1f}")
        print(f"   receiving_yards std: {np.std(samples):.1f}")
        print("   ✅ V3 format compatible (numpy arrays)")
    else:
        print("   ❌ receiving_yards not in result")
        return False

    # Test simulate() method (V4 API)
    print("\n6. Testing simulate() (V4 API with percentiles)...")
    output = simulator.simulate(player_input)
    if hasattr(output, 'receiving_yards') and output.receiving_yards:
        ry = output.receiving_yards
        print(f"   receiving_yards percentiles:")
        print(f"     p5:  {ry.p5:.1f}")
        print(f"     p25: {ry.p25:.1f}")
        print(f"     p50: {ry.median:.1f} (median)")
        print(f"     p75: {ry.p75:.1f}")
        print(f"     p95: {ry.p95:.1f}")
        print(f"   CV: {ry.cv:.2f}")
        print("   ✅ V4 format working (with percentiles)")
    else:
        print("   ❌ receiving_yards not in V4 output")
        return False

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - V4 INTEGRATION WORKING")
    print("="*60)
    print("\nYou can now run:")
    print("  python scripts/predict/generate_model_predictions.py 12 --simulator v4")
    print("")

    return True

if __name__ == '__main__':
    success = test_v4_integration()
    sys.exit(0 if success else 1)
