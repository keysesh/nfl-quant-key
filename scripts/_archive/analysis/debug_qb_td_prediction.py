#!/usr/bin/env python3
"""
Debug QB TD Rate Prediction Pipeline

Trace exactly what TD rate is being predicted at each step.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

def main():
    print("=" * 80)
    print("QB TD RATE PREDICTION DEBUGGING")
    print("=" * 80)
    print()

    # Load predictors
    print("Loading models...")
    usage_predictor, efficiency_predictor = load_predictors()
    print(f"Efficiency predictor models: {list(efficiency_predictor.models.keys())}")
    print()

    # Check if QB TD rate model exists
    if 'QB_td_rate_pass' in efficiency_predictor.models:
        model = efficiency_predictor.models['QB_td_rate_pass']
        print(f"✅ QB_td_rate_pass model found: {type(model)}")

        # Check what features it needs
        if hasattr(efficiency_predictor, 'feature_cols') and 'QB' in efficiency_predictor.feature_cols:
            print(f"   Required features: {efficiency_predictor.feature_cols['QB']}")
    else:
        print("❌ QB_td_rate_pass model NOT found")
    print()

    # Create test player input with known TD rate
    test_qbs = [
        {
            'name': 'Elite QB',
            'td_rate': 0.07,  # 7% - like Jordan Love
        },
        {
            'name': 'Good QB',
            'td_rate': 0.05,  # 5%
        },
        {
            'name': 'Average QB',
            'td_rate': 0.04,  # 4%
        },
        {
            'name': 'Below Avg QB',
            'td_rate': 0.02,  # 2% - like Bryce Young
        },
    ]

    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,
        seed=42
    )

    for test_qb in test_qbs:
        print("=" * 80)
        print(f"Testing: {test_qb['name']} (trailing_td_rate={test_qb['td_rate']*100:.0f}%)")
        print("=" * 80)
        print()

        player_input = PlayerPropInput(
            player_id='test',
            player_name=test_qb['name'],
            position='QB',
            team='TEST',
            opponent='OPP',
            week=5,
            projected_team_total=24.0,
            projected_opponent_total=22.0,
            projected_game_script=0.0,
            projected_pace=30.0,
            trailing_snap_share=1.0,
            trailing_target_share=None,
            trailing_carry_share=None,
            trailing_yards_per_opportunity=7.0,
            trailing_td_rate=test_qb['td_rate'],  # This is the key input
            opponent_def_epa_vs_position=0.0,
        )

        # Manually call efficiency predictor to see what it returns
        print("Step 1: Create efficiency features...")
        efficiency_features = pd.DataFrame([{
            'week': player_input.week,
            'trailing_completion_pct': 0.65,
            'trailing_yards_per_completion': 11.0,
            'trailing_td_rate_pass': player_input.trailing_td_rate,  # Input TD rate
            'trailing_opp_pass_def_epa': 0.0,
            'team_pace': 60.0,
        }])
        print(f"  trailing_td_rate_pass: {player_input.trailing_td_rate}")
        print()

        # Try to predict
        print("Step 2: Call efficiency predictor...")
        try:
            # Filter to just QB pass models
            pass_only_models = {k: v for k, v in efficiency_predictor.models.items()
                               if k in ['QB_comp_pct', 'QB_yards_per_completion', 'QB_td_rate_pass']}

            if 'QB_td_rate_pass' in pass_only_models:
                model = pass_only_models['QB_td_rate_pass']

                # Get feature columns for this model
                if hasattr(efficiency_predictor, 'feature_cols') and 'QB' in efficiency_predictor.feature_cols:
                    feature_cols = efficiency_predictor.feature_cols['QB']
                    print(f"  Model expects features: {feature_cols}")

                    # Check if our features match
                    missing = set(feature_cols) - set(efficiency_features.columns)
                    if missing:
                        print(f"  ⚠️  Missing features: {missing}")
                        for col in missing:
                            efficiency_features[col] = 0.0

                    # Predict
                    features_subset = efficiency_features[feature_cols]
                    predicted_td_rate = model.predict(features_subset)[0]
                    print(f"  ✅ Model predicted td_rate: {predicted_td_rate*100:.2f}%")
                else:
                    print("  ❌ Cannot find feature columns")
                    predicted_td_rate = 0.05
            else:
                print("  ❌ QB_td_rate_pass model not in pass_only_models")
                predicted_td_rate = 0.05

        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            predicted_td_rate = 0.05
        print()

        # Now run full simulation
        print("Step 3: Run full simulation...")
        result = simulator.simulate_player(player_input)

        attempts = result['passing_attempts']
        pass_tds = result['passing_tds']

        avg_attempts = np.mean(attempts)
        avg_tds = np.mean(pass_tds)
        actual_simulated_rate = avg_tds / avg_attempts if avg_attempts > 0 else 0

        print(f"  Avg attempts: {avg_attempts:.1f}")
        print(f"  Avg TDs: {avg_tds:.2f}")
        print(f"  Simulated TD rate: {actual_simulated_rate*100:.2f}%")
        print()

        # Compare
        print("Step 4: Compare...")
        print(f"  Input (trailing): {test_qb['td_rate']*100:.1f}%")
        print(f"  Predicted (model): {predicted_td_rate*100:.1f}%")
        print(f"  Simulated (output): {actual_simulated_rate*100:.1f}%")

        if abs(actual_simulated_rate - test_qb['td_rate']) > 0.01:
            print(f"  ❌ MISMATCH: Simulator not using input TD rate!")
        else:
            print(f"  ✅ MATCH: Simulator using input TD rate correctly")
        print()

    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print()

    # Check if model is actually being used
    if 'QB_td_rate_pass' not in efficiency_predictor.models:
        print("❌ PROBLEM: QB_td_rate_pass model doesn't exist")
        print("   Solution: Check if model was trained properly")
    else:
        print("Possible issues:")
        print("1. Model is predicting constant value regardless of input")
        print("2. Model features don't include trailing_td_rate_pass")
        print("3. Model was trained on insufficient data")
        print("4. Simulator is using fallback value instead of model prediction")

if __name__ == "__main__":
    main()
