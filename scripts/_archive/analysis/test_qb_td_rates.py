#!/usr/bin/env python3
"""
Test QB TD Rate Simulation

Compare predicted TD rates vs actual TD rates for QBs in our failed bets.
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
    print("QB TD RATE ANALYSIS")
    print("=" * 80)
    print()

    # Load actual outcomes for QBs from weeks 2-8, 2024
    actual_td_rates = {}

    for week in range(2, 9):
        stats_file = Path(f"data/sleeper_stats/stats_week{week}_2024.csv")
        if not stats_file.exists():
            continue

        df = pd.read_csv(stats_file)
        qbs = df[df['position'] == 'QB']

        for _, qb in qbs.iterrows():
            player_name = qb['player_name']
            pass_att = qb.get('pass_att', 0)
            pass_td = qb.get('pass_td', 0)

            if pd.notna(pass_att) and pass_att > 0:
                td_rate = pass_td / pass_att

                if player_name not in actual_td_rates:
                    actual_td_rates[player_name] = {'attempts': [], 'tds': [], 'td_rates': []}

                actual_td_rates[player_name]['attempts'].append(pass_att)
                actual_td_rates[player_name]['tds'].append(pass_td)
                actual_td_rates[player_name]['td_rates'].append(td_rate)

    print(f"Loaded actual data for {len(actual_td_rates)} QBs")
    print()

    # Calculate actual TD rates (average across weeks)
    qb_avg_rates = {}
    for qb, data in actual_td_rates.items():
        avg_rate = np.mean(data['td_rates'])
        avg_tds = np.mean(data['tds'])
        avg_att = np.mean(data['attempts'])
        qb_avg_rates[qb] = {
            'avg_td_rate': avg_rate,
            'avg_tds_per_game': avg_tds,
            'avg_attempts': avg_att,
            'weeks': len(data['td_rates'])
        }

    # Show QBs from our failed bets
    problem_qbs = [
        'C.J. Stroud',
        'Bryce Young',
        'Bo Nix',
        'Trevor Lawrence',
        'Spencer Rattler',
        'Jared Goff',
        'Jordan Love',
        'Drake Maye',
        'Baker Mayfield'
    ]

    print("=" * 80)
    print("ACTUAL TD RATES FOR QBS IN OUR BETS")
    print("=" * 80)
    print()

    for qb_name in problem_qbs:
        if qb_name in qb_avg_rates:
            data = qb_avg_rates[qb_name]
            print(f"{qb_name}:")
            print(f"  Avg TD rate: {data['avg_td_rate']*100:.1f}% per attempt")
            print(f"  Avg TDs/game: {data['avg_tds_per_game']:.1f}")
            print(f"  Avg attempts: {data['avg_attempts']:.1f}")
            print(f"  Weeks played: {data['weeks']}")
            print()

    # Calculate league averages
    all_rates = [data['avg_td_rate'] for data in qb_avg_rates.values()]
    all_tds = [data['avg_tds_per_game'] for data in qb_avg_rates.values()]

    print("=" * 80)
    print("LEAGUE STATISTICS (2024 Weeks 2-8)")
    print("=" * 80)
    print()
    print(f"Average TD rate: {np.mean(all_rates)*100:.1f}% per attempt")
    print(f"Median TD rate: {np.median(all_rates)*100:.1f}% per attempt")
    print(f"Std dev TD rate: {np.std(all_rates)*100:.1f}%")
    print()
    print(f"Average TDs per game: {np.mean(all_tds):.2f}")
    print(f"Median TDs per game: {np.median(all_tds):.2f}")
    print()

    # Show distribution
    print("TD Rate Distribution:")
    bins = [0, 0.02, 0.03, 0.04, 0.05, 0.06, 1.0]
    bin_labels = ['<2%', '2-3%', '3-4%', '4-5%', '5-6%', '>6%']

    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = sum(1 for r in all_rates if low <= r < high)
        pct = count / len(all_rates) * 100
        print(f"  {bin_labels[i]}: {count} QBs ({pct:.0f}%)")

    print()

    # Now test what our model predicts
    print("=" * 80)
    print("MODEL PREDICTIONS VS ACTUAL")
    print("=" * 80)
    print()

    # Load simulator
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,
        seed=42
    )

    # Test a few QBs with typical inputs
    test_cases = [
        {
            'name': 'Good QB (e.g., Patrick Mahomes)',
            'td_rate_trailing': 0.05,  # 5% TD rate
            'expected_actual': 0.05
        },
        {
            'name': 'Average QB',
            'td_rate_trailing': 0.04,  # 4% TD rate
            'expected_actual': 0.04
        },
        {
            'name': 'Below Average QB (e.g., Bryce Young)',
            'td_rate_trailing': 0.02,  # 2% TD rate
            'expected_actual': 0.02
        },
    ]

    for test in test_cases:
        player_input = PlayerPropInput(
            player_id='test',
            player_name=test['name'],
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
            trailing_td_rate=test['td_rate_trailing'],
            opponent_def_epa_vs_position=0.0,
        )

        # Run simulation
        result = simulator.simulate_player(player_input)

        # Calculate actual simulated TD rate
        attempts = result['passing_attempts']
        pass_tds = result['passing_tds']

        avg_attempts = np.mean(attempts)
        avg_tds = np.mean(pass_tds)
        simulated_td_rate = avg_tds / avg_attempts if avg_attempts > 0 else 0

        print(f"{test['name']}:")
        print(f"  Input TD rate: {test['td_rate_trailing']*100:.1f}%")
        print(f"  Simulated TD rate: {simulated_td_rate*100:.1f}%")
        print(f"  Avg TDs per simulation: {avg_tds:.2f}")
        print(f"  Avg attempts per simulation: {avg_attempts:.1f}")

        # Calculate prob of hitting different thresholds
        prob_over_1_5 = np.mean(pass_tds > 1.5)
        prob_over_2_5 = np.mean(pass_tds > 2.5)

        print(f"  P(>1.5 TDs): {prob_over_1_5*100:.1f}%")
        print(f"  P(>2.5 TDs): {prob_over_2_5*100:.1f}%")
        print()

    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()

    # Compare model assumptions to reality
    print("If our model is using 4-5% TD rate for average QBs:")
    print(f"  With 35 attempts: 35 × 0.045 = 1.58 TDs expected")
    print(f"  This would give ~50% chance of >1.5 TDs")
    print()

    print("But in reality (weeks 2-8):")
    print(f"  Actual avg TD rate: {np.mean(all_rates)*100:.1f}%")
    print(f"  Actual avg TDs/game: {np.mean(all_tds):.2f}")
    print()

    if np.mean(all_rates) < 0.04:
        print("⚠️  ISSUE: Actual TD rates are LOWER than typical model assumptions")
        print("   Model may be overestimating TD probability")
    else:
        print("✅ TD rates match expectations")

if __name__ == "__main__":
    main()
