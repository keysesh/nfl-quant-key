#!/usr/bin/env python3
"""
Analyze why pass TDs went from +12.41% ROI to -41.94% ROI

This script compares:
1. Old vs new calibrator predictions
2. Which specific bets failed
3. Whether it's variance or systematic issue
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def analyze_pass_td_performance():
    print("=" * 80)
    print("ANALYZING PASS TD PERFORMANCE DROP")
    print("=" * 80)
    print()

    # Load backtest audit
    audit_file = Path("reports/backtest_edge_audit_weeks1-8.csv")
    if not audit_file.exists():
        print(f"❌ Audit file not found: {audit_file}")
        return

    df = pd.read_csv(audit_file)

    # Filter to pass TDs with positive edge >= 3%
    pass_td_bets = df[
        (df['market'] == 'player_pass_tds') &
        (df['filter_status'] == 'positive_edge') &
        (df['edge'] >= 0.03)
    ].copy()

    print(f"Total pass TD bets placed: {len(pass_td_bets)}")
    print()

    if len(pass_td_bets) == 0:
        print("⚠️ No pass TD bets found")
        return

    # Load actual results to see outcomes
    props_file = Path("data/calibration/historical_props_simulated.parquet")
    if not props_file.exists():
        print("⚠️ Cannot load actual results")
        return

    props_df = pd.read_parquet(props_file)
    props_df = props_df[props_df['market'] == 'player_pass_tds'].copy()

    print("=" * 80)
    print("BET DETAILS")
    print("=" * 80)
    print()

    wins = 0
    losses = 0

    for idx, bet in pass_td_bets.iterrows():
        # Find matching actual result
        actual = props_df[
            (props_df['player_name'] == bet['player']) &
            (props_df['week'] == bet['week']) &
            (props_df['line'] == bet['line']) &
            (props_df['pick_type'].str.lower() == bet['prop_type'].lower())
        ]

        if len(actual) == 0:
            continue

        actual_row = actual.iloc[0]
        actual_tds = actual_row.get('actual_value', None)

        if actual_tds is None:
            continue

        # Determine outcome
        if bet['prop_type'].lower() == 'over':
            won = actual_tds > bet['line']
        else:
            won = actual_tds < bet['line']

        if won:
            wins += 1
            result = "✅ WIN"
        else:
            losses += 1
            result = "❌ LOSS"

        print(f"Week {bet['week']}: {bet['player']} {bet['prop_type'].upper()} {bet['line']}")
        print(f"  Model prob: {bet['model_prob_raw']:.1%}")
        print(f"  Model prob (calibrated): {bet['model_prob']:.1%}")
        print(f"  Implied prob (odds): {bet['implied_prob']:.1%}")
        print(f"  Edge: {bet['edge']:.1%}")
        print(f"  Actual TDs: {actual_tds}")
        print(f"  Outcome: {result}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")

    total_bets = wins + losses
    if total_bets == 0:
        print("⚠️ Could not match bets to actual results")
        print()
        print("This means the backtest audit CSV doesn't contain outcome data.")
        print("Need to check if backtest stores results elsewhere or run with --save-bets flag")
        return

    print(f"Win rate: {wins/total_bets*100:.1f}%")
    print()

    # Calculate ROI
    # Assume -110 average odds for simplicity
    # Win = +0.91 units, Loss = -1 unit
    total_profit = wins * 0.91 - losses * 1.0
    total_wagered = wins + losses
    roi = (total_profit / total_wagered) * 100

    print(f"Estimated ROI: {roi:.1f}%")
    print()

    # Analyze if it's variance or systematic
    expected_wins = pass_td_bets['model_prob'].mean() * len(pass_td_bets)
    actual_wins = wins

    print("=" * 80)
    print("VARIANCE ANALYSIS")
    print("=" * 80)
    print()
    print(f"Expected wins (from model): {expected_wins:.1f}")
    print(f"Actual wins: {actual_wins}")
    print(f"Difference: {actual_wins - expected_wins:.1f}")
    print()

    # Statistical significance (binomial test)
    from scipy.stats import binom
    n = wins + losses
    p = pass_td_bets['model_prob'].mean()
    p_value = binom.cdf(wins, n, p)

    print(f"P-value (one-tailed): {p_value:.4f}")
    if p_value < 0.05:
        print("❌ STATISTICALLY SIGNIFICANT underperformance")
        print("   This suggests a systematic issue, not just variance")
    else:
        print("✅ NOT statistically significant")
        print("   This could be explained by variance (small sample)")
    print()

    # Analyze model calibration quality
    print("=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)
    print()

    # Check if raw probs vs calibrated probs differ significantly
    raw_prob_mean = pass_td_bets['model_prob_raw'].mean()
    cal_prob_mean = pass_td_bets['model_prob'].mean()

    print(f"Average raw probability: {raw_prob_mean:.1%}")
    print(f"Average calibrated probability: {cal_prob_mean:.1%}")
    print(f"Calibration adjustment: {(cal_prob_mean - raw_prob_mean)*100:.1f} percentage points")
    print()

    if cal_prob_mean > 0.50:
        print("⚠️ Calibrator is shifting probabilities UP (more confident)")
        print("   If bets are losing, this means overcalibration")
    elif cal_prob_mean < 0.50:
        print("⚠️ Calibrator is shifting probabilities DOWN (less confident)")
        print("   If bets are losing at high rate, this is appropriate")
    print()

    # Check which types of bets performed worst
    print("=" * 80)
    print("BY BET TYPE")
    print("=" * 80)
    print()

    for prop_type in ['over', 'under']:
        type_bets = pass_td_bets[pass_td_bets['prop_type'].str.lower() == prop_type]
        if len(type_bets) == 0:
            continue

        type_wins = 0
        type_total = 0

        for idx, bet in type_bets.iterrows():
            actual = props_df[
                (props_df['player_name'] == bet['player']) &
                (props_df['week'] == bet['week']) &
                (props_df['line'] == bet['line']) &
                (props_df['pick_type'].str.lower() == bet['prop_type'].lower())
            ]

            if len(actual) == 0:
                continue

            actual_row = actual.iloc[0]
            actual_tds = actual_row.get('actual_value', None)

            if actual_tds is None:
                continue

            type_total += 1
            if prop_type == 'over':
                if actual_tds > bet['line']:
                    type_wins += 1
            else:
                if actual_tds < bet['line']:
                    type_wins += 1

        if type_total > 0:
            print(f"{prop_type.upper()}:")
            print(f"  Bets: {type_total}")
            print(f"  Wins: {type_wins}")
            print(f"  Win rate: {type_wins/type_total*100:.1f}%")
            print()

    # Check which QBs failed most
    print("=" * 80)
    print("BY PLAYER")
    print("=" * 80)
    print()

    player_results = {}
    for idx, bet in pass_td_bets.iterrows():
        actual = props_df[
            (props_df['player_name'] == bet['player']) &
            (props_df['week'] == bet['week']) &
            (props_df['line'] == bet['line']) &
            (props_df['pick_type'].str.lower() == bet['prop_type'].lower())
        ]

        if len(actual) == 0:
            continue

        actual_row = actual.iloc[0]
        actual_tds = actual_row.get('actual_value', None)

        if actual_tds is None:
            continue

        player = bet['player']
        if player not in player_results:
            player_results[player] = {'wins': 0, 'losses': 0}

        if bet['prop_type'].lower() == 'over':
            won = actual_tds > bet['line']
        else:
            won = actual_tds < bet['line']

        if won:
            player_results[player]['wins'] += 1
        else:
            player_results[player]['losses'] += 1

    # Sort by most bets
    sorted_players = sorted(player_results.items(), key=lambda x: x[1]['wins'] + x[1]['losses'], reverse=True)

    for player, results in sorted_players:
        total = results['wins'] + results['losses']
        win_rate = results['wins'] / total * 100 if total > 0 else 0
        print(f"{player}: {results['wins']}-{results['losses']} ({win_rate:.0f}%)")

    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if p_value < 0.05:
        print("🔴 LIKELY SYSTEMATIC ISSUE")
        print()
        print("The performance drop is statistically significant.")
        print("Possible causes:")
        print("  1. Calibrator overtrained on pass TDs")
        print("  2. QB TD simulation has a flaw")
        print("  3. Different prop mix in weeks 1-8")
        print()
        print("Recommendation: Investigate QB TD simulation logic")
    else:
        print("🟡 LIKELY VARIANCE")
        print()
        print("The performance drop is NOT statistically significant.")
        print(f"With only {wins + losses} bets, variance can easily explain this.")
        print()
        print("Recommendation: Need more data before concluding there's a problem")
        print(f"  - Run backtest on full season (weeks 1-18)")
        print(f"  - Monitor week 9+ performance")

if __name__ == "__main__":
    analyze_pass_td_performance()
