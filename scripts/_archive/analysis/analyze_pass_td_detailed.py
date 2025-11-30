#!/usr/bin/env python3
"""
Detailed Pass TD Analysis

Extract and analyze all pass TD prop data to understand:
1. What bets were placed
2. What outcomes occurred
3. Why the model is underperforming
4. What needs to be fixed
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_pass_td_bets():
    """Load all pass TD bets from backtest"""
    print("=" * 80)
    print("LOADING PASS TD BET DATA")
    print("=" * 80)
    print()

    # Load backtest audit
    audit_file = Path("reports/backtest_edge_audit_weeks1-8.csv")
    if not audit_file.exists():
        print(f"❌ Audit file not found: {audit_file}")
        return None

    df = pd.read_csv(audit_file)

    # Filter to pass TDs with positive edge >= 3%
    pass_td_bets = df[
        (df['market'] == 'player_pass_tds') &
        (df['filter_status'] == 'positive_edge') &
        (df['edge'] >= 0.03)
    ].copy()

    print(f"Found {len(pass_td_bets)} pass TD bets")
    print()

    return pass_td_bets

def load_actual_outcomes():
    """Load actual pass TD outcomes from sleeper stats"""
    print("=" * 80)
    print("LOADING ACTUAL OUTCOMES")
    print("=" * 80)
    print()

    actual_outcomes = {}

    # Load sleeper stats for weeks 2-8 (2024 season based on backtest)
    # Note: The backtest was on 2024 data but we're in 2025 now
    for week in range(2, 9):
        stats_file = Path(f"data/sleeper_stats/stats_week{week}_2024.csv")

        if not stats_file.exists():
            print(f"⚠️  Week {week}: Stats file not found")
            continue

        df = pd.read_csv(stats_file)

        # Extract pass TDs
        for _, row in df.iterrows():
            player_name = row['player_name']
            pass_tds = row.get('pass_td', 0)

            if pd.notna(pass_tds) and pass_tds >= 0:
                key = (player_name, week)
                actual_outcomes[key] = {
                    'player_name': player_name,
                    'week': week,
                    'pass_tds': int(pass_tds),
                    'pass_yds': row.get('pass_yd', 0),
                    'pass_att': row.get('pass_att', 0),
                    'position': row.get('position', 'QB')
                }

    print(f"Loaded actual outcomes for {len(actual_outcomes)} player-weeks")
    print()

    return actual_outcomes

def analyze_bet_outcomes(bets_df, actual_outcomes):
    """Match bets to actual outcomes and analyze"""
    print("=" * 80)
    print("BET-BY-BET ANALYSIS")
    print("=" * 80)
    print()

    results = []

    for idx, bet in bets_df.iterrows():
        player = bet['player']
        week = bet['week']
        line = bet['line']
        prop_type = bet['prop_type'].lower()
        model_prob_raw = bet['model_prob_raw']
        model_prob_cal = bet['model_prob']
        edge = bet['edge']

        # Find actual outcome
        key = (player, week)
        actual = actual_outcomes.get(key)

        if not actual:
            print(f"⚠️  No outcome found: {player} Week {week}")
            continue

        actual_tds = actual['pass_tds']

        # Determine outcome
        if prop_type == 'over':
            won = actual_tds > line
            expected_to_win = model_prob_cal > 0.5
        else:  # under
            won = actual_tds < line
            expected_to_win = model_prob_cal > 0.5

        # Store result
        result = {
            'week': week,
            'player': player,
            'prop_type': prop_type,
            'line': line,
            'actual_tds': actual_tds,
            'won': won,
            'expected_to_win': expected_to_win,
            'model_prob_raw': model_prob_raw,
            'model_prob_cal': model_prob_cal,
            'edge': edge,
            'pass_yds': actual.get('pass_yds', 0),
            'pass_att': actual.get('pass_att', 0),
        }

        results.append(result)

        # Print bet details
        outcome_str = "✅ WIN" if won else "❌ LOSS"
        expected_str = "Expected" if expected_to_win else "Upset"

        print(f"Week {week}: {player} {prop_type.upper()} {line}")
        print(f"  Actual: {actual_tds} TDs ({actual['pass_yds']} yds on {actual['pass_att']} att)")
        print(f"  Model prob (raw): {model_prob_raw:.1%}")
        print(f"  Model prob (cal): {model_prob_cal:.1%}")
        print(f"  Edge: {edge:.1%}")
        print(f"  Result: {outcome_str} ({expected_str})")
        print()

    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """Calculate performance metrics"""
    print("=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print()

    total_bets = len(results_df)
    wins = results_df['won'].sum()
    losses = total_bets - wins

    print(f"Total Bets: {total_bets}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {wins/total_bets*100:.1f}%")
    print()

    # Calculate ROI
    profit = wins * 0.91 - losses * 1.0
    roi = (profit / total_bets) * 100

    print(f"Profit: {profit:.2f} units")
    print(f"ROI: {roi:.1f}%")
    print()

    # Expected vs actual
    avg_model_prob = results_df['model_prob_cal'].mean()
    expected_wins = avg_model_prob * total_bets

    print(f"Expected Wins (from model): {expected_wins:.1f}")
    print(f"Actual Wins: {wins}")
    print(f"Difference: {wins - expected_wins:.1f} ({(wins - expected_wins)/expected_wins*100:+.1f}%)")
    print()

    # Statistical test
    from scipy.stats import binom
    p_value = binom.cdf(wins, total_bets, avg_model_prob)

    print(f"Binomial Test:")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: ❌ Statistically significant underperformance")
    else:
        print(f"  Result: ✅ Could be variance (not statistically significant)")
    print()

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': wins/total_bets,
        'roi': roi,
        'expected_wins': expected_wins,
        'p_value': p_value
    }

def analyze_by_line(results_df):
    """Analyze performance by prop line"""
    print("=" * 80)
    print("ANALYSIS BY PROP LINE")
    print("=" * 80)
    print()

    for line in sorted(results_df['line'].unique()):
        line_bets = results_df[results_df['line'] == line]
        wins = line_bets['won'].sum()
        total = len(line_bets)
        win_rate = wins / total if total > 0 else 0

        print(f"Line {line} TDs:")
        print(f"  Bets: {total}")
        print(f"  Wins: {wins}")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print()

def analyze_by_player(results_df):
    """Analyze performance by player"""
    print("=" * 80)
    print("ANALYSIS BY PLAYER")
    print("=" * 80)
    print()

    player_stats = results_df.groupby('player').agg({
        'won': ['count', 'sum'],
        'actual_tds': 'mean',
        'pass_yds': 'mean',
        'pass_att': 'mean'
    })

    player_stats.columns = ['bets', 'wins', 'avg_tds', 'avg_yds', 'avg_att']
    player_stats['win_rate'] = player_stats['wins'] / player_stats['bets']

    # Sort by number of bets
    player_stats = player_stats.sort_values('bets', ascending=False)

    for player, stats in player_stats.iterrows():
        print(f"{player}:")
        print(f"  Bets: {int(stats['bets'])}")
        print(f"  Record: {int(stats['wins'])}-{int(stats['bets'] - stats['wins'])} ({stats['win_rate']*100:.0f}%)")
        print(f"  Avg actual: {stats['avg_tds']:.1f} TDs, {stats['avg_yds']:.0f} yds, {stats['avg_att']:.0f} att")
        print()

def diagnose_model_issues(results_df):
    """Diagnose what's wrong with the model"""
    print("=" * 80)
    print("MODEL DIAGNOSIS")
    print("=" * 80)
    print()

    # Check calibration
    print("1. CALIBRATION CHECK")
    print("-" * 40)

    # Divide into probability bins
    results_df['prob_bin'] = pd.cut(results_df['model_prob_cal'], bins=[0, 0.4, 0.5, 0.6, 1.0])

    for bin_label, bin_data in results_df.groupby('prob_bin'):
        if len(bin_data) == 0:
            continue

        avg_prob = bin_data['model_prob_cal'].mean()
        actual_win_rate = bin_data['won'].mean()
        count = len(bin_data)

        calibration_error = abs(avg_prob - actual_win_rate)

        print(f"  Predicted {avg_prob*100:.0f}% | Actual {actual_win_rate*100:.0f}% | Error: {calibration_error*100:.1f}% | n={count}")

    print()

    # Check over vs under
    print("2. OVER VS UNDER PERFORMANCE")
    print("-" * 40)

    for prop_type in ['over', 'under']:
        type_bets = results_df[results_df['prop_type'] == prop_type]
        if len(type_bets) == 0:
            continue

        wins = type_bets['won'].sum()
        total = len(type_bets)
        win_rate = wins / total

        print(f"  {prop_type.upper()}: {wins}/{total} ({win_rate*100:.1f}%)")

    print()

    # Check model confidence
    print("3. MODEL CONFIDENCE ANALYSIS")
    print("-" * 40)

    high_conf = results_df[results_df['model_prob_cal'] > 0.6]
    low_conf = results_df[results_df['model_prob_cal'] <= 0.6]

    print(f"  High confidence (>60%): {len(high_conf)} bets, {high_conf['won'].mean()*100:.1f}% win rate")
    print(f"  Low confidence (≤60%): {len(low_conf)} bets, {low_conf['won'].mean()*100:.1f}% win rate")
    print()

    # Check if raw vs calibrated shows issue
    print("4. CALIBRATOR IMPACT")
    print("-" * 40)

    avg_raw = results_df['model_prob_raw'].mean()
    avg_cal = results_df['model_prob_cal'].mean()
    actual_wr = results_df['won'].mean()

    print(f"  Avg raw prob: {avg_raw*100:.1f}%")
    print(f"  Avg calibrated prob: {avg_cal*100:.1f}%")
    print(f"  Actual win rate: {actual_wr*100:.1f}%")
    print(f"  Calibration shift: {(avg_cal - avg_raw)*100:+.1f} percentage points")

    if avg_cal > actual_wr + 0.10:
        print(f"  ⚠️  OVERCALIBRATED - Model too confident")
    elif avg_cal < actual_wr - 0.10:
        print(f"  ⚠️  UNDERCALIBRATED - Model too conservative")
    else:
        print(f"  ✅ Calibration reasonable")

    print()

def generate_recommendations(metrics, results_df):
    """Generate recommendations for fixing the issue"""
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    if metrics['p_value'] >= 0.05:
        print("🟡 VERDICT: Likely Variance")
        print()
        print("With only 16 bets and p-value = {:.3f}, this is likely random variance.".format(metrics['p_value']))
        print()
        print("Actions:")
        print("  1. Monitor performance over next 20-30 bets")
        print("  2. No immediate fixes needed")
        print("  3. Re-evaluate once sample size reaches 50+")
        print()
        return

    print("🔴 VERDICT: Systematic Issue")
    print()
    print("The underperformance is statistically significant (p < 0.05).")
    print()

    # Determine root cause
    avg_raw = results_df['model_prob_raw'].mean()
    avg_cal = results_df['model_prob_cal'].mean()
    actual_wr = results_df['won'].mean()

    if abs(avg_cal - avg_raw) > 0.15:
        print("PRIMARY ISSUE: Calibrator Overadjustment")
        print()
        print("The calibrator is shifting probabilities too much:")
        print(f"  Raw model: {avg_raw*100:.1f}%")
        print(f"  After calibration: {avg_cal*100:.1f}%")
        print(f"  Actual: {actual_wr*100:.1f}%")
        print()
        print("Fix:")
        print("  • Use market-specific calibrators")
        print("  • Reduce calibrator influence on pass TDs")
        print("  • Or increase pass TD weight in training")
        print()

    elif avg_raw > actual_wr + 0.10:
        print("PRIMARY ISSUE: Simulator Overconfidence")
        print()
        print("The raw simulator is too optimistic about pass TDs:")
        print(f"  Model predicts: {avg_raw*100:.1f}% success")
        print(f"  Actual success: {actual_wr*100:.1f}%")
        print()
        print("Fix:")
        print("  • Check QB TD rate simulation")
        print("  • Increase TD variance in simulator")
        print("  • Validate against historical QB TD distributions")
        print()

    else:
        print("PRIMARY ISSUE: Unknown")
        print()
        print("Metrics don't point to obvious calibration or simulation issue.")
        print("May need deeper investigation into specific bet types or players.")
        print()

def export_results(results_df):
    """Export detailed results for further analysis"""
    output_file = Path("reports/pass_td_detailed_analysis.csv")
    results_df.to_csv(output_file, index=False)
    print(f"📊 Detailed results exported to: {output_file}")
    print()

def main():
    """Main analysis flow"""
    print()
    print("=" * 80)
    print("PASS TD DETAILED ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    bets_df = load_pass_td_bets()
    if bets_df is None:
        return

    actual_outcomes = load_actual_outcomes()

    # Analyze
    results_df = analyze_bet_outcomes(bets_df, actual_outcomes)

    if len(results_df) == 0:
        print("❌ No bet outcomes to analyze")
        return

    # Calculate metrics
    metrics = calculate_metrics(results_df)

    # Detailed analyses
    analyze_by_line(results_df)
    analyze_by_player(results_df)
    diagnose_model_issues(results_df)

    # Recommendations
    generate_recommendations(metrics, results_df)

    # Export
    export_results(results_df)

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
