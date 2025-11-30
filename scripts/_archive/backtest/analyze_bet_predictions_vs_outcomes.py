#!/usr/bin/env python3
"""
Detailed Backtest: Model Predictions vs Actual Outcomes

Shows for each bet:
- What the model predicted (probability)
- What actually happened (win/loss)
- Whether prediction was correct
- Edge calculation
- Profit/loss

This helps validate model accuracy and calibration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def american_to_prob(american_odds):
    """Convert American odds to implied probability."""
    if pd.isna(american_odds):
        return 0.5
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def analyze_bet_detail(df, week_num=None, limit=50):
    """Analyze detailed bet predictions vs outcomes."""

    if week_num:
        df = df[df['week'] == week_num].copy()

    if len(df) == 0:
        print(f"❌ No data found for week {week_num}")
        return

    print("=" * 100)
    print(f"📊 DETAILED BET ANALYSIS: Model Predictions vs Actual Outcomes")
    if week_num:
        print(f"   Week {week_num}")
    else:
        print(f"   All Weeks")
    print("=" * 100)
    print()

    # Calculate prediction accuracy
    df['prediction_correct'] = df['bet_won'] == True
    df['prediction_incorrect'] = df['bet_won'] == False

    # Calculate profit/loss (assuming $1 bets, -110 odds)
    df['profit'] = df['bet_won'].apply(lambda x: 0.909 if x else -1.0)

    # Sort by edge (highest first)
    df = df.sort_values('edge', ascending=False)

    # Show top bets
    print(f"📋 TOP {limit} BETS BY EDGE:")
    print("-" * 100)
    print(f"{'Week':<6} {'Player':<25} {'Pick':<20} {'Predicted':<12} {'Actual':<8} {'Result':<8} {'Edge':<8} {'Profit':<8}")
    print("-" * 100)

    shown = 0
    for _, row in df.iterrows():
        if shown >= limit:
            break

        week = int(row['week'])
        player = str(row['player'])[:24]
        pick = str(row['pick'])[:19]
        predicted = f"{row['model_prob']:.1%}" if pd.notna(row['model_prob']) else "N/A"
        actual = "WIN" if row['bet_won'] else "LOSS"
        edge = f"{row['edge']:.1%}" if pd.notna(row['edge']) else "N/A"
        profit = f"${row['profit']:+.2f}"

        # Color coding
        result_symbol = "✅" if row['bet_won'] else "❌"

        print(f"{week:<6} {player:<25} {pick:<20} {predicted:<12} {actual:<8} {result_symbol:<8} {edge:<8} {profit:<8}")
        shown += 1

    print()

    # Summary statistics
    print("=" * 100)
    print("📊 SUMMARY STATISTICS")
    print("=" * 100)
    print()

    total_bets = len(df)
    wins = df['bet_won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0

    total_profit = df['profit'].sum()
    avg_profit = df['profit'].mean()
    roi = (total_profit / total_bets) * 100 if total_bets > 0 else 0

    print(f"Total Bets:        {total_bets:,}")
    print(f"Wins:              {wins:,}")
    print(f"Losses:            {losses:,}")
    print(f"Win Rate:          {win_rate:.1%}")
    print(f"Total Profit:      ${total_profit:+.2f}")
    print(f"Average Profit:    ${avg_profit:+.2f}")
    print(f"ROI:               {roi:+.1f}%")
    print()

    # Prediction accuracy by probability bin
    if 'model_prob' in df.columns:
        print("=" * 100)
        print("📊 PREDICTION ACCURACY BY PROBABILITY BIN")
        print("=" * 100)
        print()
        print(f"{'Probability Range':<20} {'Bets':<8} {'Predicted':<12} {'Actual WR':<12} {'Error':<12} {'Correct':<10} {'Wrong':<8}")
        print("-" * 100)

        bins = [
            (0.50, 0.60, "50-60%"),
            (0.60, 0.70, "60-70%"),
            (0.70, 0.80, "70-80%"),
            (0.80, 0.85, "80-85%"),
            (0.85, 0.90, "85-90%"),
            (0.90, 1.00, "90%+"),
        ]

        for bin_min, bin_max, bin_label in bins:
            bin_data = df[
                (df['model_prob'] >= bin_min) &
                (df['model_prob'] < bin_max)
            ]

            if len(bin_data) > 0:
                bin_wins = bin_data['bet_won'].sum()
                bin_win_rate = bin_wins / len(bin_data)
                avg_predicted = bin_data['model_prob'].mean()
                error = abs(bin_win_rate - avg_predicted)
                correct = bin_wins
                wrong = len(bin_data) - bin_wins

                print(f"{bin_label:<20} {len(bin_data):<8} {avg_predicted:>11.1%} {bin_win_rate:>11.1%} {error:>+11.1%} {correct:<10} {wrong:<8}")

        print()

    # Most accurate predictions
    print("=" * 100)
    print("✅ MOST ACCURATE PREDICTIONS (High Confidence, Correct)")
    print("=" * 100)
    print()

    high_conf_correct = df[
        (df['model_prob'] >= 0.85) &
        (df['bet_won'] == True)
    ].head(20)

    if len(high_conf_correct) > 0:
        print(f"{'Player':<25} {'Pick':<20} {'Predicted':<12} {'Edge':<8} {'Profit':<8}")
        print("-" * 100)
        for _, row in high_conf_correct.iterrows():
            player = str(row['player'])[:24]
            pick = str(row['pick'])[:19]
            predicted = f"{row['model_prob']:.1%}"
            edge = f"{row['edge']:.1%}" if pd.notna(row['edge']) else "N/A"
            profit = f"${row['profit']:+.2f}"
            print(f"{player:<25} {pick:<20} {predicted:<12} {edge:<8} {profit:<8}")
        print()

    # Most inaccurate predictions
    print("=" * 100)
    print("❌ MOST INACCURATE PREDICTIONS (High Confidence, Wrong)")
    print("=" * 100)
    print()

    high_conf_wrong = df[
        (df['model_prob'] >= 0.85) &
        (df['bet_won'] == False)
    ].head(20)

    if len(high_conf_wrong) > 0:
        print(f"{'Player':<25} {'Pick':<20} {'Predicted':<12} {'Actual':<12} {'Edge':<8} {'Loss':<8}")
        print("-" * 100)
        for _, row in high_conf_wrong.iterrows():
            player = str(row['player'])[:24]
            pick = str(row['pick'])[:19]
            predicted = f"{row['model_prob']:.1%}"
            actual_value = row.get('actual_value', 'N/A')
            if actual_value != 'N/A':
                line_value = row.get('line', 0)
                if 'Over' in str(row['pick']):
                    actual_outcome = "OVER" if actual_value > line_value else "UNDER"
                else:
                    actual_outcome = "UNDER" if actual_value < line_value else "OVER"
            else:
                actual_outcome = "LOSS"
            edge = f"{row['edge']:.1%}" if pd.notna(row['edge']) else "N/A"
            loss = f"${row['profit']:+.2f}"
            print(f"{player:<25} {pick:<20} {predicted:<12} {actual_outcome:<12} {edge:<8} {loss:<8}")
        print()

    # Save detailed results with all prediction vs outcome info
    output_file = Path(f'reports/detailed_bet_analysis_week{week_num if week_num else "all"}.csv')

    # Create detailed output DataFrame
    detailed_df = df.copy()

    # Add interpretation columns
    detailed_df['predicted_outcome'] = detailed_df['pick'].apply(
        lambda x: 'OVER' if 'Over' in str(x) else 'UNDER' if 'Under' in str(x) else 'UNKNOWN'
    )

    # Determine actual outcome
    def determine_actual_outcome(row):
        pick = str(row['pick'])
        line = row['line']
        actual = row.get('actual_value', np.nan)

        if pd.isna(actual):
            return 'UNKNOWN'

        if 'Over' in pick:
            return 'OVER' if actual > line else 'UNDER'
        elif 'Under' in pick:
            return 'UNDER' if actual < line else 'OVER'
        else:
            return 'UNKNOWN'

    detailed_df['actual_outcome'] = detailed_df.apply(determine_actual_outcome, axis=1)

    # Add prediction accuracy
    detailed_df['prediction_correct'] = (
        (detailed_df['predicted_outcome'] == detailed_df['actual_outcome']) |
        (detailed_df['bet_won'] == True)
    )

    # Calculate expected vs actual
    detailed_df['expected_wins'] = detailed_df['model_prob'].apply(lambda x: 1 if x > 0.5 else 0)
    detailed_df['actual_wins'] = detailed_df['bet_won'].astype(int)

    # Select columns for output
    output_columns = [
        'week', 'player', 'position', 'market', 'pick', 'line',
        'model_prob', 'model_projection', 'market_prob', 'edge',
        'actual_value', 'actual_outcome', 'predicted_outcome',
        'bet_won', 'prediction_correct', 'profit'
    ]

    # Filter to columns that exist
    available_columns = [col for col in output_columns if col in detailed_df.columns]
    detailed_df[available_columns].to_csv(output_file, index=False)

    print(f"💾 Detailed results saved: {output_file}")
    print(f"   Columns: {', '.join(available_columns)}")
    print()


def main():
    """Main execution."""
    print("=" * 100)
    print("📊 DETAILED BACKTEST: MODEL PREDICTIONS VS ACTUAL OUTCOMES")
    print("=" * 100)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load backtest data
    backtest_file = Path('reports/BACKTEST_WEEKS_1_8_VALIDATION.csv')
    if not backtest_file.exists():
        print(f"❌ Backtest data not found: {backtest_file}")
        return

    df = pd.read_csv(backtest_file)
    print(f"✅ Loaded {len(df):,} backtest records\n")

    # Ask user which week to analyze
    import sys

    if len(sys.argv) > 1:
        week_num = int(sys.argv[1])
        print(f"📅 Analyzing Week {week_num}...\n")
        analyze_bet_detail(df, week_num=week_num)
    else:
        # Analyze all weeks
        print("📅 Analyzing All Weeks...\n")
        analyze_bet_detail(df, week_num=None, limit=100)

        # Also show each week individually
        print("\n" + "=" * 100)
        print("WEEK-BY-WEEK DETAILED ANALYSIS")
        print("=" * 100)
        print()

        available_weeks = sorted(df['week'].unique())
        for week in available_weeks:
            analyze_bet_detail(df, week_num=int(week), limit=20)


if __name__ == '__main__':
    main()
