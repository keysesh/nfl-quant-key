#!/usr/bin/env python3
"""
Quick Summary: Model Predictions vs Actual Outcomes

Shows a clean summary of what the model predicted vs what actually happened.
"""

import pandas as pd
import sys
from pathlib import Path


def show_summary(week_num=None):
    """Show summary of predictions vs outcomes."""

    if week_num:
        file_path = Path(f'reports/detailed_bet_analysis_week{week_num}.csv')
        title = f"Week {week_num}"
    else:
        file_path = Path('reports/detailed_bet_analysis_weekall.csv')
        title = "All Weeks"

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("   Run: python scripts/backtest/analyze_bet_predictions_vs_outcomes.py")
        return

    df = pd.read_csv(file_path)

    print("=" * 100)
    print(f"📊 MODEL PREDICTIONS VS ACTUAL OUTCOMES - {title}")
    print("=" * 100)
    print()

    # Sort by edge (highest first)
    df = df.sort_values('edge', ascending=False)

    print("📋 TOP 30 BETS BY EDGE:")
    print("-" * 100)
    print(f"{'Player':<20} {'Pick':<20} {'Predicted':<12} {'Actual':<15} {'Result':<8} {'Edge':<10} {'Profit':<10}")
    print("-" * 100)

    for idx, row in df.head(30).iterrows():
        player = str(row['player'])[:19]
        pick = str(row['pick'])[:19]
        predicted = f"{row['model_prob']:.1%}" if pd.notna(row['model_prob']) else "N/A"

        # Actual value
        actual_val = row['actual_value']
        if pd.notna(actual_val):
            actual_display = f"{actual_val:.1f}"
        else:
            actual_display = "N/A"

        # Result
        result = "WIN ✅" if row['bet_won'] else "LOSS ❌"

        edge = f"{row['edge']:.1%}" if pd.notna(row['edge']) else "N/A"
        profit = f"${row['profit']:+.2f}" if pd.notna(row['profit']) else "N/A"

        print(f"{player:<20} {pick:<20} {predicted:<12} {actual_display:<15} {result:<8} {edge:<10} {profit:<10}")

    print()

    # Summary stats
    total = len(df)
    wins = df['bet_won'].sum()
    losses = total - wins
    win_rate = wins / total if total > 0 else 0
    total_profit = df['profit'].sum()
    roi = (total_profit / total) * 100 if total > 0 else 0

    print("=" * 100)
    print("📊 SUMMARY")
    print("=" * 100)
    print(f"Total Bets:     {total:,}")
    print(f"Wins:           {wins:,}")
    print(f"Losses:         {losses:,}")
    print(f"Win Rate:       {win_rate:.1%}")
    print(f"Total Profit:   ${total_profit:+.2f}")
    print(f"ROI:            {roi:+.1f}%")
    print()


if __name__ == '__main__':
    week = int(sys.argv[1]) if len(sys.argv) > 1 else None
    show_summary(week)
