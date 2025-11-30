#!/usr/bin/env python3
"""
Validate Week 10 predictions against actual results.
Quick analysis to see model performance.
"""

import pandas as pd
from pathlib import Path

# Week 10: PHI @ GB on November 10, 2024
# ACTUAL RESULT: PHI 22, GB 10
# https://www.espn.com/nfl/game/_/gameId/401671709

WEEK10_RESULTS = {
    'PHI': 22,
    'GB': 10,
    'total': 32
}

def american_odds_to_decimal(odds):
    """Convert American odds to decimal."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def calculate_payout(wager, odds):
    """Calculate profit from American odds."""
    decimal_odds = american_odds_to_decimal(odds)
    return wager * (decimal_odds - 1)

def main():
    """Validate Week 10 picks."""

    picks_file = Path("reports/all_picks_ranked_week10.csv")

    if not picks_file.exists():
        print(f"ERROR: {picks_file} not found")
        return

    df = pd.read_csv(picks_file)

    print("="*100)
    print("WEEK 10 BACKTEST: PHI @ GB")
    print("="*100)
    print(f"\nACTUAL RESULT: PHI {WEEK10_RESULTS['PHI']} - GB {WEEK10_RESULTS['GB']}")
    print(f"Total Points: {WEEK10_RESULTS['total']}")
    print(f"\nTotal Picks to Evaluate: {len(df)}")

    # Filter to top recommendations (positive edge)
    top_picks = df[df['edge_pct'] > 5].copy()  # Only picks with >5% edge

    print(f"Top Recommendations (>5% edge): {len(top_picks)}")
    print("\n" + "="*100)
    print("EVALUATING TOP RECOMMENDATIONS")
    print("="*100)

    results = []
    total_wagered = 0
    total_profit = 0
    wins = 0
    losses = 0
    unknowns = 0

    for idx, pick in top_picks.iterrows():
        player = pick['player']
        bet_on = pick['pick']
        market = pick['market']
        line = pick['line']
        odds = pick['odds']
        model_prob = pick['model_prob']
        edge = pick['edge_pct']

        # Unit wager (assume $100 per pick for simplicity)
        wager = 100

        # Determine outcome
        won = None
        actual_value = None

        # Game total market
        if market == 'game_total':
            actual_value = WEEK10_RESULTS['total']
            if 'Under' in bet_on:
                won = actual_value < line
            elif 'Over' in bet_on:
                won = actual_value > line

        # Player props - we need actual player stats (not available in this quick validation)
        # For now, mark as unknown
        else:
            won = None
            unknowns += 1

        if won is not None:
            if won:
                profit = calculate_payout(wager, odds)
                total_profit += profit
                wins += 1
                status = "WIN"
                result_emoji = "✅"
            else:
                total_profit -= wager
                losses += 1
                status = "LOSS"
                result_emoji = "❌"

            total_wagered += wager

            results.append({
                'player': player if player else 'Game',
                'pick': bet_on,
                'line': line,
                'odds': odds,
                'edge': edge,
                'wager': wager,
                'actual': actual_value,
                'status': status,
                'profit': profit if won else -wager
            })

            print(f"\n{result_emoji} {player if player else 'GAME'}: {bet_on}")
            print(f"   Line: {line} | Odds: {odds:+d} | Edge: {edge:.1f}%")
            print(f"   Actual: {actual_value} | {status}")
            print(f"   P&L: ${profit if won else -wager:+.2f}")

    print("\n" + "="*100)
    print("RESULTS SUMMARY (Game Totals Only)")
    print("="*100)
    print(f"Evaluated: {wins + losses} bets")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    if wins + losses > 0:
        print(f"Win Rate: {wins/(wins+losses)*100:.1f}%")
        print(f"Total Wagered: ${total_wagered:.2f}")
        print(f"Net Profit: ${total_profit:+.2f}")
        print(f"ROI: {(total_profit/total_wagered)*100:+.1f}%")

    print(f"\nPlayer Props (Not Validated): {unknowns}")
    print("\nNOTE: Player prop validation requires actual player stat data.")
    print("To fully validate, we need:")
    print("  - Individual player passing/rushing/receiving stats")
    print("  - Data from nflverse play-by-play or box scores")
    print("  - Run scripts/fetch/pull_2024_season_data.py to get this data")

    print("\n" + "="*100)
    print("NEXT STEPS")
    print("="*100)
    print("1. Fetch complete Week 10 player stats from nflverse")
    print("2. Build comprehensive backtest script for all player props")
    print("3. Analyze calibration: model_prob vs actual hit rate")
    print("4. Calculate Brier score and log loss for model accuracy")

    return results

if __name__ == "__main__":
    main()
