#!/usr/bin/env python3
"""
⚠️  DEPRECATED: This script is week-specific and superseded by validate_week_r.R

Complete Week 10 validation: game totals + all player props.
Calculates actual ROI on full recommendation set.

DEPRECATION NOTICE:
This script is DEPRECATED as of 2025-11-14.
Use the generic, season-aware validation script instead:

    Rscript scripts/backtest/validate_week_r.R --week 10 --season 2024

See SCRIPTS_REFERENCE.md for migration guide.
"""

import warnings
warnings.warn(
    "\n"
    "⚠️  DEPRECATION WARNING ⚠️\n"
    "=========================\n"
    "This script (validate_week10_complete.py) is DEPRECATED.\n"
    "\n"
    "Use the generic validation script instead:\n"
    "  Rscript scripts/backtest/validate_week_r.R --week 10 --season 2024\n"
    "\n"
    "See SCRIPTS_REFERENCE.md for more information.\n",
    DeprecationWarning,
    stacklevel=2
)

import pandas as pd
import numpy as np
from pathlib import Path

# Week 10: PHI @ GB on November 10, 2024
# ACTUAL RESULT: PHI 22, GB 10
WEEK10_GAME_RESULTS = {
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

def calculate_profit(wager, odds, won):
    """Calculate profit from a bet."""
    if won:
        decimal_odds = american_odds_to_decimal(odds)
        return wager * (decimal_odds - 1)
    else:
        return -wager

def normalize_player_name(name):
    """Normalize player names for matching."""
    if pd.isna(name) or name == '' or name == 'nan':
        return None
    # Remove periods and apostrophes, then split
    cleaned = name.replace("'", '').strip()
    # Split on period or space
    parts = cleaned.replace('.', ' ').split()
    # Return last part as last name
    if len(parts) > 0:
        return parts[-1].lower()
    return None

def main():
    """Run complete Week 10 validation."""

    print("="*100)
    print("COMPLETE WEEK 10 VALIDATION: PHI @ GB")
    print("="*100)
    print(f"\nACTUAL GAME RESULT: PHI {WEEK10_GAME_RESULTS['PHI']} - GB {WEEK10_GAME_RESULTS['GB']}")
    print(f"Total Points: {WEEK10_GAME_RESULTS['total']}")

    # Load picks
    picks_file = Path("reports/all_picks_ranked_week10.csv")
    if not picks_file.exists():
        print(f"\n❌ ERROR: {picks_file} not found")
        return

    picks_df = pd.read_csv(picks_file)

    # Filter to top recommendations (>5% edge)
    top_picks = picks_df[picks_df['edge_pct'] > 5].copy()
    print(f"\nTotal Picks Generated: {len(picks_df)}")
    print(f"Top Recommendations (>5% edge): {len(top_picks)}")

    # Load player stats
    stats_file = Path("data/results/2024/week10_player_stats.csv")
    if not stats_file.exists():
        print(f"\n❌ ERROR: {stats_file} not found")
        print("Run: python scripts/backtest/fetch_week_player_stats.py 10 2024")
        return

    player_stats = pd.read_csv(stats_file)
    print(f"Player stats loaded: {len(player_stats)} players")

    # Create player stats lookup by last name and team
    stats_lookup_by_lastname = {}
    for _, row in player_stats.iterrows():
        lastname = normalize_player_name(row['player_name'])
        if lastname:
            key = (lastname, row['team'])
            stats_lookup_by_lastname[key] = row

    print(f"Created lookup for {len(stats_lookup_by_lastname)} player-team combinations")

    def find_player_stats(player_name_full, teams_to_check):
        """Find player stats by matching last name."""
        if pd.isna(player_name_full) or not player_name_full:
            return None, None

        lastname = normalize_player_name(player_name_full)
        if not lastname:
            return None, None

        for team in teams_to_check:
            key = (lastname, team)
            if key in stats_lookup_by_lastname:
                return stats_lookup_by_lastname[key], team

        return None, None

    print("\n" + "="*100)
    print("VALIDATING ALL RECOMMENDATIONS")
    print("="*100)

    results = []
    total_wagered = 0
    total_profit = 0
    wins = 0
    losses = 0
    no_data = 0

    # Standard wager per bet
    UNIT_SIZE = 100

    for idx, pick in top_picks.iterrows():
        player_name = pick['player']
        bet_pick = pick['pick']
        market = pick['market']
        line = pick['line']
        odds = pick['odds']
        model_prob = pick['model_prob']
        edge = pick['edge_pct']
        game = pick.get('game', 'PHI @ GB')

        wager = UNIT_SIZE
        won = None
        actual_value = None
        reason = ""

        # Game total market
        if market == 'game_total':
            actual_value = WEEK10_GAME_RESULTS['total']
            if 'Under' in bet_pick:
                won = actual_value < line
            elif 'Over' in bet_pick:
                won = actual_value > line
            reason = f"Total: {actual_value} vs line {line}"

        # Player prop markets
        elif player_name:
            # Find player stats in either PHI or GB
            player_stats_row, player_team = find_player_stats(player_name, ['PHI', 'GB'])

            if player_stats_row is not None:
                if market == 'player_pass_yds':
                    actual_value = player_stats_row['passing_yards']
                elif market == 'player_pass_tds':
                    actual_value = player_stats_row['passing_tds']
                elif market == 'player_rush_yds':
                    actual_value = player_stats_row['rushing_yards']
                elif market == 'player_rush_tds':
                    actual_value = player_stats_row['rushing_tds']
                elif market == 'player_reception_yds':
                    actual_value = player_stats_row['receiving_yards']
                elif market == 'player_receptions':
                    actual_value = player_stats_row['receptions']
                else:
                    actual_value = None
                    reason = f"Unknown market: {market}"

                if actual_value is not None:
                    if 'Under' in bet_pick:
                        won = actual_value < line
                    elif 'Over' in bet_pick:
                        won = actual_value > line
                    reason = f"Actual: {actual_value} vs line {line}"
            else:
                won = None
                reason = f"No stats found for {player_name}"
                no_data += 1

        # Calculate P&L
        if won is not None:
            profit = calculate_profit(wager, odds, won)
            total_profit += profit
            total_wagered += wager

            if won:
                wins += 1
                status = "WIN"
                emoji = "✅"
            else:
                losses += 1
                status = "LOSS"
                emoji = "❌"

            results.append({
                'player': player_name if player_name else 'Game',
                'pick': bet_pick,
                'market': market,
                'line': line,
                'odds': odds,
                'edge_pct': edge,
                'model_prob': model_prob,
                'wager': wager,
                'actual': actual_value,
                'won': won,
                'profit': profit,
                'status': status
            })

            print(f"\n{emoji} {player_name if player_name else 'GAME'}: {bet_pick}")
            print(f"   Market: {market} | Line: {line} | Odds: {odds:+d}")
            print(f"   Edge: {edge:.1f}% | Model Prob: {model_prob:.1%}")
            print(f"   {reason} → {status}")
            print(f"   P&L: ${profit:+.2f}")

        else:
            print(f"\n⚠️  {player_name if player_name else 'GAME'}: {bet_pick}")
            print(f"   Market: {market} | {reason}")

    # Summary
    print("\n" + "="*100)
    print("WEEK 10 RESULTS SUMMARY")
    print("="*100)

    evaluated = wins + losses
    print(f"\n📊 Bet Evaluation:")
    print(f"   Total Recommendations: {len(top_picks)}")
    print(f"   Evaluated: {evaluated} bets")
    print(f"   No Data Available: {no_data} bets")

    if evaluated > 0:
        print(f"\n💰 Performance:")
        print(f"   Wins: {wins}")
        print(f"   Losses: {losses}")
        print(f"   Win Rate: {wins/evaluated*100:.1f}%")
        print(f"   Total Wagered: ${total_wagered:,.2f}")
        print(f"   Total Profit: ${total_profit:+,.2f}")
        print(f"   ROI: {(total_profit/total_wagered)*100:+.1f}%")

        # Calculate expected vs actual
        expected_wins = sum([r['model_prob'] for r in results])
        print(f"\n📈 Model Calibration:")
        print(f"   Expected Wins: {expected_wins:.1f}")
        print(f"   Actual Wins: {wins}")
        print(f"   Difference: {wins - expected_wins:+.1f} ({(wins - expected_wins)/expected_wins*100:+.1f}%)")

    # Save detailed results
    if results:
        results_df = pd.DataFrame(results)
        output_file = Path("reports/WEEK10_BACKTEST_COMPLETE.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Detailed results saved to: {output_file}")

        # Best and worst picks
        if len(results_df) > 0:
            results_df_sorted = results_df.sort_values('profit', ascending=False)
            print(f"\n🏆 Top 3 Picks:")
            for i, row in results_df_sorted.head(3).iterrows():
                print(f"   {row['player']}: {row['pick']} → ${row['profit']:+.2f}")

            print(f"\n💸 Bottom 3 Picks:")
            for i, row in results_df_sorted.tail(3).iterrows():
                print(f"   {row['player']}: {row['pick']} → ${row['profit']:+.2f}")

    print("\n" + "="*100)
    print("VALIDATION COMPLETE")
    print("="*100)

    return {
        'total_picks': len(top_picks),
        'evaluated': evaluated,
        'wins': wins,
        'losses': losses,
        'win_rate': wins/evaluated if evaluated > 0 else 0,
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'roi': (total_profit/total_wagered)*100 if total_wagered > 0 else 0
    }

if __name__ == "__main__":
    main()
