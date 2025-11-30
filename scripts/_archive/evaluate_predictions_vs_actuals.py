#!/usr/bin/env python3
"""
Evaluate model predictions against actual game results.
Calculate hypothetical ROI if bets were placed.
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.utils.nflverse_loader import load_schedules

def american_odds_to_decimal(odds):
    """Convert American odds to decimal."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def calculate_payout(wager, odds):
    """Calculate payout from American odds."""
    decimal_odds = american_odds_to_decimal(odds)
    return wager * (decimal_odds - 1)

def evaluate_predictions():
    """Evaluate predictions against actual results."""

    schedule_2024 = load_schedules(seasons=2024)
    reports_dir = Path("/Users/keyonnesession/Desktop/NFL QUANT/reports")

    print("="*100)
    print("MODEL PREDICTION PERFORMANCE EVALUATION")
    print("="*100)

    # === ANALYZE ARI @ DAL (Week 10) ===
    print("\n" + "="*100)
    print("CURRENT WEEK RECOMMENDATIONS: ARI @ DAL (Week 10)")
    print("="*100)

    # Get actual game result
    dal_game = schedule_2024[(schedule_2024['week'] == 10) &
                             (schedule_2024['home_team'] == 'DAL') &
                             (schedule_2024['away_team'] == 'PHI')]

    # Actually the game in CURRENT_WEEK was listed as ARI @ DAL
    # But Week 10 shows PHI @ DAL. Let me check both
    week10_dal = schedule_2024[(schedule_2024['week'] == 10) &
                                (schedule_2024['home_team'] == 'DAL')]

    if len(week10_dal) > 0:
        game = week10_dal.iloc[0]
        print(f"\nACTUAL GAME: {game['away_team']} @ {game['home_team']}")
        print(f"Final Score: {game['away_team']} {game['away_score']} - {game['home_team']} {game['home_score']}")
        print(f"Spread Line: {game['home_team']} {game['spread_line']}")
        print(f"Total Line: {game['total_line']}")

        away_score = game['away_score']
        home_score = game['home_score']
        total = away_score + home_score

        # Load predictions
        current_rec = pd.read_csv(reports_dir / "CURRENT_WEEK_RECOMMENDATIONS.csv")

        print(f"\nMODEL RECOMMENDATIONS:")
        print("-"*100)

        total_wagered = 0
        total_won = 0
        wins = 0
        losses = 0

        for idx, bet in current_rec.iterrows():
            market = bet['market']
            pick = bet['pick']
            line = bet['line']
            odds = bet['odds']
            model_prob = bet['model_prob']
            edge_pct = bet['edge_pct']

            # Assume Kelly Criterion sizing at ~5% of bankroll per bet
            # Let's use $1000 bankroll for simulation
            bankroll = 1000
            wager = bankroll * 0.05  # 5% per bet, conservative

            result = None
            won = False

            if market == 'game_spread':
                # DAL -3.5 means DAL needs to win by more than 3.5
                if 'DAL -' in pick:
                    # Picked DAL to cover spread
                    actual_spread = home_score - away_score
                    won = actual_spread > line
                    result = f"DAL won by {actual_spread:.1f} (needed >{line})"

            elif market == 'game_total':
                if 'Under' in pick:
                    won = total < line
                    result = f"Total was {total} (needed <{line})"
                elif 'Over' in pick:
                    won = total > line
                    result = f"Total was {total} (needed >{line})"

            elif market == 'game_moneyline':
                if 'DAL ML' in pick:
                    won = home_score > away_score
                    result = f"DAL {'won' if won else 'lost'} outright"
                elif 'ARI ML' in pick or 'PHI ML' in pick:
                    won = away_score > home_score
                    result = f"{game['away_team']} {'won' if won else 'lost'} outright"

            if won:
                profit = calculate_payout(wager, odds)
                total_won += profit
                wins += 1
                status = f"WIN (+${profit:.2f})"
            else:
                total_won -= wager
                losses += 1
                status = f"LOSS (-${wager:.2f})"

            total_wagered += wager

            print(f"\n{market.upper()}: {pick}")
            print(f"  Line: {line} | Odds: {odds} | Model Prob: {model_prob:.1%} | Edge: {edge_pct:.2f}%")
            print(f"  Wager: ${wager:.2f}")
            print(f"  Result: {result}")
            print(f"  {status}")

        print(f"\n" + "-"*100)
        print(f"RESULTS SUMMARY:")
        print(f"  Total Bets: {wins + losses}")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%")
        print(f"  Total Wagered: ${total_wagered:.2f}")
        print(f"  Net Profit/Loss: ${total_won:.2f}")
        print(f"  ROI: {(total_won/total_wagered)*100:.1f}%")

    # === ANALYZE WEEK 8 RECOMMENDATIONS ===
    print("\n\n" + "="*100)
    print("WEEK 8 RECOMMENDATIONS PERFORMANCE")
    print("="*100)

    final_rec = pd.read_csv(reports_dir / "FINAL_RECOMMENDATIONS.csv")
    week8_games = schedule_2024[schedule_2024['week'] == 8]

    total_wagered = 0
    total_profit = 0
    wins = 0
    losses = 0

    print(f"\nAnalyzing {len(final_rec)} recommendations...")
    print("-"*100)

    for idx, bet in final_rec.iterrows():
        game_str = bet['game']
        bet_type = bet['bet_type']
        bet_on = bet['bet_on']
        wager = float(bet['wager'].replace('$', ''))
        odds = bet['odds']

        # Parse game string (e.g., "TEN @ IND")
        parts = game_str.split(' @ ')
        if len(parts) != 2:
            continue
        away_team, home_team = parts

        # Find the actual game
        game_result = week8_games[(week8_games['away_team'] == away_team) &
                                   (week8_games['home_team'] == home_team)]

        if len(game_result) == 0:
            print(f"\nWARNING: Could not find game {game_str}")
            continue

        game_result = game_result.iloc[0]
        away_score = game_result['away_score']
        home_score = game_result['home_score']
        spread_line = game_result['spread_line']

        # Determine if bet won
        won = False
        result_desc = ""

        if bet_type == 'Spread':
            if 'home' in bet_on.lower():
                # Bet on home team to cover
                actual_margin = home_score - away_score
                won = actual_margin > (-spread_line)
                result_desc = f"{home_team} won by {actual_margin} (spread: {spread_line})"
            else:
                # Bet on away team to cover
                actual_margin = away_score - home_score
                won = actual_margin > spread_line
                result_desc = f"{away_team} margin: {actual_margin} (spread: {spread_line})"

        if won:
            profit = calculate_payout(wager, odds)
            total_profit += profit
            wins += 1
            status = "WIN"
        else:
            total_profit -= wager
            losses += 1
            status = "LOSS"

        total_wagered += wager

        print(f"\n{game_str} - {bet_on}")
        print(f"  Score: {away_team} {away_score} - {home_team} {home_score}")
        print(f"  Wager: ${wager:.2f} at {odds} -> {status}")
        print(f"  {result_desc}")

    print(f"\n" + "="*100)
    print(f"WEEK 8 RESULTS SUMMARY:")
    print(f"  Total Bets: {wins + losses}")
    print(f"  Wins: {wins}")
    print(f"  Losses: {losses}")
    print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%")
    print(f"  Total Wagered: ${total_wagered:.2f}")
    print(f"  Net Profit/Loss: ${total_profit:.2f}")
    print(f"  ROI: {(total_profit/total_wagered)*100:.1f}%")
    print("="*100)

if __name__ == "__main__":
    evaluate_predictions()
