#!/usr/bin/env python3
"""
Generic week validation script - works for any NFL week.
Validates all picks against actual results and calculates ROI.

Usage:
    python scripts/backtest/validate_week_generic.py --week 10
    python scripts/backtest/validate_week_generic.py --week 11 --season 2025
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.season_config import CURRENT_NFL_SEASON, infer_season_from_week

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

def load_game_results(week, season=None):
    """Load actual game results for the week."""
    if season is None:
        season = infer_season_from_week(week)

    try:
        from nfl_quant.utils.nflverse_loader import load_schedules
        schedule = load_schedules(seasons=season)
        week_games = schedule[schedule['week'] == week]

        # Convert to dict format: {game_id: {home_team, away_team, home_score, away_score, total}}
        game_results = {}
        for _, game in week_games.iterrows():
            game_key = f"{game['away_team']}@{game['home_team']}"
            game_results[game_key] = {
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_score': game['home_score'] if pd.notna(game.get('home_score')) else None,
                'away_score': game['away_score'] if pd.notna(game.get('away_score')) else None,
                'total': game['home_score'] + game['away_score'] if pd.notna(game.get('home_score')) and pd.notna(game.get('away_score')) else None,
                'spread_line': game.get('spread_line'),
                'total_line': game.get('total_line')
            }

        return game_results
    except Exception as e:
        print(f"⚠️  Warning: Could not load game results from nflreadpy: {e}")
        return {}

def find_game_for_player(player_name, teams_in_week):
    """Find which game a player was in based on team matchups."""
    # This is a simplified version - in production you'd want to query rosters
    # For now, we'll rely on the team matching in player stats
    return None

def main():
    parser = argparse.ArgumentParser(description='Validate NFL picks for any week')
    parser.add_argument('--week', type=int, required=True, help='NFL week number')
    parser.add_argument('--season', type=int, default=None,
                       help='NFL season year (default: auto-infer from week, currently %(default)s)')
    parser.add_argument('--unit-size', type=float, default=100, help='Unit size for wagers ($)')
    parser.add_argument('--min-edge', type=float, default=5.0, help='Minimum edge %% to evaluate')
    args = parser.parse_args()

    week = args.week

    # Smart season inference
    if args.season is None:
        season = infer_season_from_week(week)
        print(f"ℹ️  No season specified, inferred season {season} for week {week}")
    else:
        season = args.season
    UNIT_SIZE = args.unit_size
    MIN_EDGE = args.min_edge

    print("="*100)
    print(f"WEEK {week} VALIDATION ({season} SEASON)")
    print("="*100)

    # Load picks
    picks_file = Path(f"reports/all_picks_ranked_week{week}.csv")
    if not picks_file.exists():
        print(f"\n❌ ERROR: Picks file not found: {picks_file}")
        print(f"Expected: reports/all_picks_ranked_week{week}.csv")
        return 1

    picks_df = pd.read_csv(picks_file)
    top_picks = picks_df[picks_df['edge_pct'] > MIN_EDGE].copy()

    print(f"\nTotal Picks Generated: {len(picks_df)}")
    print(f"Top Recommendations (>{MIN_EDGE}% edge): {len(top_picks)}")

    # Load player stats
    stats_file = Path(f"data/results/{season}/week{week}_player_stats.csv")
    if not stats_file.exists():
        print(f"\n❌ ERROR: Player stats not found: {stats_file}")
        print(f"Run: python scripts/backtest/fetch_week_player_stats.py {week} {season}")
        return 1

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

    # Load game results
    game_results = load_game_results(week, season)
    print(f"Loaded results for {len(game_results)} games")

    # Get all teams playing this week
    teams_in_week = set()
    for game_key in game_results.keys():
        result = game_results[game_key]
        teams_in_week.add(result['home_team'])
        teams_in_week.add(result['away_team'])

    def find_player_stats(player_name_full, teams_to_check=None):
        """Find player stats by matching last name."""
        if pd.isna(player_name_full) or not player_name_full:
            return None, None

        lastname = normalize_player_name(player_name_full)
        if not lastname:
            return None, None

        # If no teams specified, check all teams in the week
        if teams_to_check is None:
            teams_to_check = teams_in_week

        for team in teams_to_check:
            key = (lastname, team)
            if key in stats_lookup_by_lastname:
                return stats_lookup_by_lastname[key], team

        return None, None

    def get_game_result(teams_involved):
        """Get game result for teams involved."""
        for game_key, result in game_results.items():
            if any(team in game_key for team in teams_involved):
                return result
        return None

    print("\n" + "="*100)
    print("VALIDATING ALL RECOMMENDATIONS")
    print("="*100)

    results = []
    total_wagered = 0
    total_profit = 0
    wins = 0
    losses = 0
    no_data = 0

    for idx, pick in top_picks.iterrows():
        player_name = pick['player']
        bet_pick = pick['pick']
        market = pick['market']
        line = pick['line']
        odds = pick['odds']
        model_prob = pick['model_prob']
        edge = pick['edge_pct']
        game = pick.get('game', '')

        wager = UNIT_SIZE
        won = None
        actual_value = None
        reason = ""

        # Game line markets (spread, total, moneyline)
        if market in ['game_spread', 'game_total', 'game_moneyline']:
            # Parse game string to find teams
            if '@' in str(game):
                parts = game.split('@')
                away_team = parts[0].strip()
                home_team = parts[1].strip()
                game_key = f"{away_team}@{home_team}"

                if game_key in game_results:
                    game_result = game_results[game_key]

                    if market == 'game_total':
                        actual_value = game_result['total']
                        if actual_value is not None:
                            if 'Under' in bet_pick:
                                won = actual_value < line
                            elif 'Over' in bet_pick:
                                won = actual_value > line
                            reason = f"Total: {actual_value} vs line {line}"

                    elif market == 'game_spread':
                        if game_result['home_score'] is not None and game_result['away_score'] is not None:
                            actual_margin = game_result['home_score'] - game_result['away_score']
                            actual_value = actual_margin

                            # Parse spread from bet_pick (e.g., "PHI -3.5" or "GB +3.5")
                            if home_team in bet_pick:
                                won = actual_margin > line
                            elif away_team in bet_pick:
                                won = -actual_margin > line
                            reason = f"Margin: {actual_margin} vs line {line}"

                    elif market == 'game_moneyline':
                        if game_result['home_score'] is not None and game_result['away_score'] is not None:
                            winner = home_team if game_result['home_score'] > game_result['away_score'] else away_team
                            if winner in bet_pick:
                                won = True
                            else:
                                won = False
                            reason = f"Winner: {winner}"

        # Player prop markets
        elif player_name:
            # Find player stats across all teams in the week
            player_stats_row, player_team = find_player_stats(player_name)

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
                elif market == 'player_anytime_td':
                    # Check if player scored any TD
                    total_tds = (player_stats_row.get('passing_tds', 0) +
                                player_stats_row.get('rushing_tds', 0) +
                                player_stats_row.get('receiving_tds', 0))
                    actual_value = total_tds
                    if 'Yes' in bet_pick or 'Over' in bet_pick:
                        won = total_tds > 0
                    else:
                        won = total_tds == 0
                    reason = f"TDs: {total_tds}"
                else:
                    actual_value = None
                    reason = f"Unknown market: {market}"

                if actual_value is not None and won is None:
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
    print(f"WEEK {week} RESULTS SUMMARY")
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
        output_file = Path(f"reports/WEEK{week}_BACKTEST_COMPLETE.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n✅ Detailed results saved to: {output_file}")

        # Best and worst picks
        if len(results_df) > 0:
            results_df_sorted = results_df.sort_values('profit', ascending=False)
            print(f"\n🏆 Top 3 Picks:")
            for i, (_, row) in enumerate(results_df_sorted.head(3).iterrows(), 1):
                print(f"   {i}. {row['player']}: {row['pick']} → ${row['profit']:+.2f}")

            if len(results_df) >= 3:
                print(f"\n💸 Bottom 3 Picks:")
                for i, (_, row) in enumerate(results_df_sorted.tail(3).iterrows(), 1):
                    print(f"   {i}. {row['player']}: {row['pick']} → ${row['profit']:+.2f}")

    print("\n" + "="*100)
    print("VALIDATION COMPLETE")
    print("="*100)

    return 0

if __name__ == "__main__":
    sys.exit(main())
