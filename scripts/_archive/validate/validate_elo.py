#!/usr/bin/env python3
"""
Validate Elo calculations for any season/week.
Usage: python scripts/validate/validate_elo.py [--season YEAR] [--week NUM]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from nfl_quant.features.team_strength import EnhancedEloCalculator


def main():
    parser = argparse.ArgumentParser(description='Validate Elo calculations')
    parser.add_argument('--season', type=int, default=None, help='Season year (default: latest)')
    parser.add_argument('--week', type=int, default=None, help='Week number (default: latest completed + 1)')
    args = parser.parse_args()

    calc = EnhancedEloCalculator()

    # Auto-detect season/week if not specified
    schedules = pd.read_parquet('data/nflverse/schedules.parquet')

    if args.season is None:
        args.season = schedules['season'].max()

    if args.week is None:
        completed = schedules[
            (schedules['season'] == args.season) &
            (schedules['home_score'].notna()) &
            (schedules['game_type'] == 'REG')
        ]['week'].max()
        args.week = int(completed) + 1 if pd.notna(completed) else 1

    print(f"\nValidating Elo for Season {args.season}, entering Week {args.week}")
    print(f"(Using data through Week {args.week - 1})\n")

    # Show rankings
    rankings = calc.get_all_team_ratings(args.season, args.week)
    print("TEAM RANKINGS BY ELO:")
    print(rankings.to_string(index=False))

    # Run validation
    passed = calc.print_validation_report(args.season, args.week)

    # Show sample matchup predictions for upcoming week
    upcoming = schedules[
        (schedules['season'] == args.season) &
        (schedules['week'] == args.week) &
        (schedules['home_score'].isna())  # Not yet played
    ]

    if len(upcoming) > 0:
        print("\nUPCOMING GAME PREDICTIONS:")
        print("-" * 60)
        for _, game in upcoming.head(8).iterrows():
            home = game['home_team']
            away = game['away_team']
            features = calc.get_team_features(home, away, args.season, args.week, is_home=True)

            vegas_spread = game.get('spread_line', None)
            vegas_str = f"Vegas: {vegas_spread:+.1f}" if pd.notna(vegas_spread) else "Vegas: N/A"

            print(f"{away} @ {home}")
            print(f"  {home} win prob: {features['win_probability']:.1%}")
            print(f"  Elo spread: {features['expected_spread']:+.1f}")
            print(f"  {vegas_str}")
            print()

    # Compare to Vegas if available
    completed_with_vegas = schedules[
        (schedules['season'] == args.season) &
        (schedules['week'] < args.week) &
        (schedules['home_score'].notna()) &
        (schedules['spread_line'].notna())
    ]

    if len(completed_with_vegas) > 0:
        print("\nELO vs VEGAS ACCURACY (completed games):")
        print("-" * 60)

        elo_correct = 0
        vegas_correct = 0
        total = 0

        for _, game in completed_with_vegas.iterrows():
            home = game['home_team']
            away = game['away_team']
            actual_result = game['result']  # home - away
            vegas_spread = game['spread_line']  # negative = home favored

            # Get Elo prediction BEFORE the game
            features = calc.get_team_features(home, away, args.season, game['week'], is_home=True)
            elo_spread = features['expected_spread']

            # Who did each predict to win?
            elo_pick_home = elo_spread > 0
            vegas_pick_home = vegas_spread < 0  # Negative spread means home favored
            actual_home_won = actual_result > 0

            if elo_pick_home == actual_home_won:
                elo_correct += 1
            if vegas_pick_home == actual_home_won:
                vegas_correct += 1
            total += 1

        print(f"Elo correct: {elo_correct}/{total} ({elo_correct/total:.1%})")
        print(f"Vegas correct: {vegas_correct}/{total} ({vegas_correct/total:.1%})")

    return passed


if __name__ == '__main__':
    main()
