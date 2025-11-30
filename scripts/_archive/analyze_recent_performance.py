#!/usr/bin/env python3
"""
Analyze how the model performed on recent weeks where we have predictions.
Shows what would have happened if we placed bets based on the recommendations.
"""

import pandas as pd
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.utils.nflverse_loader import load_schedules

def analyze_week_performance():
    """Analyze model performance for weeks with predictions."""

    # Get actual game results from R-fetched data
    schedule_2024 = load_schedules(seasons=2024)

    # Check Week 9 and Week 10
    week9_games = schedule_2024[schedule_2024['week'] == 9].copy()
    week10_games = schedule_2024[schedule_2024['week'] == 10].copy()

    print("="*80)
    print("NFL BETTING MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    print()

    # === WEEK 9 ANALYSIS ===
    print("WEEK 9 GAMES (Nov 3-4, 2024)")
    print("-"*80)

    for idx, game in week9_games.iterrows():
        away = game['away_team']
        home = game['home_team']
        away_score = game['away_score']
        home_score = game['home_score']
        spread = game['spread_line']
        total = game['total_line']

        print(f"\n{away} @ {home}")
        print(f"  Final Score: {away} {away_score} - {home} {home_score}")
        if pd.notna(spread):
            print(f"  Spread: {home} {spread}")
            # Determine spread winner
            home_cover = (home_score + spread) > away_score
            print(f"  Spread Result: {home if home_cover else away} covered")
        if pd.notna(total):
            actual_total = away_score + home_score
            print(f"  Total: {total} (Actual: {actual_total}) - {'OVER' if actual_total > total else 'UNDER'}")

    print("\n" + "="*80)
    print("WEEK 10 GAMES (Nov 7-11, 2024)")
    print("-"*80)

    for idx, game in week10_games.iterrows():
        away = game['away_team']
        home = game['home_team']
        away_score = game['away_score']
        home_score = game['home_score']
        spread = game['spread_line']
        total = game['total_line']
        gameday = game['gameday']

        print(f"\n{away} @ {home} ({gameday})")
        print(f"  Final Score: {away} {away_score} - {home} {home_score}")
        if pd.notna(spread):
            print(f"  Spread: {home} {spread}")
            # Determine spread winner
            home_cover = (home_score + spread) > away_score
            print(f"  Spread Result: {home if home_cover else away} covered")
        if pd.notna(total):
            actual_total = away_score + home_score
            print(f"  Total: {total} (Actual: {actual_total}) - {'OVER' if actual_total > total else 'UNDER'}")

    print("\n" + "="*80)
    print("CHECKING FOR MODEL PREDICTIONS")
    print("="*80)

    # Check what predictions we have
    data_dir = Path("/Users/keyonnesession/Desktop/NFL QUANT/data")
    reports_dir = Path("/Users/keyonnesession/Desktop/NFL QUANT/reports")

    # Look for recommendations files
    recommendations_files = list(reports_dir.glob("*RECOMMENDATIONS*.csv"))
    recommendations_files += list(reports_dir.glob("*recommendations*.csv"))
    recommendations_files += list(data_dir.glob("*recommendations*.csv"))

    print(f"\nFound {len(recommendations_files)} recommendation files:")
    for f in recommendations_files:
        print(f"  - {f.name}")

    # Check for the current week recommendations
    current_rec_file = reports_dir / "CURRENT_WEEK_RECOMMENDATIONS.csv"
    if current_rec_file.exists():
        print(f"\n\nCURRENT WEEK RECOMMENDATIONS:")
        print("-"*80)
        df = pd.read_csv(current_rec_file)

        # Check if these are game-level bets
        if 'game' in df.columns:
            print(f"\nFound {len(df)} recommendations:")
            print(df[['game', 'market', 'pick', 'line', 'odds', 'model_prob', 'edge_pct']].to_string(index=False))
        else:
            print(f"\nFound {len(df)} player prop recommendations")
            print(df.head(10).to_string(index=False))

    # Check final recommendations
    final_rec_file = reports_dir / "FINAL_RECOMMENDATIONS.csv"
    if final_rec_file.exists():
        print(f"\n\nFINAL RECOMMENDATIONS (likely from Week 8):")
        print("-"*80)
        df = pd.read_csv(final_rec_file)
        if 'game' in df.columns:
            print(f"\nFound {len(df)} game recommendations:")
            print(df[['game', 'bet_type', 'bet_on', 'model_prob', 'edge', 'wager', 'potential_profit']].to_string(index=False))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY & NEXT STEPS")
    print("="*80)
    print("\nTo evaluate model performance on recent weeks, we need to:")
    print("1. Check if predictions were generated for Week 9 or Week 10")
    print("2. If not, we can generate them retrospectively for analysis")
    print("3. Compare predicted outcomes vs actual results")
    print("4. Calculate hypothetical ROI based on recommended wagers")
    print("\nTonight's game: MIA @ LA (Mon Nov 11, 2024)")
    print("  - This was the MNF game for Week 10")
    print("  - Final: MIA 23 - LA 15")
    print()

if __name__ == "__main__":
    analyze_week_performance()
