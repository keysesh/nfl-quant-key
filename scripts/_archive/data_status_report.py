#!/usr/bin/env python3
"""
Comprehensive data status report.
Shows what data we have, what's missing, and what needs to be fetched.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_quant.utils.nflverse_loader import load_schedules

def generate_status_report():
    """Generate comprehensive data status report."""

    print("="*100)
    print("NFL QUANT FRAMEWORK DATA STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    # === CURRENT SEASON STATUS ===
    print("\n" + "="*100)
    print("1. SEASON & WEEK STATUS")
    print("="*100)

    schedule_2024 = load_schedules(seasons=2024)
    now = datetime(2024, 11, 10)  # Simulating the date context

    # Find current week based on Nov 10, 2024
    completed_games = schedule_2024[pd.to_datetime(schedule_2024['gameday']) <= now]
    current_week = completed_games['week'].max() if len(completed_games) > 0 else 1

    upcoming_games = schedule_2024[pd.to_datetime(schedule_2024['gameday']) > now]
    next_week = upcoming_games['week'].min() if len(upcoming_games) > 0 else current_week + 1

    print(f"\nSeason: 2024 NFL")
    print(f"Reference Date: November 10, 2024")
    print(f"Most Recent Completed Week: {current_week}")
    print(f"Next Upcoming Week: {next_week}")

    # Check for Monday Night Football
    mnf_games = schedule_2024[(schedule_2024['week'] == current_week) &
                               (pd.to_datetime(schedule_2024['gameday']) == '2024-11-11')]
    if len(mnf_games) > 0:
        print(f"\nMonday Night Football (Week {current_week}):")
        for idx, game in mnf_games.iterrows():
            print(f"  {game['away_team']} @ {game['home_team']}")
            if pd.notna(game['away_score']):
                print(f"  Final: {game['away_score']}-{game['home_score']}")

    # === DATA FILES STATUS ===
    print("\n" + "="*100)
    print("2. DATA FILES STATUS")
    print("="*100)

    data_dir = Path("/Users/keyonnesession/Desktop/NFL QUANT/data")
    reports_dir = Path("/Users/keyonnesession/Desktop/NFL QUANT/reports")

    # Check for odds data
    print("\nODDS DATA:")
    odds_files = list(data_dir.glob("odds_week*.csv"))
    if odds_files:
        for f in sorted(odds_files):
            stats = f.stat()
            modified = datetime.fromtimestamp(stats.st_mtime)
            print(f"  {f.name} (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("  No odds files found")

    # Check for injury data
    print("\nINJURY DATA:")
    injury_files = list(data_dir.glob("injuries/injuries_week*.csv"))
    if injury_files:
        latest = max(injury_files, key=lambda f: f.stat().st_mtime)
        modified = datetime.fromtimestamp(latest.stat().st_mtime)
        print(f"  Latest: {latest.name} (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
        print(f"  Total files: {len(injury_files)}")
    else:
        print("  No injury files found")

    # Check for weather data
    print("\nWEATHER DATA:")
    weather_files = list(data_dir.glob("weather/weather_week*.csv"))
    if weather_files:
        latest = max(weather_files, key=lambda f: f.stat().st_mtime)
        modified = datetime.fromtimestamp(latest.stat().st_mtime)
        print(f"  Latest: {latest.name} (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
        print(f"  Total files: {len(weather_files)}")
    else:
        print("  No weather files found")

    # Check for predictions
    print("\nPREDICTIONS:")
    prediction_files = list(data_dir.glob("model_predictions_week*.csv"))
    if prediction_files:
        for f in sorted(prediction_files):
            stats = f.stat()
            modified = datetime.fromtimestamp(stats.st_mtime)
            print(f"  {f.name} (modified: {modified.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("  No prediction files found")

    # === RECOMMENDATIONS STATUS ===
    print("\n" + "="*100)
    print("3. BETTING RECOMMENDATIONS STATUS")
    print("="*100)

    # Check CURRENT_WEEK_RECOMMENDATIONS
    current_rec_file = reports_dir / "CURRENT_WEEK_RECOMMENDATIONS.csv"
    if current_rec_file.exists():
        df = pd.read_csv(current_rec_file)
        stats = current_rec_file.stat()
        modified = datetime.fromtimestamp(stats.st_mtime)

        print(f"\nCURRENT_WEEK_RECOMMENDATIONS.csv")
        print(f"  Last Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Total Recommendations: {len(df)}")

        if 'game' in df.columns:
            games = df['game'].unique()
            print(f"  Games: {', '.join(games)}")

            # Check if these games actually exist in schedule
            for game_str in games:
                if ' @ ' in game_str:
                    away, home = game_str.split(' @ ')
                    actual_game = schedule_2024[((schedule_2024['away_team'] == away) &
                                                  (schedule_2024['home_team'] == home))]
                    if len(actual_game) > 0:
                        week = actual_game.iloc[0]['week']
                        gameday = actual_game.iloc[0]['gameday']
                        print(f"    {game_str} -> Week {week} ({gameday})")
                    else:
                        print(f"    {game_str} -> NOT FOUND IN 2024 SCHEDULE")

    # Check FINAL_RECOMMENDATIONS
    final_rec_file = reports_dir / "FINAL_RECOMMENDATIONS.csv"
    if final_rec_file.exists():
        df = pd.read_csv(final_rec_file)
        stats = final_rec_file.stat()
        modified = datetime.fromtimestamp(stats.st_mtime)

        print(f"\nFINAL_RECOMMENDATIONS.csv")
        print(f"  Last Modified: {modified.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Total Recommendations: {len(df)}")

        if 'game' in df.columns:
            games = df['game'].unique()
            print(f"  Games Covered: {len(games)}")

    # === BACKTEST RESULTS ===
    print("\n" + "="*100)
    print("4. BACKTEST PERFORMANCE (Weeks 1-8)")
    print("="*100)

    backtest_file = reports_dir / "week_by_week_backtest_results.csv"
    if backtest_file.exists():
        df = pd.read_csv(backtest_file)
        print(f"\n{df.to_string(index=False)}")

        # Calculate cumulative performance
        total_bets = df['total_bets'].sum()
        total_wins = df['wins'].sum()
        total_profit = df['profit'].sum()

        print(f"\nCUMULATIVE PERFORMANCE (Weeks 1-8):")
        print(f"  Total Bets: {total_bets:,}")
        print(f"  Total Wins: {total_wins:,}")
        print(f"  Win Rate: {(total_wins/total_bets)*100:.2f}%")
        print(f"  Total Profit: ${total_profit:,.2f}")
        print(f"  Average ROI: {df['roi'].mean():.2f}%")

    # === ACTION ITEMS ===
    print("\n" + "="*100)
    print("5. RECOMMENDED ACTIONS")
    print("="*100)

    print("\nTo get up-to-date predictions and place bets for current week:")
    print()
    print("STEP 1: Fetch Latest Game Data")
    print("  - Update team stats through latest completed week")
    print("  - Fetch current odds for upcoming games")
    print("  - Update injury reports")
    print("  - Get weather forecasts for upcoming games")
    print()
    print("STEP 2: Generate Predictions")
    print("  - Run prediction model for next week's games")
    print("  - Generate player prop predictions")
    print("  - Calculate expected value for available bets")
    print()
    print("STEP 3: Generate Recommendations")
    print("  - Filter bets with positive expected value")
    print("  - Apply Kelly Criterion for bet sizing")
    print("  - Create actionable betting recommendations")
    print()
    print("STEP 4: Evaluate Past Performance (Optional)")
    print(f"  - Backtest Week {current_week} predictions vs actual results")
    print("  - Calculate ROI on hypothetical bets")
    print("  - Update model calibration if needed")

    print("\n" + "="*100)

if __name__ == "__main__":
    generate_status_report()
