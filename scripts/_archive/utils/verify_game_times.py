#!/usr/bin/env python3
"""Verify game times are properly loaded and connected"""

import pandas as pd
from pathlib import Path

def main():
    print("="*80)
    print("VERIFYING GAME TIMES CONNECTION")
    print("="*80)
    print()

    # Load recommendations
    recs_path = Path('reports/unified_betting_recommendations.csv')
    if not recs_path.exists():
        print(f"❌ {recs_path} not found")
        return

    df = pd.read_csv(recs_path)
    print(f"✅ Loaded {len(df)} recommendations")

    # Check columns
    print("\n📋 Column Check:")
    required_cols = ['game', 'game_time', 'commence_time']
    for col in required_cols:
        status = "✅" if col in df.columns else "❌"
        print(f"  {status} {col}")

    # Check data coverage
    print("\n📊 Data Coverage:")
    total = len(df)
    with_times = df['game_time'].notna().sum()
    print(f"  Total rows: {total}")
    print(f"  Rows with game_time: {with_times} ({with_times/total*100:.1f}%)")
    print(f"  Rows without game_time: {total - with_times} ({(total-with_times)/total*100:.1f}%)")

    # Check unique games
    print("\n🎮 Unique Games:")
    all_games = df['game'].unique()
    games_with_times = df[df['game_time'].notna()]['game'].unique()
    games_without_times = df[df['game_time'].isna()]['game'].unique()

    print(f"  Total unique games: {len(all_games)}")
    print(f"  Games WITH times: {len(games_with_times)}")
    print(f"  Games WITHOUT times: {len(games_without_times)}")

    if len(games_with_times) > 0:
        print("\n  ✅ Games with times:")
        for game in sorted(games_with_times):
            count = len(df[(df['game'] == game) & (df['game_time'].notna())])
            sample_time = df[df['game'] == game]['game_time'].iloc[0]
            print(f"    • {game} ({count} bets) - {sample_time}")

    if len(games_without_times) > 0:
        print("\n  ❌ Games without times:")
        for game in sorted(games_without_times):
            count = len(df[(df['game'] == game) & (df['game_time'].isna())])
            print(f"    • {game} ({count} bets)")

    # Check odds files for missing games
    print("\n🔍 Checking odds files for missing games...")
    data_dir = Path('data')
    found_games_in_files = set()

    if data_dir.exists():
        for odds_file in data_dir.glob('odds*.csv'):
            try:
                odf = pd.read_csv(odds_file)
                if 'commence_time' in odf.columns:
                    if 'game' in odf.columns:
                        found_games_in_files.update(odf['game'].unique())
                    elif 'home_team' in odf.columns and 'away_team' in odf.columns:
                        odf['game'] = odf['away_team'] + ' @ ' + odf['home_team']
                        found_games_in_files.update(odf['game'].unique())
            except Exception as e:
                print(f"  ⚠️  Error reading {odds_file.name}: {e}")

    # Also check player props file
    props_file = Path('data/nfl_player_props_draftkings.csv')
    if props_file.exists():
        try:
            pdf = pd.read_csv(props_file)
            if 'commence_time' in pdf.columns:
                if 'game' in pdf.columns:
                    found_games_in_files.update(pdf['game'].unique())
                elif 'home_team' in pdf.columns and 'away_team' in pdf.columns:
                    pdf['game'] = pdf['away_team'] + ' @ ' + pdf['home_team']
                    found_games_in_files.update(pdf['game'].unique())
        except Exception as e:
            print(f"  ⚠️  Error reading {props_file.name}: {e}")

    print(f"\n  Found {len(found_games_in_files)} unique games in odds files")

    # Check if missing games exist in odds files
    if len(games_without_times) > 0:
        print("\n🔗 Matching Analysis:")
        for game in sorted(games_without_times):
            if game in found_games_in_files:
                print(f"  ⚠️  {game} - EXISTS in odds files but NOT matched!")
            else:
                print(f"  ❌ {game} - NOT found in any odds files")

    # Check data consistency
    print("\n🔐 Data Consistency Check:")
    if 'commence_time' in df.columns and 'game_time' in df.columns:
        both_present = df[df['commence_time'].notna() & df['game_time'].notna()]
        if len(both_present) > 0:
            mismatched = both_present[both_present['commence_time'] != both_present['game_time']]
            if len(mismatched) > 0:
                print(f"  ⚠️  {len(mismatched)} rows have mismatched commence_time and game_time")
            else:
                print(f"  ✅ All {len(both_present)} rows have matching commence_time and game_time")

    # Sample data check
    print("\n📝 Sample Data:")
    print("  First 3 rows with game times:")
    sample = df[df['game_time'].notna()][['game', 'player', 'game_time']].head(3)
    for idx, row in sample.iterrows():
        print(f"    • {row['game']} - {row.get('player', 'N/A')} - {row['game_time']}")

    print("\n" + "="*80)
    if len(games_without_times) == 0:
        print("✅ ALL GAMES HAVE TIMES - Everything is connected!")
    else:
        print(f"⚠️  {len(games_without_times)} games missing times - Check odds files")
    print("="*80)

if __name__ == '__main__':
    main()
