#!/usr/bin/env python3
"""
Build Master Game Lines Database

Consolidates ALL game line odds from multiple fragmented files into a single
comprehensive master file. This script:

1. Scans ALL odds files recursively in the data/ directory
2. Extracts game lines (spread, total, moneyline) from each source
3. Matches every game to NFLverse schedule by (home_team, away_team)
4. Assigns correct week from schedule (NOT from filename)
5. Deduplicates based on (season, week, home_team, away_team, market, side)
6. Outputs single master file that can be built upon week-over-week

Usage:
    python scripts/data/build_master_game_lines.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_nflverse_schedule(season: int) -> pd.DataFrame:
    """Load NFLverse schedule as source of truth."""
    schedules_file = PROJECT_ROOT / "data/nflverse/schedules.parquet"
    schedules = pd.read_parquet(schedules_file)
    schedules_season = schedules[schedules['season'] == season].copy()

    # Keep only needed columns
    schedules_season = schedules_season[[
        'season', 'week', 'game_id', 'gameday',
        'home_team', 'away_team', 'home_score', 'away_score'
    ]].copy()

    return schedules_season


def standardize_market_name(market: str) -> str:
    """Standardize market names across different file formats."""
    market_map = {
        'spreads': 'spread',
        'spread': 'spread',
        'totals': 'total',
        'total': 'total',
        'h2h': 'moneyline',
        'moneyline': 'moneyline'
    }
    return market_map.get(market.lower(), market.lower())


def extract_from_draftkings_files() -> List[pd.DataFrame]:
    """Extract game lines from DraftKings odds files."""
    print("\n" + "="*70)
    print("Extracting from DraftKings odds files...")
    print("="*70)

    draftkings_files = sorted((PROJECT_ROOT / "data").glob("odds_week*_draftkings.csv"))

    if not draftkings_files:
        print("  No DraftKings files found")
        return []

    all_data = []

    for dk_file in draftkings_files:
        print(f"\n📄 {dk_file.name}")
        df = pd.read_csv(dk_file)

        # Filter to game lines only
        game_line_markets = ['spread', 'total', 'moneyline']
        df = df[df['market'].isin(game_line_markets)].copy()

        if len(df) == 0:
            print("  ⚠️  No game lines found")
            continue

        # Add source metadata
        df['data_source'] = f"draftkings_file:{dk_file.name}"
        df['collected_at'] = df.get('archived_at', datetime.now().isoformat())

        # Standardize market names
        df['market'] = df['market'].apply(standardize_market_name)

        all_data.append(df)
        print(f"  ✅ Extracted {len(df)} odds from {df['game_id'].nunique()} games")

    return all_data


def extract_from_comprehensive_files() -> List[pd.DataFrame]:
    """Extract game lines from comprehensive odds files (which include player props)."""
    print("\n" + "="*70)
    print("Extracting from comprehensive odds files...")
    print("="*70)

    comp_files = sorted((PROJECT_ROOT / "data").glob("odds_week*_comprehensive.csv"))

    if not comp_files:
        print("  No comprehensive files found")
        return []

    all_data = []

    for comp_file in comp_files:
        print(f"\n📄 {comp_file.name}")
        df = pd.read_csv(comp_file)

        # Filter to game line markets only
        game_line_markets = ['spreads', 'totals', 'h2h']
        df = df[df['market'].isin(game_line_markets)].copy()

        if len(df) == 0:
            print("  ⚠️  No game lines found")
            continue

        # Convert to standard format
        standardized = []

        for _, row in df.iterrows():
            # Determine side and market
            if row['market'] == 'h2h':
                side = row['direction']  # 'home' or 'away'
                point = None
                market = 'moneyline'
            elif row['market'] == 'spreads':
                side = row['direction']
                point = row['line']
                market = 'spread'
            elif row['market'] == 'totals':
                side = row['direction']  # 'over' or 'under'
                point = row['line']
                market = 'total'
            else:
                continue

            standardized.append({
                'game_id': row['game_id'],
                'away_team': row['away_team'],
                'home_team': row['home_team'],
                'commence_time': row['commence_time'],
                'sportsbook': row['sportsbook'],
                'market': market,
                'side': side,
                'point': point,
                'price': row['market_odds'],
                'data_source': f"comprehensive_file:{comp_file.name}",
                'collected_at': datetime.now().isoformat()
            })

        if standardized:
            std_df = pd.DataFrame(standardized)
            all_data.append(std_df)
            print(f"  ✅ Extracted {len(std_df)} odds from {std_df['game_id'].nunique()} games")

    return all_data


def extract_from_historical_game_lines() -> List[pd.DataFrame]:
    """Extract from properly archived game lines directory."""
    print("\n" + "="*70)
    print("Extracting from historical game lines archive...")
    print("="*70)

    hist_dir = PROJECT_ROOT / "data" / "historical" / "game_lines"

    if not hist_dir.exists():
        print("  Historical game lines directory not found")
        return []

    hist_files = sorted(hist_dir.glob("game_lines_*.csv"))

    if not hist_files:
        print("  No historical game line files found")
        return []

    all_data = []

    for hist_file in hist_files:
        print(f"\n📄 {hist_file.name}")
        df = pd.read_csv(hist_file)

        # Add source metadata
        df['data_source'] = f"historical_archive:{hist_file.name}"
        df['collected_at'] = df.get('archived_at', datetime.now().isoformat())

        # Standardize market names
        df['market'] = df['market'].apply(standardize_market_name)

        all_data.append(df)
        print(f"  ✅ Extracted {len(df)} odds from {df['game_id'].nunique()} games")

    return all_data


def match_to_schedule(odds_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match game lines to NFLverse schedule and assign correct weeks.

    This is CRITICAL: we match by (home_team, away_team) instead of trusting
    the week assignment from the original file, which may be incorrect.
    """
    print("\n" + "="*70)
    print("Matching games to NFLverse schedule...")
    print("="*70)

    matched = []
    unmatched = []

    # Group by unique game combinations
    game_groups = odds_df.groupby(['home_team', 'away_team'])

    for (home_team, away_team), game_odds in game_groups:
        # Find in schedule
        match = schedule_df[
            (schedule_df['home_team'] == home_team) &
            (schedule_df['away_team'] == away_team)
        ]

        if len(match) > 0:
            # Assign correct week from schedule
            schedule_row = match.iloc[0]
            game_odds = game_odds.copy()
            game_odds['season'] = int(schedule_row['season'])
            game_odds['week'] = int(schedule_row['week'])
            game_odds['gameday'] = schedule_row['gameday']
            game_odds['game_id_nflverse'] = schedule_row['game_id']

            matched.append(game_odds)
            print(f"  ✅ {away_team}@{home_team} → Week {int(schedule_row['week'])}")
        else:
            unmatched.append((away_team, home_team))
            print(f"  ⚠️  {away_team}@{home_team} NOT FOUND in schedule")

    if unmatched:
        print(f"\n⚠️  {len(unmatched)} games not found in schedule:")
        for away, home in unmatched:
            print(f"    {away}@{home}")

    if not matched:
        print("\n❌ No games matched to schedule")
        return pd.DataFrame()

    combined = pd.concat(matched, ignore_index=True)
    print(f"\n✅ Matched {len(combined)} odds from {len(matched)} unique games")

    return combined


def deduplicate_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate odds based on (season, week, home_team, away_team, market, side).

    When duplicates exist, prefer:
    1. Most recent collection time
    2. Records with non-null point values
    3. Historical archive as most trusted source
    """
    print("\n" + "="*70)
    print("Deduplicating odds...")
    print("="*70)

    initial_count = len(odds_df)

    # Add source priority (higher = more trusted)
    def get_source_priority(source: str) -> int:
        if 'historical_archive' in source:
            return 3
        elif 'draftkings_file' in source:
            return 2
        elif 'comprehensive_file' in source:
            return 1
        else:
            return 0

    odds_df['source_priority'] = odds_df['data_source'].apply(get_source_priority)

    # Sort by priority before deduplication
    odds_df = odds_df.sort_values([
        'source_priority',
        'collected_at',
        'point'
    ], ascending=[False, False, False])

    # Define deduplication key
    dedup_cols = ['season', 'week', 'home_team', 'away_team', 'market', 'side']

    # Keep first (highest priority) record
    odds_df = odds_df.drop_duplicates(subset=dedup_cols, keep='first')

    # Drop temporary column
    odds_df = odds_df.drop(columns=['source_priority'])

    final_count = len(odds_df)
    removed_count = initial_count - final_count

    print(f"  Initial records: {initial_count}")
    print(f"  Duplicates removed: {removed_count}")
    print(f"  Final records: {final_count}")

    return odds_df


def generate_coverage_report(
    master_df: pd.DataFrame,
    schedule_df: pd.DataFrame
) -> Dict:
    """Generate comprehensive coverage report showing gaps."""
    print("\n" + "="*70)
    print("COVERAGE REPORT")
    print("="*70)

    coverage = {}

    for week in sorted(schedule_df['week'].unique()):
        week_schedule = schedule_df[schedule_df['week'] == week]
        week_odds = master_df[master_df['week'] == week]

        total_games = len(week_schedule)
        games_with_odds = week_odds['game_id_nflverse'].nunique() if 'game_id_nflverse' in week_odds.columns else week_odds[['home_team', 'away_team']].drop_duplicates().shape[0]

        # Count by market
        spread_count = len(week_odds[week_odds['market'] == 'spread']) // 2  # Divide by 2 (home + away)
        total_count = len(week_odds[week_odds['market'] == 'total']) // 2
        moneyline_count = len(week_odds[week_odds['market'] == 'moneyline']) // 2

        coverage_pct = (games_with_odds / total_games * 100) if total_games > 0 else 0

        status = "✅" if coverage_pct >= 90 else "⚠️" if coverage_pct >= 50 else "❌"

        print(f"\nWeek {week:2d}: {status} {games_with_odds}/{total_games} games ({coverage_pct:.0f}%)")
        print(f"  Spread: {spread_count} games")
        print(f"  Total: {total_count} games")
        print(f"  Moneyline: {moneyline_count} games")

        coverage[week] = {
            'total_games': total_games,
            'games_with_odds': games_with_odds,
            'coverage_pct': coverage_pct,
            'spread_count': spread_count,
            'total_count': total_count,
            'moneyline_count': moneyline_count
        }

        # Show missing games
        if games_with_odds < total_games:
            if 'game_id_nflverse' in week_odds.columns:
                scheduled_games = set(week_schedule['game_id'].unique())
                games_in_odds = set(week_odds['game_id_nflverse'].unique())
                missing_game_ids = scheduled_games - games_in_odds

                print(f"  Missing games:")
                for game_id in sorted(missing_game_ids):
                    game_row = week_schedule[week_schedule['game_id'] == game_id].iloc[0]
                    print(f"    {game_row['away_team']}@{game_row['home_team']}")
            else:
                scheduled_matchups = set(week_schedule.apply(lambda r: f"{r['away_team']}@{r['home_team']}", axis=1))
                odds_matchups = set(week_odds.apply(lambda r: f"{r['away_team']}@{r['home_team']}", axis=1).unique())
                missing_matchups = scheduled_matchups - odds_matchups

                if missing_matchups:
                    print(f"  Missing games:")
                    for matchup in sorted(missing_matchups):
                        print(f"    {matchup}")

    # Overall summary
    total_scheduled = len(schedule_df)
    total_with_odds = master_df[['season', 'week', 'home_team', 'away_team']].drop_duplicates().shape[0]
    overall_pct = (total_with_odds / total_scheduled * 100) if total_scheduled > 0 else 0

    print(f"\n{'='*70}")
    print(f"OVERALL: {total_with_odds}/{total_scheduled} games ({overall_pct:.1f}%)")
    print(f"Total odds records: {len(master_df)}")
    print(f"{'='*70}")

    return coverage


def build_master_game_lines(season: int = 2025):
    """Main function to build master game lines database."""

    print("\n" + "="*70)
    print(f"BUILDING MASTER GAME LINES DATABASE - {season} Season")
    print("="*70)

    # Load schedule as source of truth
    print(f"\n📅 Loading NFLverse schedule for {season}...")
    schedule_df = load_nflverse_schedule(season)
    print(f"  ✅ Loaded {len(schedule_df)} games")

    # Extract from all sources
    all_odds = []

    all_odds.extend(extract_from_draftkings_files())
    all_odds.extend(extract_from_comprehensive_files())
    all_odds.extend(extract_from_historical_game_lines())

    if not all_odds:
        print("\n❌ No odds data found in any source")
        return

    # Combine all extracted data
    print("\n" + "="*70)
    print("Combining all extracted data...")
    print("="*70)

    combined_df = pd.concat(all_odds, ignore_index=True)
    print(f"  Total raw records: {len(combined_df)}")
    print(f"  Unique games: {combined_df[['home_team', 'away_team']].drop_duplicates().shape[0]}")

    # Match to schedule (this assigns correct weeks)
    matched_df = match_to_schedule(combined_df, schedule_df)

    if len(matched_df) == 0:
        print("\n❌ No games matched to schedule")
        return

    # Deduplicate
    master_df = deduplicate_odds(matched_df)

    # Generate coverage report
    coverage = generate_coverage_report(master_df, schedule_df)

    # Save master file
    output_dir = PROJECT_ROOT / "data" / "historical"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"game_lines_master_{season}.csv"

    # Select and order columns
    final_columns = [
        'season', 'week', 'gameday',
        'game_id_nflverse', 'home_team', 'away_team',
        'market', 'side', 'point', 'price',
        'sportsbook', 'data_source', 'collected_at'
    ]

    # Only include columns that exist
    final_columns = [col for col in final_columns if col in master_df.columns]
    master_df = master_df[final_columns]

    # Sort for readability
    master_df = master_df.sort_values(['week', 'home_team', 'market', 'side'])

    master_df.to_csv(output_file, index=False)

    print(f"\n{'='*70}")
    print(f"✅ MASTER FILE CREATED: {output_file}")
    print(f"{'='*70}")
    print(f"Total records: {len(master_df)}")
    print(f"Unique games: {master_df[['season', 'week', 'home_team', 'away_team']].drop_duplicates().shape[0]}")
    print(f"Weeks covered: {sorted(master_df['week'].unique())}")
    print(f"Markets: {master_df['market'].value_counts().to_dict()}")

    # Save coverage report (convert int32 keys to int for JSON serialization)
    coverage_file = output_dir / f"coverage_report_{season}.json"
    import json
    coverage_json = {int(k): v for k, v in coverage.items()}
    with open(coverage_file, 'w') as f:
        json.dump(coverage_json, f, indent=2)

    print(f"\n📊 Coverage report saved to: {coverage_file}")

    return master_df


if __name__ == "__main__":
    build_master_game_lines(season=2025)
