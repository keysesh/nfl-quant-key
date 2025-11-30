#!/usr/bin/env python3
"""
Fetch Complete Historical DraftKings Props for Unbiased Backtesting
=====================================================================

This script fetches historical player prop odds from The Odds API for ALL weeks
of the 2025 season, creating a complete prop universe for unbiased backtesting.

Key Benefits:
1. Tests model on FULL prop universe (not just placed bets)
2. Eliminates selection bias from consolidated bet data
3. Uses actual DraftKings lines (not synthetic approximations)
4. Matches historical odds to NFLverse actual outcomes
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.fetch.fetch_historical_player_props import (
    fetch_events_for_date,
    collect_historical_player_props,
    load_api_key,
)

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

# NFL 2025 Regular Season Week Dates (approximate kickoff windows)
# Format: week_number -> ISO timestamp for snapshot
NFL_2025_WEEK_DATES = {
    1: "2025-09-05T12:00:00Z",    # Thu Sep 4 - Mon Sep 8
    2: "2025-09-12T12:00:00Z",    # Thu Sep 11 - Mon Sep 15
    3: "2025-09-19T12:00:00Z",    # Thu Sep 18 - Mon Sep 22
    4: "2025-09-26T12:00:00Z",    # Thu Sep 25 - Mon Sep 29
    5: "2025-10-03T12:00:00Z",    # Thu Oct 2 - Mon Oct 6
    6: "2025-10-10T12:00:00Z",    # Thu Oct 9 - Mon Oct 13
    7: "2025-10-17T12:00:00Z",    # Thu Oct 16 - Mon Oct 20
    8: "2025-10-24T12:00:00Z",    # Thu Oct 23 - Mon Oct 27
    9: "2025-10-31T12:00:00Z",    # Thu Oct 30 - Mon Nov 3
    10: "2025-11-07T12:00:00Z",   # Thu Nov 6 - Mon Nov 10
    11: "2025-11-14T12:00:00Z",   # Thu Nov 13 - Mon Nov 17
}

# Core prop markets for backtesting
BACKTEST_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
]


def fetch_week_historical_props(api_key: str, week: int, snapshot_date: str) -> pd.DataFrame:
    """Fetch all DraftKings props for a single week."""
    print(f"\n{'='*60}")
    print(f"WEEK {week} - Snapshot: {snapshot_date}")
    print(f"{'='*60}")

    # Discover events for this week
    events = fetch_events_for_date(
        api_key=api_key,
        target_date=snapshot_date,
        regions="us",
        bookmaker_filter="draftkings"
    )

    if not events:
        print(f"  No events found for week {week}")
        return pd.DataFrame()

    print(f"  Found {len(events)} games")

    # Fetch player props for all events
    df = collect_historical_player_props(
        api_key=api_key,
        events=events,
        markets=BACKTEST_MARKETS,
        bookmakers="draftkings",
        regions="us",
        odds_format="american",
        snapshot=snapshot_date,
        snapshot_offsets=[1, 3, 6, 12, 24]
    )

    if not df.empty:
        df['week'] = week
        print(f"  Total props: {len(df)}")
        print(f"  Unique players: {df['player'].nunique()}")

    return df


def consolidate_props_for_backtest(all_props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate props into over/under format for backtesting.

    Each row becomes: player, week, market, line, over_odds, under_odds
    """
    if all_props_df.empty:
        return pd.DataFrame()

    # Group by event, market, player, line
    grouped = all_props_df.groupby([
        'week', 'event_id', 'home_team', 'away_team',
        'market', 'player', 'line'
    ])

    consolidated = []
    for (week, event_id, home, away, market, player, line), group in grouped:
        over_row = group[group['prop_type'] == 'over'].iloc[0] if len(group[group['prop_type'] == 'over']) > 0 else None
        under_row = group[group['prop_type'] == 'under'].iloc[0] if len(group[group['prop_type'] == 'under']) > 0 else None

        # Get American odds
        over_odds = over_row['american_price'] if over_row is not None else over_row['price'] if over_row is not None else None
        under_odds = under_row['american_price'] if under_row is not None else under_row['price'] if under_row is not None else None

        if over_odds is None or under_odds is None:
            continue

        consolidated.append({
            'week': week,
            'game_id': f"{away}@{home}",
            'home_team': home,
            'away_team': away,
            'player': player,
            'market': market,
            'line': line,
            'over_odds': over_odds,
            'under_odds': under_odds,
            'event_id': event_id,
        })

    return pd.DataFrame(consolidated)


def load_nflverse_actuals():
    """Load actual player stats from NFLverse."""
    stats_file = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    if not stats_file.exists():
        print(f"Warning: NFLverse stats not found at {stats_file}")
        return pd.DataFrame()

    df = pd.read_parquet(stats_file)
    df = df[df['season'] == 2025].copy()

    # Map market names to stat columns
    stat_cols = {
        'player_pass_yds': 'passing_yards',
        'player_rush_yds': 'rushing_yards',
        'player_reception_yds': 'receiving_yards',
        'player_receptions': 'receptions',
    }

    return df, stat_cols


def match_props_to_actuals(props_df: pd.DataFrame, stats_df: pd.DataFrame, stat_cols: dict) -> pd.DataFrame:
    """Match historical props to actual NFLverse stats."""
    if props_df.empty or stats_df.empty:
        return props_df

    print("\nMatching props to actual outcomes...")

    matched_props = []

    for _, prop in props_df.iterrows():
        week = prop['week']
        player = prop['player']
        market = prop['market']

        # Find player in NFLverse stats
        player_stats = stats_df[
            (stats_df['week'] == week) &
            (stats_df['player_display_name'] == player)
        ]

        if player_stats.empty:
            # Try partial match
            player_stats = stats_df[
                (stats_df['week'] == week) &
                (stats_df['player_display_name'].str.contains(player.split()[0], na=False))
            ]

        if player_stats.empty:
            continue

        # Get actual value
        stat_col = stat_cols.get(market)
        if not stat_col or stat_col not in player_stats.columns:
            continue

        actual_value = player_stats.iloc[0][stat_col]
        if pd.isna(actual_value):
            continue

        # Calculate went_over
        went_over = 1 if actual_value > prop['line'] else 0

        prop_row = prop.to_dict()
        prop_row['actual_value'] = actual_value
        prop_row['went_over'] = went_over
        prop_row['position'] = player_stats.iloc[0].get('position', 'UNK')
        prop_row['team'] = player_stats.iloc[0].get('recent_team', 'UNK')

        matched_props.append(prop_row)

    matched_df = pd.DataFrame(matched_props)
    print(f"  Matched {len(matched_df)} props to actual outcomes")
    print(f"  Match rate: {len(matched_df)/len(props_df)*100:.1f}%")

    return matched_df


def add_market_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate implied market probabilities from American odds."""
    if df.empty:
        return df

    def american_to_prob(odds):
        """Convert American odds to implied probability."""
        if pd.isna(odds):
            return 0.5
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    df['market_prob_over'] = df['over_odds'].apply(american_to_prob)
    df['market_prob_under'] = df['under_odds'].apply(american_to_prob)

    # Remove vig (normalize)
    total_prob = df['market_prob_over'] + df['market_prob_under']
    df['market_prob_over'] = df['market_prob_over'] / total_prob
    df['market_prob_under'] = df['market_prob_under'] / total_prob

    return df


def main():
    print("="*70)
    print("FETCHING COMPLETE HISTORICAL DRAFTKINGS PROPS")
    print("For Unbiased Walk-Forward Backtesting")
    print("="*70)

    # Use the new API key
    NEW_API_KEY = "73ec9367021badb173a0b68c35af818f"

    print(f"API Key: {NEW_API_KEY[:8]}...{NEW_API_KEY[-4:]}")

    # Fetch props for each week
    all_weeks_props = []

    for week, snapshot_date in NFL_2025_WEEK_DATES.items():
        try:
            week_df = fetch_week_historical_props(NEW_API_KEY, week, snapshot_date)
            if not week_df.empty:
                all_weeks_props.append(week_df)

            # Rate limit - be nice to the API
            time.sleep(2)

        except Exception as e:
            print(f"  Error fetching week {week}: {e}")
            continue

    if not all_weeks_props:
        print("\nNo historical props fetched. Check API key and quota.")
        return

    # Combine all weeks
    raw_props = pd.concat(all_weeks_props, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"TOTAL RAW PROPS: {len(raw_props)}")
    print(f"{'='*70}")

    # Save raw data
    raw_output = DATA_DIR / 'backtest' / 'historical_odds_raw.csv'
    DATA_DIR.joinpath('backtest').mkdir(exist_ok=True)
    raw_props.to_csv(raw_output, index=False)
    print(f"Saved raw props to: {raw_output}")

    # Consolidate into over/under format
    props_df = consolidate_props_for_backtest(raw_props)
    print(f"Consolidated into {len(props_df)} unique over/under lines")

    # Load NFLverse actuals
    stats_df, stat_cols = load_nflverse_actuals()

    if not stats_df.empty:
        # Match to actual outcomes
        matched_df = match_props_to_actuals(props_df, stats_df, stat_cols)

        # Add market probabilities
        matched_df = add_market_probabilities(matched_df)

        # Save final dataset
        output_file = DATA_DIR / 'backtest' / 'unbiased_historical_props.csv'
        matched_df.to_csv(output_file, index=False)

        print(f"\n{'='*70}")
        print(f"UNBIASED BACKTEST DATASET CREATED")
        print(f"{'='*70}")
        print(f"Total props with outcomes: {len(matched_df)}")
        print(f"Weeks covered: {sorted(matched_df['week'].unique())}")
        print(f"Props per week:")
        for week in sorted(matched_df['week'].unique()):
            count = len(matched_df[matched_df['week'] == week])
            over_rate = matched_df[matched_df['week'] == week]['went_over'].mean()
            print(f"  Week {week}: {count} props, {over_rate:.1%} OVER rate")

        print(f"\nOverall OVER rate: {matched_df['went_over'].mean():.1%}")
        print(f"Saved to: {output_file}")
    else:
        print("Could not match to actuals - NFLverse data not available")

        # Save without actuals
        output_file = DATA_DIR / 'backtest' / 'historical_props_no_actuals.csv'
        props_df.to_csv(output_file, index=False)
        print(f"Saved props (without outcomes) to: {output_file}")


if __name__ == "__main__":
    main()
