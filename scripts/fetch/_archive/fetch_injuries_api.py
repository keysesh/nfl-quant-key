#!/usr/bin/env python3
"""
Fetch NFL injury reports using nflverse API

Sources:
- nflverse: Weekly rosters with injury designations (FREE)
"""

import requests
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.season_utils import get_current_season

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_nflverse_injuries(season=None, week=None):
    """
    Fetch injury data from nflverse weekly rosters.

    Args:
        season: NFL season year (None for auto-detect current season)
        week: Specific week number (None for latest)

    Returns:
        DataFrame with injury data
    """
    if season is None:
        season = get_current_season()

    logger.info(f"Fetching injury data from nflverse (season {season})...")

    try:
        # Fetch weekly rosters
        url = f'https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_weekly_{season}.parquet'
        rosters = pd.read_parquet(url)

        # Filter to specific week if provided
        if week:
            rosters = rosters[rosters['week'] == week]
        else:
            # Get most recent week
            latest_week = rosters['week'].max()
            rosters = rosters[rosters['week'] == latest_week]
            logger.info(f"   Using Week {latest_week} data")

        # Filter to injured players
        injured = rosters[rosters['injury_status'].notna()].copy()

        # Rename columns for consistency
        injured = injured.rename(columns={
            'player_name': 'name',
            'gsis_id': 'player_id'
        })

        # Add source
        injured['source'] = 'nflverse'

        logger.info(f"✅ Found {len(injured)} injured players from nflverse")

        return injured[['player_id', 'name', 'team', 'position', 'injury_status', 'status', 'source']]

    except Exception as e:
        logger.error(f"❌ nflverse fetch failed: {e}")
        return pd.DataFrame()




def parse_injury_severity(status):
    """
    Convert injury status to numeric severity.

    Args:
        status: Injury designation (Out, Doubtful, Questionable, etc.)

    Returns:
        float: 0.0 (healthy) to 1.0 (out)
    """
    if pd.isna(status):
        return 0.0

    status_lower = str(status).lower()

    if 'out' in status_lower or 'ir' in status_lower:
        return 1.0
    elif 'doubtful' in status_lower:
        return 0.85
    elif 'questionable' in status_lower or 'q' == status_lower:
        return 0.40
    elif 'probable' in status_lower or 'gtd' in status_lower:
        return 0.15
    else:
        return 0.0


def filter_key_positions(injuries_df):
    """
    Filter to positions relevant for prop betting (QB, RB, WR, TE).

    Args:
        injuries_df: Full injury DataFrame

    Returns:
        Filtered DataFrame
    """
    key_positions = ['QB', 'RB', 'WR', 'TE']
    return injuries_df[injuries_df['position'].isin(key_positions)].copy()


def create_injury_report(week=None, save=True, season=None):
    """
    Generate comprehensive injury report.

    Args:
        week: Week number (None for current week)
        save: Whether to save to file
        season: NFL season year (None for auto-detect)

    Returns:
        DataFrame with injury data
    """
    if season is None:
        season = get_current_season()

    print("="*80)
    print("NFL INJURY REPORT (API-BASED)")
    print("="*80)
    print(f"Timestamp: {datetime.now()}")
    print(f"Season: {season}")
    print()

    # Fetch from nflverse
    injuries = fetch_nflverse_injuries(season=season, week=week)

    if len(injuries) == 0:
        print("❌ No injury data retrieved")
        return None

    # Add severity score
    injuries['severity'] = injuries['injury_status'].apply(parse_injury_severity)

    # Sort by severity (most severe first)
    injuries = injuries.sort_values('severity', ascending=False)

    # Save to file
    if save:
        output_dir = Path('data/injuries')
        output_dir.mkdir(exist_ok=True, parents=True)

        week_str = f"_week{week}" if week else ""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = output_dir / f'injuries{week_str}_{timestamp}.csv'

        injuries.to_csv(output_file, index=False)
        print(f"💾 Saved to: {output_file}")
        print()

    # Print summary
    print(f"✅ Total injuries: {len(injuries)}")
    print(f"   nflverse: {len(injuries)}")
    print()

    # Key positions only (QB/RB/WR/TE)
    key_positions = filter_key_positions(injuries)

    print("="*80)
    print(f"KEY POSITION INJURIES (QB/RB/WR/TE): {len(key_positions)}")
    print("="*80)
    print()

    # Group by status
    status_counts = key_positions.groupby('injury_status').size().sort_values(ascending=False)
    print("By Status:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print()

    # Show OUT players (most critical)
    out_players = key_positions[key_positions['severity'] >= 1.0]

    if len(out_players) > 0:
        print("🚨 PLAYERS OUT:")
        print("-"*80)
        for _, player in out_players.iterrows():
            team = player['team'] if pd.notna(player['team']) else 'FA'
            print(f"  {player['name']:25s} {player['position']:3s} {team:3s} - {player['injury_status']}")
        print()

    # Show DOUBTFUL players
    doubtful_players = key_positions[
        (key_positions['severity'] >= 0.75) &
        (key_positions['severity'] < 1.0)
    ]

    if len(doubtful_players) > 0:
        print("⚠️  DOUBTFUL:")
        print("-"*80)
        for _, player in doubtful_players.iterrows():
            team = player['team'] if pd.notna(player['team']) else 'FA'
            print(f"  {player['name']:25s} {player['position']:3s} {team:3s} - {player['injury_status']}")
        print()

    # Show QUESTIONABLE players
    questionable_players = key_positions[
        (key_positions['severity'] >= 0.3) &
        (key_positions['severity'] < 0.75)
    ]

    if len(questionable_players) > 0:
        print("❓ QUESTIONABLE:")
        print("-"*80)
        for _, player in questionable_players.head(15).iterrows():
            team = player['team'] if pd.notna(player['team']) else 'FA'
            print(f"  {player['name']:25s} {player['position']:3s} {team:3s} - {player['injury_status']}")
        if len(questionable_players) > 15:
            print(f"  ... and {len(questionable_players) - 15} more")
        print()

    print("="*80)

    return injuries


def get_player_injury_status(player_name, team=None, season=None, week=None):
    """
    Check if a specific player is injured.

    Args:
        player_name: Player full name
        team: Team abbreviation (optional, for disambiguation)
        season: NFL season year (None for auto-detect)
        week: Specific week number

    Returns:
        dict: {'injured': bool, 'status': str, 'severity': float}
    """
    if season is None:
        season = get_current_season()

    # Fetch current injuries
    injuries = fetch_nflverse_injuries(season=season, week=week)

    # Look for player
    matches = injuries[injuries['name'].str.contains(player_name, case=False, na=False)]

    if team:
        matches = matches[matches['team'] == team]

    if len(matches) == 0:
        return {'injured': False, 'status': 'Healthy', 'severity': 0.0}

    player = matches.iloc[0]

    return {
        'injured': True,
        'status': player['injury_status'],
        'severity': parse_injury_severity(player['injury_status']),
        'team': player['team'],
        'position': player['position']
    }


def main():
    import sys

    week = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # Generate report
    injuries = create_injury_report(week=week, save=True)

    if injuries is not None:
        print("\n💡 TIP: Use this data to filter out injured players from predictions")
        print("   Example: injuries[injuries['severity'] >= 1.0] for OUT players")

    print("\n" + "="*80)
    print("⚠️  ALWAYS verify injury status before betting!")
    print("   Updates: nflverse weekly rosters")
    print("="*80)


if __name__ == "__main__":
    main()
