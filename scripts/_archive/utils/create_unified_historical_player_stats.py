#!/usr/bin/env python3
"""
Unified Trailing Stats Calculator

Combines 2025 Sleeper stats (current season, continuously updated) with
historical NFLverse/Sleeper data (2024 and earlier) to provide comprehensive
trailing statistics for all players.

Strategy:
- Primary: 2025 Sleeper stats (most recent, prop-friendly format)
- Fallback: Historical NFLverse/Sleeper stats (2024, 2023, etc.)
- Weighting: Recent data weighted more heavily than older data
- Team totals: Calculated from actual team statistics in both sources
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.stats_loader import load_weekly_stats, load_season_stats, is_data_available


def load_2025_sleeper_stats() -> pd.DataFrame:
    """Load all available 2025 stats using unified interface (prioritizes Sleeper)."""
    all_stats = []

    # Check available weeks for 2025
    for week in range(1, 19):
        if is_data_available(week, 2025, source='auto'):
            try:
                df = load_weekly_stats(week, 2025, source='auto')
                df["week"] = week
                df["season"] = 2025
                # Source column already added by load_weekly_stats
                all_stats.append(df)
            except Exception as e:
                print(f"  ⚠️  Failed to load week {week}, season 2025: {e}")
                continue

    if not all_stats:
        return pd.DataFrame()

    combined = pd.concat(all_stats, ignore_index=True)

    # Map canonical columns to internal format
    column_mapping = {
        'receiving_yards': 'rec_yd',
        'rushing_yards': 'rush_yd',
        'passing_yards': 'pass_yd',
        'receptions': 'rec',
        'targets': 'rec_tgt',
        'rushing_attempts': 'rush_att',
        'passing_attempts': 'pass_att',
        'passing_completions': 'pass_cmp',
        'passing_tds': 'pass_td',
        'rushing_tds': 'rush_td',
        'receiving_tds': 'rec_td',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in combined.columns}
    combined = combined.rename(columns=rename_dict)

    return combined


def load_historical_nflverse_stats() -> pd.DataFrame:
    """Load historical NFLverse weekly stats (2024, 2023, etc.) using unified interface."""
    all_stats = []

    # Load historical seasons (2024 and earlier)
    for season in range(2020, 2025):  # 2020-2024
        try:
            # Use unified interface - it will prioritize NFLverse for historical seasons
            season_df = load_season_stats(season, source='auto')
            if len(season_df) > 0:
                season_df['source'] = season_df.get('source', 'nflverse')
                all_stats.append(season_df)
        except Exception as e:
            print(f"  ⚠️  Failed to load season {season}: {e}")
            continue

    if not all_stats:
        return pd.DataFrame()

    combined = pd.concat(all_stats, ignore_index=True)

    # Map canonical columns to internal format
    column_mapping = {
        'receiving_yards': 'rec_yd',
        'rushing_yards': 'rush_yd',
        'passing_yards': 'pass_yd',
        'receptions': 'rec',
        'targets': 'rec_tgt',
        'rushing_attempts': 'rush_att',
        'passing_attempts': 'pass_att',
        'passing_completions': 'pass_cmp',
        'passing_tds': 'pass_td',
        'rushing_tds': 'rush_td',
        'receiving_tds': 'rec_td',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in combined.columns}
    combined = combined.rename(columns=rename_dict)

    return combined


def load_historical_sleeper_stats() -> pd.DataFrame:
    """Load historical Sleeper stats (2024 and earlier) using unified interface."""
    all_stats = []

    # Load historical seasons (2024 and earlier) from Sleeper if available
    for season in range(2020, 2025):  # 2020-2024
        for week in range(1, 19):
            if is_data_available(week, season, source='sleeper'):
                try:
                    df = load_weekly_stats(week, season, source='sleeper')
                    df["week"] = week
                    df["season"] = season
                    df["source"] = "sleeper"
                    all_stats.append(df)
                except Exception as e:
                    continue

    if not all_stats:
        return pd.DataFrame()

    combined = pd.concat(all_stats, ignore_index=True)

    # Map canonical columns to internal format
    column_mapping = {
        'receiving_yards': 'rec_yd',
        'rushing_yards': 'rush_yd',
        'passing_yards': 'pass_yd',
        'receptions': 'rec',
        'targets': 'rec_tgt',
        'rushing_attempts': 'rush_att',
        'passing_attempts': 'pass_att',
        'passing_completions': 'pass_cmp',
        'passing_tds': 'pass_td',
        'rushing_tds': 'rush_td',
        'receiving_tds': 'rec_td',
    }

    rename_dict = {k: v for k, v in column_mapping.items() if k in combined.columns}
    combined = combined.rename(columns=rename_dict)

    return combined


def calculate_team_totals_per_week(df: pd.DataFrame) -> Dict[Tuple[str, int, int], Dict]:
    """
    Calculate team totals (targets, carries) per week for accurate share calculations.

    Returns:
        Dict keyed by (team, week, season) -> {'total_targets': int, 'total_carries': int}
    """
    team_week_totals = {}

    for (team, week, season), week_data in df.groupby(['team', 'week', 'season']):
        if pd.isna(team):
            continue

        # Calculate team total targets (sum of all WR/TE/RB targets)
        team_total_targets = week_data[
            week_data['position'].isin(['WR', 'TE', 'RB'])
        ]['rec_tgt'].sum()

        # Calculate team total carries (sum of all RB/QB rush attempts)
        team_total_carries = week_data[
            week_data['position'].isin(['RB', 'QB'])
        ]['rush_att'].sum()

        key = (team, week, season)
        team_week_totals[key] = {
            'total_targets': team_total_targets,
            'total_carries': team_total_carries
        }

    return team_week_totals


def calculate_weighted_trailing_stats(
    player_weeks: pd.DataFrame,
    team_week_totals: Dict[Tuple[str, int, int], Dict],
    lookback_weeks: int = 4,
    current_week: int = None,
    current_season: int = None
) -> Dict:
    """
    Calculate trailing stats with weighting: recent data (2025) weighted more heavily.

    Strategy:
    - Filter to weeks BEFORE current week (if current_week/season provided)
    - Use most recent lookback_weeks of eligible data
    - Weight 2025 data at 1.0, 2024 at 0.7, 2023 at 0.5, etc.
    - Calculate team totals from actual team statistics
    - Track metadata: which weeks used, team/position consistency, games played

    Args:
        player_weeks: DataFrame with player's historical weeks
        team_week_totals: Dict mapping (team, week, season) -> team totals
        lookback_weeks: Number of weeks to include in lookback window (default 4)
        current_week: Current week number (for filtering, None = week-agnostic)
        current_season: Current season year (for filtering, None = week-agnostic)

    Returns:
        Dict with trailing stats, weighted averages, simple averages, and metadata
    """
    if len(player_weeks) == 0:
        return {}

    # Phase 1.2: Filter to weeks BEFORE current week (if week context provided)
    if current_week is not None and current_season is not None:
        # Filter to weeks before current week
        eligible_weeks = player_weeks[
            (player_weeks['season'] < current_season) |
            ((player_weeks['season'] == current_season) & (player_weeks['week'] < current_week))
        ].copy()
    else:
        # Week-agnostic: use all weeks (for unified stats generation)
        eligible_weeks = player_weeks.copy()

    if len(eligible_weeks) == 0:
        return {}

    # Sort by season, then week (most recent first)
    eligible_weeks = eligible_weeks.sort_values(['season', 'week'], ascending=[False, False])

    # Get most recent lookback_weeks from eligible weeks
    recent_weeks = eligible_weeks.head(lookback_weeks).copy()

    if len(recent_weeks) == 0:
        return {}

    # Phase 4: Filter out zero-activity weeks (BYE weeks, injuries)
    # Only include weeks where player actually played (had some activity)
    active_weeks = recent_weeks[
        (recent_weeks['rec_tgt'] > 0) |
        (recent_weeks['rush_att'] > 0) |
        (recent_weeks['pass_att'] > 0)
    ].copy()

    # Use active_weeks for calculations, but track both recent_weeks and active_weeks
    weeks_for_calc = active_weeks if len(active_weeks) > 0 else recent_weeks

    if len(weeks_for_calc) == 0:
        return {}

    position = weeks_for_calc.iloc[0]['position']
    team = weeks_for_calc.iloc[0]['team']

    # Phase 3: Track team/position consistency
    teams_in_lookback = sorted(weeks_for_calc['team'].unique().tolist())
    positions_in_lookback = sorted(weeks_for_calc['position'].unique().tolist())
    team_changed_in_lookback = len(teams_in_lookback) > 1
    position_changed_in_lookback = len(positions_in_lookback) > 1

    # Calculate weighted totals (using weeks_for_calc which excludes zero-activity weeks)
    total_rec_tgt = 0
    total_rush_att = 0
    total_rec_yd = 0
    total_rush_yd = 0
    total_pass_yd = 0
    total_attempts = 0
    total_tds = 0
    total_team_targets = 0
    total_team_carries = 0
    total_weight = 0

    # Phase 3: Use team-specific totals for each week (handles team changes)
    for idx, row in weeks_for_calc.iterrows():
        season = row['season']
        week = row['week']
        row_team = row['team']  # Use this week's team (handles trades)

        # Weight: 2025 = 1.0, 2024 = 0.7, 2023 = 0.5, older = 0.3
        if season == 2025:
            weight = 1.0
        elif season == 2024:
            weight = 0.7
        elif season == 2023:
            weight = 0.5
        else:
            weight = 0.3

        # Accumulate weighted player stats
        total_rec_tgt += row.get('rec_tgt', 0) * weight
        total_rush_att += row.get('rush_att', 0) * weight
        total_rec_yd += row.get('rec_yd', 0) * weight
        total_rush_yd += row.get('rush_yd', 0) * weight
        total_pass_yd += row.get('pass_yd', 0) * weight
        total_attempts += row.get('pass_att', 0) * weight

        # TDs
        total_tds += (
            row.get('pass_td', 0) +
            row.get('rush_td', 0) +
            row.get('rec_td', 0)
        ) * weight

        # Phase 3: Accumulate weighted team totals using THIS week's team (handles trades)
        team_key = (row_team, week, season)
        if team_key in team_week_totals:
            total_team_targets += team_week_totals[team_key]['total_targets'] * weight
            total_team_carries += team_week_totals[team_key]['total_carries'] * weight

        total_weight += weight

    # Phase 2: Calculate simple totals for per-game averages
    simple_total_rec_yd = weeks_for_calc['rec_yd'].sum()
    simple_total_rush_yd = weeks_for_calc['rush_yd'].sum()
    simple_total_pass_yd = weeks_for_calc['pass_yd'].sum()
    simple_total_rec_tgt = weeks_for_calc['rec_tgt'].sum()
    simple_total_rush_att = weeks_for_calc['rush_att'].sum()
    simple_total_pass_att = weeks_for_calc['pass_att'].sum()
    num_games_played = len(weeks_for_calc)

    if total_weight == 0:
        return {}

    # Phase 2: Normalize by total weight (weighted averages for trailing stats)
    avg_rec_tgt_weighted = total_rec_tgt / total_weight
    avg_rush_att_weighted = total_rush_att / total_weight
    avg_rec_yd_weighted = total_rec_yd / total_weight
    avg_rush_yd_weighted = total_rush_yd / total_weight
    avg_pass_yd_weighted = total_pass_yd / total_weight
    avg_attempts_weighted = total_attempts / total_weight
    avg_team_targets = total_team_targets / total_weight
    avg_team_carries = total_team_carries / total_weight

    # Phase 2: Calculate simple per-game averages (for filtering/display)
    avg_rec_yd_per_game = simple_total_rec_yd / num_games_played if num_games_played > 0 else 0
    avg_rush_yd_per_game = simple_total_rush_yd / num_games_played if num_games_played > 0 else 0
    avg_pass_yd_per_game = simple_total_pass_yd / num_games_played if num_games_played > 0 else 0
    avg_rec_tgt_per_game = simple_total_rec_tgt / num_games_played if num_games_played > 0 else 0
    avg_rush_att_per_game = simple_total_rush_att / num_games_played if num_games_played > 0 else 0
    avg_pass_att_per_game = simple_total_pass_att / num_games_played if num_games_played > 0 else 0

    # Calculate shares from actual team totals (using weighted averages)
    trailing_target_share = None
    if avg_rec_tgt_weighted > 0 and avg_team_targets > 0:
        trailing_target_share = avg_rec_tgt_weighted / avg_team_targets
    elif position in ['RB', 'WR', 'TE']:
        trailing_target_share = 0.0  # Player exists but has 0 targets

    trailing_carry_share = None
    if avg_rush_att_weighted > 0 and avg_team_carries > 0:
        trailing_carry_share = avg_rush_att_weighted / avg_team_carries
    elif position == 'RB':
        trailing_carry_share = 0.0  # RB exists but has 0 carries

    # Calculate yards per opportunity (using weighted averages)
    total_opportunities = avg_attempts_weighted + avg_rush_att_weighted + avg_rec_tgt_weighted
    if total_opportunities > 0:
        trailing_yards_per_opportunity = (avg_pass_yd_weighted + avg_rush_yd_weighted + avg_rec_yd_weighted) / total_opportunities
    else:
        trailing_yards_per_opportunity = 0.0

    # Calculate TD rate
    trailing_td_rate = total_tds / total_weight / max(total_opportunities, 1)

    # Phase 6.2: Validation checks - verify no future weeks, verify window completeness
    if current_week is not None and current_season is not None:
        # Verify no future weeks in lookback window
        for _, row in recent_weeks.iterrows():
            if row['season'] > current_season or (row['season'] == current_season and row['week'] >= current_week):
                import warnings
                # Get player name from first row if available
                player_name = player_weeks.iloc[0].get('player_name', 'Unknown') if len(player_weeks) > 0 else 'Unknown'
                warnings.warn(
                    f"Data leakage detected for {player_name}: "
                    f"Found week {row['season']}-{row['week']} in lookback window for current week {current_season}-{current_week}"
                )

    # Phase 1.3: Build lookback weeks metadata
    lookback_weeks_used = sorted([
        (int(row['season']), int(row['week']))
        for _, row in recent_weeks.iterrows()
    ])

    return {
        'position': position,
        'team': team,  # Most recent team in lookback window

        # Phase 1.3: Lookback window metadata
        'weeks_played': len(player_weeks),  # Total career weeks across all seasons
        'lookback_weeks_played': len(recent_weeks),  # Weeks in lookback window (includes BYE/injury)
        'games_played_in_lookback': len(weeks_for_calc),  # Actual games played (excludes BYE/injury)
        'lookback_weeks_used': lookback_weeks_used,  # List of (season, week) tuples
        'lookback_window_complete': len(recent_weeks) >= lookback_weeks,  # True if full window
        'lookback_window_complete_games': len(weeks_for_calc) >= lookback_weeks,  # True if full window of games

        # Phase 3: Team/position consistency metadata
        'teams_in_lookback': teams_in_lookback,
        'positions_in_lookback': positions_in_lookback,
        'team_changed_in_lookback': team_changed_in_lookback,
        'position_changed_in_lookback': position_changed_in_lookback,

        # Trailing stats (using weighted averages)
        'trailing_snap_share': 0.6,  # Would need snap count data to calculate properly
        'trailing_target_share': trailing_target_share,
        'trailing_carry_share': trailing_carry_share,
        'trailing_yards_per_opportunity': trailing_yards_per_opportunity,
        'trailing_td_rate': min(trailing_td_rate, 0.15),  # Cap at 15%

        # Phase 2: Weighted averages (for trailing stats calculations)
        'avg_pass_yd_weighted': avg_pass_yd_weighted,
        'avg_rush_yd_weighted': avg_rush_yd_weighted,
        'avg_rec_yd_weighted': avg_rec_yd_weighted,
        'avg_rec_tgt_weighted': avg_rec_tgt_weighted,
        'avg_rush_att_weighted': avg_rush_att_weighted,
        'avg_pass_att_weighted': avg_attempts_weighted,

        # Phase 2: Simple per-game averages (for filtering/display) - BACKWARD COMPATIBILITY
        'avg_pass_yd': avg_pass_yd_per_game,  # Keep for backward compatibility
        'avg_rush_yd': avg_rush_yd_per_game,  # Keep for backward compatibility
        'avg_rec_yd': avg_rec_yd_per_game,  # Keep for backward compatibility
        'avg_rec_yd_per_game': avg_rec_yd_per_game,
        'avg_rush_yd_per_game': avg_rush_yd_per_game,
        'avg_pass_yd_per_game': avg_pass_yd_per_game,
        'avg_rec_tgt_per_game': avg_rec_tgt_per_game,
        'avg_rush_att_per_game': avg_rush_att_per_game,
        'avg_pass_att_per_game': avg_pass_att_per_game,

        'data_sources': list(recent_weeks['source'].unique()),
        'seasons': sorted(recent_weeks['season'].unique().tolist()),
    }


def build_unified_player_stats_history():
    """
    Build comprehensive player stats history combining:
    - 2025 Sleeper stats (current season, continuously updated)
    - Historical NFLverse stats (2024, 2023, etc.)
    - Historical Sleeper stats (2024, etc.)
    """
    print("="*80)
    print("BUILDING UNIFIED HISTORICAL PLAYER STATS")
    print("(Combining 2025 Sleeper + Historical NFLverse/Sleeper)")
    print("="*80)

    # Load all data sources
    print("\n1. Loading data sources...")

    print("   Loading 2025 Sleeper stats...")
    df_2025 = load_2025_sleeper_stats()
    print(f"      ✅ Loaded {len(df_2025)} player-week records from 2025")

    print("   Loading historical NFLverse stats...")
    df_nflverse = load_historical_nflverse_stats()
    print(f"      ✅ Loaded {len(df_nflverse)} player-week records from NFLverse")

    print("   Loading historical Sleeper stats...")
    df_sleeper_hist = load_historical_sleeper_stats()
    print(f"      ✅ Loaded {len(df_sleeper_hist)} player-week records from historical Sleeper")

    # Ensure all DataFrames have the same columns before combining
    required_cols = ['player_name', 'position', 'team', 'week', 'season', 'source',
                     'rec_tgt', 'rush_att', 'rec_yd', 'rush_yd', 'pass_yd', 'pass_att',
                     'pass_cmp', 'pass_td', 'rush_td', 'rec_td', 'rec']

    # Add missing columns to each DataFrame and ensure clean index
    all_data = []
    if len(df_2025) > 0:
        df_2025_clean = df_2025.copy()
        for col in required_cols:
            if col not in df_2025_clean.columns:
                df_2025_clean[col] = 0
        df_2025_clean = df_2025_clean[required_cols].copy()
        df_2025_clean.reset_index(drop=True, inplace=True)
        all_data.append(df_2025_clean)

    if len(df_nflverse) > 0:
        df_nflverse_clean = df_nflverse.copy()
        for col in required_cols:
            if col not in df_nflverse_clean.columns:
                df_nflverse_clean[col] = 0
        df_nflverse_clean = df_nflverse_clean[required_cols].copy()
        df_nflverse_clean.reset_index(drop=True, inplace=True)
        all_data.append(df_nflverse_clean)

    if len(df_sleeper_hist) > 0:
        df_sleeper_clean = df_sleeper_hist.copy()
        for col in required_cols:
            if col not in df_sleeper_clean.columns:
                df_sleeper_clean[col] = 0
        df_sleeper_clean = df_sleeper_clean[required_cols].copy()
        df_sleeper_clean.reset_index(drop=True, inplace=True)
        all_data.append(df_sleeper_clean)

    if not all_data:
        raise ValueError("No data sources found!")

    # Build combined DataFrame using simple list append (avoids pandas concat issues)
    if not all_data:
        raise ValueError("No data sources found!")

    # Convert each DataFrame to list of dicts, then create new DataFrame
    all_rows = []
    for df in all_data:
        for _, row in df.iterrows():
            all_rows.append(row.to_dict())

    # Create combined DataFrame from list of dicts
    combined = pd.DataFrame(all_rows)
    print(f"\n   ✅ Combined total: {len(combined)} player-week records")
    print(f"      Seasons: {sorted(combined['season'].unique())}")

    # Calculate team totals per week
    print("\n2. Calculating team totals per week...")
    team_week_totals = calculate_team_totals_per_week(combined)
    print(f"   ✅ Calculated totals for {len(team_week_totals)} team-week combinations")

    # Build player database
    print("\n3. Calculating trailing stats for each player...")
    player_db = {}

    for player_name in combined['player_name'].unique():
        player_weeks = combined[combined['player_name'] == player_name].sort_values(['season', 'week'])

        if len(player_weeks) == 0:
            continue

        # Calculate weighted trailing stats
        trailing_stats = calculate_weighted_trailing_stats(player_weeks, team_week_totals)

        if trailing_stats:
            player_db[player_name] = trailing_stats

    # Save to file
    output_file = Path('data/historical_player_stats.json')
    with open(output_file, 'w') as f:
        json.dump(player_db, f, indent=2)

    print(f"\n✅ Created unified historical player database: {len(player_db)} players")
    print(f"   Saved to {output_file}")

    # Show sample
    print("\n📊 Sample players:")
    for i, (name, stats) in enumerate(list(player_db.items())[:5]):
        print(f"  {name} ({stats['position']}):")
        print(f"    Seasons: {stats.get('seasons', [])}")
        print(f"    Sources: {stats.get('data_sources', [])}")
        if stats.get('trailing_target_share') is not None:
            print(f"    Target share: {stats['trailing_target_share']:.1%}")
        if stats.get('trailing_carry_share') is not None:
            print(f"    Carry share: {stats['trailing_carry_share']:.1%}")
        print(f"    Yards/opp: {stats['trailing_yards_per_opportunity']:.2f}")

    print("\n" + "="*80)
    print("✅ Unified historical stats complete!")
    print("="*80)

    return player_db


if __name__ == '__main__':
    build_unified_player_stats_history()
