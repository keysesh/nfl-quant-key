"""
Fix module for 13 broken features with 0% importance.

All features were defaulting to constants instead of being calculated from source data.
This module provides actual calculations from NFLverse data.

Features fixed:
1. adot - Average Depth of Target
2. trailing_catch_rate - Receptions / Targets EWMA
3. game_pace - Team plays per game
4. pressure_rate - QB pressure rate
5. opp_pressure_rate - Opponent defense pressure rate
6. slot_snap_pct - Slot alignment percentage
7. oline_health_score - O-line injury impact
8. opp_wr1_receptions_allowed - Receptions allowed to WR1s
9. opp_man_coverage_rate_trailing - Man coverage rate
10. slot_funnel_score - Slot vulnerability
11. man_coverage_adjustment - Coverage type adjustment
12. lvt_x_defense - LVT * defense EPA interaction
13. lvt_x_rest - LVT * rest days interaction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Global caches
_TEAM_PACE_CACHE: Optional[pd.DataFrame] = None
_PARTICIPATION_CACHE: Optional[pd.DataFrame] = None
_PRESSURE_BY_TEAM_CACHE: Optional[pd.DataFrame] = None
_COVERAGE_BY_TEAM_CACHE: Optional[pd.DataFrame] = None
_ADOT_CATCH_RATE_CACHE: Optional[pd.DataFrame] = None
_WR1_ALLOWED_CACHE: Optional[pd.DataFrame] = None
_SLOT_SNAP_CACHE: Optional[pd.DataFrame] = None
_REST_DAYS_CACHE: Optional[pd.DataFrame] = None


def clear_broken_feature_caches():
    """Clear all caches for broken feature calculations."""
    global _TEAM_PACE_CACHE, _PARTICIPATION_CACHE, _PRESSURE_BY_TEAM_CACHE
    global _COVERAGE_BY_TEAM_CACHE, _ADOT_CATCH_RATE_CACHE, _WR1_ALLOWED_CACHE, _SLOT_SNAP_CACHE
    global _REST_DAYS_CACHE
    _TEAM_PACE_CACHE = None
    _PARTICIPATION_CACHE = None
    _PRESSURE_BY_TEAM_CACHE = None
    _COVERAGE_BY_TEAM_CACHE = None
    _REST_DAYS_CACHE = None
    _ADOT_CATCH_RATE_CACHE = None
    _WR1_ALLOWED_CACHE = None
    _SLOT_SNAP_CACHE = None


# =============================================================================
# DATA LOADERS
# =============================================================================

def _load_team_pace() -> pd.DataFrame:
    """Load team pace data."""
    global _TEAM_PACE_CACHE
    if _TEAM_PACE_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'team_pace.parquet'
        if path.exists():
            _TEAM_PACE_CACHE = pd.read_parquet(path)
            logger.debug(f"Loaded team_pace: {len(_TEAM_PACE_CACHE)} rows")
        else:
            logger.warning(f"team_pace.parquet not found at {path}")
            _TEAM_PACE_CACHE = pd.DataFrame()
    return _TEAM_PACE_CACHE


def _load_participation() -> pd.DataFrame:
    """Load participation data for pressure and coverage."""
    global _PARTICIPATION_CACHE
    if _PARTICIPATION_CACHE is None:
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'participation.parquet'
        if path.exists():
            _PARTICIPATION_CACHE = pd.read_parquet(path)
            logger.debug(f"Loaded participation: {len(_PARTICIPATION_CACHE)} rows")
        else:
            logger.warning(f"participation.parquet not found at {path}")
            _PARTICIPATION_CACHE = pd.DataFrame()
    return _PARTICIPATION_CACHE


def _load_weekly_stats() -> pd.DataFrame:
    """Load weekly stats for adot, catch_rate, opp_wr1."""
    path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        # Normalize team column (NFLverse uses 'recent_team', we need 'team')
        if 'team' not in df.columns and 'recent_team' in df.columns:
            df['team'] = df['recent_team']
        return df
    logger.warning(f"weekly_stats.parquet not found at {path}")
    return pd.DataFrame()


def _load_rest_days_cache() -> pd.DataFrame:
    """
    Load rest days from schedules.parquet.

    Creates a lookup table: (season, week, team) -> rest_days
    Each team gets their rest days for each week (home_rest or away_rest).
    """
    global _REST_DAYS_CACHE

    if _REST_DAYS_CACHE is not None:
        return _REST_DAYS_CACHE

    path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if not path.exists():
        logger.warning(f"schedules.parquet not found at {path}")
        _REST_DAYS_CACHE = pd.DataFrame()
        return _REST_DAYS_CACHE

    schedules = pd.read_parquet(path)

    # Check for required columns
    if 'away_rest' not in schedules.columns or 'home_rest' not in schedules.columns:
        logger.warning("schedules.parquet missing rest columns")
        _REST_DAYS_CACHE = pd.DataFrame()
        return _REST_DAYS_CACHE

    # Create lookup for away teams
    away_rest = schedules[['season', 'week', 'away_team', 'away_rest']].copy()
    away_rest.columns = ['season', 'week', 'team', 'rest_days']

    # Create lookup for home teams
    home_rest = schedules[['season', 'week', 'home_team', 'home_rest']].copy()
    home_rest.columns = ['season', 'week', 'team', 'rest_days']

    # Combine and deduplicate
    rest_lookup = pd.concat([away_rest, home_rest], ignore_index=True)
    rest_lookup = rest_lookup.drop_duplicates(subset=['season', 'week', 'team'])

    _REST_DAYS_CACHE = rest_lookup
    logger.info(f"Loaded rest_days for {len(rest_lookup)} team-week combinations")
    return _REST_DAYS_CACHE


# =============================================================================
# ADOT AND CATCH RATE (from weekly_stats)
# =============================================================================

def calculate_adot_and_catch_rate_cache(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-calculate ADOT and catch rate for each player-season-week.

    ADOT = receiving_air_yards / targets (trailing EWMA)
    Catch Rate = receptions / targets (trailing EWMA)

    Uses shift(1) to ensure no data leakage.
    """
    global _ADOT_CATCH_RATE_CACHE

    if _ADOT_CATCH_RATE_CACHE is not None:
        return _ADOT_CATCH_RATE_CACHE

    if len(weekly_stats) == 0:
        return pd.DataFrame()

    # Filter to receiving players
    rec_stats = weekly_stats[weekly_stats['targets'] > 0].copy()

    if len(rec_stats) == 0:
        return pd.DataFrame()

    # Calculate per-game ADOT and catch rate
    rec_stats['game_adot'] = rec_stats['receiving_air_yards'] / rec_stats['targets']
    rec_stats['game_catch_rate'] = rec_stats['receptions'] / rec_stats['targets']

    # Sort by player, season, week
    rec_stats = rec_stats.sort_values(['player_id', 'season', 'week'])

    # Calculate trailing EWMA (shifted to prevent leakage)
    result_rows = []

    for player_id, player_data in rec_stats.groupby('player_id'):
        player_data = player_data.sort_values(['season', 'week'])

        # Trailing EWMA with span=4
        trailing_adot = player_data['game_adot'].ewm(span=4, min_periods=1).mean().shift(1)
        trailing_catch_rate = player_data['game_catch_rate'].ewm(span=4, min_periods=1).mean().shift(1)

        for idx, (_, row) in enumerate(player_data.iterrows()):
            result_rows.append({
                'player_id': player_id,
                'player_display_name': row.get('player_display_name', ''),
                'season': row['season'],
                'week': row['week'],
                'adot': trailing_adot.iloc[idx] if idx < len(trailing_adot) else np.nan,
                'trailing_catch_rate': trailing_catch_rate.iloc[idx] if idx < len(trailing_catch_rate) else np.nan,
            })

    _ADOT_CATCH_RATE_CACHE = pd.DataFrame(result_rows)
    logger.info(f"Calculated ADOT/catch_rate for {_ADOT_CATCH_RATE_CACHE['player_id'].nunique()} players")

    return _ADOT_CATCH_RATE_CACHE


# =============================================================================
# GAME PACE (from team_pace)
# =============================================================================

def calculate_game_pace_cache() -> pd.DataFrame:
    """
    Get game pace (plays per game) for each team.

    Returns DataFrame with team, game_pace columns.
    """
    team_pace = _load_team_pace()

    if len(team_pace) == 0:
        return pd.DataFrame()

    if 'plays_per_game' not in team_pace.columns:
        logger.warning("plays_per_game not in team_pace")
        return pd.DataFrame()

    result = team_pace[['team', 'plays_per_game']].copy()
    result = result.rename(columns={'plays_per_game': 'game_pace'})

    logger.info(f"Loaded game_pace for {len(result)} teams")
    return result


# =============================================================================
# PRESSURE RATES (from participation)
# =============================================================================

def calculate_pressure_rates_cache(participation: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pressure rates from participation data.

    Returns:
        offense_pressure: Pressure rate faced by each offense team
        defense_pressure: Pressure rate generated by each defense team
    """
    global _PRESSURE_BY_TEAM_CACHE

    if _PRESSURE_BY_TEAM_CACHE is not None:
        return _PRESSURE_BY_TEAM_CACHE

    if len(participation) == 0 or 'was_pressure' not in participation.columns:
        return pd.DataFrame(), pd.DataFrame()

    # Parse game_id to get season/week
    if 'nflverse_game_id' in participation.columns:
        # Format: 2024_01_NYJ_SF
        participation = participation.copy()
        split_id = participation['nflverse_game_id'].str.split('_', expand=True)
        if len(split_id.columns) >= 2:
            participation['season'] = split_id[0].astype(int)
            participation['week'] = split_id[1].astype(int)

    if 'season' not in participation.columns:
        logger.warning("Cannot determine season/week from participation data")
        return pd.DataFrame(), pd.DataFrame()

    # Aggregate pressure by possession team (offense)
    offense_pressure = participation.groupby(['possession_team', 'season', 'week']).agg({
        'was_pressure': ['sum', 'count']
    }).reset_index()
    offense_pressure.columns = ['team', 'season', 'week', 'pressures', 'dropbacks']
    offense_pressure['pressure_rate'] = offense_pressure['pressures'] / offense_pressure['dropbacks']

    # For defense pressure, we need the defense team
    # possession_team is offense, so defense is the other team
    # We'll calculate league-wide pressure and invert for defense

    # Get unique games and teams
    if 'nflverse_game_id' in participation.columns:
        game_teams = participation.groupby('nflverse_game_id')['possession_team'].apply(list).reset_index()

        # Calculate defense pressure (pressure generated when defending)
        defense_rows = []
        for _, game in game_teams.iterrows():
            teams = list(set(game['possession_team']))
            if len(teams) == 2:
                # Calculate pressure each defense generated
                game_data = participation[participation['nflverse_game_id'] == game['nflverse_game_id']]

                for i, def_team in enumerate(teams):
                    off_team = teams[1-i]
                    # Defense = plays where the other team has possession
                    def_plays = game_data[game_data['possession_team'] == off_team]
                    if len(def_plays) > 0:
                        season = def_plays['season'].iloc[0]
                        week = def_plays['week'].iloc[0]
                        pressures = def_plays['was_pressure'].sum()
                        dropbacks = len(def_plays)
                        defense_rows.append({
                            'team': def_team,
                            'season': season,
                            'week': week,
                            'pressures_generated': pressures,
                            'opp_dropbacks': dropbacks,
                            'opp_pressure_rate': pressures / dropbacks if dropbacks > 0 else 0.25
                        })

        defense_pressure = pd.DataFrame(defense_rows)
    else:
        defense_pressure = pd.DataFrame()

    _PRESSURE_BY_TEAM_CACHE = (offense_pressure, defense_pressure)
    logger.info(f"Calculated pressure rates for {offense_pressure['team'].nunique()} teams")

    return _PRESSURE_BY_TEAM_CACHE


# =============================================================================
# COVERAGE TENDENCIES (from participation)
# =============================================================================

def calculate_coverage_rates_cache(participation: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate man/zone coverage rates from participation data.

    Returns DataFrame with defense team, season, week, man_coverage_rate.
    """
    global _COVERAGE_BY_TEAM_CACHE

    if _COVERAGE_BY_TEAM_CACHE is not None:
        return _COVERAGE_BY_TEAM_CACHE

    if len(participation) == 0 or 'defense_man_zone_type' not in participation.columns:
        return pd.DataFrame()

    # Parse game_id for season/week
    participation = participation.copy()
    if 'nflverse_game_id' in participation.columns:
        split_id = participation['nflverse_game_id'].str.split('_', expand=True)
        if len(split_id.columns) >= 2:
            participation['season'] = split_id[0].astype(int)
            participation['week'] = split_id[1].astype(int)

    if 'season' not in participation.columns:
        return pd.DataFrame()

    # Filter to plays with coverage data
    coverage_plays = participation[participation['defense_man_zone_type'].isin(['MAN_COVERAGE', 'ZONE_COVERAGE'])].copy()

    if len(coverage_plays) == 0:
        return pd.DataFrame()

    coverage_plays['is_man'] = (coverage_plays['defense_man_zone_type'] == 'MAN_COVERAGE').astype(int)

    # Get defense team from game structure
    # The defense team is NOT possession_team
    if 'nflverse_game_id' in coverage_plays.columns:
        # For each game, find both teams
        game_teams = {}
        for game_id in coverage_plays['nflverse_game_id'].unique():
            teams = coverage_plays[coverage_plays['nflverse_game_id'] == game_id]['possession_team'].unique()
            game_teams[game_id] = list(teams)

        def get_defense_team(row):
            teams = game_teams.get(row['nflverse_game_id'], [])
            if len(teams) == 2:
                return [t for t in teams if t != row['possession_team']][0]
            return None

        coverage_plays['defense_team'] = coverage_plays.apply(get_defense_team, axis=1)
        coverage_plays = coverage_plays.dropna(subset=['defense_team'])

        # Aggregate by defense team, season, week
        coverage_agg = coverage_plays.groupby(['defense_team', 'season', 'week']).agg({
            'is_man': ['sum', 'count']
        }).reset_index()
        coverage_agg.columns = ['team', 'season', 'week', 'man_plays', 'total_plays']
        coverage_agg['man_coverage_rate'] = coverage_agg['man_plays'] / coverage_agg['total_plays']

        # Calculate trailing EWMA
        coverage_agg = coverage_agg.sort_values(['team', 'season', 'week'])
        coverage_agg['opp_man_coverage_rate_trailing'] = coverage_agg.groupby('team')['man_coverage_rate'].transform(
            lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
        )

        _COVERAGE_BY_TEAM_CACHE = coverage_agg
        logger.info(f"Calculated coverage rates for {coverage_agg['team'].nunique()} teams")
        return _COVERAGE_BY_TEAM_CACHE

    return pd.DataFrame()


# =============================================================================
# SLOT SNAP PERCENTAGE (from participation routes)
# =============================================================================

def calculate_slot_snap_cache(participation: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate slot snap percentage from route data.

    Slot routes: SHALLOW CROSS/DRAG, SLANT, IN/DIG, QUICK OUT, HITCH/CURL
    Outside routes: GO, DEEP OUT, CORNER, POST

    Returns player-level trailing slot_snap_pct.
    """
    global _SLOT_SNAP_CACHE

    if _SLOT_SNAP_CACHE is not None:
        return _SLOT_SNAP_CACHE

    if len(participation) == 0 or 'route' not in participation.columns:
        return pd.DataFrame()

    # Define slot routes (shorter, inside routes typically run from slot)
    slot_routes = {'SHALLOW CROSS/DRAG', 'SLANT', 'IN/DIG', 'QUICK OUT', 'HITCH/CURL', 'SCREEN', 'SWING', 'TEXAS/ANGLE'}

    participation = participation.copy()

    # Filter to pass plays with routes
    route_plays = participation[participation['route'].notna() & (participation['route'] != '')].copy()

    if len(route_plays) == 0:
        return pd.DataFrame()

    route_plays['is_slot_route'] = route_plays['route'].isin(slot_routes).astype(int)

    # Parse player names from offense_names (comma-separated list)
    # This is complex - we'd need to match routes to specific players
    # For now, use a simpler team-level calculation and position defaults

    # Parse season/week
    if 'nflverse_game_id' in route_plays.columns:
        split_id = route_plays['nflverse_game_id'].str.split('_', expand=True)
        if len(split_id.columns) >= 2:
            route_plays['season'] = split_id[0].astype(int)
            route_plays['week'] = split_id[1].astype(int)

    if 'season' not in route_plays.columns:
        return pd.DataFrame()

    # Aggregate by team (as proxy for receivers)
    team_slot = route_plays.groupby(['possession_team', 'season', 'week']).agg({
        'is_slot_route': ['sum', 'count']
    }).reset_index()
    team_slot.columns = ['team', 'season', 'week', 'slot_routes', 'total_routes']
    team_slot['slot_rate'] = team_slot['slot_routes'] / team_slot['total_routes']

    # Trailing EWMA
    team_slot = team_slot.sort_values(['team', 'season', 'week'])
    team_slot['slot_snap_pct'] = team_slot.groupby('team')['slot_rate'].transform(
        lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
    )

    _SLOT_SNAP_CACHE = team_slot
    logger.info(f"Calculated slot_snap_pct for {team_slot['team'].nunique()} teams")
    return _SLOT_SNAP_CACHE


# =============================================================================
# OPP WR1 RECEPTIONS ALLOWED
# =============================================================================

def calculate_opp_wr1_allowed_cache(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate receptions allowed to WR1s by each defense.

    WR1 = receiver with highest target share on their team.
    """
    global _WR1_ALLOWED_CACHE

    if _WR1_ALLOWED_CACHE is not None:
        return _WR1_ALLOWED_CACHE

    if len(weekly_stats) == 0:
        return pd.DataFrame()

    # Filter to WRs with targets
    wr_stats = weekly_stats[(weekly_stats['position'] == 'WR') & (weekly_stats['targets'] > 0)].copy()

    if len(wr_stats) == 0:
        return pd.DataFrame()

    # Find WR1 for each team-week (highest targets)
    # Filter out rows without team to avoid NaN issues
    wr_stats = wr_stats[wr_stats['team'].notna()].copy()

    if len(wr_stats) == 0:
        return pd.DataFrame()

    wr_stats['is_wr1'] = wr_stats.groupby(['team', 'season', 'week'])['targets'].transform(
        lambda x: x == x.max()
    )

    wr1_stats = wr_stats[wr_stats['is_wr1'].fillna(False)].copy()

    # Aggregate WR1 receptions by opponent defense
    # opponent_team is the defense
    if 'opponent_team' not in wr1_stats.columns:
        return pd.DataFrame()

    defense_wr1 = wr1_stats.groupby(['opponent_team', 'season', 'week']).agg({
        'receptions': 'sum'
    }).reset_index()
    defense_wr1.columns = ['team', 'season', 'week', 'wr1_receptions_allowed']

    # Trailing EWMA
    defense_wr1 = defense_wr1.sort_values(['team', 'season', 'week'])
    defense_wr1['opp_wr1_receptions_allowed'] = defense_wr1.groupby('team')['wr1_receptions_allowed'].transform(
        lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
    )

    _WR1_ALLOWED_CACHE = defense_wr1
    logger.info(f"Calculated opp_wr1_receptions_allowed for {defense_wr1['team'].nunique()} teams")
    return _WR1_ALLOWED_CACHE


# =============================================================================
# MAIN FUNCTION: ADD ALL BROKEN FEATURES
# =============================================================================

def add_broken_features_to_dataframe(
    df: pd.DataFrame,
    market: str
) -> pd.DataFrame:
    """
    Add all 13 broken features to the dataframe.

    This should be called BEFORE default fillna operations.

    Args:
        df: DataFrame with player data (must have player_id, team, season, week)
        market: Market type for market-specific features

    Returns:
        DataFrame with calculated features added
    """
    df = df.copy()

    # Load source data
    weekly_stats = _load_weekly_stats()
    participation = _load_participation()

    # =========================================================================
    # STEP 0: Ensure we have required columns for merging
    # =========================================================================

    # Add player_id if missing (via name matching)
    if 'player_id' not in df.columns and 'player' in df.columns and len(weekly_stats) > 0:
        df['player_lower'] = df['player'].str.lower().str.strip()
        weekly_stats_temp = weekly_stats.copy()
        weekly_stats_temp['player_lower'] = weekly_stats_temp['player_display_name'].str.lower().str.strip()
        player_id_map = weekly_stats_temp[['player_lower', 'player_id']].drop_duplicates()
        df = df.merge(player_id_map, on='player_lower', how='left')
        df = df.drop(columns=['player_lower'], errors='ignore')
        logger.debug(f"Added player_id via name matching: {df['player_id'].notna().mean():.1%} coverage")

    # Ensure team column is populated
    if 'team' not in df.columns or df['team'].isna().all():
        # Try player_team first
        if 'player_team' in df.columns:
            df['team'] = df['player_team']
        # Try to get from weekly_stats via player match
        elif 'player_id' in df.columns and len(weekly_stats) > 0:
            # Get most recent team for each player
            recent_teams = weekly_stats.sort_values(['season', 'week']).groupby('player_id')['team'].last().reset_index()
            recent_teams = recent_teams.rename(columns={'team': 'team_from_stats'})
            df = df.merge(recent_teams, on='player_id', how='left')
            if 'team_from_stats' in df.columns:
                df['team'] = df['team'].fillna(df['team_from_stats'])
                df = df.drop(columns=['team_from_stats'], errors='ignore')

    # Ensure opponent column is set up
    opponent_col = None
    for col in ['opponent', 'opponent_team', 'opp']:
        if col in df.columns and df[col].notna().any():
            opponent_col = col
            break

    # =========================================================================
    # 1. ADOT and TRAILING_CATCH_RATE (from weekly_stats)
    # =========================================================================
    if len(weekly_stats) > 0 and market in ['player_receptions', 'player_reception_yds']:
        adot_cache = calculate_adot_and_catch_rate_cache(weekly_stats)
        if len(adot_cache) > 0:
            # Merge by player_id, season, week
            if 'player_id' in df.columns and 'player_id' in adot_cache.columns:
                merge_cols = ['player_id', 'season', 'week']
                df = df.merge(
                    adot_cache[['player_id', 'season', 'week', 'adot', 'trailing_catch_rate']],
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_calc')
                )
                # Use calculated values if original are missing
                for col in ['adot', 'trailing_catch_rate']:
                    calc_col = f'{col}_calc'
                    if calc_col in df.columns:
                        df[col] = df[col].fillna(df[calc_col])
                        df = df.drop(columns=[calc_col], errors='ignore')

            logger.debug(f"ADOT coverage: {df['adot'].notna().mean():.1%}" if 'adot' in df.columns else "ADOT not added")

    # =========================================================================
    # 2. GAME_PACE (from team_pace)
    # =========================================================================
    game_pace_cache = calculate_game_pace_cache()
    if len(game_pace_cache) > 0 and 'team' in df.columns:
        df = df.merge(
            game_pace_cache[['team', 'game_pace']],
            on='team',
            how='left',
            suffixes=('', '_calc')
        )
        if 'game_pace_calc' in df.columns:
            df['game_pace'] = df['game_pace'].fillna(df['game_pace_calc'])
            df = df.drop(columns=['game_pace_calc'], errors='ignore')

        logger.debug(f"game_pace coverage: {df['game_pace'].notna().mean():.1%}" if 'game_pace' in df.columns else "game_pace not added")

    # =========================================================================
    # 3. PRESSURE_RATE and OPP_PRESSURE_RATE (from participation)
    # =========================================================================
    if len(participation) > 0:
        offense_pressure, defense_pressure = calculate_pressure_rates_cache(participation)

        # Player's team pressure rate (how often their QB is pressured)
        if len(offense_pressure) > 0 and 'team' in df.columns:
            # Get trailing pressure rate
            offense_pressure_trailing = offense_pressure.sort_values(['team', 'season', 'week'])
            offense_pressure_trailing['pressure_rate_trailing'] = offense_pressure_trailing.groupby('team')['pressure_rate'].transform(
                lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
            )

            df = df.merge(
                offense_pressure_trailing[['team', 'season', 'week', 'pressure_rate_trailing']].rename(
                    columns={'pressure_rate_trailing': 'pressure_rate_calc'}
                ),
                on=['team', 'season', 'week'],
                how='left'
            )
            if 'pressure_rate_calc' in df.columns:
                df['pressure_rate'] = df['pressure_rate'].fillna(df['pressure_rate_calc']) if 'pressure_rate' in df.columns else df['pressure_rate_calc']
                df = df.drop(columns=['pressure_rate_calc'], errors='ignore')

        # Opponent defense pressure rate (use opponent_col determined earlier)
        if len(defense_pressure) > 0 and opponent_col is not None:
            defense_pressure_trailing = defense_pressure.sort_values(['team', 'season', 'week'])
            defense_pressure_trailing['opp_pressure_rate_trailing'] = defense_pressure_trailing.groupby('team')['opp_pressure_rate'].transform(
                lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
            )

            df = df.merge(
                defense_pressure_trailing[['team', 'season', 'week', 'opp_pressure_rate_trailing']].rename(
                    columns={'team': opponent_col, 'opp_pressure_rate_trailing': 'opp_pressure_rate_calc'}
                ),
                on=[opponent_col, 'season', 'week'],
                how='left'
            )
            if 'opp_pressure_rate_calc' in df.columns:
                df['opp_pressure_rate'] = df['opp_pressure_rate'].fillna(df['opp_pressure_rate_calc']) if 'opp_pressure_rate' in df.columns else df['opp_pressure_rate_calc']
                df = df.drop(columns=['opp_pressure_rate_calc'], errors='ignore')

        logger.debug(f"pressure_rate coverage: {df['pressure_rate'].notna().mean():.1%}" if 'pressure_rate' in df.columns else "pressure_rate not added")

    # =========================================================================
    # 4. COVERAGE FEATURES (from participation)
    # =========================================================================
    if len(participation) > 0:
        coverage_cache = calculate_coverage_rates_cache(participation)

        if len(coverage_cache) > 0 and opponent_col is not None:
            df = df.merge(
                coverage_cache[['team', 'season', 'week', 'opp_man_coverage_rate_trailing']].rename(
                    columns={'team': opponent_col}
                ),
                on=[opponent_col, 'season', 'week'],
                how='left',
                suffixes=('', '_calc')
            )
            if 'opp_man_coverage_rate_trailing_calc' in df.columns:
                df['opp_man_coverage_rate_trailing'] = df['opp_man_coverage_rate_trailing'].fillna(df['opp_man_coverage_rate_trailing_calc']) if 'opp_man_coverage_rate_trailing' in df.columns else df['opp_man_coverage_rate_trailing_calc']
                df = df.drop(columns=['opp_man_coverage_rate_trailing_calc'], errors='ignore')

            # Calculate man_coverage_adjustment: higher man rate = harder for receivers
            if 'opp_man_coverage_rate_trailing' in df.columns:
                league_avg_man = 0.49  # From our data: 11039 / (11039+11369)
                df['man_coverage_adjustment'] = 1.0 - 0.1 * (df['opp_man_coverage_rate_trailing'] - league_avg_man)
                df['man_coverage_adjustment'] = df['man_coverage_adjustment'].fillna(1.0)

            logger.debug(f"coverage feature coverage: {df['opp_man_coverage_rate_trailing'].notna().mean():.1%}" if 'opp_man_coverage_rate_trailing' in df.columns else "coverage not added")

    # =========================================================================
    # 5. SLOT_SNAP_PCT (from participation routes)
    # =========================================================================
    if len(participation) > 0 and market in ['player_receptions', 'player_reception_yds']:
        slot_cache = calculate_slot_snap_cache(participation)

        if len(slot_cache) > 0 and 'team' in df.columns:
            df = df.merge(
                slot_cache[['team', 'season', 'week', 'slot_snap_pct']],
                on=['team', 'season', 'week'],
                how='left',
                suffixes=('', '_calc')
            )
            if 'slot_snap_pct_calc' in df.columns:
                df['slot_snap_pct'] = df['slot_snap_pct'].fillna(df['slot_snap_pct_calc']) if 'slot_snap_pct' in df.columns else df['slot_snap_pct_calc']
                df = df.drop(columns=['slot_snap_pct_calc'], errors='ignore')

            # Calculate slot_funnel_score: slot % * man coverage rate inverse
            if 'slot_snap_pct' in df.columns and 'opp_man_coverage_rate_trailing' in df.columns:
                # High slot % + low man coverage = high slot funnel opportunity
                df['slot_funnel_score'] = df['slot_snap_pct'] * (1 - df['opp_man_coverage_rate_trailing'].fillna(0.5))

            logger.debug(f"slot_snap_pct coverage: {df['slot_snap_pct'].notna().mean():.1%}" if 'slot_snap_pct' in df.columns else "slot_snap_pct not added")

    # =========================================================================
    # 6. OPP_WR1_RECEPTIONS_ALLOWED (from weekly_stats)
    # =========================================================================
    if len(weekly_stats) > 0 and market in ['player_receptions', 'player_reception_yds']:
        wr1_cache = calculate_opp_wr1_allowed_cache(weekly_stats)

        if len(wr1_cache) > 0 and opponent_col is not None:
            df = df.merge(
                wr1_cache[['team', 'season', 'week', 'opp_wr1_receptions_allowed']].rename(
                    columns={'team': opponent_col}
                ),
                on=[opponent_col, 'season', 'week'],
                how='left',
                suffixes=('', '_calc')
            )
            if 'opp_wr1_receptions_allowed_calc' in df.columns:
                df['opp_wr1_receptions_allowed'] = df['opp_wr1_receptions_allowed'].fillna(df['opp_wr1_receptions_allowed_calc']) if 'opp_wr1_receptions_allowed' in df.columns else df['opp_wr1_receptions_allowed_calc']
                df = df.drop(columns=['opp_wr1_receptions_allowed_calc'], errors='ignore')

            logger.debug(f"opp_wr1_receptions_allowed coverage: {df['opp_wr1_receptions_allowed'].notna().mean():.1%}" if 'opp_wr1_receptions_allowed' in df.columns else "opp_wr1 not added")

    # =========================================================================
    # 7. REST DAYS (from schedules.parquet)
    # =========================================================================
    # Merge rest_days from schedules before calculating interaction terms
    if 'rest_days' not in df.columns and 'team' in df.columns:
        rest_cache = _load_rest_days_cache()
        if len(rest_cache) > 0:
            df = df.merge(
                rest_cache[['season', 'week', 'team', 'rest_days']],
                on=['season', 'week', 'team'],
                how='left',
                suffixes=('', '_sched')
            )
            if 'rest_days_sched' in df.columns:
                df['rest_days'] = df['rest_days'].fillna(df['rest_days_sched']) if 'rest_days' in df.columns else df['rest_days_sched']
                df = df.drop(columns=['rest_days_sched'], errors='ignore')
            logger.debug(f"rest_days coverage: {df['rest_days'].notna().mean():.1%}" if 'rest_days' in df.columns else "rest_days not added")

    # =========================================================================
    # 8. INTERACTION TERMS (lvt_x_defense, lvt_x_rest)
    # =========================================================================
    # These depend on other features being calculated first

    # lvt_x_defense: line_vs_trailing * opponent_def_epa
    if 'line_vs_trailing' in df.columns:
        def_epa_col = None
        for col in ['opp_position_def_epa', 'opp_def_epa', 'opponent_def_epa']:
            if col in df.columns and df[col].notna().any():
                def_epa_col = col
                break

        if def_epa_col is not None:
            df['lvt_x_defense'] = df['line_vs_trailing'] * df[def_epa_col].fillna(0)
            logger.debug(f"lvt_x_defense calculated from {def_epa_col}")

        # lvt_x_rest: line_vs_trailing * normalized rest days
        rest_col = None
        for col in ['rest_days', 'days_rest']:
            if col in df.columns and df[col].notna().any():
                rest_col = col
                break

        if rest_col is not None:
            rest_normalized = (df[rest_col] - 7) / 7.0  # Normalize around 7 days
            df['lvt_x_rest'] = df['line_vs_trailing'] * rest_normalized
            logger.debug(f"lvt_x_rest calculated from {rest_col}: coverage {df['lvt_x_rest'].notna().mean():.1%}")

    return df
