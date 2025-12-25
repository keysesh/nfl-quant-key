"""
ATS (Against The Spread) Tracker Module.

Calculates and tracks team ATS records from NFLverse schedules data:
- Season ATS record per team
- Last N games ATS
- Home/Away ATS splits
- Over/Under records

Uses data/nflverse/schedules.parquet which contains:
- spread_line: Home team spread (negative = home favored)
- result: Home margin (home_score - away_score)
- total_line: O/U line
- total: Actual combined score
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Module-level cache
_ATS_CACHE = {}
_SCHEDULES_CACHE = None


def load_schedules(force_reload: bool = False) -> pd.DataFrame:
    """Load schedules data with caching."""
    global _SCHEDULES_CACHE

    if _SCHEDULES_CACHE is not None and not force_reload:
        return _SCHEDULES_CACHE

    path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'

    if not path.exists():
        logger.warning(f"Schedules data not found at {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    _SCHEDULES_CACHE = df
    logger.info(f"Loaded schedules: {len(df)} games")

    return df


def calculate_team_ats_record(
    team: str,
    season: int,
    max_week: int
) -> Dict:
    """
    Calculate full season ATS record for a team.

    Args:
        team: Team abbreviation (e.g., 'KC', 'PHI')
        season: Season year
        max_week: Only include games up to this week (exclusive)

    Returns:
        Dict with:
        - ats_wins: int
        - ats_losses: int
        - ats_pushes: int
        - ats_pct: float (win %)
        - ats_record_str: str (e.g., "10-4 ATS")
    """
    schedules = load_schedules()

    if len(schedules) == 0:
        return _empty_ats_record()

    # Filter to completed games for this season up to max_week
    games = schedules[
        (schedules['season'] == season) &
        (schedules['week'] < max_week) &
        (schedules['home_score'].notna()) &
        (schedules['spread_line'].notna())
    ].copy()

    if len(games) == 0:
        return _empty_ats_record()

    # Get games where team played (home or away)
    home_games = games[games['home_team'] == team].copy()
    away_games = games[games['away_team'] == team].copy()

    wins = 0
    losses = 0
    pushes = 0

    # Home games: team covers if result > -spread_line
    # spread_line is from home team's perspective
    # If spread_line = -3.5, home favored by 3.5
    # Home covers if they win by more than 3.5 (result > 3.5)
    for _, game in home_games.iterrows():
        spread = game['spread_line']
        result = game['result']  # home_score - away_score

        if pd.isna(spread) or pd.isna(result):
            continue

        # Home team covers if actual margin > spread line
        # (remember: negative spread means home favored)
        cover_threshold = -spread  # If spread = -3.5, need to win by > 3.5

        if result > cover_threshold:
            wins += 1
        elif result < cover_threshold:
            losses += 1
        else:
            pushes += 1

    # Away games: team covers if they beat the spread
    # From away team's perspective: if spread_line = -3.5 (home favored)
    # Away gets +3.5, so they cover if result < 3.5 (they don't lose by more than 3.5)
    for _, game in away_games.iterrows():
        spread = game['spread_line']
        result = game['result']  # home_score - away_score

        if pd.isna(spread) or pd.isna(result):
            continue

        # Away team covers if home doesn't cover
        # Away cover threshold is the negative of home cover
        cover_threshold = -spread

        if result < cover_threshold:
            wins += 1
        elif result > cover_threshold:
            losses += 1
        else:
            pushes += 1

    total_games = wins + losses
    ats_pct = wins / total_games if total_games > 0 else 0.5

    return {
        'ats_wins': wins,
        'ats_losses': losses,
        'ats_pushes': pushes,
        'ats_pct': ats_pct,
        'ats_record_str': f"{wins}-{losses}" + (f"-{pushes}" if pushes > 0 else "") + " ATS"
    }


def calculate_team_last_n_ats(
    team: str,
    season: int,
    week: int,
    n: int = 6
) -> Dict:
    """
    Calculate ATS record for last N games.

    Args:
        team: Team abbreviation
        season: Season year
        week: Current week
        n: Number of games to look back

    Returns:
        Same structure as calculate_team_ats_record
    """
    schedules = load_schedules()

    if len(schedules) == 0:
        return _empty_ats_record()

    # Get completed games for this team in current season
    games = schedules[
        (schedules['season'] == season) &
        (schedules['week'] < week) &
        (schedules['home_score'].notna()) &
        (schedules['spread_line'].notna()) &
        ((schedules['home_team'] == team) | (schedules['away_team'] == team))
    ].copy()

    if len(games) == 0:
        return _empty_ats_record()

    # Sort by week descending and take last N
    games = games.sort_values('week', ascending=False).head(n)

    wins = 0
    losses = 0
    pushes = 0

    for _, game in games.iterrows():
        spread = game['spread_line']
        result = game['result']
        is_home = game['home_team'] == team

        if pd.isna(spread) or pd.isna(result):
            continue

        cover_threshold = -spread

        if is_home:
            if result > cover_threshold:
                wins += 1
            elif result < cover_threshold:
                losses += 1
            else:
                pushes += 1
        else:  # Away
            if result < cover_threshold:
                wins += 1
            elif result > cover_threshold:
                losses += 1
            else:
                pushes += 1

    total_games = wins + losses
    ats_pct = wins / total_games if total_games > 0 else 0.5

    return {
        'ats_wins': wins,
        'ats_losses': losses,
        'ats_pushes': pushes,
        'ats_pct': ats_pct,
        'ats_record_str': f"{wins}-{losses}" + (f"-{pushes}" if pushes > 0 else "") + " ATS"
    }


def calculate_team_ou_record(
    team: str,
    season: int,
    max_week: int
) -> Dict:
    """
    Calculate Over/Under record for games involving this team.

    Returns:
        Dict with:
        - over_wins: int
        - under_wins: int
        - pushes: int
        - over_pct: float
        - ou_record_str: str
    """
    schedules = load_schedules()

    if len(schedules) == 0:
        return _empty_ou_record()

    # Get completed games with O/U data
    games = schedules[
        (schedules['season'] == season) &
        (schedules['week'] < max_week) &
        (schedules['total'].notna()) &
        (schedules['total_line'].notna()) &
        ((schedules['home_team'] == team) | (schedules['away_team'] == team))
    ].copy()

    if len(games) == 0:
        return _empty_ou_record()

    overs = 0
    unders = 0
    pushes = 0

    for _, game in games.iterrows():
        total = game['total']
        total_line = game['total_line']

        if pd.isna(total) or pd.isna(total_line):
            continue

        if total > total_line:
            overs += 1
        elif total < total_line:
            unders += 1
        else:
            pushes += 1

    total_games = overs + unders
    over_pct = overs / total_games if total_games > 0 else 0.5

    return {
        'over_wins': overs,
        'under_wins': unders,
        'pushes': pushes,
        'over_pct': over_pct,
        'ou_record_str': f"{overs}O-{unders}U" + (f"-{pushes}P" if pushes > 0 else "")
    }


def get_all_teams_ats_summary(season: int, week: int) -> pd.DataFrame:
    """
    Generate ATS summary for all 32 NFL teams.

    Args:
        season: Season year
        week: Current week (calculate up to this week)

    Returns:
        DataFrame with columns:
        - team
        - season_ats_record, season_ats_pct
        - last_6_ats_record, last_6_ats_pct
        - ou_over_pct, ou_record
    """
    cache_key = f"summary_{season}_{week}"

    if cache_key in _ATS_CACHE:
        return _ATS_CACHE[cache_key]

    schedules = load_schedules()

    if len(schedules) == 0:
        return pd.DataFrame()

    # Get all teams from schedules
    teams = set(schedules[schedules['season'] == season]['home_team'].unique())
    teams.update(schedules[schedules['season'] == season]['away_team'].unique())
    teams = sorted([t for t in teams if pd.notna(t)])

    records = []
    for team in teams:
        season_ats = calculate_team_ats_record(team, season, week)
        last_6_ats = calculate_team_last_n_ats(team, season, week, n=6)
        ou_record = calculate_team_ou_record(team, season, week)

        records.append({
            'team': team,
            'season_ats_record': season_ats['ats_record_str'],
            'season_ats_pct': season_ats['ats_pct'],
            'last_6_ats_record': last_6_ats['ats_record_str'],
            'last_6_ats_pct': last_6_ats['ats_pct'],
            'ou_over_pct': ou_record['over_pct'],
            'ou_record': ou_record['ou_record_str']
        })

    result = pd.DataFrame(records)
    _ATS_CACHE[cache_key] = result

    return result


def get_team_ats_context(
    team: str,
    season: int,
    week: int
) -> Dict:
    """
    Get complete ATS context for a team (for dashboard display).

    Returns:
        Dict with all ATS metrics for dashboard rendering
    """
    season_ats = calculate_team_ats_record(team, season, week)
    last_6_ats = calculate_team_last_n_ats(team, season, week, n=6)
    ou_record = calculate_team_ou_record(team, season, week)

    return {
        'team': team,
        'season_ats': season_ats['ats_record_str'],
        'season_ats_pct': season_ats['ats_pct'],
        'last_6_ats': last_6_ats['ats_record_str'],
        'last_6_ats_pct': last_6_ats['ats_pct'],
        'ou_over_pct': ou_record['over_pct'],
        'ou_record': ou_record['ou_record_str'],
        'has_data': season_ats['ats_wins'] + season_ats['ats_losses'] > 0
    }


def _empty_ats_record() -> Dict:
    """Return empty ATS record."""
    return {
        'ats_wins': 0,
        'ats_losses': 0,
        'ats_pushes': 0,
        'ats_pct': 0.5,
        'ats_record_str': "0-0 ATS"
    }


def _empty_ou_record() -> Dict:
    """Return empty O/U record."""
    return {
        'over_wins': 0,
        'under_wins': 0,
        'pushes': 0,
        'over_pct': 0.5,
        'ou_record_str': "0O-0U"
    }


def clear_caches():
    """Clear module-level caches."""
    global _ATS_CACHE, _SCHEDULES_CACHE
    _ATS_CACHE = {}
    _SCHEDULES_CACHE = None
    logger.info("ATS tracker caches cleared")
