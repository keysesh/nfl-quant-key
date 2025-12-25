"""
NFL QUANT - Situational Features
================================

Tier 4 Blueprint features for situational adjustments:
- Rest days (days since last game)
- Home field advantage adjustments

These features capture game context factors that affect player performance
beyond pure skill metrics.

Usage:
    from nfl_quant.features.situational_features import (
        get_rest_days,
        get_hfa_adjustment,
        get_situational_features,
    )
"""

import logging
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from nfl_quant.config_paths import PROJECT_ROOT

logger = logging.getLogger(__name__)


# =============================================================================
# REST DAYS CALCULATION
# =============================================================================

@lru_cache(maxsize=256)
def _load_schedule() -> pd.DataFrame:
    """Load and cache nflverse schedule data."""
    schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.parquet'
    if not schedule_path.exists():
        schedule_path = PROJECT_ROOT / 'data' / 'nflverse' / 'schedules.csv'

    if not schedule_path.exists():
        logger.warning(f"Schedule file not found: {schedule_path}")
        return pd.DataFrame()

    if schedule_path.suffix == '.parquet':
        df = pd.read_parquet(schedule_path)
    else:
        df = pd.read_csv(schedule_path)

    # Ensure gameday is datetime
    if 'gameday' in df.columns:
        df['gameday'] = pd.to_datetime(df['gameday'], errors='coerce')

    return df


def get_rest_days(
    team: str,
    season: int,
    week: int,
) -> int:
    """
    Calculate days of rest since team's last game.

    Uses nflverse schedule data which already has rest_days column,
    but we calculate manually if needed for consistency.

    Args:
        team: Team abbreviation (e.g., 'KC', 'BUF')
        season: Season year
        week: Week number

    Returns:
        Days since last game (typically 4-14):
        - 4 days: Thursday night game
        - 7 days: Normal week
        - 10-14 days: Coming off bye week
        - 17+ days: Very long rest (rare)
    """
    schedule = _load_schedule()
    if len(schedule) == 0:
        return 7  # Default to normal week

    # If schedule has rest columns, use those
    if 'home_rest' in schedule.columns and 'away_rest' in schedule.columns:
        game = schedule[
            (schedule['season'] == season) &
            (schedule['week'] == week) &
            ((schedule['home_team'] == team) | (schedule['away_team'] == team))
        ]

        if len(game) > 0:
            game = game.iloc[0]
            if game['home_team'] == team:
                rest = game.get('home_rest')
            else:
                rest = game.get('away_rest')

            if pd.notna(rest):
                return int(rest)

    # Calculate manually from gameday dates
    try:
        team_games = schedule[
            (schedule['season'] == season) &
            ((schedule['home_team'] == team) | (schedule['away_team'] == team)) &
            schedule['gameday'].notna()
        ].sort_values('week')

        if len(team_games) < 2:
            return 7  # Not enough data

        # Find current week and previous week games
        current_week_idx = team_games[team_games['week'] == week].index
        if len(current_week_idx) == 0:
            return 7

        current_idx = team_games.index.get_loc(current_week_idx[0])
        if current_idx == 0:
            return 7  # First game of season

        current_game = team_games.iloc[current_idx]
        prev_game = team_games.iloc[current_idx - 1]

        if pd.notna(current_game['gameday']) and pd.notna(prev_game['gameday']):
            rest_days = (current_game['gameday'] - prev_game['gameday']).days
            return max(4, min(21, rest_days))  # Bound to reasonable range

    except Exception as e:
        logger.warning(f"Error calculating rest days: {e}")

    return 7  # Default


def get_rest_advantage(
    home_team: str,
    away_team: str,
    season: int,
    week: int,
) -> float:
    """
    Calculate rest advantage (home_rest - away_rest).

    Positive value means home team has more rest.

    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        season: Season year
        week: Week number

    Returns:
        Rest day differential (typically -3 to +7)
    """
    home_rest = get_rest_days(home_team, season, week)
    away_rest = get_rest_days(away_team, season, week)
    return home_rest - away_rest


# =============================================================================
# HOME FIELD ADVANTAGE
# =============================================================================

# Team-specific HFA adjustments based on historical home/away performance
# Positive = stronger home advantage, negative = weaker
# Based on 2021-2024 data analysis
TEAM_HFA_ADJUSTMENTS: Dict[str, float] = {
    # Strong home field (altitude, crowd noise, travel)
    'DEN': 1.3,   # Altitude factor
    'SEA': 1.2,   # 12th man noise
    'KC': 1.15,   # Arrowhead noise
    'NO': 1.1,    # Superdome noise
    'BUF': 1.1,   # Weather + passionate fans

    # Average home field
    'GB': 1.0,
    'DAL': 1.0,
    'SF': 1.0,
    'PHI': 1.0,
    'BAL': 1.0,
    'CIN': 1.0,
    'MIN': 1.0,
    'PIT': 1.0,
    'CLE': 1.0,
    'NYG': 1.0,
    'NYJ': 1.0,
    'NE': 1.0,
    'CHI': 1.0,
    'DET': 1.0,
    'ATL': 1.0,
    'TB': 1.0,
    'IND': 1.0,
    'HOU': 1.0,
    'TEN': 1.0,
    'MIA': 1.0,
    'ARI': 1.0,

    # Weaker home field (travel-friendly, dome, new stadium)
    'LAC': 0.85,  # LA teams have visiting fan issues
    'LA': 0.85,
    'LV': 0.9,    # Vegas attracts visiting fans
    'JAX': 0.9,   # Smaller fanbase
    'CAR': 0.95,
    'WAS': 0.95,
}


def get_hfa_adjustment(
    team: str,
    is_home: bool = True,
) -> float:
    """
    Get team-specific home field advantage adjustment.

    Returns a multiplier that can be applied to expected performance.
    Values > 1.0 indicate stronger HFA, < 1.0 indicate weaker.

    Args:
        team: Team abbreviation
        is_home: Whether team is playing at home

    Returns:
        HFA multiplier (0.85 to 1.3 range)
    """
    base_hfa = TEAM_HFA_ADJUSTMENTS.get(team, 1.0)

    if is_home:
        return base_hfa
    else:
        # Away team gets inverse adjustment
        return 2.0 - base_hfa  # If home = 1.2, away = 0.8


def calculate_hfa_from_history(
    team: str,
    season: int,
    trailing_seasons: int = 2,
) -> float:
    """
    Calculate dynamic HFA based on historical home/away EPA splits.

    This provides a data-driven HFA adjustment rather than static lookup.

    Args:
        team: Team abbreviation
        season: Current season
        trailing_seasons: Seasons to include in calculation

    Returns:
        HFA multiplier based on home vs away EPA differential
    """
    schedule = _load_schedule()
    if len(schedule) == 0:
        return 1.0

    try:
        # Get completed games for trailing seasons
        games = schedule[
            (schedule['season'] >= season - trailing_seasons) &
            (schedule['season'] < season) &
            schedule['home_score'].notna() &
            schedule['away_score'].notna()
        ]

        # Home games
        home_games = games[games['home_team'] == team]
        # Away games
        away_games = games[games['away_team'] == team]

        if len(home_games) == 0 or len(away_games) == 0:
            return TEAM_HFA_ADJUSTMENTS.get(team, 1.0)

        # Calculate win rates
        home_wins = (home_games['home_score'] > home_games['away_score']).mean()
        away_wins = (away_games['away_score'] > away_games['home_score']).mean()

        # HFA is the difference in win rates, converted to multiplier
        # Average NFL HFA is ~3% win rate boost (~0.03)
        hfa_boost = home_wins - away_wins

        # Convert to multiplier (center at 1.0, range 0.8 to 1.2)
        # 0% diff = 1.0, +10% diff = 1.1, +20% diff = 1.2
        multiplier = 1.0 + min(0.2, max(-0.2, hfa_boost))

        return float(multiplier)

    except Exception as e:
        logger.warning(f"Error calculating HFA from history: {e}")
        return TEAM_HFA_ADJUSTMENTS.get(team, 1.0)


# =============================================================================
# COMBINED SITUATIONAL FEATURES
# =============================================================================

def get_situational_features(
    team: str,
    opponent: str,
    season: int,
    week: int,
    is_home: bool = True,
) -> Dict[str, float]:
    """
    Get all situational features for a team in a specific game.

    Args:
        team: Team abbreviation
        opponent: Opponent team abbreviation
        season: Season year
        week: Week number
        is_home: Whether team is playing at home

    Returns:
        Dictionary with:
        - rest_days: Days since last game
        - rest_advantage: Rest differential vs opponent
        - hfa_adjustment: Team-specific HFA multiplier
        - is_primetime: Whether game is primetime (TNF/SNF/MNF)
        - is_divisional: Whether it's a divisional game
    """
    rest = get_rest_days(team, season, week)
    opp_rest = get_rest_days(opponent, season, week)
    hfa = get_hfa_adjustment(team, is_home=is_home)

    # Check for primetime/divisional from schedule
    schedule = _load_schedule()
    is_primetime = False
    is_divisional = False

    if len(schedule) > 0:
        game = schedule[
            (schedule['season'] == season) &
            (schedule['week'] == week) &
            ((schedule['home_team'] == team) | (schedule['away_team'] == team))
        ]

        if len(game) > 0:
            game = game.iloc[0]
            # Check gametime for primetime (after 6 PM ET on weekends, any weeknight)
            if 'gametime' in game and pd.notna(game.get('gametime')):
                gametime = str(game['gametime'])
                # Simple heuristic: games after 7 PM are primetime
                try:
                    hour = int(gametime.split(':')[0])
                    is_primetime = hour >= 19 or hour <= 3  # 7 PM+ or late night
                except (ValueError, TypeError, IndexError):
                    pass

            # Check for divisional game
            if 'div_game' in game:
                is_divisional = bool(game.get('div_game', 0))

    return {
        'rest_days': rest,
        'rest_advantage': rest - opp_rest,
        'hfa_adjustment': hfa,
        'is_primetime': 1.0 if is_primetime else 0.0,
        'is_divisional': 1.0 if is_divisional else 0.0,
    }


def clear_cache() -> None:
    """Clear cached schedule data."""
    _load_schedule.cache_clear()
