"""
V25 Team Synergy Feature Extractor for Batch Processing.

This module provides vectorized extraction of team synergy features
for use in the model training pipeline.

Features extracted:
- team_synergy_multiplier: Compound multiplier from all synergy/degradation conditions
- oline_health_score_v25: Weighted O-line unit health percentage
- wr_corps_health: Weighted WR corps health (WR1=0.5, WR2=0.3, WR3=0.2)
- has_synergy_bonus: Flag indicating positive synergy active
- cascade_efficiency_boost: Max efficiency boost from teammate returning
- wr_coverage_reduction: Coverage reduction due to healthy WR corps
- returning_player_count: Number of key players returning from injury
- has_synergy_context: Flag indicating HIGH confidence synergy data

Usage:
    from nfl_quant.features.team_synergy_extractor import extract_team_synergy_features

    df = extract_team_synergy_features(df, week=15, season=2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# V25 synergy feature names
V25_SYNERGY_FEATURES = [
    'team_synergy_multiplier',
    'oline_health_score_v25',
    'wr_corps_health',
    'has_synergy_bonus',
    'cascade_efficiency_boost',
    'wr_coverage_reduction',
    'returning_player_count',
    'has_synergy_context',
]

# Default values for synergy features (neutral)
SYNERGY_DEFAULTS = {
    'team_synergy_multiplier': 1.0,
    'oline_health_score_v25': 0.85,
    'wr_corps_health': 0.80,
    'has_synergy_bonus': 0.0,
    'cascade_efficiency_boost': 0.0,
    'wr_coverage_reduction': 0.0,
    'returning_player_count': 0.0,
    'has_synergy_context': 0.0,
}


def _load_injuries_data() -> Optional[pd.DataFrame]:
    """Load current injuries data."""
    injuries_paths = [
        PROJECT_ROOT / 'data' / 'injuries' / 'injuries_latest.csv',
        PROJECT_ROOT / 'data' / 'injuries' / 'current_injuries.csv',
    ]

    for path in injuries_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                logger.debug(f"Loaded injuries from {path}: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Failed to load injuries from {path}: {e}")

    logger.warning("No injuries data found")
    return None


def _load_rosters_data(season: int = 2025) -> Optional[pd.DataFrame]:
    """Load roster data from NFLverse."""
    paths = [
        PROJECT_ROOT / 'data' / 'nflverse' / 'rosters.parquet',
        PROJECT_ROOT / 'data' / 'nflverse' / f'rosters_{season}.parquet',
    ]

    for path in paths:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                # Filter to current season if possible
                if 'season' in df.columns:
                    df = df[df['season'] == season]
                logger.debug(f"Loaded rosters from {path}: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Failed to load rosters from {path}: {e}")

    logger.warning("No roster data found")
    return None


def _load_depth_charts_data(season: int = 2025, week: int = None) -> Optional[pd.DataFrame]:
    """Load depth chart data using canonical loader."""
    try:
        from nfl_quant.data.depth_chart_loader import get_depth_charts
        df = get_depth_charts(season=season, week=week)
        logger.debug(f"Loaded depth charts: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Failed to load depth charts: {e}")
        return None


def _load_weekly_stats_data(season: int = 2025) -> Optional[pd.DataFrame]:
    """Load weekly stats data from NFLverse."""
    paths = [
        PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet',
        PROJECT_ROOT / 'data' / 'nflverse' / f'weekly_stats_{season}.parquet',
    ]

    for path in paths:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.debug(f"Loaded weekly stats from {path}: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Failed to load weekly stats from {path}: {e}")

    logger.warning("No weekly stats data found")
    return None


def _extract_features_from_synergy_result(result: Any) -> Dict[str, float]:
    """
    Extract 8 V25 features from a SynergyResult.

    Args:
        result: SynergyResult from calculate_team_synergy_adjustment

    Returns:
        Dict mapping feature name to value
    """
    # Get O-line health from unit scores
    oline_health = 0.85  # default
    if hasattr(result, 'unit_health_scores') and result.unit_health_scores:
        oline_unit = result.unit_health_scores.get('oline')
        if oline_unit and hasattr(oline_unit, 'health_pct'):
            oline_health = oline_unit.health_pct

    # Calculate WR corps health from unit scores
    wr_corps_health = 0.80  # default
    if hasattr(result, 'unit_health_scores') and result.unit_health_scores:
        wr_unit = result.unit_health_scores.get('wr')
        if wr_unit and hasattr(wr_unit, 'health_pct'):
            wr_corps_health = wr_unit.health_pct

    # Get max cascade efficiency boost
    cascade_boost = 0.0
    coverage_reduction = 0.0
    if hasattr(result, 'player_cascades') and result.player_cascades:
        for player, effects in result.player_cascades.items():
            if 'efficiency_boost' in effects:
                boost = effects['efficiency_boost'] - 1.0  # Convert from multiplier to boost
                cascade_boost = max(cascade_boost, boost)
            if 'coverage_reduction' in effects:
                coverage_reduction = max(coverage_reduction, effects['coverage_reduction'])

    # Count returning players
    returning_count = len(result.player_cascades) if hasattr(result, 'player_cascades') else 0

    return {
        'team_synergy_multiplier': getattr(result, 'team_multiplier', 1.0),
        'oline_health_score_v25': oline_health,
        'wr_corps_health': wr_corps_health,
        'has_synergy_bonus': 1.0 if (hasattr(result, 'active_synergies') and result.active_synergies) else 0.0,
        'cascade_efficiency_boost': cascade_boost,
        'wr_coverage_reduction': coverage_reduction,
        'returning_player_count': float(returning_count),
        'has_synergy_context': 1.0 if getattr(result, 'confidence', 'LOW') == 'HIGH' else 0.0,
    }


@lru_cache(maxsize=64)
def _get_team_synergy_cached(team: str, week: int, season: int) -> Dict[str, float]:
    """
    Get synergy features for a team (cached).

    Uses lru_cache to avoid recalculating for the same team/week combo.
    """
    try:
        from nfl_quant.features.team_synergy import (
            calculate_team_synergy_adjustment,
            load_player_statuses_from_injuries,
            detect_returning_players_from_stats,
        )

        # Load required data
        injuries_df = _load_injuries_data()
        rosters_df = _load_rosters_data(season)
        depth_charts_df = _load_depth_charts_data(season, week)
        weekly_stats_df = _load_weekly_stats_data(season)

        if rosters_df is None:
            logger.debug(f"No roster data for {team}, using defaults")
            return SYNERGY_DEFAULTS.copy()

        # Build player statuses
        player_statuses = load_player_statuses_from_injuries(
            injuries_df=injuries_df if injuries_df is not None else pd.DataFrame(),
            rosters_df=rosters_df,
            depth_charts_df=depth_charts_df,
            snap_counts_df=None,
            week=week,
            season=season
        )

        if not player_statuses:
            logger.debug(f"No player statuses for {team}, using defaults")
            return SYNERGY_DEFAULTS.copy()

        # Detect returning players
        returning_players = []
        if weekly_stats_df is not None:
            try:
                returning_players = detect_returning_players_from_stats(
                    weekly_stats_df, player_statuses, week, season
                )
            except Exception as e:
                logger.debug(f"Could not detect returning players: {e}")

        # Calculate synergy
        synergy_result = calculate_team_synergy_adjustment(
            player_statuses=player_statuses,
            team=team,
            returning_players=returning_players,
            include_cascades=True
        )

        return _extract_features_from_synergy_result(synergy_result)

    except ImportError as e:
        logger.warning(f"Team synergy module not available: {e}")
        return SYNERGY_DEFAULTS.copy()
    except Exception as e:
        logger.debug(f"Synergy calculation failed for {team}: {e}")
        return SYNERGY_DEFAULTS.copy()


def extract_team_synergy_features(
    df: pd.DataFrame,
    week: int = None,
    season: int = 2025,
    team_col: str = 'recent_team'
) -> pd.DataFrame:
    """
    Add V25 synergy features to a predictions/training DataFrame.

    This function calculates team-level synergy metrics and maps them
    to individual player rows based on their team.

    Args:
        df: DataFrame with player predictions/features
        week: Current NFL week (uses max from data if not specified)
        season: NFL season year
        team_col: Column containing team abbreviation

    Returns:
        DataFrame with 8 synergy feature columns added
    """
    if len(df) == 0:
        return df

    df = df.copy()

    # Determine team column
    if team_col not in df.columns:
        for alt_col in ['team', 'recent_team', 'player_team']:
            if alt_col in df.columns:
                team_col = alt_col
                break
        else:
            logger.warning("No team column found, adding default synergy features")
            for feature in V25_SYNERGY_FEATURES:
                df[feature] = SYNERGY_DEFAULTS[feature]
            return df

    # Determine week
    if week is None:
        if 'week' in df.columns:
            week = int(df['week'].max())
        else:
            week = 15  # Default to week 15 if unknown

    # Get unique teams
    teams = df[team_col].dropna().unique()
    logger.info(f"    Calculating V25 synergy features for {len(teams)} teams (week {week})")

    # Calculate synergy for each team
    team_synergy_cache = {}
    for team in teams:
        if pd.isna(team):
            continue
        team_synergy_cache[str(team)] = _get_team_synergy_cached(str(team), week, season)

    # Initialize feature columns with defaults
    for feature in V25_SYNERGY_FEATURES:
        df[feature] = SYNERGY_DEFAULTS[feature]

    # Map features to rows
    for idx, row in df.iterrows():
        team = row.get(team_col)
        if pd.notna(team) and str(team) in team_synergy_cache:
            features = team_synergy_cache[str(team)]
            for feature, value in features.items():
                df.loc[idx, feature] = value

    # Log summary
    teams_with_data = sum(1 for t in teams if pd.notna(t) and str(t) in team_synergy_cache
                          and team_synergy_cache[str(t)].get('has_synergy_context', 0) > 0)
    logger.info(f"    V25 synergy: {teams_with_data}/{len(teams)} teams with HIGH confidence data")

    return df


def clear_synergy_cache():
    """Clear the synergy calculation cache."""
    _get_team_synergy_cached.cache_clear()
    logger.info("Synergy cache cleared")


if __name__ == '__main__':
    # Test the extractor
    print("V25 Team Synergy Feature Extractor")
    print("=" * 50)

    # Test with sample data
    test_df = pd.DataFrame({
        'player_name': ['Patrick Mahomes', 'Travis Kelce', 'Tyreek Hill'],
        'recent_team': ['KC', 'KC', 'MIA'],
        'week': [15, 15, 15],
    })

    result = extract_team_synergy_features(test_df, week=15, season=2025)

    print("\nExtracted features:")
    for feature in V25_SYNERGY_FEATURES:
        print(f"  {feature}: {result[feature].tolist()}")
