"""
Coverage Tendencies for V24 Defensive Matchup Analysis.

Calculates man/zone coverage tendencies from participation.parquet:
- Man coverage rate by team
- Zone coverage rate by team
- Slot funnel score (how much defense allows to slot vs outside)

Uses participation.parquet field: defense_man_zone_type
Values: 'MAN_COVERAGE', 'ZONE_COVERAGE', or NaN

All calculations use shift(1) EWMA for no data leakage.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Module-level caches
_PARTICIPATION_CACHE = None
_COVERAGE_TENDENCIES_CACHE = None


def load_participation(force_reload: bool = False) -> pd.DataFrame:
    """
    Load participation parquet file with caching.

    Key columns: defense_man_zone_type, defense_coverage_type
    """
    global _PARTICIPATION_CACHE

    if _PARTICIPATION_CACHE is not None and not force_reload:
        return _PARTICIPATION_CACHE

    path = PROJECT_ROOT / 'data' / 'nflverse' / 'participation.parquet'

    if not path.exists():
        logger.warning(f"Participation data not found at {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    _PARTICIPATION_CACHE = df
    logger.info(f"Loaded participation: {len(df):,} records")

    return df


def calculate_team_coverage_tendencies(
    participation: pd.DataFrame = None,
    n_weeks: int = 4
) -> pd.DataFrame:
    """
    Calculate each team's man/zone coverage tendencies.

    Args:
        participation: Participation DataFrame (loaded if None)
        n_weeks: Weeks for EWMA calculation

    Returns:
        DataFrame with:
        - defense_team, week, season
        - man_coverage_rate (% of snaps in man coverage)
        - zone_coverage_rate
        - *_trailing (EWMA with shift(1))
    """
    logger.info("Calculating team coverage tendencies...")

    if participation is None:
        participation = load_participation()

    if len(participation) == 0:
        logger.warning("No participation data available")
        return pd.DataFrame()

    # Check for coverage type column
    if 'defense_man_zone_type' not in participation.columns:
        logger.warning("defense_man_zone_type column not found in participation data")
        return pd.DataFrame()

    # Filter to plays with coverage type data
    coverage_plays = participation[participation['defense_man_zone_type'].notna()].copy()

    if len(coverage_plays) == 0:
        logger.warning("No plays with coverage type data")
        return pd.DataFrame()

    logger.info(f"Found {len(coverage_plays)} plays with coverage type data")

    # Extract week and season from nflverse_game_id if not present
    # Format: 2024_01_MIN_NYG
    if 'week' not in coverage_plays.columns or 'season' not in coverage_plays.columns:
        if 'nflverse_game_id' in coverage_plays.columns:
            game_parts = coverage_plays['nflverse_game_id'].str.split('_')
            coverage_plays['season'] = game_parts.str[0].astype(float)
            coverage_plays['week'] = game_parts.str[1].astype(float)

    # Need to determine which team is on defense
    # The 'possession_team' is on offense, so defense is the other team

    # Extract teams from game_id
    if 'defteam' not in coverage_plays.columns:
        if 'nflverse_game_id' in coverage_plays.columns:
            # Game ID format: 2024_01_TEAM1_TEAM2
            game_parts = coverage_plays['nflverse_game_id'].str.split('_')
            home_team = game_parts.str[2]
            away_team = game_parts.str[3]

            # Defense is the team that doesn't have possession
            coverage_plays['defteam'] = np.where(
                coverage_plays['possession_team'] == home_team,
                away_team,
                home_team
            )

    if 'defteam' not in coverage_plays.columns:
        logger.warning("Could not determine defensive team")
        return pd.DataFrame()

    # Encode coverage types
    coverage_plays['is_man'] = (coverage_plays['defense_man_zone_type'] == 'MAN_COVERAGE').astype(int)
    coverage_plays['is_zone'] = (coverage_plays['defense_man_zone_type'] == 'ZONE_COVERAGE').astype(int)

    # Calculate coverage rates by team/game
    coverage_rates = coverage_plays.groupby(
        ['defteam', 'week', 'season']
    ).agg({
        'is_man': 'mean',
        'is_zone': 'mean',
        'defense_man_zone_type': 'count'  # Number of plays with data
    }).reset_index()

    coverage_rates = coverage_rates.rename(columns={
        'defteam': 'defense_team',
        'is_man': 'man_coverage_rate',
        'is_zone': 'zone_coverage_rate',
        'defense_man_zone_type': 'coverage_plays'
    })

    # Sort for trailing calculation
    coverage_rates = coverage_rates.sort_values(['defense_team', 'season', 'week'])

    # Calculate trailing EWMA with shift(1) for no leakage
    for col in ['man_coverage_rate', 'zone_coverage_rate']:
        coverage_rates[f'{col}_trailing'] = (
            coverage_rates
            .groupby('defense_team')[col]
            .transform(lambda x: x.shift(1).ewm(span=n_weeks, min_periods=1).mean())
        )

    logger.info(f"Calculated coverage tendencies for {coverage_rates['defense_team'].nunique()} teams")

    return coverage_rates


def calculate_slot_funnel_score(
    weekly_stats: pd.DataFrame,
    ngs_receiving: pd.DataFrame,
    n_weeks: int = 4
) -> pd.DataFrame:
    """
    Calculate "slot funnel" score: how much does this defense allow to slot vs outside receivers?

    High score = defense allows more to slot (funnel effect)

    Args:
        weekly_stats: Weekly stats DataFrame
        ngs_receiving: NGS receiving DataFrame with avg_cushion
        n_weeks: Weeks for EWMA calculation

    Returns:
        DataFrame with defense_team, week, season, slot_funnel_score
    """
    logger.info("Calculating slot funnel scores...")

    if len(weekly_stats) == 0:
        return pd.DataFrame()

    # Join stats with NGS to get slot classification
    wr_stats = weekly_stats[weekly_stats['position'] == 'WR'].copy()

    if len(ngs_receiving) > 0:
        # Classify receivers as slot based on NGS cushion
        ngs_avg = ngs_receiving.groupby(['player_gsis_id', 'season']).agg({
            'avg_cushion': 'mean',
            'avg_intended_air_yards': 'mean'
        }).reset_index()

        ngs_avg['is_slot'] = (
            (ngs_avg['avg_cushion'] > 5.5) &
            (ngs_avg['avg_intended_air_yards'] < 8.0)
        )

        # Join to stats
        wr_stats = wr_stats.merge(
            ngs_avg[['player_gsis_id', 'season', 'is_slot']].rename(
                columns={'player_gsis_id': 'player_id'}
            ),
            on=['player_id', 'season'],
            how='left'
        )
        wr_stats['is_slot'] = wr_stats['is_slot'].fillna(False)
    else:
        # Without NGS, estimate from depth chart position (fallback)
        # WR3+ are often slot receivers
        wr_stats['is_slot'] = False

    # Calculate yards allowed to slot vs outside by defense
    # Group without unstacking to preserve all columns
    slot_agg = wr_stats.groupby(
        ['opponent_team', 'is_slot', 'week', 'season']
    )['receiving_yards'].sum().reset_index()

    # Pivot to get slot vs outside columns
    slot_pivot = slot_agg.pivot_table(
        index=['opponent_team', 'week', 'season'],
        columns='is_slot',
        values='receiving_yards',
        fill_value=0
    ).reset_index()

    # Flatten column names and rename
    slot_pivot.columns.name = None
    col_mapping = {}
    for col in slot_pivot.columns:
        if col == True:
            col_mapping[col] = 'slot_yards_allowed'
        elif col == False:
            col_mapping[col] = 'outside_yards_allowed'

    slot_pivot = slot_pivot.rename(columns=col_mapping)

    # Ensure both columns exist
    if 'slot_yards_allowed' not in slot_pivot.columns:
        slot_pivot['slot_yards_allowed'] = 0
    if 'outside_yards_allowed' not in slot_pivot.columns:
        slot_pivot['outside_yards_allowed'] = 0

    slot_vs_outside = slot_pivot.rename(columns={'opponent_team': 'defense_team'})

    # Calculate slot funnel score (proportion of yards to slot)
    total_yards = slot_vs_outside['slot_yards_allowed'] + slot_vs_outside['outside_yards_allowed']
    slot_vs_outside['slot_funnel_score'] = np.where(
        total_yards > 0,
        slot_vs_outside['slot_yards_allowed'] / total_yards,
        0.5  # Default to neutral
    )

    # Sort and calculate trailing
    slot_vs_outside = slot_vs_outside.sort_values(['defense_team', 'season', 'week'])

    slot_vs_outside['slot_funnel_score_trailing'] = (
        slot_vs_outside
        .groupby('defense_team')['slot_funnel_score']
        .transform(lambda x: x.shift(1).ewm(span=n_weeks, min_periods=1).mean())
    )

    logger.info(f"Calculated slot funnel scores for {slot_vs_outside['defense_team'].nunique()} teams")

    return slot_vs_outside[['defense_team', 'week', 'season', 'slot_funnel_score', 'slot_funnel_score_trailing']]


def calculate_man_coverage_adjustment(
    is_wr1: bool,
    is_slot: bool,
    opp_man_coverage_rate: float
) -> float:
    """
    Adjust projection based on opponent's man coverage tendency.

    - WR1 vs high man coverage = slight reduction (CB1 shadow more likely)
    - Slot receivers vs high man coverage = slight boost (harder to bracket)

    Args:
        is_wr1: Is player the team's WR1?
        is_slot: Does player align in slot?
        opp_man_coverage_rate: Opponent's man coverage rate (0-1)

    Returns:
        Adjustment multiplier (0.93-1.07 range)
    """
    if pd.isna(opp_man_coverage_rate):
        return 1.0

    # Zone-heavy defenses (< 30% man) = no adjustment
    if opp_man_coverage_rate < 0.30:
        return 1.0

    # Calculate adjustment based on man rate excess over baseline
    man_excess = opp_man_coverage_rate - 0.30

    if is_wr1 and not is_slot:
        # Outside WR1 vs man-heavy = harder matchup (CB1 shadow)
        # Max 7% reduction at 100% man rate
        adjustment = 1.0 - (man_excess * 0.10)
        return max(0.93, adjustment)

    if is_slot:
        # Slot receivers may benefit vs man-heavy (harder to bracket, can rub routes)
        # Max 7% boost at 100% man rate
        adjustment = 1.0 + (man_excess * 0.10)
        return min(1.07, adjustment)

    # WR2/WR3 outside receivers - slight reduction but less than WR1
    if not is_slot:
        adjustment = 1.0 - (man_excess * 0.05)
        return max(0.96, adjustment)

    return 1.0


def add_coverage_features(
    df: pd.DataFrame,
    coverage_tendencies: pd.DataFrame,
    slot_funnel: pd.DataFrame = None,
    opponent_col: str = 'opponent'
) -> pd.DataFrame:
    """
    Add coverage tendency features to a DataFrame.

    Args:
        df: DataFrame with opponent, week, season columns
        coverage_tendencies: Coverage tendencies by team/week
        slot_funnel: Optional slot funnel scores
        opponent_col: Name of opponent column in df

    Returns:
        DataFrame with coverage features added
    """
    df = df.copy()

    # Initialize columns
    new_cols = [
        'opp_man_coverage_rate_trailing',
        'opp_zone_coverage_rate_trailing',
        'slot_funnel_score',
        'man_coverage_adjustment'
    ]

    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan

    if len(coverage_tendencies) == 0:
        return df

    # Merge coverage tendencies
    coverage_cols = ['defense_team', 'week', 'season', 'man_coverage_rate_trailing', 'zone_coverage_rate_trailing']
    coverage_cols = [c for c in coverage_cols if c in coverage_tendencies.columns]

    if len(coverage_cols) >= 3:
        df = df.merge(
            coverage_tendencies[coverage_cols].rename(columns={
                'defense_team': opponent_col,
                'man_coverage_rate_trailing': 'opp_man_coverage_rate_trailing',
                'zone_coverage_rate_trailing': 'opp_zone_coverage_rate_trailing'
            }),
            on=[opponent_col, 'week', 'season'],
            how='left'
        )

    # Merge slot funnel if available
    if slot_funnel is not None and len(slot_funnel) > 0:
        funnel_cols = [c for c in ['defense_team', 'week', 'season', 'slot_funnel_score_trailing']
                       if c in slot_funnel.columns]

        if len(funnel_cols) >= 3:
            df = df.merge(
                slot_funnel[funnel_cols].rename(columns={
                    'defense_team': opponent_col,
                    'slot_funnel_score_trailing': 'slot_funnel_score'
                }),
                on=[opponent_col, 'week', 'season'],
                how='left'
            )

    # Calculate man coverage adjustment
    if 'is_starter' in df.columns and 'is_slot_receiver' in df.columns:
        df['man_coverage_adjustment'] = df.apply(
            lambda row: calculate_man_coverage_adjustment(
                is_wr1=bool(row.get('is_starter', False)),
                is_slot=bool(row.get('is_slot_receiver', False)),
                opp_man_coverage_rate=row.get('opp_man_coverage_rate_trailing', np.nan)
            ),
            axis=1
        )
    else:
        df['man_coverage_adjustment'] = 1.0

    return df


def build_coverage_tendencies_cache(
    participation: pd.DataFrame = None,
    weekly_stats: pd.DataFrame = None,
    ngs_receiving: pd.DataFrame = None,
    output_path: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build and cache coverage tendencies and slot funnel scores.

    Args:
        participation: Participation DataFrame
        weekly_stats: Weekly stats DataFrame
        ngs_receiving: NGS receiving DataFrame
        output_path: Optional base path to save caches

    Returns:
        Tuple of (coverage_tendencies, slot_funnel) DataFrames
    """
    global _COVERAGE_TENDENCIES_CACHE

    # Calculate coverage tendencies
    coverage_tendencies = calculate_team_coverage_tendencies(participation)

    # Calculate slot funnel
    slot_funnel = pd.DataFrame()
    if weekly_stats is not None and ngs_receiving is not None:
        slot_funnel = calculate_slot_funnel_score(weekly_stats, ngs_receiving)

    # Cache
    _COVERAGE_TENDENCIES_CACHE = {
        'coverage': coverage_tendencies,
        'slot_funnel': slot_funnel
    }

    # Save if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(coverage_tendencies) > 0:
            coverage_tendencies.to_parquet(output_path.parent / 'coverage_tendencies.parquet', index=False)

        if len(slot_funnel) > 0:
            slot_funnel.to_parquet(output_path.parent / 'slot_funnel.parquet', index=False)

        logger.info(f"Saved coverage caches to {output_path.parent}")

    return coverage_tendencies, slot_funnel


def load_coverage_cache(cache_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached coverage data."""
    global _COVERAGE_TENDENCIES_CACHE

    if _COVERAGE_TENDENCIES_CACHE is not None:
        return _COVERAGE_TENDENCIES_CACHE['coverage'], _COVERAGE_TENDENCIES_CACHE['slot_funnel']

    if cache_dir is None:
        cache_dir = PROJECT_ROOT / 'data' / 'cache'

    coverage = pd.DataFrame()
    slot_funnel = pd.DataFrame()

    coverage_path = cache_dir / 'coverage_tendencies.parquet'
    if coverage_path.exists():
        coverage = pd.read_parquet(coverage_path)
        logger.info(f"Loaded coverage tendencies cache: {len(coverage)} rows")

    funnel_path = cache_dir / 'slot_funnel.parquet'
    if funnel_path.exists():
        slot_funnel = pd.read_parquet(funnel_path)
        logger.info(f"Loaded slot funnel cache: {len(slot_funnel)} rows")

    _COVERAGE_TENDENCIES_CACHE = {
        'coverage': coverage,
        'slot_funnel': slot_funnel
    }

    return coverage, slot_funnel


def clear_cache():
    """Clear module-level caches."""
    global _PARTICIPATION_CACHE, _COVERAGE_TENDENCIES_CACHE
    _PARTICIPATION_CACHE = None
    _COVERAGE_TENDENCIES_CACHE = None
    logger.info("Coverage tendencies caches cleared")
