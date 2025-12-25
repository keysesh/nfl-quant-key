"""
Defense-vs-Position Statistics for V24 Defensive Matchup Analysis.

Calculates what each defense allows to each position role:
- WR1 yards/receptions allowed
- WR2 yards allowed
- RB1 rushing yards allowed
- TE1 yards allowed
- Over/under rates at common line buckets

Uses join of weekly_stats.parquet (has opponent_team + stats) with
position role data to answer: "What did Houston allow to WR1s this season?"

All calculations use shift(1) EWMA for no data leakage.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Module-level cache
_DEFENSE_VS_POSITION_CACHE = None


def calculate_defense_vs_position_stats(
    weekly_stats: pd.DataFrame,
    position_roles: pd.DataFrame,
    n_weeks: int = 4
) -> pd.DataFrame:
    """
    Calculate what each defense allows to each position role.

    Join weekly_stats (has opponent_team, receiving_yards, receptions)
    with position_roles (has pos_rank, position) to compute:
    "What did Houston allow to WR1s this week?"

    Args:
        weekly_stats: Weekly stats DataFrame with opponent_team column
        position_roles: DataFrame with player_id, team, week, season, pos_rank, position
        n_weeks: Weeks for EWMA calculation

    Returns:
        DataFrame with:
        - defense_team, pos_abb, pos_rank, week, season
        - receiving_yards_allowed, receptions_allowed, targets_allowed
        - rushing_yards_allowed, carries_allowed
        - *_allowed_trailing (EWMA with shift(1))
    """
    logger.info("Calculating defense-vs-position stats...")

    if len(weekly_stats) == 0 or len(position_roles) == 0:
        logger.warning("Empty input data for defense-vs-position calculation")
        return pd.DataFrame()

    # Ensure we have needed columns
    required_stats_cols = ['player_id', 'team', 'opponent_team', 'week', 'season', 'position']
    missing = [c for c in required_stats_cols if c not in weekly_stats.columns]
    if missing:
        logger.error(f"Missing columns in weekly_stats: {missing}")
        return pd.DataFrame()

    # Join weekly stats with position roles
    # Position roles tell us if this player was WR1, WR2, etc.
    stats_with_roles = weekly_stats.merge(
        position_roles[['player_id', 'team', 'week', 'season', 'pos_rank', 'position_role']],
        on=['player_id', 'team', 'week', 'season'],
        how='left'
    )

    # Filter to players with known roles (WR1, WR2, WR3, RB1, RB2, TE1, TE2)
    stats_with_roles = stats_with_roles[stats_with_roles['pos_rank'].notna()]
    stats_with_roles['pos_rank'] = stats_with_roles['pos_rank'].astype(int)

    if len(stats_with_roles) == 0:
        logger.warning("No players with known position roles")
        return pd.DataFrame()

    logger.info(f"Found {len(stats_with_roles)} player-games with position roles")

    # Aggregate what each defense allowed to each position role per game
    # Group by opponent_team (the defense), position, pos_rank, week, season
    agg_cols = {
        'receiving_yards': 'sum',
        'receptions': 'sum',
        'targets': 'sum',
        'rushing_yards': 'sum',
        'carries': 'sum'
    }

    # Filter to columns that exist
    agg_cols = {k: v for k, v in agg_cols.items() if k in stats_with_roles.columns}

    defense_allowed = stats_with_roles.groupby(
        ['opponent_team', 'position', 'pos_rank', 'week', 'season']
    ).agg(agg_cols).reset_index()

    # Rename opponent_team to defense_team for clarity
    defense_allowed = defense_allowed.rename(columns={'opponent_team': 'defense_team'})

    # Sort for trailing calculation
    defense_allowed = defense_allowed.sort_values(
        ['defense_team', 'position', 'pos_rank', 'season', 'week']
    )

    # Calculate trailing EWMA with shift(1) for no leakage
    for stat in agg_cols.keys():
        if stat in defense_allowed.columns:
            defense_allowed[f'{stat}_allowed_trailing'] = (
                defense_allowed
                .groupby(['defense_team', 'position', 'pos_rank'])[stat]
                .transform(lambda x: x.shift(1).ewm(span=n_weeks, min_periods=1).mean())
            )

    # Calculate defense-vs-position vs league average (z-scores)
    for stat in ['receiving_yards', 'rushing_yards', 'receptions']:
        if stat in defense_allowed.columns:
            league_avg = defense_allowed[stat].mean()
            league_std = defense_allowed[stat].std()

            if league_std > 0:
                trailing_col = f'{stat}_allowed_trailing'
                if trailing_col in defense_allowed.columns:
                    defense_allowed[f'{stat}_vs_avg'] = (
                        (defense_allowed[trailing_col] - league_avg) / league_std
                    )
            else:
                defense_allowed[f'{stat}_vs_avg'] = 0.0

    logger.info(f"Calculated defense stats for {defense_allowed['defense_team'].nunique()} teams")

    return defense_allowed


def pivot_defense_vs_position(
    defense_vs_position: pd.DataFrame,
    positions_to_include: List[str] = None
) -> pd.DataFrame:
    """
    Pivot defense-vs-position data to wide format for easy merge.

    Creates columns like:
    - opp_wr1_yards_allowed_trailing
    - opp_wr2_yards_allowed_trailing
    - opp_rb1_yards_allowed_trailing
    - opp_te1_yards_allowed_trailing

    Args:
        defense_vs_position: Long-form defense vs position stats
        positions_to_include: List of positions to include (default: WR, RB, TE)

    Returns:
        Wide DataFrame indexed by defense_team, week, season
    """
    if len(defense_vs_position) == 0:
        return pd.DataFrame()

    if positions_to_include is None:
        positions_to_include = ['WR', 'RB', 'TE']

    df = defense_vs_position[
        defense_vs_position['position'].isin(positions_to_include)
    ].copy()

    # Filter to pos_rank 1-3 for WR, 1-2 for others
    df = df[
        ((df['position'] == 'WR') & (df['pos_rank'] <= 3)) |
        ((df['position'] != 'WR') & (df['pos_rank'] <= 2))
    ]

    # Create position_rank label
    df['pos_label'] = df['position'].str.lower() + df['pos_rank'].astype(str)

    # Columns to pivot
    stat_cols = [c for c in df.columns if '_trailing' in c or '_vs_avg' in c]

    result_dfs = []

    for stat_col in stat_cols:
        pivot = df.pivot_table(
            index=['defense_team', 'week', 'season'],
            columns='pos_label',
            values=stat_col,
            aggfunc='first'
        ).reset_index()

        # Rename columns with prefix
        stat_name = stat_col.replace('_allowed_trailing', '').replace('_vs_avg', '_vs_avg')
        pivot.columns = [
            f'opp_{c}_{stat_col}' if c not in ['defense_team', 'week', 'season'] else c
            for c in pivot.columns
        ]

        result_dfs.append(pivot)

    if len(result_dfs) == 0:
        return pd.DataFrame()

    # Merge all stat pivots
    result = result_dfs[0]
    for df in result_dfs[1:]:
        result = result.merge(df, on=['defense_team', 'week', 'season'], how='outer')

    return result


def get_defense_vs_position_for_matchup(
    pivoted_defense: pd.DataFrame,
    opponent: str,
    week: int,
    season: int,
    position: str,
    pos_rank: int
) -> Dict:
    """
    Get defense-vs-position stats for a specific matchup.

    Args:
        pivoted_defense: Wide-format defense vs position data
        opponent: Opponent team abbreviation
        week: Current week
        season: Season year
        position: Position (WR, RB, TE)
        pos_rank: Position rank (1, 2, 3)

    Returns:
        Dict with defense-vs-position metrics
    """
    result = {
        'opp_position_yards_allowed_trailing': np.nan,
        'opp_position_volume_allowed_trailing': np.nan,
        'opp_position_target_share_trailing': np.nan,
        'has_position_defense_context': False
    }

    if len(pivoted_defense) == 0:
        return result

    pos_label = f"{position.lower()}{pos_rank}"

    # Look up the defense data for this opponent/week
    matchup_row = pivoted_defense[
        (pivoted_defense['defense_team'] == opponent) &
        (pivoted_defense['week'] == week) &
        (pivoted_defense['season'] == season)
    ]

    if len(matchup_row) == 0:
        # Try previous week if current not available
        matchup_row = pivoted_defense[
            (pivoted_defense['defense_team'] == opponent) &
            (pivoted_defense['week'] == week - 1) &
            (pivoted_defense['season'] == season)
        ]

    if len(matchup_row) == 0:
        return result

    row = matchup_row.iloc[0]

    # Get yards allowed trailing
    yards_col = f'opp_{pos_label}_receiving_yards_allowed_trailing'
    if position == 'RB':
        yards_col = f'opp_{pos_label}_rushing_yards_allowed_trailing'

    if yards_col in row.index and pd.notna(row[yards_col]):
        result['opp_position_yards_allowed_trailing'] = row[yards_col]

    # Get volume (receptions or carries)
    if position in ['WR', 'TE']:
        volume_col = f'opp_{pos_label}_receptions_allowed_trailing'
    else:
        volume_col = f'opp_{pos_label}_carries_allowed_trailing'

    if volume_col in row.index and pd.notna(row[volume_col]):
        result['opp_position_volume_allowed_trailing'] = row[volume_col]

    # Get targets for target share
    targets_col = f'opp_{pos_label}_targets_allowed_trailing'
    if targets_col in row.index and pd.notna(row[targets_col]):
        result['opp_position_target_share_trailing'] = row[targets_col]

    result['has_position_defense_context'] = (
        pd.notna(result['opp_position_yards_allowed_trailing']) or
        pd.notna(result['opp_position_volume_allowed_trailing'])
    )

    return result


def calculate_position_over_rate(
    historical_props: pd.DataFrame,
    position_roles: pd.DataFrame,
    defense_team: str,
    position_role: str,
    stat_type: str,
    line_bucket: float,
    bucket_range: float = 5.0
) -> float:
    """
    Calculate historical over rate for a position role vs a defense at a line bucket.

    Example:
    calculate_position_over_rate(props, roles, "HOU", "WR1", "receiving_yards", 69.5)
    → 0.42 (42% of WR1s went over 69.5 vs Houston)

    Args:
        historical_props: Historical props with actuals
        position_roles: Position role assignments
        defense_team: Defense team abbreviation
        position_role: Position role (WR1, WR2, etc.)
        stat_type: Stat type (receiving_yards, receptions, rushing_yards)
        line_bucket: Line to calculate over rate at
        bucket_range: +/- range around line bucket

    Returns:
        Over rate (0-1), or 0.5 if insufficient data
    """
    if len(historical_props) == 0 or len(position_roles) == 0:
        return 0.5

    # Map stat_type to actual column
    stat_col_map = {
        'receiving_yards': 'receiving_yards',
        'player_reception_yds': 'receiving_yards',
        'receptions': 'receptions',
        'player_receptions': 'receptions',
        'rushing_yards': 'rushing_yards',
        'player_rush_yds': 'rushing_yards',
    }
    actual_col = stat_col_map.get(stat_type, stat_type)

    # Need to join props with position roles
    # This requires player_id in props
    if 'player_id' not in historical_props.columns:
        logger.debug("No player_id in historical props for over rate calculation")
        return 0.5

    # Join with position roles
    props_with_role = historical_props.merge(
        position_roles[['player_id', 'team', 'week', 'season', 'position_role']],
        on=['player_id', 'team', 'week', 'season'],
        how='left'
    )

    # Filter to:
    # - This defense as opponent
    # - This position role
    # - Similar line range
    # - Has actual stat
    relevant = props_with_role[
        (props_with_role['opponent'] == defense_team) &
        (props_with_role['position_role'] == position_role) &
        (abs(props_with_role['line'] - line_bucket) <= bucket_range)
    ]

    if actual_col not in relevant.columns:
        return 0.5

    relevant = relevant[relevant[actual_col].notna()]

    if len(relevant) < 3:
        return 0.5  # Not enough data

    over_rate = (relevant[actual_col] > relevant['line']).mean()
    return float(over_rate)


def add_defense_vs_position_features(
    df: pd.DataFrame,
    defense_vs_position: pd.DataFrame,
    position_roles: pd.DataFrame,
    opponent_col: str = 'opponent'
) -> pd.DataFrame:
    """
    Add defense-vs-position features to a DataFrame.

    Args:
        df: DataFrame with opponent, week, season columns
        defense_vs_position: Pivoted defense vs position stats
        position_roles: Position role assignments
        opponent_col: Name of opponent column in df

    Returns:
        DataFrame with position matchup features added
    """
    df = df.copy()

    # Initialize feature columns
    new_cols = [
        'pos_rank',
        'is_starter',
        'opp_position_yards_allowed_trailing',
        'opp_position_volume_allowed_trailing',
        'position_role_x_opp_yards',
        'has_position_context'
    ]

    for col in new_cols:
        if col not in df.columns:
            df[col] = np.nan if 'has_' not in col else 0

    if len(defense_vs_position) == 0 or len(position_roles) == 0:
        return df

    # Join position roles
    if 'player_id' in df.columns:
        df = df.merge(
            position_roles[['player_id', 'team', 'week', 'season', 'pos_rank', 'is_starter', 'position']],
            on=['player_id', 'team', 'week', 'season'],
            how='left',
            suffixes=('', '_role')
        )

    # For each row, lookup defense-vs-position stats
    # This is vectorized via merge

    # First, create lookup key in defense_vs_position
    # We need to match: opponent -> defense_team, week, season, position, pos_rank

    # Merge defense stats based on opponent
    df['lookup_key'] = df[opponent_col].astype(str) + '_' + df['week'].astype(str) + '_' + df['season'].astype(str)

    defense_vs_position = defense_vs_position.copy()
    defense_vs_position['lookup_key'] = (
        defense_vs_position['defense_team'].astype(str) + '_' +
        defense_vs_position['week'].astype(str) + '_' +
        defense_vs_position['season'].astype(str)
    )

    # For receiving markets (WR/TE), use receiving_yards_allowed_trailing
    # For rushing markets (RB), use rushing_yards_allowed_trailing

    # Get position-specific columns from pivoted data
    if 'position' in df.columns and 'pos_rank' in df.columns:
        for idx, row in df.iterrows():
            pos = row.get('position', 'WR')
            rank = row.get('pos_rank', 1)

            if pd.isna(pos) or pd.isna(rank):
                continue

            matchup_stats = get_defense_vs_position_for_matchup(
                defense_vs_position,
                row[opponent_col],
                row['week'],
                row['season'],
                pos,
                int(rank)
            )

            for key, val in matchup_stats.items():
                if key in df.columns:
                    df.at[idx, key] = val

    # Calculate interaction feature
    df['position_role_x_opp_yards'] = (
        df['pos_rank'].fillna(2) * df['opp_position_yards_allowed_trailing'].fillna(0)
    )

    # Set has_position_context flag
    df['has_position_context'] = (
        df['pos_rank'].notna() &
        df['opp_position_yards_allowed_trailing'].notna()
    ).astype(int)

    # Clean up temp columns
    df = df.drop(columns=['lookup_key'], errors='ignore')

    return df


def build_defense_vs_position_cache(
    weekly_stats: pd.DataFrame,
    position_roles: pd.DataFrame,
    output_path: Path = None
) -> pd.DataFrame:
    """
    Build and cache defense-vs-position statistics.

    Args:
        weekly_stats: Weekly stats DataFrame
        position_roles: Position role assignments
        output_path: Optional path to save cache

    Returns:
        Pivoted defense-vs-position DataFrame
    """
    global _DEFENSE_VS_POSITION_CACHE

    # Calculate long-form stats
    defense_stats = calculate_defense_vs_position_stats(weekly_stats, position_roles)

    if len(defense_stats) == 0:
        return pd.DataFrame()

    # Pivot to wide format
    pivoted = pivot_defense_vs_position(defense_stats)

    # Cache
    _DEFENSE_VS_POSITION_CACHE = pivoted

    # Save if path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pivoted.to_parquet(output_path, index=False)
        logger.info(f"Saved defense-vs-position cache to {output_path}")

    return pivoted


def load_defense_vs_position_cache(cache_path: Path = None) -> pd.DataFrame:
    """Load cached defense-vs-position data."""
    global _DEFENSE_VS_POSITION_CACHE

    if _DEFENSE_VS_POSITION_CACHE is not None:
        return _DEFENSE_VS_POSITION_CACHE

    if cache_path is None:
        cache_path = PROJECT_ROOT / 'data' / 'cache' / 'defense_vs_position.parquet'

    if cache_path.exists():
        _DEFENSE_VS_POSITION_CACHE = pd.read_parquet(cache_path)
        logger.info(f"Loaded defense-vs-position cache: {len(_DEFENSE_VS_POSITION_CACHE)} rows")
        return _DEFENSE_VS_POSITION_CACHE

    return pd.DataFrame()


def clear_cache():
    """Clear the module-level cache."""
    global _DEFENSE_VS_POSITION_CACHE
    _DEFENSE_VS_POSITION_CACHE = None
    logger.info("Defense-vs-position cache cleared")
