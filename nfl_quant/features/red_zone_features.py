"""
Red Zone Features for TD Prediction

Parses play-by-play data to compute red zone snap allocation,
which is critical for TD prediction (85% of rushing TDs come from red zone).

Usage:
    from nfl_quant.features.red_zone_features import compute_red_zone_features
    rz_features = compute_red_zone_features(season=2024)
"""
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import DATA_DIR


def load_pbp_data(season: int) -> pd.DataFrame:
    """
    Load play-by-play data for a season.

    Args:
        season: NFL season year

    Returns:
        DataFrame with play-by-play data
    """
    # Try different path patterns
    paths_to_try = [
        DATA_DIR / 'nflverse' / 'pbp.parquet',
        DATA_DIR / 'nflverse' / f'pbp_{season}.parquet',
        DATA_DIR / 'processed' / f'pbp_{season}.parquet',
    ]

    for path in paths_to_try:
        if path.exists():
            df = pd.read_parquet(path)
            # Filter to season
            if 'season' in df.columns:
                df = df[df['season'] == season]
            return df

    raise FileNotFoundError(f"No PBP data found for season {season}")


def compute_red_zone_snaps(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute red zone snap counts per player-week.

    Red zone = opponent's 20-yard line or closer (yardline_100 <= 20).

    Args:
        pbp: Play-by-play DataFrame

    Returns:
        DataFrame with player red zone snap counts
    """
    # Filter to red zone plays (yardline_100 <= 20 means within 20 yards of endzone)
    rz_plays = pbp[pbp['yardline_100'] <= 20].copy()

    if rz_plays.empty:
        return pd.DataFrame()

    # Count rushing plays per player in red zone
    rush_rz = rz_plays[rz_plays['play_type'] == 'run'].copy()
    if not rush_rz.empty and 'rusher_player_id' in rush_rz.columns:
        rush_counts = rush_rz.groupby(
            ['rusher_player_id', 'week', 'season', 'posteam']
        ).size().reset_index(name='rz_rush_snaps')
        rush_counts = rush_counts.rename(columns={'rusher_player_id': 'player_id'})
    else:
        rush_counts = pd.DataFrame()

    # Count receiving targets per player in red zone
    pass_rz = rz_plays[rz_plays['play_type'] == 'pass'].copy()
    if not pass_rz.empty and 'receiver_player_id' in pass_rz.columns:
        rec_counts = pass_rz.groupby(
            ['receiver_player_id', 'week', 'season', 'posteam']
        ).size().reset_index(name='rz_targets')
        rec_counts = rec_counts.rename(columns={'receiver_player_id': 'player_id'})
    else:
        rec_counts = pd.DataFrame()

    # Count passing attempts per QB in red zone
    if not pass_rz.empty and 'passer_player_id' in pass_rz.columns:
        pass_counts = pass_rz.groupby(
            ['passer_player_id', 'week', 'season', 'posteam']
        ).size().reset_index(name='rz_pass_attempts')
        pass_counts = pass_counts.rename(columns={'passer_player_id': 'player_id'})
    else:
        pass_counts = pd.DataFrame()

    # Combine all red zone data
    dfs = [df for df in [rush_counts, rec_counts, pass_counts] if not df.empty]

    if not dfs:
        return pd.DataFrame()

    # Merge all together
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(
            df,
            on=['player_id', 'week', 'season', 'posteam'],
            how='outer'
        )

    # Fill NaN with 0
    for col in ['rz_rush_snaps', 'rz_targets', 'rz_pass_attempts']:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    return result


def compute_team_red_zone_totals(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total team red zone plays per week.

    Args:
        pbp: Play-by-play DataFrame

    Returns:
        DataFrame with team red zone totals
    """
    rz_plays = pbp[pbp['yardline_100'] <= 20].copy()

    if rz_plays.empty:
        return pd.DataFrame()

    # Count rush plays per team
    rush_totals = rz_plays[rz_plays['play_type'] == 'run'].groupby(
        ['posteam', 'week', 'season']
    ).size().reset_index(name='team_rz_rush_plays')

    # Count pass plays per team
    pass_totals = rz_plays[rz_plays['play_type'] == 'pass'].groupby(
        ['posteam', 'week', 'season']
    ).size().reset_index(name='team_rz_pass_plays')

    # Merge
    totals = rush_totals.merge(
        pass_totals,
        on=['posteam', 'week', 'season'],
        how='outer'
    )

    for col in ['team_rz_rush_plays', 'team_rz_pass_plays']:
        totals[col] = totals[col].fillna(0).astype(int)

    totals['team_rz_total_plays'] = totals['team_rz_rush_plays'] + totals['team_rz_pass_plays']

    return totals


def compute_red_zone_shares(
    player_rz: pd.DataFrame,
    team_rz: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute player's share of team red zone opportunities.

    Args:
        player_rz: Player red zone snap counts
        team_rz: Team red zone totals

    Returns:
        DataFrame with red zone share features
    """
    if player_rz.empty or team_rz.empty:
        return pd.DataFrame()

    # Merge player and team data
    merged = player_rz.merge(
        team_rz,
        on=['posteam', 'week', 'season'],
        how='left'
    )

    # Calculate shares
    if 'rz_rush_snaps' in merged.columns and 'team_rz_rush_plays' in merged.columns:
        merged['rz_rush_share'] = np.where(
            merged['team_rz_rush_plays'] > 0,
            merged['rz_rush_snaps'] / merged['team_rz_rush_plays'],
            0
        )
    else:
        merged['rz_rush_share'] = 0

    if 'rz_targets' in merged.columns and 'team_rz_pass_plays' in merged.columns:
        merged['rz_target_share'] = np.where(
            merged['team_rz_pass_plays'] > 0,
            merged['rz_targets'] / merged['team_rz_pass_plays'],
            0
        )
    else:
        merged['rz_target_share'] = 0

    return merged


def compute_trailing_rz_features(
    rz_data: pd.DataFrame,
    ewma_span: int = 4,
) -> pd.DataFrame:
    """
    Compute trailing red zone features with EWMA smoothing.

    Args:
        rz_data: Red zone data with shares
        ewma_span: EWMA span for smoothing

    Returns:
        DataFrame with trailing red zone features
    """
    if rz_data.empty:
        return rz_data

    # Sort by player and time
    rz_data = rz_data.sort_values(['player_id', 'season', 'week'])

    # Compute trailing features (shifted to prevent leakage)
    for col in ['rz_rush_share', 'rz_target_share', 'rz_rush_snaps', 'rz_targets']:
        if col in rz_data.columns:
            rz_data[f'trailing_{col}'] = (
                rz_data.groupby('player_id')[col]
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

    return rz_data


def compute_red_zone_features(
    season: int,
    weeks: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute all red zone features for a season.

    Main entry point for red zone feature computation.

    Args:
        season: NFL season year
        weeks: Optional list of weeks to process

    Returns:
        DataFrame with red zone features per player-week
    """
    print(f"Computing red zone features for {season}...")

    # Load PBP data
    try:
        pbp = load_pbp_data(season)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return pd.DataFrame()

    if weeks:
        pbp = pbp[pbp['week'].isin(weeks)]

    # Compute player red zone snaps
    player_rz = compute_red_zone_snaps(pbp)

    if player_rz.empty:
        print("  No red zone plays found")
        return pd.DataFrame()

    # Compute team totals
    team_rz = compute_team_red_zone_totals(pbp)

    # Compute shares
    rz_shares = compute_red_zone_shares(player_rz, team_rz)

    # Compute trailing features
    rz_features = compute_trailing_rz_features(rz_shares)

    print(f"  Computed red zone features for {len(rz_features)} player-weeks")

    return rz_features


def merge_red_zone_features(
    df: pd.DataFrame,
    rz_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge red zone features into main DataFrame.

    Args:
        df: Main DataFrame with player_id, season, week
        rz_features: Red zone features DataFrame

    Returns:
        DataFrame with red zone features merged
    """
    if rz_features.empty:
        # Add default columns
        for col in ['trailing_rz_rush_share', 'trailing_rz_target_share']:
            df[col] = 0.0
        return df

    # Standardize player ID column
    if 'player_id' not in df.columns and 'gsis_id' in df.columns:
        df['player_id'] = df['gsis_id']

    # Merge on player_id, season, week
    merge_cols = ['trailing_rz_rush_share', 'trailing_rz_target_share',
                  'trailing_rz_rush_snaps', 'trailing_rz_targets']
    available_cols = [c for c in merge_cols if c in rz_features.columns]

    if not available_cols:
        return df

    merged = df.merge(
        rz_features[['player_id', 'season', 'week'] + available_cols],
        on=['player_id', 'season', 'week'],
        how='left'
    )

    # Fill missing with 0
    for col in available_cols:
        merged[col] = merged[col].fillna(0)

    return merged


if __name__ == '__main__':
    # Test computation
    rz = compute_red_zone_features(2024)
    print(rz.head())
