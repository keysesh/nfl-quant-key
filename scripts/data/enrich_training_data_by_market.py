#!/usr/bin/env python3
"""
Enrich Training Data with Market-Specific Features

This script properly enriches the training data with features specific to each market:
- player_receptions: target_share, snap_share, trailing_receptions
- player_rush_attempts: trailing_carries, rb_snap_share, team_run_rate
- player_pass_completions: trailing_completions, completion_rate, pressure_rate

CRITICAL: All features use shift(1) BEFORE expanding/rolling to prevent data leakage.

Usage:
    python scripts/data/enrich_training_data_by_market.py
    python scripts/data/enrich_training_data_by_market.py --market player_rush_attempts
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import DATA_DIR, BACKTEST_DIR
from nfl_quant.utils.player_names import normalize_player_name
from configs.model_config import EWMA_SPAN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# MARKET-SPECIFIC FEATURE DEFINITIONS
# =============================================================================

# Stat column mapping per market
MARKET_STAT_MAP = {
    'player_receptions': 'receptions',
    'player_reception_yds': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_rush_attempts': 'carries',
    'player_pass_yds': 'passing_yards',
    'player_pass_completions': 'completions',
    'player_pass_attempts': 'attempts',
}


def load_stats_data() -> pd.DataFrame:
    """Load player stats with all required columns."""
    # Try parquet first, then CSV
    stats_path = DATA_DIR / 'nflverse' / 'player_stats.parquet'
    if not stats_path.exists():
        stats_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'

    if stats_path.suffix == '.parquet':
        stats = pd.read_parquet(stats_path)
    else:
        stats = pd.read_csv(stats_path, low_memory=False)

    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    return stats


def load_snap_counts() -> pd.DataFrame:
    """Load snap count data."""
    snap_path = DATA_DIR / 'nflverse' / 'snap_counts.parquet'
    if not snap_path.exists():
        logger.warning("snap_counts.parquet not found")
        return pd.DataFrame()

    snaps = pd.read_parquet(snap_path)
    snaps['player_norm'] = snaps['player'].apply(normalize_player_name)
    snaps['global_week'] = (snaps['season'] - 2023) * 18 + snaps['week']

    return snaps


def load_pbp_data() -> pd.DataFrame:
    """Load play-by-play data for team-level stats."""
    pbp_path = DATA_DIR / 'nflverse' / 'pbp.parquet'
    if not pbp_path.exists():
        logger.warning("pbp.parquet not found")
        return pd.DataFrame()

    pbp = pd.read_parquet(pbp_path)
    return pbp


# =============================================================================
# ANTI-LEAKAGE TRAILING COMPUTATION
# =============================================================================

def compute_trailing_stat_no_leakage(
    df: pd.DataFrame,
    group_col: str,
    stat_col: str,
    output_col: str,
    ewma_span: int = EWMA_SPAN
) -> pd.DataFrame:
    """
    Compute trailing stat with NO data leakage.

    CRITICAL: Uses shift(1) BEFORE ewm() to exclude current row.
    """
    df = df.sort_values([group_col, 'global_week'])

    # ANTI-LEAKAGE: shift(1) BEFORE ewm
    df[output_col] = df.groupby(group_col)[stat_col].transform(
        lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean()
    )

    return df


def compute_trailing_cv_no_leakage(
    df: pd.DataFrame,
    group_col: str,
    stat_col: str,
    output_col: str,
    window: int = 4
) -> pd.DataFrame:
    """
    Compute coefficient of variation (consistency metric) with no leakage.
    """
    df = df.sort_values([group_col, 'global_week'])

    # ANTI-LEAKAGE: shift(1) BEFORE rolling
    mean = df.groupby(group_col)[stat_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    std = df.groupby(group_col)[stat_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).std()
    )

    df[output_col] = (std / (mean + 0.01)).clip(0, 2)

    return df


# =============================================================================
# MARKET-SPECIFIC FEATURE EXTRACTION
# =============================================================================

def enrich_receptions_features(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    snaps: pd.DataFrame
) -> pd.DataFrame:
    """Enrich player_receptions with WR-specific features."""
    logger.info("Enriching player_receptions features...")

    market_odds = odds[odds['market'] == 'player_receptions'].copy()

    if len(market_odds) == 0:
        return odds

    # Compute trailing receptions
    stats = compute_trailing_stat_no_leakage(
        stats, 'player_norm', 'receptions', 'trailing_receptions'
    )

    # Compute target share trailing
    if 'target_share' in stats.columns:
        stats = compute_trailing_stat_no_leakage(
            stats, 'player_norm', 'target_share', 'trailing_target_share'
        )

    # Compute trailing CV (consistency)
    stats = compute_trailing_cv_no_leakage(
        stats, 'player_norm', 'receptions', 'trailing_cv_receptions'
    )

    # Get latest stats per player for each week
    stats_latest = stats.sort_values('global_week').groupby(
        ['player_norm', 'season', 'week']
    ).last().reset_index()

    # Merge
    merge_cols = ['trailing_receptions', 'trailing_target_share', 'trailing_cv_receptions']
    merge_cols = [c for c in merge_cols if c in stats_latest.columns]

    market_odds = market_odds.merge(
        stats_latest[['player_norm', 'season', 'week'] + merge_cols],
        on=['player_norm', 'season', 'week'],
        how='left',
        suffixes=('', '_new')
    )

    # Add snap share if available
    if len(snaps) > 0 and 'offense_pct' in snaps.columns:
        snaps_latest = snaps.sort_values('global_week').groupby(
            ['player_norm', 'season', 'week']
        ).last().reset_index()

        # Compute trailing snap share
        snaps_latest = compute_trailing_stat_no_leakage(
            snaps_latest, 'player_norm', 'offense_pct', 'trailing_snap_share'
        )

        market_odds = market_odds.merge(
            snaps_latest[['player_norm', 'season', 'week', 'trailing_snap_share']],
            on=['player_norm', 'season', 'week'],
            how='left',
            suffixes=('', '_snap')
        )

    # Update original odds
    for col in merge_cols + ['trailing_snap_share']:
        if col in market_odds.columns and col not in odds.columns:
            odds[col] = np.nan
        if col in market_odds.columns:
            mask = odds['market'] == 'player_receptions'
            odds.loc[mask, col] = market_odds[col].values

    logger.info(f"  Enriched {len(market_odds):,} receptions rows")
    return odds


def enrich_rush_attempts_features(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    snaps: pd.DataFrame,
    pbp: pd.DataFrame
) -> pd.DataFrame:
    """Enrich player_rush_attempts with RB-specific features."""
    logger.info("Enriching player_rush_attempts features...")

    market_odds = odds[odds['market'] == 'player_rush_attempts'].copy()

    if len(market_odds) == 0:
        return odds

    # Compute trailing carries
    stats = compute_trailing_stat_no_leakage(
        stats, 'player_norm', 'carries', 'trailing_carries'
    )

    # Compute yards per carry trailing
    if 'rushing_yards' in stats.columns and 'carries' in stats.columns:
        stats['ypc'] = stats['rushing_yards'] / (stats['carries'] + 0.1)
        stats = compute_trailing_stat_no_leakage(
            stats, 'player_norm', 'ypc', 'trailing_ypc'
        )

    # Compute trailing CV (consistency)
    stats = compute_trailing_cv_no_leakage(
        stats, 'player_norm', 'carries', 'trailing_cv_carries'
    )

    # Team run rate from PBP
    if len(pbp) > 0:
        team_run_rate = pbp.groupby(['posteam', 'season', 'week']).apply(
            lambda x: (x['play_type'] == 'run').mean()
        ).reset_index(name='team_run_rate')

        # Compute trailing team run rate
        team_run_rate['global_week'] = (team_run_rate['season'] - 2023) * 18 + team_run_rate['week']
        team_run_rate = team_run_rate.sort_values(['posteam', 'global_week'])
        team_run_rate['trailing_team_run_rate'] = team_run_rate.groupby('posteam')['team_run_rate'].transform(
            lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
        )

        # Merge with odds
        if 'player_team' in market_odds.columns:
            market_odds = market_odds.merge(
                team_run_rate[['posteam', 'season', 'week', 'trailing_team_run_rate']].rename(
                    columns={'posteam': 'player_team'}
                ),
                on=['player_team', 'season', 'week'],
                how='left'
            )

    # Get latest stats per player for each week
    stats_latest = stats.sort_values('global_week').groupby(
        ['player_norm', 'season', 'week']
    ).last().reset_index()

    # Merge
    merge_cols = ['trailing_carries', 'trailing_ypc', 'trailing_cv_carries']
    merge_cols = [c for c in merge_cols if c in stats_latest.columns]

    market_odds = market_odds.merge(
        stats_latest[['player_norm', 'season', 'week'] + merge_cols],
        on=['player_norm', 'season', 'week'],
        how='left',
        suffixes=('', '_new')
    )

    # Add RB snap share if available
    if len(snaps) > 0 and 'offense_pct' in snaps.columns:
        # Filter to RB position if position column exists
        rb_snaps = snaps.copy()
        if 'position' in rb_snaps.columns:
            rb_snaps = rb_snaps[rb_snaps['position'].isin(['RB', 'HB', 'FB'])]

        rb_snaps_latest = rb_snaps.sort_values('global_week').groupby(
            ['player_norm', 'season', 'week']
        ).last().reset_index()

        rb_snaps_latest = compute_trailing_stat_no_leakage(
            rb_snaps_latest, 'player_norm', 'offense_pct', 'trailing_rb_snap_share'
        )

        market_odds = market_odds.merge(
            rb_snaps_latest[['player_norm', 'season', 'week', 'trailing_rb_snap_share']],
            on=['player_norm', 'season', 'week'],
            how='left',
            suffixes=('', '_snap')
        )
        merge_cols.append('trailing_rb_snap_share')

    if 'trailing_team_run_rate' in market_odds.columns:
        merge_cols.append('trailing_team_run_rate')

    # Update original odds
    for col in merge_cols:
        if col in market_odds.columns and col not in odds.columns:
            odds[col] = np.nan
        if col in market_odds.columns:
            mask = odds['market'] == 'player_rush_attempts'
            odds.loc[mask, col] = market_odds[col].values

    logger.info(f"  Enriched {len(market_odds):,} rush_attempts rows")
    return odds


def enrich_pass_completions_features(
    odds: pd.DataFrame,
    stats: pd.DataFrame
) -> pd.DataFrame:
    """Enrich player_pass_completions with QB-specific features."""
    logger.info("Enriching player_pass_completions features...")

    market_odds = odds[odds['market'] == 'player_pass_completions'].copy()

    if len(market_odds) == 0:
        return odds

    # Filter to QBs
    qb_stats = stats[stats['position'].isin(['QB'])] if 'position' in stats.columns else stats

    # Compute trailing completions
    qb_stats = compute_trailing_stat_no_leakage(
        qb_stats.copy(), 'player_norm', 'completions', 'trailing_completions'
    )

    # Compute trailing attempts
    if 'attempts' in qb_stats.columns:
        qb_stats = compute_trailing_stat_no_leakage(
            qb_stats, 'player_norm', 'attempts', 'trailing_attempts'
        )

    # Compute completion rate trailing
    if 'completions' in qb_stats.columns and 'attempts' in qb_stats.columns:
        qb_stats['completion_rate'] = qb_stats['completions'] / (qb_stats['attempts'] + 0.1)
        qb_stats = compute_trailing_stat_no_leakage(
            qb_stats, 'player_norm', 'completion_rate', 'trailing_completion_rate'
        )

    # Compute trailing CV
    qb_stats = compute_trailing_cv_no_leakage(
        qb_stats, 'player_norm', 'completions', 'trailing_cv_completions'
    )

    # Get latest stats per player for each week
    qb_latest = qb_stats.sort_values('global_week').groupby(
        ['player_norm', 'season', 'week']
    ).last().reset_index()

    # Merge
    merge_cols = ['trailing_completions', 'trailing_attempts', 'trailing_completion_rate', 'trailing_cv_completions']
    merge_cols = [c for c in merge_cols if c in qb_latest.columns]

    market_odds = market_odds.merge(
        qb_latest[['player_norm', 'season', 'week'] + merge_cols],
        on=['player_norm', 'season', 'week'],
        how='left',
        suffixes=('', '_new')
    )

    # Update original odds
    for col in merge_cols:
        if col in market_odds.columns and col not in odds.columns:
            odds[col] = np.nan
        if col in market_odds.columns:
            mask = odds['market'] == 'player_pass_completions'
            odds.loc[mask, col] = market_odds[col].values

    logger.info(f"  Enriched {len(market_odds):,} pass_completions rows")
    return odds


# =============================================================================
# MAIN ENRICHMENT PIPELINE
# =============================================================================

def enrich_all_markets(output_path: Path = None):
    """Enrich training data with market-specific features."""
    logger.info("="*60)
    logger.info("MARKET-SPECIFIC FEATURE ENRICHMENT")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading data...")
    odds_path = BACKTEST_DIR / 'combined_odds_actuals_ENRICHED.csv'
    odds = pd.read_csv(odds_path, low_memory=False)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)
    logger.info(f"  Loaded {len(odds):,} odds rows")

    stats = load_stats_data()
    logger.info(f"  Loaded {len(stats):,} stats rows")

    snaps = load_snap_counts()
    logger.info(f"  Loaded {len(snaps):,} snap rows")

    pbp = load_pbp_data()
    logger.info(f"  Loaded {len(pbp):,} PBP rows")

    # Enrich each market
    logger.info("\nEnriching markets...")

    odds = enrich_receptions_features(odds, stats.copy(), snaps.copy())
    odds = enrich_rush_attempts_features(odds, stats.copy(), snaps.copy(), pbp)
    odds = enrich_pass_completions_features(odds, stats.copy())

    # Save
    if output_path is None:
        output_path = BACKTEST_DIR / 'combined_odds_actuals_MARKET_ENRICHED.csv'

    odds.to_csv(output_path, index=False)
    logger.info(f"\nSaved to: {output_path}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("ENRICHMENT SUMMARY")
    logger.info("="*60)

    new_cols = [c for c in odds.columns if 'trailing_' in c]
    for market in ['player_receptions', 'player_rush_attempts', 'player_pass_completions']:
        mkt = odds[odds['market'] == market]
        logger.info(f"\n{market} ({len(mkt):,} rows):")
        for col in new_cols:
            if col in mkt.columns:
                pct = mkt[col].notna().mean() * 100
                if pct > 0:
                    logger.info(f"  {col}: {pct:.0f}% populated")

    return odds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Enrich training data with market-specific features')
    parser.add_argument('--output', type=str, help='Output path')
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    enrich_all_markets(output)
