#!/usr/bin/env python3
"""
Build Position and Defensive Matchup Caches for V24.

Pre-computes all position-specific features for fast prediction lookup:
- Position roles (WR1/WR2/WR3, RB1/RB2, TE1)
- Defense-vs-position stats (what each defense allows to each role)
- Coverage tendencies (man/zone rates)
- Slot funnel scores

Run weekly after nflverse data updates:
    python scripts/cache/build_position_cache.py --season 2025

Output files:
    data/cache/position_roles_{season}.parquet
    data/cache/defense_vs_position_{season}.parquet
    data/cache/coverage_tendencies_{season}.parquet
    data/cache/slot_funnel_{season}.parquet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR

# Import V24 feature modules
from nfl_quant.features.position_role import (
    load_depth_charts,
    load_ngs_receiving,
    get_all_position_roles_vectorized,
    add_slot_detection_vectorized,
    clear_caches as clear_position_caches
)
from nfl_quant.features.defense_vs_position import (
    calculate_defense_vs_position_stats,
    pivot_defense_vs_position,
    clear_cache as clear_defense_cache
)
from nfl_quant.features.coverage_tendencies import (
    load_participation,
    calculate_team_coverage_tendencies,
    calculate_slot_funnel_score,
    clear_cache as clear_coverage_cache
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_position_roles_cache(season: int = None) -> pd.DataFrame:
    """
    Build position roles cache.

    Args:
        season: Optional specific season (default: all)

    Returns:
        DataFrame with position roles for all players
    """
    logger.info("Building position roles cache...")
    start = time.time()

    # Load weekly stats
    ws_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    if not ws_path.exists():
        logger.error(f"Weekly stats not found at {ws_path}")
        return pd.DataFrame()

    weekly_stats = pd.read_parquet(ws_path)
    logger.info(f"Loaded {len(weekly_stats):,} weekly stats records")

    # Load depth charts
    depth_charts = load_depth_charts()

    # Load NGS receiving for slot detection
    ngs_receiving = load_ngs_receiving()

    # Calculate position roles
    position_roles = get_all_position_roles_vectorized(weekly_stats, depth_charts, season)
    logger.info(f"Calculated {len(position_roles):,} position role assignments")

    # Add slot detection
    if len(ngs_receiving) > 0:
        position_roles = add_slot_detection_vectorized(position_roles, ngs_receiving)
        logger.info(f"Added slot detection")

    elapsed = time.time() - start
    logger.info(f"Position roles cache built in {elapsed:.1f}s")

    return position_roles


def build_defense_vs_position_cache(
    weekly_stats: pd.DataFrame,
    position_roles: pd.DataFrame,
    season: int = None
) -> pd.DataFrame:
    """
    Build defense-vs-position cache.

    Args:
        weekly_stats: Weekly stats DataFrame
        position_roles: Position roles DataFrame
        season: Optional specific season

    Returns:
        Pivoted defense-vs-position DataFrame
    """
    logger.info("Building defense-vs-position cache...")
    start = time.time()

    if season is not None:
        weekly_stats = weekly_stats[weekly_stats['season'] == season]
        position_roles = position_roles[position_roles['season'] == season]

    # Calculate defense-vs-position stats
    defense_stats = calculate_defense_vs_position_stats(weekly_stats, position_roles)
    logger.info(f"Calculated {len(defense_stats):,} defense-vs-position records")

    # Pivot to wide format
    pivoted = pivot_defense_vs_position(defense_stats)
    logger.info(f"Pivoted to {len(pivoted):,} rows with {len(pivoted.columns)} columns")

    elapsed = time.time() - start
    logger.info(f"Defense-vs-position cache built in {elapsed:.1f}s")

    return pivoted


def build_coverage_tendencies_cache(season: int = None) -> tuple:
    """
    Build coverage tendencies and slot funnel caches.

    Args:
        season: Optional specific season

    Returns:
        Tuple of (coverage_tendencies, slot_funnel) DataFrames
    """
    logger.info("Building coverage tendencies cache...")
    start = time.time()

    # Load participation data
    participation = load_participation()

    if len(participation) == 0:
        logger.warning("No participation data available")
        return pd.DataFrame(), pd.DataFrame()

    # Calculate coverage tendencies
    coverage_tendencies = calculate_team_coverage_tendencies(participation)
    logger.info(f"Calculated {len(coverage_tendencies):,} coverage tendency records")

    # Calculate slot funnel
    ws_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    weekly_stats = pd.read_parquet(ws_path) if ws_path.exists() else pd.DataFrame()

    ngs_receiving = load_ngs_receiving()

    slot_funnel = pd.DataFrame()
    if len(weekly_stats) > 0 and len(ngs_receiving) > 0:
        if season is not None:
            weekly_stats = weekly_stats[weekly_stats['season'] == season]
            ngs_receiving = ngs_receiving[ngs_receiving['season'] == season]

        slot_funnel = calculate_slot_funnel_score(weekly_stats, ngs_receiving)
        logger.info(f"Calculated {len(slot_funnel):,} slot funnel records")

    elapsed = time.time() - start
    logger.info(f"Coverage tendencies cache built in {elapsed:.1f}s")

    return coverage_tendencies, slot_funnel


def save_caches(
    position_roles: pd.DataFrame,
    defense_vs_position: pd.DataFrame,
    coverage_tendencies: pd.DataFrame,
    slot_funnel: pd.DataFrame,
    season: int = None
):
    """Save all caches to parquet files."""
    cache_dir = DATA_DIR / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{season}" if season else ""

    # Save position roles
    if len(position_roles) > 0:
        path = cache_dir / f'position_roles{suffix}.parquet'
        position_roles.to_parquet(path, index=False)
        logger.info(f"Saved position roles to {path}")

    # Save defense-vs-position
    if len(defense_vs_position) > 0:
        path = cache_dir / f'defense_vs_position{suffix}.parquet'
        defense_vs_position.to_parquet(path, index=False)
        logger.info(f"Saved defense-vs-position to {path}")

    # Save coverage tendencies
    if len(coverage_tendencies) > 0:
        path = cache_dir / f'coverage_tendencies{suffix}.parquet'
        coverage_tendencies.to_parquet(path, index=False)
        logger.info(f"Saved coverage tendencies to {path}")

    # Save slot funnel
    if len(slot_funnel) > 0:
        path = cache_dir / f'slot_funnel{suffix}.parquet'
        slot_funnel.to_parquet(path, index=False)
        logger.info(f"Saved slot funnel to {path}")

    # Also save combined "latest" versions without season suffix
    if season is not None:
        if len(position_roles) > 0:
            position_roles.to_parquet(cache_dir / 'position_roles.parquet', index=False)
        if len(defense_vs_position) > 0:
            defense_vs_position.to_parquet(cache_dir / 'defense_vs_position.parquet', index=False)
        if len(coverage_tendencies) > 0:
            coverage_tendencies.to_parquet(cache_dir / 'coverage_tendencies.parquet', index=False)
        if len(slot_funnel) > 0:
            slot_funnel.to_parquet(cache_dir / 'slot_funnel.parquet', index=False)


def main():
    parser = argparse.ArgumentParser(description='Build position and defensive matchup caches')
    parser.add_argument('--season', type=int, default=None,
                        help='Specific season to build cache for (default: all available)')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild even if cache exists')
    args = parser.parse_args()

    total_start = time.time()
    logger.info("="*60)
    logger.info("V24 POSITION CACHE BUILDER")
    logger.info("="*60)

    if args.season:
        logger.info(f"Building cache for season {args.season}")
    else:
        logger.info("Building cache for all seasons")

    # Clear existing caches
    clear_position_caches()
    clear_defense_cache()
    clear_coverage_cache()

    # Build caches
    position_roles = build_position_roles_cache(args.season)

    # Load weekly stats for defense-vs-position
    ws_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    weekly_stats = pd.read_parquet(ws_path) if ws_path.exists() else pd.DataFrame()

    defense_vs_position = build_defense_vs_position_cache(
        weekly_stats, position_roles, args.season
    )

    coverage_tendencies, slot_funnel = build_coverage_tendencies_cache(args.season)

    # Save caches
    save_caches(
        position_roles,
        defense_vs_position,
        coverage_tendencies,
        slot_funnel,
        args.season
    )

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("="*60)
    logger.info("CACHE BUILD COMPLETE")
    logger.info("="*60)
    logger.info(f"Position roles: {len(position_roles):,} records")
    logger.info(f"Defense-vs-position: {len(defense_vs_position):,} records")
    logger.info(f"Coverage tendencies: {len(coverage_tendencies):,} records")
    logger.info(f"Slot funnel: {len(slot_funnel):,} records")
    logger.info(f"Total time: {total_elapsed:.1f}s")


if __name__ == '__main__':
    main()
