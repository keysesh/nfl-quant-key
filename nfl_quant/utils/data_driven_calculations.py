"""
Data-Driven Calculation Utilities

All calculations use actual data - no hardcoded assumptions.
Uses R-generated NFLverse parquet files as the canonical data source.
"""

from typing import Optional, Dict, List
import pandas as pd

# Use centralized path configuration
from nfl_quant.config_paths import NFLVERSE_DIR

# NFLverse parquet data directory (from centralized config)
NFLVERSE_DATA_DIR = NFLVERSE_DIR


def calculate_rb_targets_from_historical_data(
    avg_rec_yd: Optional[float],
    avg_rec_tgt: Optional[float],
    historical_stats: Optional[Dict] = None
) -> float:
    """
    Calculate RB targets from actual historical data.
    
    Priority:
    1. Use avg_rec_tgt if available (direct target average)
    2. Calculate from avg_rec_yd using actual catch rate and yards per reception
    3. Use minimum 2.0 for active RBs if no data
    
    Args:
        avg_rec_yd: Average receiving yards per game from historical stats
        avg_rec_tgt: Average targets per game from historical stats (if available)
        historical_stats: Full historical stats dict (for catch rate calculation)
    
    Returns:
        Estimated targets per game
    """
    # Priority 1: Use actual average targets if available
    if avg_rec_tgt and avg_rec_tgt > 0:
        return avg_rec_yd
    
    # Priority 2: Calculate from receiving yards using actual catch rate
    if avg_rec_yd and avg_rec_yd > 0:
        # Calculate catch rate from historical data if available
        if historical_stats:
            # Try to get actual catch rate from historical data
            total_receptions = historical_stats.get('total_receptions', 0)
            total_targets = historical_stats.get('total_rec_tgt', 0)
            if total_targets > 0:
                catch_rate = total_receptions / total_targets
            else:
                # Use position average from current season data
                catch_rate = get_position_catch_rate('RB')
        else:
            catch_rate = get_position_catch_rate('RB')
        
        # Calculate yards per reception from historical data
        if historical_stats:
            total_rec_yd = historical_stats.get('total_rec_yd', 0)
            total_receptions = historical_stats.get('total_receptions', 0)
            if total_receptions > 0:
                yards_per_reception = total_rec_yd / total_receptions
            else:
                yards_per_reception = get_position_yards_per_reception('RB')
        else:
            yards_per_reception = get_position_yards_per_reception('RB')
        
        # Estimate targets: receptions = rec_yd / ypr, targets = receptions / catch_rate
        estimated_receptions = avg_rec_yd / yards_per_reception
        estimated_targets = estimated_receptions / catch_rate
        
        return max(estimated_targets, 2.0)  # Minimum 2 targets for active RBs
    
    # Priority 3: Minimum for active RBs
    return 2.0


def get_position_catch_rate(position: str, season: int = 2025, weeks: Optional[List[int]] = None) -> float:
    """
    Get actual catch rate for a position from R-generated NFLverse parquet files.

    Args:
        position: Player position (RB, WR, TE)
        season: Season year (default: 2025)
        weeks: Optional list of weeks to include (default: all weeks)

    Returns:
        Actual catch rate from NFLverse parquet data
    """
    try:
        # Load from R-generated NFLverse parquet files
        weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_stats.parquet'
        if not weekly_stats_file.exists():
            weekly_stats_file = NFLVERSE_DATA_DIR / f'weekly_{season}.parquet'

        if not weekly_stats_file.exists():
            return get_league_catch_rate_from_history(position)

        stats = pd.read_parquet(weekly_stats_file)

        # Filter by season
        stats = stats[stats['season'] == season]

        # Filter by weeks if specified
        if weeks is not None:
            stats = stats[stats['week'].isin(weeks)]

        # Filter by position
        pos_stats = stats[stats['position'] == position].copy()

        if len(pos_stats) == 0:
            return get_league_catch_rate_from_history(position)

        # Calculate actual catch rate: receptions / targets
        if 'receptions' in pos_stats.columns and 'targets' in pos_stats.columns:
            total_receptions = pos_stats['receptions'].sum()
            total_targets = pos_stats['targets'].sum()
            if total_targets > 0:
                return total_receptions / total_targets

    except Exception:
        # If parquet load fails, use historical fallback
        pass

    return get_league_catch_rate_from_history(position)


def get_position_yards_per_reception(position: str, season: int = 2025, weeks: Optional[List[int]] = None) -> float:
    """
    Get actual yards per reception for a position from R-generated NFLverse parquet files.

    Args:
        position: Player position (RB, WR, TE)
        season: Season year (default: 2025)
        weeks: Optional list of weeks to include (default: all weeks)

    Returns:
        Actual yards per reception from NFLverse parquet data
    """
    try:
        # Load from R-generated NFLverse parquet files
        weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_stats.parquet'
        if not weekly_stats_file.exists():
            weekly_stats_file = NFLVERSE_DATA_DIR / f'weekly_{season}.parquet'

        if not weekly_stats_file.exists():
            return get_league_yards_per_reception_from_history(position)

        stats = pd.read_parquet(weekly_stats_file)

        # Filter by season
        stats = stats[stats['season'] == season]

        # Filter by weeks if specified
        if weeks is not None:
            stats = stats[stats['week'].isin(weeks)]

        # Filter by position
        pos_stats = stats[stats['position'] == position].copy()

        if len(pos_stats) == 0:
            return get_league_yards_per_reception_from_history(position)

        # Calculate actual yards per reception
        # NFLverse uses 'receiving_yards' column
        if 'receiving_yards' in pos_stats.columns and 'receptions' in pos_stats.columns:
            total_rec_yd = pos_stats['receiving_yards'].sum()
            total_receptions = pos_stats['receptions'].sum()
            if total_receptions > 0:
                return total_rec_yd / total_receptions

    except Exception:
        # If parquet load fails, use historical fallback
        pass

    return get_league_yards_per_reception_from_history(position)


def get_league_catch_rate_from_history(position: str) -> float:
    """
    Get league average catch rate from R-generated NFLverse historical parquet files.

    Args:
        position: Player position

    Returns:
        League average catch rate calculated from NFLverse parquet data
    """
    try:
        # Load historical NFLverse data from R-generated parquet
        weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_stats.parquet'
        if not weekly_stats_file.exists():
            weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_historical.parquet'

        if weekly_stats_file.exists():
            stats = pd.read_parquet(weekly_stats_file)

            # Filter by position
            pos_stats = stats[stats['position'] == position].copy()

            if len(pos_stats) > 0 and 'receptions' in pos_stats.columns and 'targets' in pos_stats.columns:
                total_receptions = pos_stats['receptions'].sum()
                total_targets = pos_stats['targets'].sum()
                if total_targets > 0:
                    return total_receptions / total_targets
    except Exception as e:
        raise ValueError(
            f"Failed to calculate catch rate from NFLverse data: {e}. "
            f"NO HARDCODED DEFAULTS - ensure NFLverse weekly_stats.parquet exists."
        )

    # NO HARDCODED DEFAULTS - require actual data
    raise ValueError(
        f"No catch rate data found for {position} in NFLverse data. "
        f"NO HARDCODED DEFAULTS - ensure NFLverse weekly_stats.parquet exists."
    )


def get_league_yards_per_reception_from_history(position: str) -> float:
    """
    Get league average yards per reception from R-generated NFLverse historical parquet files.

    Args:
        position: Player position

    Returns:
        League average yards per reception calculated from NFLverse parquet data
    """
    try:
        # Load historical NFLverse data from R-generated parquet
        weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_stats.parquet'
        if not weekly_stats_file.exists():
            weekly_stats_file = NFLVERSE_DATA_DIR / 'weekly_historical.parquet'

        if weekly_stats_file.exists():
            stats = pd.read_parquet(weekly_stats_file)

            # Filter by position
            pos_stats = stats[stats['position'] == position].copy()

            if len(pos_stats) > 0 and 'receiving_yards' in pos_stats.columns and 'receptions' in pos_stats.columns:
                total_rec_yd = pos_stats['receiving_yards'].sum()
                total_receptions = pos_stats['receptions'].sum()
                if total_receptions > 0:
                    return total_rec_yd / total_receptions
    except Exception as e:
        raise ValueError(
            f"Failed to calculate yards per reception from NFLverse data: {e}. "
            f"NO HARDCODED DEFAULTS - ensure NFLverse weekly_stats.parquet exists."
        )

    # NO HARDCODED DEFAULTS - require actual data
    raise ValueError(
        f"No yards per reception data found for {position} in NFLverse data. "
        f"NO HARDCODED DEFAULTS - ensure NFLverse weekly_stats.parquet exists."
    )

