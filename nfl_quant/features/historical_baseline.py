"""
Historical Baseline Provider

Populates hist_count and hist_over_rate fields for recommendations.
Uses player_stats to calculate historical performance vs prop lines.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class HistoricalBaselineProvider:
    """
    Provide historical baseline statistics for player props.

    Calculates:
    - How many times player hit OVER/UNDER for given line
    - Historical over rate
    - Average performance vs line
    """

    def __init__(self, data_dir: Path = None):
        """
        Initialize with player stats data.

        Args:
            data_dir: Path to nflverse data directory
        """
        if data_dir is None:
            data_dir = Path('data/nflverse')

        self.data_dir = Path(data_dir)
        self.player_stats = None
        self.weekly_stats = None
        self._load_data()

    def _load_data(self):
        """Load player stats parquet files."""
        # Try player_stats first (aggregated)
        stats_path = self.data_dir / 'player_stats.parquet'
        if stats_path.exists():
            self.player_stats = pd.read_parquet(stats_path)
            logger.info(f"Loaded player_stats: {len(self.player_stats):,} records")

        # Also load weekly stats if available (more granular)
        weekly_path = self.data_dir / 'weekly_stats.parquet'
        if weekly_path.exists():
            self.weekly_stats = pd.read_parquet(weekly_path)
            logger.info(f"Loaded weekly_stats: {len(self.weekly_stats):,} records")
        else:
            # Fallback to player_stats_2024_2025.parquet
            alt_path = self.data_dir / 'player_stats_2024_2025.parquet'
            if alt_path.exists():
                self.weekly_stats = pd.read_parquet(alt_path)
                logger.info(f"Loaded player_stats_2024_2025: {len(self.weekly_stats):,} records")

        if self.player_stats is None and self.weekly_stats is None:
            logger.warning("No player stats data found")

    def get_historical_baseline(
        self,
        player_name: str,
        market_type: str,
        line: float,
        season: int,
        week: int,
        lookback_weeks: int = None
    ) -> Dict[str, float]:
        """
        Get historical baseline for a player prop.

        Args:
            player_name: Player name
            market_type: Type of prop (player_pass_yds, player_rush_yds, etc.)
            line: The prop line value
            season: Current season
            week: Current week
            lookback_weeks: How many weeks to look back (None = all season)

        Returns:
            Dict with:
                - hist_count: Number of historical games
                - hist_over_count: Number of times hit OVER
                - hist_over_rate: Percentage hitting OVER
                - hist_avg: Historical average for stat
                - hist_avg_vs_line: Average minus line
        """
        baseline = {
            'hist_count': 0,
            'hist_over_count': 0,
            'hist_over_rate': 0.0,
            'hist_avg': 0.0,
            'hist_avg_vs_line': 0.0,
            'hist_std': 0.0,
        }

        df = self.weekly_stats if self.weekly_stats is not None else self.player_stats
        if df is None:
            return baseline

        # Map market type to column name
        stat_column = self._map_market_to_column(market_type)
        if stat_column is None:
            logger.debug(f"Unknown market type: {market_type}")
            return baseline

        # Filter to player's historical games
        player_data = df[
            (df['player_name'].str.contains(player_name, case=False, na=False)) &
            (df['season'] == season) &
            (df['week'] < week)
        ]

        if len(player_data) == 0:
            # Try partial name match
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                last_name = name_parts[-1]
                player_data = df[
                    (df['player_name'].str.contains(last_name, case=False, na=False)) &
                    (df['season'] == season) &
                    (df['week'] < week)
                ]

        if len(player_data) == 0:
            logger.debug(f"No historical data for {player_name}")
            return baseline

        # Apply lookback window if specified
        if lookback_weeks is not None:
            min_week = max(1, week - lookback_weeks)
            player_data = player_data[player_data['week'] >= min_week]

        # Check if column exists
        if stat_column not in player_data.columns:
            logger.debug(f"Column {stat_column} not in data")
            return baseline

        # Extract stat values
        stat_values = player_data[stat_column].dropna().values

        if len(stat_values) == 0:
            return baseline

        # Calculate baseline metrics
        baseline['hist_count'] = len(stat_values)
        baseline['hist_over_count'] = int((stat_values > line).sum())
        baseline['hist_over_rate'] = baseline['hist_over_count'] / baseline['hist_count']
        baseline['hist_avg'] = float(stat_values.mean())
        baseline['hist_avg_vs_line'] = baseline['hist_avg'] - line
        baseline['hist_std'] = float(stat_values.std())

        logger.debug(f"Historical baseline for {player_name} {market_type} {line}: {baseline}")
        return baseline

    def _map_market_to_column(self, market_type: str) -> Optional[str]:
        """Map market type to player_stats column name."""
        mapping = {
            'player_pass_yds': 'passing_yards',
            'player_passing_yards': 'passing_yards',
            'passing_yards': 'passing_yards',
            'player_pass_tds': 'passing_tds',
            'player_passing_tds': 'passing_tds',
            'passing_tds': 'passing_tds',
            'player_pass_attempts': 'attempts',
            'player_completions': 'completions',
            'player_interceptions': 'interceptions',
            'player_rush_yds': 'rushing_yards',
            'player_rushing_yards': 'rushing_yards',
            'rushing_yards': 'rushing_yards',
            'player_rush_attempts': 'carries',
            'player_carries': 'carries',
            'player_rushing_tds': 'rushing_tds',
            'player_reception_yds': 'receiving_yards',
            'player_receiving_yards': 'receiving_yards',
            'receiving_yards': 'receiving_yards',
            'player_receptions': 'receptions',
            'receptions': 'receptions',
            'player_targets': 'targets',
            'player_receiving_tds': 'receiving_tds',
            'player_anytime_td': 'total_tds',  # May need special handling
        }

        # Try direct match
        if market_type in mapping:
            return mapping[market_type]

        # Try case-insensitive
        for key, val in mapping.items():
            if market_type.lower() == key.lower():
                return val

        # Try partial match
        for key, val in mapping.items():
            if key in market_type.lower():
                return val

        return None

    def get_recent_trend(
        self,
        player_name: str,
        market_type: str,
        season: int,
        week: int,
        num_weeks: int = 3
    ) -> Dict[str, float]:
        """
        Get recent performance trend.

        Args:
            player_name: Player name
            market_type: Prop market type
            season: Season
            week: Current week
            num_weeks: Number of recent weeks

        Returns:
            Trend metrics dict
        """
        trend = {
            'trend_direction': 0.0,  # Positive = improving, negative = declining
            'trend_consistency': 0.0,  # 0-1, higher = more consistent
            'recent_avg': 0.0,
            'recent_vs_season_avg': 0.0,
        }

        df = self.weekly_stats if self.weekly_stats is not None else self.player_stats
        if df is None:
            return trend

        stat_column = self._map_market_to_column(market_type)
        if stat_column is None:
            return trend

        # Get recent weeks
        player_data = df[
            (df['player_name'].str.contains(player_name, case=False, na=False)) &
            (df['season'] == season) &
            (df['week'] < week) &
            (df['week'] >= max(1, week - num_weeks))
        ].sort_values('week')

        if len(player_data) < 2:
            return trend

        if stat_column not in player_data.columns:
            return trend

        values = player_data[stat_column].dropna().values
        if len(values) < 2:
            return trend

        # Calculate trend direction (slope)
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        trend['trend_direction'] = float(slope)

        # Calculate consistency (inverse of coefficient of variation)
        mean_val = values.mean()
        std_val = values.std()
        if mean_val > 0:
            cv = std_val / mean_val
            trend['trend_consistency'] = max(0, 1 - cv)
        else:
            trend['trend_consistency'] = 0.0

        # Recent average
        trend['recent_avg'] = float(values.mean())

        # Compare to season average
        season_data = df[
            (df['player_name'].str.contains(player_name, case=False, na=False)) &
            (df['season'] == season) &
            (df['week'] < week)
        ]
        if len(season_data) > 0 and stat_column in season_data.columns:
            season_avg = season_data[stat_column].mean()
            trend['recent_vs_season_avg'] = trend['recent_avg'] - season_avg

        return trend

    def get_league_percentile(
        self,
        player_name: str,
        position: str,
        market_type: str,
        season: int,
        week: int
    ) -> float:
        """
        Get player's percentile ranking within position for this stat.

        Args:
            player_name: Player name
            position: Position
            market_type: Stat type
            season: Season
            week: Current week

        Returns:
            Percentile (0-100)
        """
        df = self.weekly_stats if self.weekly_stats is not None else self.player_stats
        if df is None:
            return 50.0

        stat_column = self._map_market_to_column(market_type)
        if stat_column is None:
            return 50.0

        # Get player's average
        player_avg = df[
            (df['player_name'].str.contains(player_name, case=False, na=False)) &
            (df['season'] == season) &
            (df['week'] < week)
        ][stat_column].mean() if stat_column in df.columns else 0

        # Get all players at position
        if 'position' in df.columns:
            pos_data = df[
                (df['position'] == position) &
                (df['season'] == season) &
                (df['week'] < week)
            ]
        else:
            pos_data = df[
                (df['season'] == season) &
                (df['week'] < week)
            ]

        if stat_column not in pos_data.columns:
            return 50.0

        # Group by player and get averages
        player_avgs = pos_data.groupby('player_name')[stat_column].mean()

        # Calculate percentile
        percentile = (player_avgs < player_avg).mean() * 100

        return float(percentile)


def populate_historical_baselines(
    recommendations_df: pd.DataFrame,
    season: int = 2025,
    week: int = 11,
    data_dir: Path = None
) -> pd.DataFrame:
    """
    Populate historical baseline columns in recommendations DataFrame.

    Args:
        recommendations_df: DataFrame with recommendations
        season: Current season
        week: Current week
        data_dir: Path to nflverse data

    Returns:
        DataFrame with populated hist_count, hist_over_rate, etc.
    """
    provider = HistoricalBaselineProvider(data_dir)

    # Add columns if they don't exist
    if 'hist_count' not in recommendations_df.columns:
        recommendations_df['hist_count'] = 0
    if 'hist_over_rate' not in recommendations_df.columns:
        recommendations_df['hist_over_rate'] = 0.0
    if 'hist_avg' not in recommendations_df.columns:
        recommendations_df['hist_avg'] = 0.0

    for idx, row in recommendations_df.iterrows():
        player = row.get('player', '')
        market = row.get('market', '')
        line = row.get('line', 0)

        baseline = provider.get_historical_baseline(
            player, market, line, season, week
        )

        recommendations_df.at[idx, 'hist_count'] = baseline['hist_count']
        recommendations_df.at[idx, 'hist_over_rate'] = baseline['hist_over_rate']
        recommendations_df.at[idx, 'hist_avg'] = baseline['hist_avg']

    logger.info(f"Populated historical baselines for {len(recommendations_df)} recommendations")
    return recommendations_df
