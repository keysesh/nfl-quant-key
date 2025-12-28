"""
Extract actual trailing statistics from play-by-play data for predictor inputs.

This solves the feature mismatch problem:
- Predictors were trained on actual per-game counts (Josh Allen: 31 attempts/game)
- We were passing shares × constants (0.97 × 70 = 67.9)
- This module extracts real trailing stats from PBP data
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Feature flag for regime detection
# Set via environment variable: ENABLE_REGIME_DETECTION=1
ENABLE_REGIME_DETECTION = os.environ.get('ENABLE_REGIME_DETECTION', '0') == '1'


class TrailingStatsExtractor:
    """Extract trailing 4-week averages from play-by-play data."""

    def __init__(self, pbp_path: Path = None, season: Optional[int] = None):
        """
        Initialize with PBP data.

        Args:
            pbp_path: Path to play-by-play parquet file
            season: Season to load (defaults to current season)
        """
        # Use FRESH generic PBP as primary source (not stale season-specific files)
        if pbp_path is None:
            pbp_path = Path('data/nflverse/pbp.parquet')
            if not pbp_path.exists():
                raise FileNotFoundError(
                    f"PBP file not found: {pbp_path}. "
                    "Run 'Rscript scripts/fetch/fetch_nflverse_data.R' to fetch fresh data."
                )

        logger.info(f"Loading PBP data from {pbp_path}")
        self.pbp_df = pd.read_parquet(pbp_path)

        # Filter to requested season if provided
        if season is not None and 'season' in self.pbp_df.columns:
            self.pbp_df = self.pbp_df[self.pbp_df['season'] == season]
            logger.info(f"Filtered to season {season}")
        logger.info(f"Loaded {len(self.pbp_df):,} plays")

        # Pre-compute player-week aggregations for fast lookup
        self._compute_player_week_stats()

    def _compute_player_week_stats(self):
        """Pre-compute all player-week statistics."""
        logger.info("Computing player-week statistics...")

        # QB passing stats
        qb_passing = (
            self.pbp_df[self.pbp_df['play_type'] == 'pass']
            .groupby(['passer_player_id', 'passer_player_name', 'week'])
            .size()
            .reset_index(name='pass_attempts')
        )

        # QB/RB rushing stats
        rushing = (
            self.pbp_df[self.pbp_df['play_type'] == 'run']
            .groupby(['rusher_player_id', 'rusher_player_name', 'week'])
            .size()
            .reset_index(name='rush_attempts')
        )

        # WR/TE/RB receiving stats (targets)
        receiving = (
            self.pbp_df[self.pbp_df['receiver_player_id'].notna()]
            .groupby(['receiver_player_id', 'receiver_player_name', 'week'])
            .size()
            .reset_index(name='targets')
        )

        # Store for lookup
        self.qb_passing_stats = qb_passing
        self.rushing_stats = rushing
        self.receiving_stats = receiving

        logger.info(f"  QB passing: {len(qb_passing)} player-weeks")
        logger.info(f"  Rushing: {len(rushing)} player-weeks")
        logger.info(f"  Receiving: {len(receiving)} player-weeks")

    def _calculate_weighted_average(
        self,
        values: pd.Series,
        use_ewma: bool = True,
        span: int = 4
    ) -> float:
        """
        Calculate weighted average of values.

        Args:
            values: Series of values to average
            use_ewma: Use exponential weighting
            span: Span for EWMA (default 4)

        Returns:
            Weighted average value
        """
        if len(values) == 0:
            return 0.0

        if use_ewma:
            # EWMA: Recent values weighted more heavily
            # With span=4: Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
            alpha = 2.0 / (span + 1)  # Smoothing factor
            weights = []
            for i in range(len(values)):
                weight = alpha * (1 - alpha) ** i
                weights.append(weight)

            # Reverse weights (most recent gets highest weight)
            weights = list(reversed(weights))

            # Normalize weights to sum to 1
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]

            # Calculate weighted average
            weighted_avg = sum(v * w for v, w in zip(values, weights))
            return float(weighted_avg)
        else:
            # Simple mean
            return float(values.mean())

    def get_trailing_stats(
        self,
        player_name: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4,
        use_ewma: bool = True
    ) -> Dict[str, float]:
        """
        Get trailing N-week averages for a player.

        Args:
            player_name: Player name (e.g., "J.Allen")
            position: Position (QB, RB, WR, TE)
            current_week: Current week number
            trailing_weeks: Number of weeks to average (default 4)
            use_ewma: Use exponential weighting (default True)
                     If True, recent weeks weighted higher:
                       Week N-1: 40%, N-2: 27%, N-3: 18%, N-4: 12%
                     If False, all weeks weighted equally (25% each)

        Returns:
            Dict with trailing_snaps, trailing_attempts, trailing_carries
        """
        # Week range for trailing stats
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            # No trailing data available (Week 1)
            logger.warning(f"No trailing data for {player_name} in week {current_week}")
            return {
                'trailing_snaps': 0.0,
                'trailing_attempts': 0.0,
                'trailing_carries': 0.0,
            }

        # Position-specific stats
        if position == 'QB':
            # Get passing attempts
            qb_data = self.qb_passing_stats[
                (self.qb_passing_stats['passer_player_name'] == player_name) &
                (self.qb_passing_stats['week'] >= start_week) &
                (self.qb_passing_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            # Get rushing attempts
            rush_data = self.rushing_stats[
                (self.rushing_stats['rusher_player_name'] == player_name) &
                (self.rushing_stats['week'] >= start_week) &
                (self.rushing_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            trailing_attempts = self._calculate_weighted_average(
                qb_data['pass_attempts'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(qb_data) > 0 else 0.0

            trailing_carries = self._calculate_weighted_average(
                rush_data['rush_attempts'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rush_data) > 0 else 0.0

            trailing_snaps = trailing_attempts + trailing_carries

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),
                'trailing_carries': float(trailing_carries),
            }

        elif position == 'RB':
            # Get rushing attempts
            rush_data = self.rushing_stats[
                (self.rushing_stats['rusher_player_name'] == player_name) &
                (self.rushing_stats['week'] >= start_week) &
                (self.rushing_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            # Get targets
            rec_data = self.receiving_stats[
                (self.receiving_stats['receiver_player_name'] == player_name) &
                (self.receiving_stats['week'] >= start_week) &
                (self.receiving_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            trailing_carries = self._calculate_weighted_average(
                rush_data['rush_attempts'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rush_data) > 0 else 0.0

            trailing_attempts = self._calculate_weighted_average(
                rec_data['targets'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rec_data) > 0 else 0.0  # attempts = targets for RB

            trailing_snaps = trailing_carries + trailing_attempts

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),  # targets
                'trailing_carries': float(trailing_carries),
            }

        elif position in ['WR', 'TE']:
            # Get targets
            rec_data = self.receiving_stats[
                (self.receiving_stats['receiver_player_name'] == player_name) &
                (self.receiving_stats['week'] >= start_week) &
                (self.receiving_stats['week'] <= end_week)
            ].sort_values('week')  # Sort by week for EWMA

            trailing_attempts = self._calculate_weighted_average(
                rec_data['targets'], use_ewma=use_ewma, span=trailing_weeks
            ) if len(rec_data) > 0 else 0.0  # attempts = targets for WR/TE

            trailing_snaps = trailing_attempts  # Approximation

            return {
                'trailing_snaps': float(trailing_snaps),
                'trailing_attempts': float(trailing_attempts),  # targets
                'trailing_carries': 0.0,
            }

        else:
            raise ValueError(f"Unsupported position: {position}")

    def get_regime_features(
        self,
        player_name: str,
        team: str,
        current_week: int,
        position: str
    ) -> Dict[str, float]:
        """
        Extract regime-related features for ML models.

        Args:
            player_name: Player name
            team: Team abbreviation
            current_week: Current week number
            position: Player position

        Returns:
            Dict with regime features:
            - weeks_since_regime_change: Weeks since last QB/coaching change
            - is_in_regime: 1.0 if in an active regime, 0.0 otherwise
            - regime_confidence: Confidence score (0-1) if in regime
        """
        if not ENABLE_REGIME_DETECTION:
            # Regime detection disabled - return neutral features
            return {
                'weeks_since_regime_change': 999.0,  # No regime
                'is_in_regime': 0.0,
                'regime_confidence': 0.0,
            }

        try:
            # Use regime-aware extractor to detect regime
            from ..regime.integration import get_regime_aware_extractor
            regime_extractor = get_regime_aware_extractor(enable_regime=True)

            # Get regime info for player's team
            regime_info = regime_extractor._detect_team_regime(team, current_week)

            if regime_info is None:
                # No active regime
                return {
                    'weeks_since_regime_change': 999.0,  # No regime
                    'is_in_regime': 0.0,
                    'regime_confidence': 0.0,
                }

            # Active regime detected
            weeks_since_change = current_week - regime_info['start_week']

            return {
                'weeks_since_regime_change': float(weeks_since_change),
                'is_in_regime': 1.0,
                'regime_confidence': float(regime_info['confidence']),
            }

        except Exception as e:
            logger.warning(f"Error extracting regime features for {player_name}: {e}")
            # Return neutral features on error
            return {
                'weeks_since_regime_change': 999.0,
                'is_in_regime': 0.0,
                'regime_confidence': 0.0,
            }

    def get_game_script_features(
        self,
        player_name: str,
        position: str,
        current_week: int,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Extract game script-related features from historical data.

        Args:
            player_name: Player name
            position: Player position
            current_week: Current week number
            trailing_weeks: Number of weeks to look back

        Returns:
            Dict with game script features:
            - usage_when_leading: Avg usage when team leading by 7+
            - usage_when_trailing: Avg usage when team trailing by 7+
            - usage_when_close: Avg usage when within 1 score
            - game_script_sensitivity: Std dev of usage across game scripts
        """
        start_week = max(1, current_week - trailing_weeks)
        end_week = current_week - 1

        if end_week < 1:
            return {
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            }

        # Filter plays for this player in the trailing window
        # This requires score differential data from PBP
        player_plays = self.pbp_df[
            (self.pbp_df['week'] >= start_week) &
            (self.pbp_df['week'] <= end_week) &
            (
                (self.pbp_df['passer_player_name'] == player_name) |
                (self.pbp_df['rusher_player_name'] == player_name) |
                (self.pbp_df['receiver_player_name'] == player_name)
            )
        ].copy()

        if len(player_plays) == 0:
            return {
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            }

        # Calculate score differential (positive = team leading)
        # score_differential may already exist in PBP data
        if 'score_differential' in player_plays.columns:
            player_plays['score_diff'] = player_plays['score_differential']
        elif 'posteam_score' in player_plays.columns and 'defteam_score' in player_plays.columns:
            player_plays['score_diff'] = player_plays['posteam_score'] - player_plays['defteam_score']
        else:
            # Can't calculate game script features without score data
            logger.debug(f"No score data available for game script features for {player_name}")
            return {
                'usage_when_leading': 0.0,
                'usage_when_trailing': 0.0,
                'usage_when_close': 0.0,
                'game_script_sensitivity': 0.0,
            }

        # Group plays by game script
        leading_plays = player_plays[player_plays['score_diff'] >= 7]
        trailing_plays = player_plays[player_plays['score_diff'] <= -7]
        close_plays = player_plays[player_plays['score_diff'].abs() < 7]

        # Count plays per game script per week
        def count_plays_per_week(plays_df):
            if len(plays_df) == 0:
                return 0.0
            # Group by week and count, then average
            weekly_counts = plays_df.groupby('week').size()
            return float(weekly_counts.mean())

        usage_leading = count_plays_per_week(leading_plays)
        usage_trailing = count_plays_per_week(trailing_plays)
        usage_close = count_plays_per_week(close_plays)

        # Calculate sensitivity (variance in usage)
        usage_values = [usage_leading, usage_trailing, usage_close]
        usage_values = [v for v in usage_values if v > 0]
        if len(usage_values) >= 2:
            sensitivity = float(pd.Series(usage_values).std())
        else:
            sensitivity = 0.0

        return {
            'usage_when_leading': usage_leading,
            'usage_when_trailing': usage_trailing,
            'usage_when_close': usage_close,
            'game_script_sensitivity': sensitivity,
        }


# Singleton instance for reuse
_EXTRACTOR = None


def get_trailing_stats_extractor() -> TrailingStatsExtractor:
    """
    Get or create trailing stats extractor singleton.

    If ENABLE_REGIME_DETECTION=1, returns regime-aware extractor instead.
    """
    global _EXTRACTOR

    if ENABLE_REGIME_DETECTION:
        # Use regime-aware extractor (drop-in replacement)
        try:
            from ..regime.integration import get_regime_aware_extractor
            logger.info("✓ Using regime-aware trailing stats extractor")
            return get_regime_aware_extractor(enable_regime=True)
        except ImportError as e:
            logger.warning(f"Regime detection not available: {e}")
            logger.warning("Falling back to standard 4-week extractor")
            # Fall through to standard extractor

    # Standard extractor
    if _EXTRACTOR is None:
        _EXTRACTOR = TrailingStatsExtractor()
        if ENABLE_REGIME_DETECTION:
            logger.info("Using standard 4-week trailing stats extractor (fallback)")
        else:
            logger.info("Using standard 4-week trailing stats extractor")
    return _EXTRACTOR


# =============================================================================
# EDGE SYSTEM TRAILING STATS - Functions for LVT edge computation
# =============================================================================

# Market -> stat column mapping for edge system
EDGE_MARKET_STAT_MAP: Dict[str, str] = {
    'player_receptions': 'receptions',
    'player_rush_yds': 'rushing_yards',
    'player_reception_yds': 'receiving_yards',
    'player_rush_attempts': 'carries',
    'player_pass_attempts': 'attempts',  # NFL pass attempts
    'player_pass_completions': 'completions',  # NEW - 55.4% UNDER historical
    # player_pass_yds: REMOVED - -14.1% ROI in walk-forward
}

# Trailing column names for edge system
EDGE_TRAILING_COL_MAP: Dict[str, str] = {
    'player_receptions': 'trailing_receptions',
    'player_rush_yds': 'trailing_rushing_yards',
    'player_reception_yds': 'trailing_receiving_yards',
    'player_rush_attempts': 'trailing_carries',
    'player_pass_attempts': 'trailing_pass_attempts',
    'player_pass_completions': 'trailing_completions',  # NEW
    # player_pass_yds: REMOVED
}

# Default EWMA span for edge system
EDGE_EWMA_SPAN = 6

# Market-specific EWMA spans (some stats are more volatile)
MARKET_EWMA_SPANS: Dict[str, int] = {
    'player_receptions': 6,
    'player_rush_yds': 6,
    'player_reception_yds': 4,  # Shorter span - receiving yards are volatile
    'player_rush_attempts': 6,
    'player_pass_attempts': 4,  # Shorter span - attempts vary by game script
    'player_pass_completions': 4,  # NEW - shorter span like attempts
    # player_pass_yds: REMOVED
}

# Deflation factors (imported from model_config)
try:
    from configs.model_config import TRAILING_DEFLATION_FACTORS
except ImportError:
    TRAILING_DEFLATION_FACTORS = {
        'player_receptions': 0.92,
        'player_reception_yds': 0.83,
        'player_rush_yds': 0.95,
        'player_pass_yds': 1.02,
        'player_rush_attempts': 0.95,
    }


def load_player_stats_for_edge() -> pd.DataFrame:
    """
    Load NFLverse player stats for edge trailing calculation.

    Returns:
        DataFrame with player stats across seasons
    """
    from nfl_quant.config_paths import DATA_DIR

    stats_files = [
        DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv',
        DATA_DIR / 'nflverse' / 'player_stats_2023.csv',
    ]

    dfs = []
    for path in stats_files:
        if path.exists():
            df = pd.read_csv(path, low_memory=False)
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No player stats files found")

    stats = pd.concat(dfs, ignore_index=True)

    # Normalize player names
    from nfl_quant.utils.player_names import normalize_player_name
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

    return stats.sort_values(['player_norm', 'season', 'week'])


def compute_edge_trailing_stats(
    stats_df: pd.DataFrame,
    markets: Optional[list] = None,
    ewma_span: int = None,
    use_market_specific_spans: bool = True,
) -> pd.DataFrame:
    """
    Compute trailing stats for edge system (all markets).

    Args:
        stats_df: NFLverse player stats with player_norm, season, week columns
        markets: List of markets to compute (default: all)
        ewma_span: Override EWMA span for all markets (default: use market-specific)
        use_market_specific_spans: If True, use MARKET_EWMA_SPANS per market

    Returns:
        DataFrame with trailing_* columns added
    """
    if markets is None:
        markets = list(EDGE_MARKET_STAT_MAP.keys())

    stats_df = stats_df.copy()
    stats_df = stats_df.sort_values(['player_norm', 'season', 'week'])

    for market in markets:
        stat_col = EDGE_MARKET_STAT_MAP.get(market)
        if stat_col is None or stat_col not in stats_df.columns:
            continue

        trailing_col = EDGE_TRAILING_COL_MAP.get(market, f'trailing_{stat_col}')

        # Use market-specific EWMA span or override
        if ewma_span is not None:
            span = ewma_span
        elif use_market_specific_spans and market in MARKET_EWMA_SPANS:
            span = MARKET_EWMA_SPANS[market]
        else:
            span = EDGE_EWMA_SPAN

        # EWMA with shift(1) to prevent leakage
        stats_df[trailing_col] = (
            stats_df.groupby('player_norm')[stat_col]
            .transform(lambda x: x.ewm(span=span, min_periods=1).mean().shift(1))
        )

    # =========================================================================
    # RB-SPECIFIC FEATURES for player_rush_attempts (V32 Dec 2025)
    # These are required by the LVT Edge model trained with RB features
    # =========================================================================
    if 'player_rush_attempts' in markets:
        # trailing_ypc: yards per carry
        if 'rushing_yards' in stats_df.columns and 'carries' in stats_df.columns:
            # Compute YPC per game first
            stats_df['_ypc'] = stats_df['rushing_yards'] / stats_df['carries'].replace(0, np.nan)
            stats_df['trailing_ypc'] = (
                stats_df.groupby('player_norm')['_ypc']
                .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
            )
            stats_df.drop(columns=['_ypc'], inplace=True)
        else:
            stats_df['trailing_ypc'] = 0.0

        # trailing_cv_carries: coefficient of variation of carries
        if 'carries' in stats_df.columns:
            def compute_cv(x):
                shifted = x.shift(1)
                roll_mean = shifted.rolling(6, min_periods=2).mean()
                roll_std = shifted.rolling(6, min_periods=2).std()
                return roll_std / roll_mean.replace(0, np.nan)
            stats_df['trailing_cv_carries'] = (
                stats_df.groupby('player_norm')['carries']
                .transform(compute_cv)
            )
        else:
            stats_df['trailing_cv_carries'] = 0.0

        # trailing_rb_snap_share: requires snap count data (use default if unavailable)
        if 'offense_pct' in stats_df.columns:
            stats_df['trailing_rb_snap_share'] = (
                stats_df.groupby('player_norm')['offense_pct']
                .transform(lambda x: x.shift(1).ewm(span=6, min_periods=1).mean())
            )
        else:
            stats_df['trailing_rb_snap_share'] = 0.0

    return stats_df


def merge_edge_trailing_stats(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    markets: Optional[list] = None,
) -> pd.DataFrame:
    """
    Merge trailing stats from stats_df into main DataFrame.

    For predictions, we need the PRIOR week's trailing stats to avoid leakage.
    Uses MOST RECENT available trailing stats per player (not exact week match).
    This handles incomplete weeks (e.g., week 17 in progress -> uses week 16 stats).

    Args:
        df: Main DataFrame with player_norm, season, week columns
        stats_df: Stats DataFrame with trailing_* columns
        markets: List of markets to merge (default: all)

    Returns:
        df with trailing_* columns merged
    """
    import warnings

    if markets is None:
        markets = list(EDGE_TRAILING_COL_MAP.keys())

    # Get trailing columns to merge
    trailing_cols = [EDGE_TRAILING_COL_MAP.get(m) for m in markets if m in EDGE_TRAILING_COL_MAP]
    trailing_cols = [c for c in trailing_cols if c in stats_df.columns]

    if not trailing_cols:
        warnings.warn("No trailing columns found in stats_df")
        return df

    # Get most recent trailing stats per player (handles incomplete weeks)
    # Sort by season, week descending to get most recent first
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols
    stats_subset = stats_df[merge_cols].copy()
    stats_subset = stats_subset.sort_values(['player_norm', 'season', 'week'], ascending=[True, False, False])

    # Get the most recent row per player (first after sorting desc)
    latest_stats = stats_subset.groupby('player_norm').first().reset_index()

    # Drop season/week from latest_stats - we just want player_norm + trailing_cols
    latest_stats = latest_stats[['player_norm'] + trailing_cols]

    # Merge on player_norm only (uses most recent available stats)
    result = df.merge(
        latest_stats,
        on=['player_norm'],
        how='left',
        suffixes=('', '_stats')
    )

    # Handle duplicate columns (prefer new values)
    for col in trailing_cols:
        if f'{col}_stats' in result.columns:
            result[col] = result[col].fillna(result[f'{col}_stats'])
            result.drop(columns=[f'{col}_stats'], inplace=True)

    return result


def compute_line_vs_trailing(
    df: pd.DataFrame,
    market: str,
) -> pd.Series:
    """
    Compute line_vs_trailing (LVT) for a specific market.

    LVT = ((line - deflated_trailing) / deflated_trailing) * 100

    Args:
        df: DataFrame with 'line' and trailing_* columns
        market: Market to compute LVT for

    Returns:
        Series with LVT values (percentage)
    """
    import warnings
    import numpy as np

    trailing_col = EDGE_TRAILING_COL_MAP.get(market)
    if trailing_col is None:
        return pd.Series(0, index=df.index)

    if trailing_col not in df.columns:
        warnings.warn(f"Missing {trailing_col} column, LVT will be 0")
        return pd.Series(0, index=df.index)

    if 'line' not in df.columns:
        warnings.warn("Missing 'line' column, LVT will be 0")
        return pd.Series(0, index=df.index)

    # Get trailing values, fill missing with line (neutral LVT)
    trailing = df[trailing_col].fillna(df['line'])

    # Apply deflation factor
    deflation = TRAILING_DEFLATION_FACTORS.get(market, 0.90)
    deflated = trailing * deflation

    # Calculate LVT as percentage
    lvt = np.where(
        deflated > 0,
        (df['line'] - deflated) / deflated * 100,
        0
    )

    # Clip extreme values
    return pd.Series(np.clip(lvt, -100, 100), index=df.index)


def prepare_edge_data_with_trailing(
    df: pd.DataFrame,
    week: Optional[int] = None,
    season: int = 2025,
) -> pd.DataFrame:
    """
    Prepare a DataFrame with all trailing stats computed for edge system.

    This is the main entry point for preparing data for edge inference.

    Args:
        df: DataFrame with player_norm (or player), season, week, line columns
        week: Current week (for filtering historical data)
        season: Current season

    Returns:
        DataFrame with trailing_* and line_vs_trailing columns
    """
    from nfl_quant.utils.player_names import normalize_player_name

    df = df.copy()

    # Ensure player_norm exists
    if 'player_norm' not in df.columns:
        if 'player' in df.columns:
            df['player_norm'] = df['player'].apply(normalize_player_name)
        else:
            raise ValueError("DataFrame must have 'player' or 'player_norm' column")

    # Load and compute trailing stats
    stats = load_player_stats_for_edge()

    # Filter to historical data only (prevent leakage)
    if week is not None:
        historical_mask = (
            (stats['season'] < season) |
            ((stats['season'] == season) & (stats['week'] < week))
        )
        stats = stats[historical_mask]

    # Compute trailing stats
    stats = compute_edge_trailing_stats(stats)

    # Merge into main DataFrame
    df = merge_edge_trailing_stats(df, stats)

    return df


def get_edge_trailing_col(market: str) -> str:
    """Get trailing column name for a market (edge system)."""
    return EDGE_TRAILING_COL_MAP.get(market, f'trailing_{market}')


def get_edge_stat_col(market: str) -> str:
    """Get stat column name for a market (edge system)."""
    return EDGE_MARKET_STAT_MAP.get(market, market)
