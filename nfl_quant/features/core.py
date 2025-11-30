"""
NFL QUANT - Core Feature Engineering Layer

THIS IS THE SINGLE SOURCE OF TRUTH FOR ALL FEATURE CALCULATIONS.

All scripts MUST import from here. No inline feature calculations allowed.

Design Principles:
1. NO DATA LEAKAGE - all functions use shift(1) by default
2. CONSISTENT - same calculation everywhere
3. AUDITABLE - clear documentation of what each feature does
4. CACHED - expensive calculations cached to avoid redundant work

Usage:
    from nfl_quant.features.core import FeatureEngine

    engine = FeatureEngine()
    features = engine.get_player_features(player_name, week, season, market)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union, Any
from functools import lru_cache, cache
from dataclasses import dataclass
import logging

# Type aliases for clarity
DataFrame = pd.DataFrame
Series = pd.Series

logger = logging.getLogger(__name__)


# =============================================================================
# LEAGUE AVERAGES (Single Source of Truth for Fallback Values)
# =============================================================================

@dataclass(frozen=True)
class LeagueAverages:
    """
    NFL league-wide averages for fallback values.

    These are used when player-specific data is unavailable.
    All values based on 2023-2024 NFL data analysis.
    """
    # Receiving metrics
    separation: float = 2.5       # Avg yards of separation at catch
    cushion: float = 6.0          # Avg yards between WR and DB at snap
    completion_pct: float = 0.64  # League average completion %

    # Rushing metrics
    eight_box_rate: float = 0.20  # ~20% of rushes face 8+ box (corrected from 0.25)

    # EPA (efficiency metrics)
    rush_epa: float = -0.05       # Slight negative is true average
    pass_epa: float = 0.03        # Slight positive for pass plays

    # Snap metrics
    snap_share: float = 0.0       # Default to 0 when no data (conservative)

    # Betting defaults
    under_rate: float = 0.5       # Neutral default
    racr: float = 1.0             # Neutral RACR (yards = air yards)


# Singleton instance for use throughout codebase
LEAGUE_AVG = LeagueAverages()


# =============================================================================
# MARKET CONFIGURATION (Single Source of Truth for Market Mappings)
# =============================================================================

@dataclass(frozen=True)
class MarketConfig:
    """Configuration for a single betting market."""
    stat_col: str           # Column name in stats DataFrames
    epa_type: str           # 'pass' or 'run' for defense EPA lookup
    ngs_type: Optional[str] # 'receiving', 'rushing', or None


# Market-to-configuration mapping - USE THIS EVERYWHERE
MARKET_CONFIG: Dict[str, MarketConfig] = {
    'player_receptions': MarketConfig(
        stat_col='receptions',
        epa_type='pass',
        ngs_type='receiving'
    ),
    'player_reception_yds': MarketConfig(
        stat_col='receiving_yards',
        epa_type='pass',
        ngs_type='receiving'
    ),
    'player_rush_yds': MarketConfig(
        stat_col='rushing_yards',
        epa_type='run',
        ngs_type='rushing'
    ),
    'player_pass_yds': MarketConfig(
        stat_col='passing_yards',
        epa_type='pass',
        ngs_type=None
    ),
}


def get_stat_col_for_market(market: str) -> str:
    """Get the stat column name for a market."""
    if market not in MARKET_CONFIG:
        raise ValueError(f"Unknown market: {market}. Valid: {list(MARKET_CONFIG.keys())}")
    return MARKET_CONFIG[market].stat_col


def get_epa_type_for_market(market: str) -> str:
    """Get the EPA type ('pass' or 'run') for a market."""
    if market not in MARKET_CONFIG:
        raise ValueError(f"Unknown market: {market}. Valid: {list(MARKET_CONFIG.keys())}")
    return MARKET_CONFIG[market].epa_type


PROJECT_ROOT = Path(__file__).parent.parent.parent


class FeatureEngine:
    """
    Centralized feature engineering for all NFL QUANT models.

    This class is the ONLY place where features should be calculated.
    All training scripts, prediction scripts, and analysis scripts
    should use this class instead of inline calculations.
    """

    # Class-level defaults - single source of truth
    DEFAULT_TRAILING_WEEKS: int = 4
    DEFAULT_EWMA_SPAN: int = 4
    DEFAULT_MIN_PERIODS: int = 1

    def __init__(
        self,
        cache_enabled: bool = True,
        default_trailing_weeks: int = DEFAULT_TRAILING_WEEKS
    ):
        self.cache_enabled = cache_enabled
        self.default_trailing_weeks = default_trailing_weeks
        self._pbp_cache: Dict[int, pd.DataFrame] = {}
        self._weekly_stats_cache: Optional[pd.DataFrame] = None
        self._defense_epa_cache: Dict[str, pd.DataFrame] = {}
        # NEW V2 caches for additional NFLverse data
        self._snap_counts_cache: Optional[pd.DataFrame] = None
        self._ngs_receiving_cache: Optional[pd.DataFrame] = None
        self._ngs_rushing_cache: Optional[pd.DataFrame] = None

    def _get_trailing_window(
        self,
        df: pd.DataFrame,
        player_col: str,
        player_value: str,
        season: int,
        week: int,
        trailing_weeks: int = None
    ) -> pd.DataFrame:
        """
        Get trailing window data, handling cross-season boundaries.

        For early-season weeks (1-4), includes prior season data to ensure
        sufficient historical context.

        Args:
            df: DataFrame with 'season' and 'week' columns
            player_col: Column name for player identifier
            player_value: Value to filter player by
            season: Target season
            week: Target week (get data BEFORE this week)
            trailing_weeks: Number of weeks to look back (default: self.default_trailing_weeks)

        Returns:
            DataFrame filtered to trailing window for this player
        """
        if trailing_weeks is None:
            trailing_weeks = self.default_trailing_weeks

        # Get current season data (strictly before target week)
        current_season_data = df[
            (df[player_col] == player_value) &
            (df['season'] == season) &
            (df['week'] < week)
        ]

        weeks_in_current = len(current_season_data)

        # If we have enough data from current season, use simple filter
        if weeks_in_current >= trailing_weeks:
            return current_season_data[
                current_season_data['week'] >= week - trailing_weeks
            ]

        # Early season: pull from prior season if available
        min_season = df['season'].min() if len(df) > 0 else season

        if weeks_in_current < trailing_weeks and season > min_season:
            # Get remaining weeks from prior season (latest weeks)
            weeks_needed = trailing_weeks - weeks_in_current
            prior_season_data = df[
                (df[player_col] == player_value) &
                (df['season'] == season - 1)
            ].nlargest(weeks_needed, 'week')

            return pd.concat([prior_season_data, current_season_data], ignore_index=True)

        # Not enough data even with prior season - return what we have
        return current_season_data

    # =========================================================================
    # TRAILING STATS (Single Source of Truth)
    # =========================================================================

    def calculate_trailing_stat(
        self,
        df: pd.DataFrame,
        stat_col: str,
        player_col: str = 'player_norm',
        span: int = 4,
        min_periods: int = 1,
        no_leakage: bool = True
    ) -> pd.Series:
        """
        Calculate EWMA trailing stat with proper temporal separation.

        THIS IS THE CANONICAL IMPLEMENTATION. All scripts must use this.

        Args:
            df: DataFrame sorted by player and time
            stat_col: Column to calculate trailing stat for
            player_col: Column identifying players
            span: EWMA span (default 4 = 40%-27%-18%-12% weights)
            min_periods: Minimum periods for EWMA
            no_leakage: If True, shift(1) to prevent data leakage

        Returns:
            Series with trailing stat values

        Example:
            >>> engine = FeatureEngine()
            >>> df['trailing_receptions'] = engine.calculate_trailing_stat(
            ...     df, 'receptions', no_leakage=True
            ... )
        """
        if stat_col not in df.columns:
            logger.warning(f"Column {stat_col} not in DataFrame")
            return pd.Series(index=df.index, dtype=float)

        # Sort by player and time
        df = df.sort_values([player_col, 'season', 'week'])

        # Calculate EWMA per player - compute once with optional shift
        def _compute_ewma(x):
            ewma = x.ewm(span=span, min_periods=min_periods).mean()
            return ewma.shift(1) if no_leakage else ewma

        return df.groupby(player_col)[stat_col].transform(_compute_ewma)

    # =========================================================================
    # DEFENSIVE EPA (Single Source of Truth)
    # =========================================================================

    def calculate_defense_epa(
        self,
        pbp: pd.DataFrame,
        play_type: str,  # 'pass' or 'run'
        trailing_weeks: int = 4,
        no_leakage: bool = True
    ) -> pd.DataFrame:
        """
        Calculate trailing defensive EPA per team.

        THIS IS THE CANONICAL IMPLEMENTATION for defense EPA.

        Args:
            pbp: Play-by-play DataFrame
            play_type: 'pass' for pass defense, 'run' for rush defense
            trailing_weeks: Number of weeks for rolling average
            no_leakage: If True, use only prior weeks' data

        Returns:
            DataFrame with columns: [defteam, season, week, trailing_def_epa]
        """
        cache_key = f"{play_type}_{trailing_weeks}"
        if self.cache_enabled and cache_key in self._defense_epa_cache:
            return self._defense_epa_cache[cache_key]

        # Filter to play type
        plays = pbp[pbp['play_type'] == play_type].copy()

        # Aggregate EPA per team per week
        def_epa = plays.groupby(['defteam', 'season', 'week']).agg(
            def_epa=('epa', 'mean'),
            plays=('epa', 'count')
        ).reset_index()

        # Sort for proper rolling calculation
        def_epa = def_epa.sort_values(['defteam', 'season', 'week'])

        # Calculate trailing EPA with shift to prevent leakage
        if no_leakage:
            def_epa['trailing_def_epa'] = def_epa.groupby('defteam')['def_epa'].transform(
                lambda x: x.shift(1).rolling(trailing_weeks, min_periods=1).mean()
            )
        else:
            def_epa['trailing_def_epa'] = def_epa.groupby('defteam')['def_epa'].transform(
                lambda x: x.rolling(trailing_weeks, min_periods=1).mean()
            )

        if self.cache_enabled:
            self._defense_epa_cache[cache_key] = def_epa

        return def_epa

    def get_rush_defense_epa(
        self,
        opponent: str,
        season: int,
        week: int,
        pbp: pd.DataFrame = None
    ) -> float:
        """
        Get trailing rush defense EPA for opponent.

        Args:
            opponent: Team abbreviation (e.g., 'KC')
            season: Season year
            week: Week number (gets EPA BEFORE this week)
            pbp: Play-by-play data (loads if not provided)

        Returns:
            Trailing rush defense EPA (negative = good defense)
        """
        if pbp is None:
            pbp = self._load_pbp(season)

        def_epa = self.calculate_defense_epa(pbp, 'run', no_leakage=True)

        # Get EPA for opponent before this week
        row = def_epa[
            (def_epa['defteam'] == opponent) &
            (def_epa['season'] == season) &
            (def_epa['week'] == week)
        ]

        if len(row) == 0:
            logger.debug(f"No rush defense EPA for {opponent} week {week}, using league average")
            return LEAGUE_AVG.rush_epa

        return float(row['trailing_def_epa'].iloc[0])

    def get_pass_defense_epa(
        self,
        opponent: str,
        season: int,
        week: int,
        pbp: pd.DataFrame = None
    ) -> float:
        """
        Get trailing pass defense EPA for opponent.

        Args:
            opponent: Team abbreviation
            season: Season year
            week: Week number
            pbp: Play-by-play data

        Returns:
            Trailing pass defense EPA (negative = good defense)
        """
        if pbp is None:
            pbp = self._load_pbp(season)

        def_epa = self.calculate_defense_epa(pbp, 'pass', no_leakage=True)

        row = def_epa[
            (def_epa['defteam'] == opponent) &
            (def_epa['season'] == season) &
            (def_epa['week'] == week)
        ]

        if len(row) == 0:
            logger.debug(f"No pass defense EPA for {opponent} week {week}, using league average")
            return LEAGUE_AVG.pass_epa

        return float(row['trailing_def_epa'].iloc[0])

    # =========================================================================
    # LINE VS TRAILING (Primary Feature)
    # =========================================================================

    def calculate_line_vs_trailing(
        self,
        line: float,
        trailing_stat: float,
        method: str = 'difference'
    ) -> float:
        """
        Calculate line vs trailing stat.

        THIS IS THE PRIMARY FEATURE for all classifiers.

        Args:
            line: Vegas prop line
            trailing_stat: Player's trailing average
            method: 'difference' (line - trailing) or 'ratio' (line / trailing)

        Returns:
            Line vs trailing value
        """
        if trailing_stat == 0 or pd.isna(trailing_stat):
            return 0.0

        if method == 'ratio':
            return line / trailing_stat
        else:
            return line - trailing_stat

    # =========================================================================
    # WEEKLY STATS FEATURES
    # =========================================================================

    def get_target_share(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get player's trailing target share.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            trailing_weeks: Weeks to average

        Returns:
            Trailing target share (0.0 to 1.0)
        """
        weekly = self._load_weekly_stats()

        player_data = weekly[
            (weekly['player_id'] == player_id) &
            (weekly['season'] == season) &
            (weekly['week'] < week) &
            (weekly['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0 or 'target_share' not in player_data.columns:
            return 0.0

        return float(player_data['target_share'].mean())

    def get_air_yards_share(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get player's trailing air yards share.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            trailing_weeks: Weeks to average

        Returns:
            Trailing air yards share (0.0 to 1.0)
        """
        weekly = self._load_weekly_stats()

        player_data = weekly[
            (weekly['player_id'] == player_id) &
            (weekly['season'] == season) &
            (weekly['week'] < week) &
            (weekly['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0 or 'air_yards_share' not in player_data.columns:
            return 0.0

        return float(player_data['air_yards_share'].mean())

    # =========================================================================
    # COMPLETION PERCENTAGE ALLOWED
    # =========================================================================

    def get_completion_pct_allowed(
        self,
        opponent: str,
        season: int,
        week: int,
        pbp: pd.DataFrame = None
    ) -> float:
        """
        Get opponent's trailing completion percentage allowed.

        Args:
            opponent: Team abbreviation
            season: Season year
            week: Week number
            pbp: Play-by-play data

        Returns:
            Completion percentage allowed (0.0 to 1.0)
        """
        if pbp is None:
            pbp = self._load_pbp(season)

        # Filter to pass plays before this week
        pass_plays = pbp[
            (pbp['play_type'] == 'pass') &
            (pbp['defteam'] == opponent) &
            (pbp['season'] == season) &
            (pbp['week'] < week)
        ]

        if len(pass_plays) == 0:
            return LEAGUE_AVG.completion_pct

        completions = pass_plays['complete_pass'].sum()
        attempts = len(pass_plays[pass_plays['incomplete_pass'].notna() | pass_plays['complete_pass'].notna()])

        if attempts == 0:
            return LEAGUE_AVG.completion_pct

        return completions / attempts

    # =========================================================================
    # SNAP COUNT FEATURES (V2 - NEW)
    # =========================================================================

    def get_snap_share(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = None
    ) -> float:
        """
        Get player's trailing offensive snap percentage.

        USES shift(1) - only data from weeks BEFORE target week.
        Handles cross-season boundaries for early-season weeks.

        Args:
            player_name: Player name (matched against 'player' column)
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average (default: self.default_trailing_weeks)

        Returns:
            Trailing snap percentage (0.0 to 1.0, e.g., 0.85 = 85% of snaps)
        """
        if trailing_weeks is None:
            trailing_weeks = self.default_trailing_weeks

        snap = self._load_snap_counts()
        if len(snap) == 0:
            return LEAGUE_AVG.snap_share

        # Add player_lower for consistent matching
        snap_lower = snap.copy()
        snap_lower['player_lower'] = snap_lower['player'].str.lower()
        player_lower = player_name.lower()

        # Use cross-season aware window
        player_data = self._get_trailing_window(
            snap_lower, 'player_lower', player_lower, season, week, trailing_weeks
        )

        if len(player_data) == 0 or 'offense_pct' not in player_data.columns:
            return LEAGUE_AVG.snap_share

        # offense_pct is already 0-1 in NFLverse data
        snap_pct = player_data['offense_pct'].mean()
        return float(snap_pct) if pd.notna(snap_pct) else LEAGUE_AVG.snap_share

    def get_snap_trend(
        self,
        player_name: str,
        season: int,
        week: int,
        trailing_weeks: int = None
    ) -> float:
        """
        Calculate snap share trend (increasing/decreasing usage).

        Compares last 2 weeks vs prior 2 weeks.

        Args:
            player_name: Player name
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to look back (default: self.default_trailing_weeks)

        Returns:
            Positive = increasing usage, Negative = decreasing usage
            Typical range: -0.3 to +0.3
        """
        if trailing_weeks is None:
            trailing_weeks = self.default_trailing_weeks

        snap = self._load_snap_counts()
        if len(snap) == 0:
            return 0.0

        # Add player_lower for consistent matching
        snap_lower = snap.copy()
        snap_lower['player_lower'] = snap_lower['player'].str.lower()
        player_lower = player_name.lower()

        # Use cross-season aware window
        player_data = self._get_trailing_window(
            snap_lower, 'player_lower', player_lower, season, week, trailing_weeks
        ).sort_values('week')

        if len(player_data) < 3:
            return 0.0  # Not enough data for trend

        # Split into recent vs older
        mid = len(player_data) // 2
        older = player_data.iloc[:mid]['offense_pct'].mean()
        recent = player_data.iloc[mid:]['offense_pct'].mean()

        return float(recent - older) if (pd.notna(recent) and pd.notna(older)) else 0.0

    # =========================================================================
    # NGS RECEIVING FEATURES (V2 - NEW)
    # =========================================================================

    def get_avg_separation(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get receiver's trailing average separation at catch point.

        Higher separation = easier catches = more likely to hit receptions.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Average separation in yards (typically 2.0-4.0)
        """
        ngs = self._load_ngs_receiving()
        if len(ngs) == 0:
            return LEAGUE_AVG.separation

        # Filter to player's prior weeks
        player_data = ngs[
            (ngs['player_gsis_id'] == player_id) &
            (ngs['season'] == season) &
            (ngs['week'] < week) &
            (ngs['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0:
            return LEAGUE_AVG.separation

        sep = player_data['avg_separation'].mean()
        return float(sep) if pd.notna(sep) else LEAGUE_AVG.separation

    def get_avg_cushion(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get average cushion (yards between WR and DB at snap).

        Higher cushion = more space to work with.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Average cushion in yards (typically 4.0-8.0)
        """
        ngs = self._load_ngs_receiving()
        if len(ngs) == 0:
            return LEAGUE_AVG.cushion

        player_data = ngs[
            (ngs['player_gsis_id'] == player_id) &
            (ngs['season'] == season) &
            (ngs['week'] < week) &
            (ngs['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0:
            return LEAGUE_AVG.cushion

        cushion = player_data['avg_cushion'].mean()
        return float(cushion) if pd.notna(cushion) else LEAGUE_AVG.cushion

    def get_yac_above_expectation(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get yards after catch above expectation.

        Positive = player beats expected YAC.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            YAC above expectation (can be negative)
        """
        ngs = self._load_ngs_receiving()
        if len(ngs) == 0:
            return 0.0

        player_data = ngs[
            (ngs['player_gsis_id'] == player_id) &
            (ngs['season'] == season) &
            (ngs['week'] < week) &
            (ngs['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0:
            return 0.0

        yac_ae = player_data['avg_yac_above_expectation'].mean()
        return float(yac_ae) if pd.notna(yac_ae) else 0.0

    # =========================================================================
    # NGS RUSHING FEATURES (V2 - NEW)
    # =========================================================================

    def get_eight_box_rate(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get percentage of rush attempts facing 8+ defenders in box.

        Higher = harder rushing conditions.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            8+ box rate (0.0 to 1.0)
        """
        ngs = self._load_ngs_rushing()
        if len(ngs) == 0:
            return LEAGUE_AVG.eight_box_rate

        player_data = ngs[
            (ngs['player_gsis_id'] == player_id) &
            (ngs['season'] == season) &
            (ngs['week'] < week) &
            (ngs['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0:
            return LEAGUE_AVG.eight_box_rate

        # Column is percent_attempts_gte_eight_defenders (0-100 in NFLverse)
        box_rate = player_data['percent_attempts_gte_eight_defenders'].mean()
        if pd.notna(box_rate):
            # Convert from percentage to ratio if needed
            return float(box_rate / 100.0 if box_rate > 1 else box_rate)
        return LEAGUE_AVG.eight_box_rate

    def get_rush_efficiency(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get NGS rushing efficiency metric.

        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player GSIS ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Rush efficiency (typically -5 to +5)
        """
        ngs = self._load_ngs_rushing()
        if len(ngs) == 0:
            return 0.0

        player_data = ngs[
            (ngs['player_gsis_id'] == player_id) &
            (ngs['season'] == season) &
            (ngs['week'] < week) &
            (ngs['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0:
            return 0.0

        eff = player_data['efficiency'].mean()
        return float(eff) if pd.notna(eff) else 0.0

    def get_opponent_eight_box_rate(
        self,
        opponent: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get how often opponent defense uses 8+ man box.

        Useful for predicting rush difficulty.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            opponent: Team abbreviation
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Opponent's 8+ box usage rate (0.0 to 1.0)
        """
        ngs = self._load_ngs_rushing()
        if len(ngs) == 0:
            return LEAGUE_AVG.eight_box_rate

        # Aggregate by team opponent faced
        # This requires knowing which team each rusher played against
        # For simplicity, use league average from the data
        recent_data = ngs[
            (ngs['season'] == season) &
            (ngs['week'] < week) &
            (ngs['week'] >= week - trailing_weeks)
        ]

        if len(recent_data) == 0:
            return LEAGUE_AVG.eight_box_rate

        # Get average 8-box rate across all rushers
        avg_rate = recent_data['percent_attempts_gte_eight_defenders'].mean()
        if pd.notna(avg_rate):
            return float(avg_rate / 100.0 if avg_rate > 1 else avg_rate)
        return LEAGUE_AVG.eight_box_rate

    # =========================================================================
    # WOPR, RACR, EPA FEATURES (V2 - NEW)
    # =========================================================================

    def get_trailing_wopr(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get trailing WOPR (Weighted Opportunity Rating).

        WOPR = 1.5 × target_share + 0.7 × air_yards_share
        Higher WOPR = more receiving opportunity.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Trailing WOPR (typically 0.1-0.6 for relevant receivers)
        """
        weekly = self._load_weekly_stats()

        player_data = weekly[
            (weekly['player_id'] == player_id) &
            (weekly['season'] == season) &
            (weekly['week'] < week) &
            (weekly['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0 or 'wopr' not in player_data.columns:
            return 0.0

        wopr = player_data['wopr'].mean()
        return float(wopr) if pd.notna(wopr) else 0.0

    def get_trailing_racr(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get trailing RACR (Receiver Air Conversion Ratio).

        RACR = receiving_yards / receiving_air_yards
        >1.0 = gains more yards than air yards (good YAC)
        <1.0 = loses yards from air yards (poor YAC)
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Trailing RACR (typically 0.7-1.3)
        """
        weekly = self._load_weekly_stats()

        player_data = weekly[
            (weekly['player_id'] == player_id) &
            (weekly['season'] == season) &
            (weekly['week'] < week) &
            (weekly['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0 or 'racr' not in player_data.columns:
            return 1.0  # Neutral default

        racr = player_data['racr'].mean()
        return float(racr) if pd.notna(racr) else 1.0

    def get_trailing_receiving_epa(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get player's trailing receiving EPA.

        Measures efficiency of receiving production.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Trailing receiving EPA (can be negative)
        """
        weekly = self._load_weekly_stats()

        player_data = weekly[
            (weekly['player_id'] == player_id) &
            (weekly['season'] == season) &
            (weekly['week'] < week) &
            (weekly['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0 or 'receiving_epa' not in player_data.columns:
            return 0.0

        epa = player_data['receiving_epa'].mean()
        return float(epa) if pd.notna(epa) else 0.0

    def get_trailing_rushing_epa(
        self,
        player_id: str,
        season: int,
        week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Get player's trailing rushing EPA.

        Measures efficiency of rushing production.
        USES shift(1) - only data from weeks BEFORE target week.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            trailing_weeks: Number of weeks to average

        Returns:
            Trailing rushing EPA (can be negative)
        """
        weekly = self._load_weekly_stats()

        player_data = weekly[
            (weekly['player_id'] == player_id) &
            (weekly['season'] == season) &
            (weekly['week'] < week) &
            (weekly['week'] >= week - trailing_weeks)
        ]

        if len(player_data) == 0 or 'rushing_epa' not in player_data.columns:
            return 0.0

        epa = player_data['rushing_epa'].mean()
        return float(epa) if pd.notna(epa) else 0.0

    # =========================================================================
    # V12 BETTING HISTORY FEATURES
    # =========================================================================

    def calculate_player_under_rate(
        self,
        historical_odds: pd.DataFrame,
        player_norm: str,
        target_global_week: int,
        rolling_games: int = 10,
        min_games: int = 3
    ) -> float:
        """
        Calculate player's historical under hit rate.

        THIS IS THE CANONICAL IMPLEMENTATION for player_under_rate.

        Args:
            historical_odds: Historical odds DataFrame with 'under_hit' column
            player_norm: Normalized player name
            target_global_week: Week to calculate for (uses ONLY prior weeks)
            rolling_games: Number of games for rolling average
            min_games: Minimum games required

        Returns:
            Player's under rate (0.0 to 1.0), defaults to 0.5 if insufficient data
        """
        # Filter to strictly prior weeks
        hist = historical_odds[
            (historical_odds['player_norm'] == player_norm) &
            (historical_odds['global_week'] < target_global_week)
        ].sort_values('global_week')

        if len(hist) < min_games:
            return 0.5  # Neutral default

        # Rolling under rate
        under_rate = hist['under_hit'].rolling(rolling_games, min_periods=min_games).mean().iloc[-1]
        return float(under_rate) if pd.notna(under_rate) else 0.5

    def calculate_player_bias(
        self,
        historical_odds: pd.DataFrame,
        player_norm: str,
        target_global_week: int,
        rolling_games: int = 10,
        min_games: int = 3
    ) -> float:
        """
        Calculate player's historical bias (actual - line).

        THIS IS THE CANONICAL IMPLEMENTATION for player_bias.

        Args:
            historical_odds: Historical odds DataFrame with 'actual_stat' and 'line' columns
            player_norm: Normalized player name
            target_global_week: Week to calculate for (uses ONLY prior weeks)
            rolling_games: Number of games for rolling average
            min_games: Minimum games required

        Returns:
            Player's average (actual - line), defaults to 0.0 if insufficient data
        """
        hist = historical_odds[
            (historical_odds['player_norm'] == player_norm) &
            (historical_odds['global_week'] < target_global_week)
        ].sort_values('global_week')

        if len(hist) < min_games:
            return 0.0  # Neutral default

        hist = hist.copy()
        hist['actual_minus_line'] = hist['actual_stat'] - hist['line']
        bias = hist['actual_minus_line'].rolling(rolling_games, min_periods=min_games).mean().iloc[-1]
        return float(bias) if pd.notna(bias) else 0.0

    def calculate_market_under_rate(
        self,
        historical_odds: pd.DataFrame,
        target_global_week: int,
        trailing_weeks: int = 4
    ) -> float:
        """
        Calculate market-wide under hit rate (regime indicator).

        THIS IS THE CANONICAL IMPLEMENTATION for market_under_rate.

        Args:
            historical_odds: Historical odds DataFrame for a specific market
            target_global_week: Week to calculate for (uses ONLY prior weeks)
            trailing_weeks: Number of weeks to average

        Returns:
            Market under rate (0.0 to 1.0), defaults to 0.5 if no data
        """
        hist = historical_odds[historical_odds['global_week'] < target_global_week]

        if len(hist) == 0:
            return 0.5

        # Weekly under rates
        weekly_rates = hist.groupby('global_week')['under_hit'].mean()

        # Get last N weeks
        recent_weeks = sorted(weekly_rates.index)[-trailing_weeks:]
        if len(recent_weeks) == 0:
            return 0.5

        return float(weekly_rates.loc[recent_weeks].mean())

    def calculate_v12_features(
        self,
        line: float,
        trailing_stat: float,
        player_under_rate: float,
        player_bias: float,
        market_under_rate: float
    ) -> Dict[str, float]:
        """
        Calculate all V12 features including interactions.

        THIS IS THE CANONICAL IMPLEMENTATION for V12 feature set.

        Args:
            line: Vegas line
            trailing_stat: Player's trailing average
            player_under_rate: Player's historical under rate
            player_bias: Player's historical (actual - line) average
            market_under_rate: Market-wide under rate

        Returns:
            Dictionary with all 12 V12 features
        """
        # Primary features
        lvt = self.calculate_line_vs_trailing(line, trailing_stat, method='difference')
        line_in_sweet_spot = 1.0 if 3.5 <= line <= 7.5 else 0.0

        # Interaction features
        features = {
            'line_vs_trailing': lvt,
            'line_level': line,
            'line_in_sweet_spot': line_in_sweet_spot,
            'player_under_rate': player_under_rate,
            'player_bias': player_bias,
            'market_under_rate': market_under_rate,
            'LVT_x_player_tendency': lvt * (player_under_rate - 0.5),
            'LVT_x_player_bias': lvt * player_bias,
            'LVT_x_regime': lvt * (market_under_rate - 0.5),
            'LVT_in_sweet_spot': lvt * line_in_sweet_spot,
            'market_bias_strength': abs(market_under_rate - 0.5) * 2,
            'player_market_aligned': (player_under_rate - 0.5) * (market_under_rate - 0.5),
        }

        return features

    def extract_v12_features_for_week(
        self,
        odds_with_trailing: pd.DataFrame,
        all_historical_odds: pd.DataFrame,
        target_global_week: int,
        market: str
    ) -> pd.DataFrame:
        """
        Extract all V12 features for a specific week.

        THIS IS THE CANONICAL IMPLEMENTATION for V12 feature extraction.
        Uses ONLY data from prior weeks - NO DATA LEAKAGE.

        Args:
            odds_with_trailing: Odds data with trailing stats already merged
            all_historical_odds: Full historical odds for calculating betting features
            target_global_week: Week to extract features for
            market: Market name (player_receptions, player_rush_yds, etc.)

        Returns:
            DataFrame with all V12 features for the target week

        Raises:
            ValueError: If required columns are missing from input DataFrames
        """
        # Validate required columns
        required_cols = ['player_norm', 'global_week', 'market', 'line', 'under_hit', 'season', 'week']
        missing = [c for c in required_cols if c not in odds_with_trailing.columns]
        if missing:
            raise ValueError(f"Missing required columns in odds_with_trailing: {missing}")

        hist_required = ['player_norm', 'global_week', 'market', 'under_hit', 'actual_stat', 'line']
        hist_missing = [c for c in hist_required if c not in all_historical_odds.columns]
        if hist_missing:
            raise ValueError(f"Missing required columns in all_historical_odds: {hist_missing}")

        # Map market to stat column
        stat_col_map = {
            'player_receptions': 'receptions',
            'player_reception_yds': 'receiving_yards',
            'player_rush_yds': 'rushing_yards',
            'player_pass_yds': 'passing_yards',
        }
        stat_col = stat_col_map.get(market)
        trailing_col = f'trailing_{stat_col}'

        if trailing_col not in odds_with_trailing.columns:
            return pd.DataFrame()

        # Get data for target week
        target_data = odds_with_trailing[
            (odds_with_trailing['global_week'] == target_global_week) &
            (odds_with_trailing['market'] == market)
        ].copy()

        if len(target_data) == 0:
            return pd.DataFrame()

        # Get historical odds for this market (strictly before target week)
        hist_odds = all_historical_odds[
            (all_historical_odds['market'] == market) &
            (all_historical_odds['global_week'] < target_global_week)
        ].copy()

        # Calculate features for each player
        all_features = []

        for idx, row in target_data.iterrows():
            player_norm = row['player_norm']
            line = row['line']
            trailing_stat = row[trailing_col]

            # Calculate betting history features
            if len(hist_odds) == 0:
                player_under_rate = 0.5
                player_bias = 0.0
                market_under_rate = 0.5
            else:
                player_under_rate = self.calculate_player_under_rate(
                    hist_odds, player_norm, target_global_week
                )
                player_bias = self.calculate_player_bias(
                    hist_odds, player_norm, target_global_week
                )
                market_under_rate = self.calculate_market_under_rate(
                    hist_odds, target_global_week
                )

            # Calculate all V12 features
            features = self.calculate_v12_features(
                line, trailing_stat, player_under_rate, player_bias, market_under_rate
            )

            # Add row identifiers
            features['player_norm'] = player_norm
            features['global_week'] = target_global_week
            features['season'] = row['season']
            features['week'] = row['week']
            features['line'] = line
            features['under_hit'] = row['under_hit']
            features['actual_stat'] = row.get('actual_stat', row.get(stat_col, np.nan))

            all_features.append(features)

        return pd.DataFrame(all_features)

    def precompute_v2_features(
        self,
        odds_data: pd.DataFrame,
        market: str,
        player_id_map: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Pre-compute V2 features for all rows in odds data using vectorized operations.

        THIS IS THE CANONICAL IMPLEMENTATION for V2 feature precomputation.
        Uses merge operations instead of iterrows() for 100-1000x speedup.

        Args:
            odds_data: Odds data with player, season, week columns
            market: Market name (player_receptions, player_rush_yds, etc.)
            player_id_map: Optional dict mapping player names (lowercase) to player IDs

        Returns:
            DataFrame with V2 features added
        """
        market_odds = odds_data[odds_data['market'] == market].copy()

        if len(market_odds) == 0:
            return market_odds

        # Load V2 data sources once
        snap_counts = self._load_snap_counts()
        ngs_rec = self._load_ngs_receiving()
        ngs_rush = self._load_ngs_rushing()
        weekly_stats = self._load_weekly_stats()

        # Add player_id if mapping provided
        if player_id_map:
            market_odds['player_id'] = market_odds['player'].str.lower().str.strip().map(player_id_map)
        market_odds['player_lower'] = market_odds['player'].str.lower().str.strip()

        # Initialize V2 feature columns with defaults
        market_odds['snap_share'] = LEAGUE_AVG.snap_share
        market_odds['snap_trend'] = 0.0

        if market in ['player_receptions', 'player_reception_yds']:
            market_odds['avg_separation'] = LEAGUE_AVG.separation
            market_odds['avg_cushion'] = LEAGUE_AVG.cushion
            market_odds['yac_above_expectation'] = 0.0
            market_odds['trailing_wopr'] = 0.0
            market_odds['trailing_racr'] = 1.0
            market_odds['trailing_receiving_epa'] = 0.0
        elif market == 'player_rush_yds':
            market_odds['eight_box_rate'] = LEAGUE_AVG.eight_box_rate
            market_odds['rush_efficiency'] = 0.0
            market_odds['opp_eight_box_rate'] = LEAGUE_AVG.eight_box_rate
            market_odds['trailing_rushing_epa'] = 0.0

        # =====================================================================
        # VECTORIZED SNAP FEATURES (replaces iterrows loop)
        # =====================================================================
        if len(snap_counts) > 0 and 'player' in snap_counts.columns and 'offense_pct' in snap_counts.columns:
            snap_counts = snap_counts.copy()
            snap_counts['player_lower'] = snap_counts['player'].str.lower().str.strip()
            snap_counts = snap_counts.sort_values(['player_lower', 'season', 'week'])

            # Pre-compute trailing snap share using groupby + transform with shift
            snap_counts['trailing_snap_share'] = snap_counts.groupby(['player_lower', 'season'])['offense_pct'].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            )

            # Compute snap trend: recent 2 weeks vs older 2 weeks
            snap_counts['snap_recent_2'] = snap_counts.groupby(['player_lower', 'season'])['offense_pct'].transform(
                lambda x: x.shift(1).rolling(2, min_periods=1).mean()
            )
            snap_counts['snap_older_2'] = snap_counts.groupby(['player_lower', 'season'])['offense_pct'].transform(
                lambda x: x.shift(3).rolling(2, min_periods=1).mean()
            )
            snap_counts['snap_trend_computed'] = snap_counts['snap_recent_2'] - snap_counts['snap_older_2']

            # Merge trailing features into market_odds
            snap_lookup = snap_counts[['player_lower', 'season', 'week', 'trailing_snap_share', 'snap_trend_computed']].drop_duplicates()
            market_odds = market_odds.merge(
                snap_lookup,
                on=['player_lower', 'season', 'week'],
                how='left'
            )

            # Update columns with merged values, keeping defaults where null
            market_odds['snap_share'] = market_odds['trailing_snap_share'].fillna(LEAGUE_AVG.snap_share)
            market_odds['snap_trend'] = market_odds['snap_trend_computed'].fillna(0.0)
            market_odds = market_odds.drop(columns=['trailing_snap_share', 'snap_trend_computed'], errors='ignore')

        # =====================================================================
        # VECTORIZED NGS RECEIVING FEATURES (replaces iterrows loop)
        # =====================================================================
        if market in ['player_receptions', 'player_reception_yds'] and len(ngs_rec) > 0 and 'player_gsis_id' in ngs_rec.columns:
            ngs_rec = ngs_rec.copy()
            ngs_rec = ngs_rec.sort_values(['player_gsis_id', 'season', 'week'])

            # Pre-compute trailing averages for NGS metrics
            for col, default in [('avg_separation', LEAGUE_AVG.separation),
                                 ('avg_cushion', LEAGUE_AVG.cushion),
                                 ('avg_yac_above_expectation', 0.0)]:
                if col in ngs_rec.columns:
                    ngs_rec[f'trailing_{col}'] = ngs_rec.groupby(['player_gsis_id', 'season'])[col].transform(
                        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                    )

            # Build lookup table
            ngs_cols = ['player_gsis_id', 'season', 'week']
            ngs_cols += [f'trailing_{c}' for c in ['avg_separation', 'avg_cushion', 'avg_yac_above_expectation']
                        if f'trailing_{c}' in ngs_rec.columns]
            ngs_lookup = ngs_rec[ngs_cols].drop_duplicates()
            ngs_lookup = ngs_lookup.rename(columns={'player_gsis_id': 'player_id'})

            if 'player_id' in market_odds.columns:
                market_odds = market_odds.merge(ngs_lookup, on=['player_id', 'season', 'week'], how='left')

                # Update with merged values
                if 'trailing_avg_separation' in market_odds.columns:
                    market_odds['avg_separation'] = market_odds['trailing_avg_separation'].fillna(LEAGUE_AVG.separation)
                    market_odds = market_odds.drop(columns=['trailing_avg_separation'], errors='ignore')
                if 'trailing_avg_cushion' in market_odds.columns:
                    market_odds['avg_cushion'] = market_odds['trailing_avg_cushion'].fillna(LEAGUE_AVG.cushion)
                    market_odds = market_odds.drop(columns=['trailing_avg_cushion'], errors='ignore')
                if 'trailing_avg_yac_above_expectation' in market_odds.columns:
                    market_odds['yac_above_expectation'] = market_odds['trailing_avg_yac_above_expectation'].fillna(0.0)
                    market_odds = market_odds.drop(columns=['trailing_avg_yac_above_expectation'], errors='ignore')

        # =====================================================================
        # VECTORIZED WEEKLY STATS FEATURES (WOPR, RACR, EPA) - replaces iterrows
        # =====================================================================
        if market in ['player_receptions', 'player_reception_yds'] and len(weekly_stats) > 0:
            ws = weekly_stats.copy()
            ws = ws.sort_values(['player_id', 'season', 'week'])

            # Pre-compute trailing averages
            for col, default in [('wopr', 0.0), ('racr', 1.0), ('receiving_epa', 0.0)]:
                if col in ws.columns:
                    ws[f'trailing_{col}'] = ws.groupby(['player_id', 'season'])[col].transform(
                        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                    )

            # Build lookup table
            ws_cols = ['player_id', 'season', 'week']
            ws_cols += [f'trailing_{c}' for c in ['wopr', 'racr', 'receiving_epa'] if f'trailing_{c}' in ws.columns]
            ws_lookup = ws[ws_cols].drop_duplicates()

            if 'player_id' in market_odds.columns:
                market_odds = market_odds.merge(ws_lookup, on=['player_id', 'season', 'week'], how='left')

                # Update with merged values
                if 'trailing_wopr' in market_odds.columns and 'trailing_wopr_y' not in market_odds.columns:
                    pass  # Already has default
                elif 'trailing_wopr_y' in market_odds.columns:
                    market_odds['trailing_wopr'] = market_odds['trailing_wopr_y'].fillna(0.0)
                    market_odds = market_odds.drop(columns=['trailing_wopr_x', 'trailing_wopr_y'], errors='ignore')

                if 'trailing_racr' in market_odds.columns and 'trailing_racr_y' not in market_odds.columns:
                    pass
                elif 'trailing_racr_y' in market_odds.columns:
                    market_odds['trailing_racr'] = market_odds['trailing_racr_y'].fillna(1.0)
                    market_odds = market_odds.drop(columns=['trailing_racr_x', 'trailing_racr_y'], errors='ignore')

                if 'trailing_receiving_epa' in market_odds.columns and 'trailing_receiving_epa_y' not in market_odds.columns:
                    pass
                elif 'trailing_receiving_epa_y' in market_odds.columns:
                    market_odds['trailing_receiving_epa'] = market_odds['trailing_receiving_epa_y'].fillna(0.0)
                    market_odds = market_odds.drop(columns=['trailing_receiving_epa_x', 'trailing_receiving_epa_y'], errors='ignore')

        # =====================================================================
        # VECTORIZED NGS RUSHING FEATURES (replaces iterrows loop)
        # =====================================================================
        if market == 'player_rush_yds' and len(ngs_rush) > 0 and 'player_gsis_id' in ngs_rush.columns:
            ngs = ngs_rush.copy()
            ngs = ngs.sort_values(['player_gsis_id', 'season', 'week'])

            # Pre-compute trailing averages
            if 'percent_attempts_gte_eight_defenders' in ngs.columns:
                # Normalize to 0-1 range
                ngs['eight_box_pct'] = ngs['percent_attempts_gte_eight_defenders'].apply(
                    lambda x: x / 100.0 if pd.notna(x) and x > 1 else x
                )
                ngs['trailing_eight_box'] = ngs.groupby(['player_gsis_id', 'season'])['eight_box_pct'].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )
            if 'efficiency' in ngs.columns:
                ngs['trailing_efficiency'] = ngs.groupby(['player_gsis_id', 'season'])['efficiency'].transform(
                    lambda x: x.shift(1).rolling(4, min_periods=1).mean()
                )

            # Build lookup table
            ngs_cols = ['player_gsis_id', 'season', 'week']
            if 'trailing_eight_box' in ngs.columns:
                ngs_cols.append('trailing_eight_box')
            if 'trailing_efficiency' in ngs.columns:
                ngs_cols.append('trailing_efficiency')
            ngs_lookup = ngs[ngs_cols].drop_duplicates()
            ngs_lookup = ngs_lookup.rename(columns={'player_gsis_id': 'player_id'})

            if 'player_id' in market_odds.columns:
                market_odds = market_odds.merge(ngs_lookup, on=['player_id', 'season', 'week'], how='left')

                if 'trailing_eight_box' in market_odds.columns:
                    market_odds['eight_box_rate'] = market_odds['trailing_eight_box'].fillna(LEAGUE_AVG.eight_box_rate)
                    market_odds = market_odds.drop(columns=['trailing_eight_box'], errors='ignore')
                if 'trailing_efficiency' in market_odds.columns:
                    market_odds['rush_efficiency'] = market_odds['trailing_efficiency'].fillna(0.0)
                    market_odds = market_odds.drop(columns=['trailing_efficiency'], errors='ignore')

        # =====================================================================
        # VECTORIZED RUSHING EPA FROM WEEKLY STATS
        # =====================================================================
        if market == 'player_rush_yds' and len(weekly_stats) > 0 and 'rushing_epa' in weekly_stats.columns:
            ws = weekly_stats.copy()
            ws = ws.sort_values(['player_id', 'season', 'week'])
            ws['trailing_rush_epa'] = ws.groupby(['player_id', 'season'])['rushing_epa'].transform(
                lambda x: x.shift(1).rolling(4, min_periods=1).mean()
            )

            ws_lookup = ws[['player_id', 'season', 'week', 'trailing_rush_epa']].drop_duplicates()

            if 'player_id' in market_odds.columns:
                market_odds = market_odds.merge(ws_lookup, on=['player_id', 'season', 'week'], how='left')
                market_odds['trailing_rushing_epa'] = market_odds['trailing_rush_epa'].fillna(0.0)
                market_odds = market_odds.drop(columns=['trailing_rush_epa'], errors='ignore')

        # Drop helper columns
        market_odds = market_odds.drop(columns=['player_lower'], errors='ignore')

        return market_odds

    def add_v2_features_to_df(
        self,
        df: pd.DataFrame,
        precomputed_v2: pd.DataFrame,
        market: str
    ) -> pd.DataFrame:
        """
        Add V2 features to a DataFrame using precomputed features.

        Args:
            df: DataFrame with V12 features
            precomputed_v2: Precomputed V2 features from precompute_v2_features()
            market: Market name

        Returns:
            DataFrame with V2 features added
        """
        result = df.copy()

        # V2 feature columns by market
        v2_snap = ['snap_share', 'snap_trend']
        v2_ngs_rec = ['avg_separation', 'avg_cushion', 'yac_above_expectation']
        v2_ngs_rush = ['eight_box_rate', 'rush_efficiency', 'opp_eight_box_rate']
        v2_opportunity = ['trailing_wopr', 'trailing_racr']
        v2_epa_rec = ['trailing_receiving_epa']
        v2_epa_rush = ['trailing_rushing_epa']

        all_v2 = v2_snap + v2_ngs_rec + v2_ngs_rush + v2_opportunity + v2_epa_rec + v2_epa_rush

        v2_cols = [c for c in precomputed_v2.columns if c in all_v2]

        if v2_cols:
            v2_lookup = precomputed_v2[['player_norm', 'season', 'week'] + v2_cols].drop_duplicates()
            result = result.merge(
                v2_lookup,
                on=['player_norm', 'season', 'week'],
                how='left',
                suffixes=('', '_v2')
            )

            # Fill missing with defaults
            for col in v2_cols:
                if col in result.columns:
                    default = 0.0
                    if 'racr' in col:
                        default = 1.0
                    elif 'rate' in col:
                        default = 0.5
                    result[col] = result[col].fillna(default)

        return result

    # =========================================================================
    # FULL FEATURE EXTRACTION
    # =========================================================================

    def extract_features_for_bet(
        self,
        player_name: str,
        player_id: str,
        team: str,
        opponent: str,
        position: str,
        market: str,
        line: float,
        season: int,
        week: int,
        trailing_stat: float,
        pbp: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Extract ALL features for a single bet.

        This is the main entry point for feature extraction.
        Returns a dictionary of all features needed for prediction.

        V2 UPDATE (2025-11-29): Added snap counts, NGS metrics, WOPR, RACR, EPA

        Args:
            player_name: Player name
            player_id: Player ID (for weekly stats lookup)
            team: Player's team
            opponent: Opponent team
            position: Player position
            market: Bet market (player_receptions, player_rush_yds, etc.)
            line: Vegas line
            season: Season year
            week: Week number
            trailing_stat: Pre-calculated trailing stat for this market
            pbp: Play-by-play data (optional, loads if not provided)

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # PRIMARY FEATURE
        features['line_vs_trailing'] = self.calculate_line_vs_trailing(line, trailing_stat)
        features['line_level'] = line
        features['line_in_sweet_spot'] = 1.0 if 3.5 <= line <= 7.5 else 0.0

        # DEFENSIVE EPA
        if market in ['player_rush_yds']:
            features['trailing_def_epa'] = self.get_rush_defense_epa(opponent, season, week, pbp)
        elif market in ['player_receptions', 'player_reception_yds', 'player_pass_yds']:
            features['trailing_def_epa'] = self.get_pass_defense_epa(opponent, season, week, pbp)

        # WEEKLY STATS FEATURES (if applicable)
        if market in ['player_receptions', 'player_reception_yds'] and player_id:
            features['target_share'] = self.get_target_share(player_id, season, week)
            features['air_yards_share'] = self.get_air_yards_share(player_id, season, week)

        # COMPLETION PCT ALLOWED (for receptions)
        if market == 'player_receptions':
            features['completion_pct_allowed'] = self.get_completion_pct_allowed(opponent, season, week, pbp)

        # =====================================================================
        # V2 FEATURES (NEW - 2025-11-29)
        # =====================================================================

        # SNAP COUNTS - All markets benefit from usage information
        if player_name:
            features['snap_share'] = self.get_snap_share(player_name, season, week)
            features['snap_trend'] = self.get_snap_trend(player_name, season, week)

        # NGS RECEIVING FEATURES - For receiving markets
        if market in ['player_receptions', 'player_reception_yds'] and player_id:
            features['avg_separation'] = self.get_avg_separation(player_id, season, week)
            features['avg_cushion'] = self.get_avg_cushion(player_id, season, week)
            features['yac_above_expectation'] = self.get_yac_above_expectation(player_id, season, week)
            features['trailing_wopr'] = self.get_trailing_wopr(player_id, season, week)
            features['trailing_racr'] = self.get_trailing_racr(player_id, season, week)
            features['trailing_receiving_epa'] = self.get_trailing_receiving_epa(player_id, season, week)

        # NGS RUSHING FEATURES - For rushing markets
        if market in ['player_rush_yds'] and player_id:
            features['eight_box_rate'] = self.get_eight_box_rate(player_id, season, week)
            features['rush_efficiency'] = self.get_rush_efficiency(player_id, season, week)
            features['opp_eight_box_rate'] = self.get_opponent_eight_box_rate(opponent, season, week)
            features['trailing_rushing_epa'] = self.get_trailing_rushing_epa(player_id, season, week)

        # PASSING EPA - For passing markets
        if market in ['player_pass_yds'] and player_id:
            # Use passing_epa from weekly stats if available
            weekly = self._load_weekly_stats()
            player_data = weekly[
                (weekly['player_id'] == player_id) &
                (weekly['season'] == season) &
                (weekly['week'] < week) &
                (weekly['week'] >= week - 4)
            ]
            if len(player_data) > 0 and 'passing_epa' in player_data.columns:
                epa = player_data['passing_epa'].mean()
                features['trailing_passing_epa'] = float(epa) if pd.notna(epa) else 0.0
            else:
                features['trailing_passing_epa'] = 0.0

        return features

    # =========================================================================
    # DATA LOADERS (with caching)
    # =========================================================================

    def _load_pbp(self, season: int) -> pd.DataFrame:
        """Load play-by-play data for a season."""
        if self.cache_enabled and season in self._pbp_cache:
            return self._pbp_cache[season]

        path = PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{season}.parquet'
        if not path.exists():
            raise FileNotFoundError(f"PBP file not found: {path}")

        pbp = pd.read_parquet(path)

        if self.cache_enabled:
            self._pbp_cache[season] = pbp

        return pbp

    def _load_weekly_stats(self) -> pd.DataFrame:
        """Load weekly player stats."""
        if self.cache_enabled and self._weekly_stats_cache is not None:
            return self._weekly_stats_cache

        path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if not path.exists():
            raise FileNotFoundError(f"Weekly stats not found: {path}")

        weekly = pd.read_parquet(path)

        if self.cache_enabled:
            self._weekly_stats_cache = weekly

        return weekly

    def _load_snap_counts(self) -> pd.DataFrame:
        """Load snap count data."""
        if self.cache_enabled and self._snap_counts_cache is not None:
            return self._snap_counts_cache

        path = PROJECT_ROOT / 'data' / 'nflverse' / 'snap_counts.parquet'
        if not path.exists():
            logger.warning(f"Snap counts not found: {path}")
            return pd.DataFrame()

        snap = pd.read_parquet(path)

        if self.cache_enabled:
            self._snap_counts_cache = snap

        return snap

    def _load_ngs_receiving(self) -> pd.DataFrame:
        """Load Next Gen Stats receiving data."""
        if self.cache_enabled and self._ngs_receiving_cache is not None:
            return self._ngs_receiving_cache

        # Try current season first, fall back to historical
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_receiving.parquet'
        if not path.exists():
            path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_receiving_historical.parquet'
        if not path.exists():
            logger.warning(f"NGS receiving data not found")
            return pd.DataFrame()

        ngs = pd.read_parquet(path)

        if self.cache_enabled:
            self._ngs_receiving_cache = ngs

        return ngs

    def _load_ngs_rushing(self) -> pd.DataFrame:
        """Load Next Gen Stats rushing data."""
        if self.cache_enabled and self._ngs_rushing_cache is not None:
            return self._ngs_rushing_cache

        path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_rushing.parquet'
        if not path.exists():
            path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_rushing_historical.parquet'
        if not path.exists():
            logger.warning(f"NGS rushing data not found")
            return pd.DataFrame()

        ngs = pd.read_parquet(path)

        if self.cache_enabled:
            self._ngs_rushing_cache = ngs

        return ngs

    # =========================================================================
    # V12 MODEL ARCHITECTURE
    # =========================================================================

    def build_v12_model_params(self, feature_cols: List[str]) -> Dict:
        """
        Build XGBoost parameters for V12 model architecture.

        THIS IS THE CANONICAL IMPLEMENTATION for V12 model architecture.
        All training and backtest scripts must use this.

        V12 Architecture:
        - LVT (line_vs_trailing) is the "hub" feature
        - All other features can only interact with LVT
        - Monotone constraints preserve directional relationships

        Args:
            feature_cols: List of feature column names

        Returns:
            Dictionary of XGBoost parameters including constraints
        """
        feature_indices = {f: i for i, f in enumerate(feature_cols)}
        lvt_idx = feature_indices.get('line_vs_trailing', 0)

        # Interaction constraints: LVT can interact with everything,
        # other features can only interact with LVT
        all_idx = list(range(len(feature_cols)))
        interaction_constraints = [[lvt_idx] + all_idx]
        for i in range(len(feature_cols)):
            if i != lvt_idx:
                interaction_constraints.append([lvt_idx, i])

        # Monotone constraints based on feature meaning
        # +1 = higher value → higher probability of UNDER
        # -1 = higher value → lower probability of UNDER
        # 0 = no constraint
        monotonic = []
        for feat in feature_cols:
            if feat == 'line_vs_trailing':
                monotonic.append(1)  # Higher LVT → more likely UNDER
            elif feat == 'player_under_rate':
                monotonic.append(1)  # Higher under rate → more likely UNDER
            elif feat == 'market_under_rate':
                monotonic.append(1)  # Higher market under → more likely UNDER
            elif feat == 'LVT_x_player_tendency':
                monotonic.append(1)
            elif feat == 'LVT_x_regime':
                monotonic.append(1)
            else:
                monotonic.append(0)  # No constraint

        monotonic_str = '(' + ','.join(map(str, monotonic)) + ')'

        return {
            # Base parameters
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            # max_bin: Increase from default 256 when using monotonic constraints
            # with tree_method='hist' to prevent overly shallow trees
            'max_bin': 512,
            'seed': 42,
            'verbosity': 0,
            # V12 constraints
            'interaction_constraints': str(interaction_constraints),
            'monotone_constraints': monotonic_str,
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._pbp_cache.clear()
        self._weekly_stats_cache = None
        self._defense_epa_cache.clear()
        self._snap_counts_cache = None
        self._ngs_receiving_cache = None
        self._ngs_rushing_cache = None


# =============================================================================
# SINGLETON INSTANCE (Thread-Safe)
# =============================================================================

@cache
def get_feature_engine() -> FeatureEngine:
    """
    Get the singleton FeatureEngine instance.

    Uses functools.cache for thread-safe lazy initialization.
    The first call creates the instance, subsequent calls return the cached instance.
    """
    return FeatureEngine()


# =============================================================================
# CONVENIENCE FUNCTIONS (for backwards compatibility)
# =============================================================================

def calculate_trailing_stat(
    df: pd.DataFrame,
    stat_col: str,
    **kwargs
) -> pd.Series:
    """Convenience wrapper for FeatureEngine.calculate_trailing_stat()"""
    return get_feature_engine().calculate_trailing_stat(df, stat_col, **kwargs)


def get_rush_defense_epa(opponent: str, season: int, week: int) -> float:
    """Convenience wrapper for FeatureEngine.get_rush_defense_epa()"""
    return get_feature_engine().get_rush_defense_epa(opponent, season, week)


def get_pass_defense_epa(opponent: str, season: int, week: int) -> float:
    """Convenience wrapper for FeatureEngine.get_pass_defense_epa()"""
    return get_feature_engine().get_pass_defense_epa(opponent, season, week)


# =============================================================================
# ROUTE METRICS (Integrated from route_metrics.py)
# =============================================================================

def calculate_tprr(targets: int, routes_run: float) -> float:
    """
    Calculate targets per route run (TPRR).

    TPRR = targets / routes_run
    Typical range: 0.15-0.35 for WR/TE
    """
    if routes_run == 0:
        return 0.0
    return targets / routes_run


def calculate_yards_per_route(receiving_yards: int, routes_run: float) -> float:
    """
    Calculate yards per route run (Y/RR).

    Y/RR = receiving_yards / routes_run
    Typical range: 1.5-3.5 for WR/TE
    """
    if routes_run == 0:
        return 0.0
    return receiving_yards / routes_run


def estimate_routes_run(snap_count: int, team_pass_attempts: int, team_total_plays: int) -> float:
    """
    Estimate routes run from snap counts.

    Formula: routes_run ≈ snaps × (team_pass_attempts / team_plays)
    """
    if team_total_plays == 0:
        return 0.0
    pass_play_rate = team_pass_attempts / team_total_plays
    return snap_count * pass_play_rate


def calculate_route_participation(routes_run: float, team_pass_attempts: int) -> float:
    """
    Calculate route participation percentage (RP).

    RP = routes_run / team_pass_attempts
    Returns: 0.0 to 1.0
    """
    if team_pass_attempts == 0:
        return 0.0
    rp = routes_run / team_pass_attempts
    return min(rp, 1.0)  # Cap at 100%


# =============================================================================
# BATCH TRAILING STATS CALCULATION
# =============================================================================

def calculate_all_trailing_stats(
    stats_df: pd.DataFrame,
    player_col: str = 'player_norm',
    stat_cols: List[str] = None
) -> pd.DataFrame:
    """
    Calculate trailing stats for multiple columns at once.

    THIS IS THE CANONICAL IMPLEMENTATION for batch trailing calculations.
    Use this instead of inline EWMA calculations.

    Args:
        stats_df: DataFrame with player stats
        player_col: Column identifying players
        stat_cols: List of columns to calculate trailing stats for

    Returns:
        DataFrame with trailing_{col} columns added
    """
    if stat_cols is None:
        stat_cols = ['receptions', 'receiving_yards', 'rushing_yards',
                     'passing_yards', 'targets', 'carries']

    engine = get_feature_engine()
    result = stats_df.copy()

    for col in stat_cols:
        if col in result.columns:
            result[f'trailing_{col}'] = engine.calculate_trailing_stat(
                df=result,
                stat_col=col,
                player_col=player_col,
                span=4,
                min_periods=1,
                no_leakage=True
            )

    return result
