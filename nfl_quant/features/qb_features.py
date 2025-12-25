"""
QB Feature Extraction Module

Extracts QB-specific features for the player_pass_yds market.
Unlike WR/RB features, this module focuses on:
- QB trailing stats (attempts, completion %, EPA)
- NGS passing metrics (time to throw, CPOE, aggressiveness)
- Game script estimation (spread-based pass attempt adjustment)
- QB starter detection

This replaces the receiver-centric features (avg_separation, target_share, etc.)
that are meaningless for QB props.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from configs.qb_model_config import (
    QB_FEATURES,
    QB_EWMA_SPAN,
    MIN_STARTER_CONFIDENCE,
    GAME_SCRIPT_MULTIPLIERS,
    GAME_SCRIPT_VOLATILITY,
    BASELINE_PASS_ATTEMPTS,
    QB_SWEET_SPOT_CENTER,
    QB_SWEET_SPOT_WIDTH,
)
from nfl_quant.features.qb_starter import (
    get_qb_starter_detector,
    QBRole,
)
from nfl_quant.features.ngs_features import NGSFeatureExtractor
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import PROJECT_ROOT, NFLVERSE_DIR

logger = logging.getLogger(__name__)


class QBFeatureExtractor:
    """
    Extracts QB-specific features for passing yards predictions.

    This is a separate extractor from the main batch_extractor because
    QB props require fundamentally different features than WR/RB props.
    """

    def __init__(self, data_dir: Path = None):
        """Initialize with data directory."""
        self.data_dir = data_dir or NFLVERSE_DIR
        self._weekly_stats: Optional[pd.DataFrame] = None
        self._ngs_extractor: Optional[NGSFeatureExtractor] = None
        self._qb_detector = get_qb_starter_detector()

    def _load_weekly_stats(self) -> pd.DataFrame:
        """Load weekly player stats."""
        if self._weekly_stats is not None:
            return self._weekly_stats

        # Try parquet first, then CSV
        parquet_path = self.data_dir / 'weekly_stats.parquet'
        csv_path = self.data_dir / 'player_stats_2024_2025.csv'

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            logger.warning(f"Weekly stats not found at {parquet_path} or {csv_path}")
            return pd.DataFrame()

        # Normalize player names
        if 'player_display_name' in df.columns:
            df['player_norm'] = df['player_display_name'].apply(normalize_player_name)

        # Filter to QBs only
        if 'position' in df.columns:
            df = df[df['position'] == 'QB'].copy()

        # Sort for proper EWMA calculation
        df = df.sort_values(['player_norm', 'season', 'week'])

        # Add global week for cross-season comparisons
        df['global_week'] = (df['season'] - 2023) * 18 + df['week']

        self._weekly_stats = df
        return df

    def _get_ngs_extractor(self) -> NGSFeatureExtractor:
        """Get or create NGS feature extractor."""
        if self._ngs_extractor is None:
            self._ngs_extractor = NGSFeatureExtractor(self.data_dir)
        return self._ngs_extractor

    def calculate_qb_trailing_stats(
        self,
        player_norm: str,
        season: int,
        week: int,
    ) -> Dict[str, float]:
        """
        Calculate QB-specific trailing stats.

        Uses EWMA with shift(1) to avoid data leakage.

        Args:
            player_norm: Normalized player name
            season: Season year
            week: Current week

        Returns:
            Dict of trailing stats
        """
        stats = self._load_weekly_stats()

        defaults = {
            'qb_trailing_pass_yds': 220.0,
            'qb_trailing_attempts': 32.0,
            'qb_trailing_completion_pct': 0.65,
            'qb_passing_epa_trailing': 0.0,
            'qb_passing_cpoe_trailing': 0.0,
            'sacks_suffered_trailing': 2.0,
        }

        if len(stats) == 0:
            return defaults

        # Get player's historical data (before current week)
        global_week = (season - 2023) * 18 + week
        player_stats = stats[
            (stats['player_norm'] == player_norm) &
            (stats['global_week'] < global_week)
        ].copy()

        if len(player_stats) < 3:
            return defaults

        player_stats = player_stats.sort_values('global_week')

        # Calculate EWMA for each stat
        result = {}

        # Passing yards
        if 'passing_yards' in player_stats.columns:
            ewma = player_stats['passing_yards'].ewm(span=QB_EWMA_SPAN, min_periods=1).mean()
            result['qb_trailing_pass_yds'] = float(ewma.iloc[-1])
        else:
            result['qb_trailing_pass_yds'] = defaults['qb_trailing_pass_yds']

        # Pass attempts
        if 'attempts' in player_stats.columns:
            ewma = player_stats['attempts'].ewm(span=QB_EWMA_SPAN, min_periods=1).mean()
            result['qb_trailing_attempts'] = float(ewma.iloc[-1])
        else:
            result['qb_trailing_attempts'] = defaults['qb_trailing_attempts']

        # Completion percentage
        if 'completions' in player_stats.columns and 'attempts' in player_stats.columns:
            player_stats['comp_pct'] = player_stats['completions'] / player_stats['attempts'].replace(0, np.nan)
            ewma = player_stats['comp_pct'].ewm(span=QB_EWMA_SPAN, min_periods=1).mean()
            result['qb_trailing_completion_pct'] = float(ewma.iloc[-1]) if not pd.isna(ewma.iloc[-1]) else defaults['qb_trailing_completion_pct']
        else:
            result['qb_trailing_completion_pct'] = defaults['qb_trailing_completion_pct']

        # Passing EPA
        if 'passing_epa' in player_stats.columns:
            ewma = player_stats['passing_epa'].ewm(span=QB_EWMA_SPAN, min_periods=1).mean()
            result['qb_passing_epa_trailing'] = float(ewma.iloc[-1]) if not pd.isna(ewma.iloc[-1]) else 0.0
        else:
            result['qb_passing_epa_trailing'] = defaults['qb_passing_epa_trailing']

        # CPOE (dakota rating if available)
        if 'dakota' in player_stats.columns:
            ewma = player_stats['dakota'].ewm(span=QB_EWMA_SPAN, min_periods=1).mean()
            result['qb_passing_cpoe_trailing'] = float(ewma.iloc[-1]) if not pd.isna(ewma.iloc[-1]) else 0.0
        else:
            result['qb_passing_cpoe_trailing'] = defaults['qb_passing_cpoe_trailing']

        # Sacks suffered
        if 'sacks' in player_stats.columns:
            ewma = player_stats['sacks'].ewm(span=QB_EWMA_SPAN, min_periods=1).mean()
            result['sacks_suffered_trailing'] = float(ewma.iloc[-1]) if not pd.isna(ewma.iloc[-1]) else defaults['sacks_suffered_trailing']
        else:
            result['sacks_suffered_trailing'] = defaults['sacks_suffered_trailing']

        return result

    def get_ngs_qb_features(
        self,
        player_name: str,
        season: int,
        week: int,
    ) -> Dict[str, float]:
        """
        Get NGS passing features for a QB.

        Args:
            player_name: Player name
            season: Season year
            week: Current week

        Returns:
            Dict of NGS features
        """
        ngs = self._get_ngs_extractor()
        features = ngs.get_qb_features(player_name, season, week)

        return {
            'avg_time_to_throw': features.get('avg_time_to_throw', 2.7),
            'completion_pct_above_exp': features.get('completion_pct_above_exp', 0.0),
            'aggressiveness': features.get('aggressiveness', 0.18),
            'avg_intended_air_yards': features.get('avg_air_yards_to_sticks', 0.0),
        }

    def get_starter_features(
        self,
        player_name: str,
        team: str,
        season: int,
    ) -> Dict[str, float]:
        """
        Get QB starter detection features.

        Args:
            player_name: Player name
            team: Team abbreviation
            season: Season year

        Returns:
            Dict with starter features
        """
        info = self._qb_detector.classify_qb(player_name, team, season)

        return {
            'qb_is_starter': 1.0 if info.role == QBRole.STARTER else 0.0,
            'qb_starter_confidence': info.confidence,
        }

    def calculate_game_script_features(
        self,
        vegas_spread: float,
    ) -> Dict[str, float]:
        """
        Calculate game script features from Vegas spread.

        Args:
            vegas_spread: Point spread (negative = favorite)

        Returns:
            Dict with game script features
        """
        # Categorize spread
        if vegas_spread < -7:
            category = 'big_underdog'
        elif vegas_spread < -3:
            category = 'slight_underdog'
        elif vegas_spread <= 3:
            category = 'neutral'
        elif vegas_spread <= 7:
            category = 'slight_favorite'
        else:
            category = 'big_favorite'

        multiplier = GAME_SCRIPT_MULTIPLIERS.get(category, 1.0)
        volatility = GAME_SCRIPT_VOLATILITY.get(category, 0.15)

        return {
            'expected_game_script_multiplier': multiplier,
            'implied_pass_attempts': BASELINE_PASS_ATTEMPTS * multiplier,
            'game_script_volatility': volatility,
        }

    def calculate_line_features(
        self,
        line: float,
        trailing_pass_yds: float,
        player_under_rate: float = 0.5,
        player_bias: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate line-based features.

        Args:
            line: Betting line
            trailing_pass_yds: Trailing passing yards
            player_under_rate: Historical under hit rate
            player_bias: Player's tendency vs line

        Returns:
            Dict with line features
        """
        # Line vs trailing
        line_vs_trailing = line - trailing_pass_yds

        # Sweet spot (Gaussian decay)
        sweet_spot = np.exp(-((line - QB_SWEET_SPOT_CENTER) ** 2) / (2 * QB_SWEET_SPOT_WIDTH ** 2))

        return {
            'line_vs_trailing': line_vs_trailing,
            'line_level': line,
            'line_in_sweet_spot': sweet_spot,
            'player_under_rate': player_under_rate,
            'player_bias': player_bias,
        }

    def extract_features_for_prop(
        self,
        player_name: str,
        player_norm: str,
        team: str,
        line: float,
        season: int,
        week: int,
        vegas_total: float = 45.0,
        vegas_spread: float = 0.0,
        opponent: str = None,
        historical_odds: pd.DataFrame = None,
    ) -> Dict[str, float]:
        """
        Extract all QB features for a single prop bet.

        Args:
            player_name: Player display name
            player_norm: Normalized player name
            team: Team abbreviation
            line: Betting line
            season: Season year
            week: Week number
            vegas_total: Game total
            vegas_spread: Point spread
            opponent: Opponent team
            historical_odds: Historical odds data for player rates

        Returns:
            Dict with all QB features
        """
        features = {}

        # 1. QB trailing stats
        trailing = self.calculate_qb_trailing_stats(player_norm, season, week)
        features.update(trailing)

        # 2. NGS features
        ngs = self.get_ngs_qb_features(player_name, season, week)
        features.update(ngs)

        # 3. Starter detection
        starter = self.get_starter_features(player_name, team, season)
        features.update(starter)

        # 4. Game context
        features['vegas_total'] = vegas_total
        features['vegas_spread'] = vegas_spread
        features['implied_team_total'] = (vegas_total / 2) - (vegas_spread / 2)

        # 5. Game script features
        game_script = self.calculate_game_script_features(vegas_spread)
        features.update(game_script)

        # 6. Opponent defense features
        features['opp_pass_defense_epa'] = 0.0  # TODO: Calculate from opponent data
        features['opp_pass_yds_def_vs_avg'] = 0.0  # TODO: Calculate from opponent data

        # 7. Calculate player-specific rates from historical data
        player_under_rate = 0.5
        player_bias = 0.0
        if historical_odds is not None and len(historical_odds) > 0:
            player_hist = historical_odds[
                (historical_odds['player_norm'] == player_norm) &
                (historical_odds['market'] == 'player_pass_yds')
            ]
            if len(player_hist) >= 5:
                player_under_rate = player_hist['under_hit'].mean()
                if 'line' in player_hist.columns and 'actual_stat' in player_hist.columns:
                    actual_vs_line = player_hist['actual_stat'] - player_hist['line']
                    player_bias = actual_vs_line.mean()

        # 8. Line features
        line_features = self.calculate_line_features(
            line=line,
            trailing_pass_yds=trailing['qb_trailing_pass_yds'],
            player_under_rate=player_under_rate,
            player_bias=player_bias,
        )
        features.update(line_features)

        return features


def extract_qb_features_batch(
    df: pd.DataFrame,
    historical_odds: pd.DataFrame,
    market: str = 'player_pass_yds',
    target_global_week: int = None,
) -> pd.DataFrame:
    """
    Extract QB features for a batch of props.

    This is the main entry point called by batch_extractor.py when
    market == 'player_pass_yds'.

    Args:
        df: DataFrame with prop data (must have player, line, season, week, team)
        historical_odds: Historical odds for rate calculations
        market: Market name (should be 'player_pass_yds')
        target_global_week: Current week for temporal filtering

    Returns:
        DataFrame with QB features added
    """
    if market != 'player_pass_yds':
        logger.warning(f"QB features called for non-QB market: {market}")
        return df

    if len(df) == 0:
        return df

    extractor = QBFeatureExtractor()

    # Ensure required columns exist
    if 'player_norm' not in df.columns:
        if 'player' in df.columns:
            df['player_norm'] = df['player'].apply(normalize_player_name)
        else:
            logger.error("No player column in DataFrame")
            return df

    features_list = []

    for idx, row in df.iterrows():
        try:
            features = extractor.extract_features_for_prop(
                player_name=row.get('player', row.get('player_display_name', '')),
                player_norm=row['player_norm'],
                team=row.get('team', ''),
                line=row.get('line', 225.0),
                season=row.get('season', 2025),
                week=row.get('week', 1),
                vegas_total=row.get('vegas_total', 45.0),
                vegas_spread=row.get('vegas_spread', 0.0),
                opponent=row.get('opponent', None),
                historical_odds=historical_odds,
            )
            features['_idx'] = idx
            features_list.append(features)
        except Exception as e:
            logger.warning(f"Error extracting QB features for row {idx}: {e}")
            # Add default features
            features = {col: 0.0 for col in QB_FEATURES}
            features['_idx'] = idx
            features_list.append(features)

    if not features_list:
        return df

    # Convert to DataFrame and merge back
    features_df = pd.DataFrame(features_list)
    features_df = features_df.set_index('_idx')

    # Add features to original DataFrame
    for col in QB_FEATURES:
        if col in features_df.columns:
            df.loc[features_df.index, col] = features_df[col]
        else:
            df[col] = 0.0

    return df


# Convenience function for single player
def get_qb_features(
    player_name: str,
    team: str,
    line: float,
    season: int = 2025,
    week: int = 15,
    vegas_total: float = 45.0,
    vegas_spread: float = 0.0,
) -> Dict[str, float]:
    """
    Get QB features for a single player.

    Args:
        player_name: Player name
        team: Team abbreviation
        line: Betting line
        season: Season year
        week: Week number
        vegas_total: Game total
        vegas_spread: Point spread

    Returns:
        Dict with all QB features
    """
    extractor = QBFeatureExtractor()
    return extractor.extract_features_for_prop(
        player_name=player_name,
        player_norm=normalize_player_name(player_name),
        team=team,
        line=line,
        season=season,
        week=week,
        vegas_total=vegas_total,
        vegas_spread=vegas_spread,
    )
