"""
Enhanced Player Features - Unified Integration

Combines all new feature extractors:
- Next Gen Stats (separation, RYOE, CPOE)
- FF Opportunity (expected vs actual, regression)
- Snap Counts (usage trends, workload)
- Historical Baseline (past performance vs line)

This module provides a single interface for the prediction pipeline.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
import json

from nfl_quant.features.ngs_features import NGSFeatureExtractor
from nfl_quant.features.ff_opportunity_features import FFOpportunityFeatureExtractor
from nfl_quant.features.snap_count_features import SnapCountFeatureExtractor
from nfl_quant.features.historical_baseline import HistoricalBaselineProvider

logger = logging.getLogger(__name__)


class EnhancedPlayerFeatureEngine:
    """
    Unified feature extraction engine for player props.

    Integrates all advanced nflverse data sources to generate
    comprehensive feature sets for prediction models.
    """

    def __init__(self, data_dir: Path = None):
        """
        Initialize all feature extractors.

        Args:
            data_dir: Path to nflverse data directory
        """
        if data_dir is None:
            data_dir = Path('data/nflverse')

        self.data_dir = Path(data_dir)

        # Initialize extractors (lazy loading - only load if needed)
        self._ngs_extractor = None
        self._ff_opportunity_extractor = None
        self._snap_count_extractor = None
        self._historical_provider = None

        # Load threshold configs
        self._load_thresholds()

        logger.info("EnhancedPlayerFeatureEngine initialized")

    def _load_thresholds(self):
        """Load position-market threshold configuration."""
        config_path = Path('configs/position_market_thresholds.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.thresholds = json.load(f)
            logger.info("Loaded position-market thresholds")
        else:
            self.thresholds = {}
            logger.warning("Position-market thresholds not found")

    @property
    def ngs_extractor(self):
        """Lazy load NGS extractor."""
        if self._ngs_extractor is None:
            self._ngs_extractor = NGSFeatureExtractor(self.data_dir)
        return self._ngs_extractor

    @property
    def ff_opportunity_extractor(self):
        """Lazy load FF opportunity extractor."""
        if self._ff_opportunity_extractor is None:
            self._ff_opportunity_extractor = FFOpportunityFeatureExtractor(self.data_dir)
        return self._ff_opportunity_extractor

    @property
    def snap_count_extractor(self):
        """Lazy load snap count extractor."""
        if self._snap_count_extractor is None:
            self._snap_count_extractor = SnapCountFeatureExtractor(self.data_dir)
        return self._snap_count_extractor

    @property
    def historical_provider(self):
        """Lazy load historical baseline provider."""
        if self._historical_provider is None:
            self._historical_provider = HistoricalBaselineProvider(self.data_dir)
        return self._historical_provider

    def get_comprehensive_features(
        self,
        player_name: str,
        position: str,
        team: str,
        market_type: str,
        line: float,
        season: int = 2025,
        week: int = 11,
        trailing_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get all available features for a player prop prediction.

        Args:
            player_name: Player name
            position: Position (QB, RB, WR, TE)
            team: Team abbreviation
            market_type: Prop market type
            line: Prop line value
            season: Current season
            week: Current week
            trailing_weeks: Weeks to look back

        Returns:
            Comprehensive feature dictionary
        """
        features = {}

        # 1. Next Gen Stats Features (skill metrics)
        try:
            ngs_features = self.ngs_extractor.get_player_features(
                player_name, position, season, week, trailing_weeks
            )
            features.update({f'ngs_{k}': v for k, v in ngs_features.items()})

            # Add skill score
            skill_score = self.ngs_extractor.calculate_skill_score(
                player_name, position, season, week
            )
            features['ngs_skill_score'] = skill_score

            # Add regression indicators
            regression = self.ngs_extractor.get_regression_indicators(
                player_name, position, season, week
            )
            features.update({f'ngs_{k}': v for k, v in regression.items()})

        except Exception as e:
            logger.warning(f"Error getting NGS features for {player_name}: {e}")

        # 2. FF Opportunity Features (expected vs actual)
        try:
            opp_features = self.ff_opportunity_extractor.get_player_opportunity_features(
                player_name, season, week, trailing_weeks
            )
            features.update({f'ffo_{k}': v for k, v in opp_features.items()})

            # Add betting adjustment
            betting_adj = self.ff_opportunity_extractor.get_betting_adjustment(
                player_name, season, week, market_type
            )
            features['ffo_betting_adjustment'] = betting_adj

            # Add sustainability score
            sustainability = self.ff_opportunity_extractor.calculate_sustainable_production_score(
                player_name, season, week
            )
            features['ffo_sustainable_production'] = sustainability

        except Exception as e:
            logger.warning(f"Error getting FF opportunity features for {player_name}: {e}")

        # 3. Snap Count Features (usage)
        try:
            snap_features = self.snap_count_extractor.get_usage_prediction_features(
                player_name, position, season, week
            )
            features.update({f'snap_{k}': v for k, v in snap_features.items()})

        except Exception as e:
            logger.warning(f"Error getting snap count features for {player_name}: {e}")

        # 4. Historical Baseline Features
        try:
            hist_baseline = self.historical_provider.get_historical_baseline(
                player_name, market_type, line, season, week
            )
            features.update({f'hist_{k}': v for k, v in hist_baseline.items()})

            # Add trend features
            trend = self.historical_provider.get_recent_trend(
                player_name, market_type, season, week, num_weeks=3
            )
            features.update({f'hist_{k}': v for k, v in trend.items()})

            # Add league percentile
            percentile = self.historical_provider.get_league_percentile(
                player_name, position, market_type, season, week
            )
            features['hist_league_percentile'] = percentile

        except Exception as e:
            logger.warning(f"Error getting historical features for {player_name}: {e}")

        # 5. Mean Bias Correction
        bias_key = f'{position}_{market_type}'
        bias_corrections = self.thresholds.get('mean_bias_corrections', {})
        features['mean_bias_correction'] = bias_corrections.get(bias_key, 0.0)

        # 6. Market-specific edge threshold
        features['min_edge_threshold'] = self._get_min_edge_threshold(position, market_type)

        logger.debug(f"Generated {len(features)} features for {player_name} ({position})")
        return features

    def _get_min_edge_threshold(self, position: str, market_type: str) -> float:
        """Get minimum edge threshold for position-market combo."""
        edge_thresholds = self.thresholds.get('edge_thresholds', {})

        if position in edge_thresholds:
            pos_thresholds = edge_thresholds[position]
            if market_type in pos_thresholds:
                return pos_thresholds[market_type].get('min_edge', 0.10)

        # Default thresholds by market type
        defaults = {
            'player_pass_yds': 0.05,
            'player_pass_tds': 0.10,
            'player_rush_yds': 0.20,
            'player_reception_yds': 0.10,
            'player_receptions': 0.10,
            'player_anytime_td': 0.20,
        }

        return defaults.get(market_type, 0.10)

    def adjust_prediction_with_features(
        self,
        base_prediction: float,
        features: Dict[str, float],
        market_type: str
    ) -> float:
        """
        Adjust base prediction using enhanced features.

        Args:
            base_prediction: Initial model prediction
            features: Feature dictionary
            market_type: Market type

        Returns:
            Adjusted prediction
        """
        adjusted = base_prediction

        # Apply mean bias correction
        bias = features.get('mean_bias_correction', 0.0)
        adjusted += bias

        # Apply regression adjustment
        # If player is overperforming (ffo_betting_adjustment < 0), reduce prediction
        # If underperforming (ffo_betting_adjustment > 0), increase prediction
        betting_adj = features.get('ffo_betting_adjustment', 0.0)

        # Convert adjustment to percentage (adjustment is -1 to +1)
        regression_factor = 1.0 + (betting_adj * 0.1)  # Up to 10% adjustment
        adjusted *= regression_factor

        # Apply NGS skill adjustment
        skill_score = features.get('ngs_skill_score', 0.0)
        # Skilled players (score > 0) get small boost, unskilled get penalty
        skill_factor = 1.0 + (skill_score * 0.05)  # Up to 5% adjustment
        adjusted *= skill_factor

        # Apply snap trend adjustment
        snap_trend = features.get('snap_snap_trend', 0.0)
        if snap_trend > 0.05:  # Increasing snap share
            adjusted *= 1.05
        elif snap_trend < -0.05:  # Decreasing snap share
            adjusted *= 0.95

        # Apply historical trend adjustment
        trend_direction = features.get('hist_trend_direction', 0.0)
        if 'rush' in market_type:
            # RB rushing trends matter more
            adjusted += trend_direction * 0.5
        elif 'reception' in market_type or 'rec' in market_type:
            adjusted += trend_direction * 0.3

        logger.debug(f"Adjusted prediction: {base_prediction:.2f} -> {adjusted:.2f}")
        return adjusted

    def calculate_confidence_score(
        self,
        features: Dict[str, float],
        raw_edge: float,
        position: str,
        market_type: str
    ) -> Dict[str, float]:
        """
        Calculate confidence score incorporating all features.

        Args:
            features: Feature dictionary
            raw_edge: Raw edge percentage
            position: Player position
            market_type: Market type

        Returns:
            Confidence metrics dict
        """
        min_threshold = features.get('min_edge_threshold', 0.10)

        # Base confidence from edge vs threshold
        if raw_edge >= min_threshold * 1.5:
            base_confidence = 0.85
        elif raw_edge >= min_threshold:
            base_confidence = 0.70
        elif raw_edge >= min_threshold * 0.8:
            base_confidence = 0.55
        else:
            base_confidence = 0.40

        # Adjust for sustainability
        sustainability = features.get('ffo_sustainable_production', 0.5)
        sustainability_adj = (sustainability - 0.5) * 0.1  # Up to +/- 5%

        # Adjust for historical consistency
        consistency = features.get('hist_trend_consistency', 0.5)
        consistency_adj = (consistency - 0.5) * 0.1

        # Adjust for skill score
        skill = features.get('ngs_skill_score', 0.0)
        skill_adj = skill * 0.05

        # Combined confidence
        confidence = base_confidence + sustainability_adj + consistency_adj + skill_adj
        confidence = np.clip(confidence, 0.30, 0.95)

        # Determine tier
        if confidence >= 0.75 and raw_edge >= min_threshold * 1.3:
            tier = 'High'
        elif confidence >= 0.60 and raw_edge >= min_threshold:
            tier = 'Medium'
        else:
            tier = 'Low'

        return {
            'confidence_score': float(confidence),
            'confidence_tier': tier,
            'meets_threshold': raw_edge >= min_threshold,
            'edge_vs_threshold': raw_edge / min_threshold if min_threshold > 0 else 0,
        }


def enhance_recommendations(
    recommendations_df: pd.DataFrame,
    season: int = 2025,
    week: int = 11,
    data_dir: Path = None
) -> pd.DataFrame:
    """
    Enhance recommendations DataFrame with all advanced features.

    Args:
        recommendations_df: DataFrame with recommendations
        season: Current season
        week: Current week
        data_dir: Path to nflverse data

    Returns:
        Enhanced DataFrame with NGS, opportunity, snap features
    """
    engine = EnhancedPlayerFeatureEngine(data_dir)

    # Add feature columns
    enhanced_df = recommendations_df.copy()

    # Track which features we successfully extract
    ngs_cols = []
    ffo_cols = []
    snap_cols = []
    hist_cols = []

    for idx, row in enhanced_df.iterrows():
        player = row.get('player', '')
        position = row.get('position', 'UNK')
        team = row.get('team', 'UNK')
        market = row.get('market', '')
        line = row.get('line', 0)

        # Infer position from market if not available
        if position == 'UNK':
            if 'pass' in market.lower():
                position = 'QB'
            elif 'rush' in market.lower():
                position = 'RB'
            elif 'rec' in market.lower():
                position = 'WR'

        features = engine.get_comprehensive_features(
            player, position, team, market, line, season, week
        )

        # Update DataFrame with features
        for key, value in features.items():
            if key not in enhanced_df.columns:
                enhanced_df[key] = np.nan

            enhanced_df.at[idx, key] = value

        # Calculate adjusted confidence
        raw_edge = row.get('edge_pct', 0) / 100  # Convert to decimal
        confidence_metrics = engine.calculate_confidence_score(
            features, raw_edge, position, market
        )
        enhanced_df.at[idx, 'enhanced_confidence'] = confidence_metrics['confidence_score']
        enhanced_df.at[idx, 'enhanced_tier'] = confidence_metrics['confidence_tier']
        enhanced_df.at[idx, 'meets_edge_threshold'] = confidence_metrics['meets_threshold']

    logger.info(f"Enhanced {len(enhanced_df)} recommendations with {len(features)} features")
    return enhanced_df
