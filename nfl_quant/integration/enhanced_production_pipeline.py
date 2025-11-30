#!/usr/bin/env python3
"""
Enhanced Production Pipeline with Full Feature Integration

This pipeline builds on the base ProductionPipeline by adding:
1. Defensive EPA matchup adjustments
2. Weather/environment factors
3. Rest/travel context
4. Snap count trends
5. Injury redistribution
6. Target share velocity
7. Next Gen Stats
8. Team pace

All contextual features are systematically integrated into predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.data.dynamic_parameters import get_parameter_provider, DynamicParameterProvider
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.integration.enhanced_prediction import EnhancedPrediction, AllFeatures
from nfl_quant.integration.feature_aggregator import FeatureAggregator, create_feature_aggregator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CALIBRATION_DIR = PROJECT_ROOT / "models" / "calibration"


class EnhancedProductionPipeline:
    """
    Enhanced production pipeline with full contextual feature integration.

    This pipeline:
    1. Computes base statistics from trailing player data
    2. Aggregates all contextual features (weather, matchups, injuries, etc.)
    3. Applies multiplicative adjustments to projections
    4. Calibrates probabilities with isotonic regression
    5. Returns predictions with full feature visibility
    """

    def __init__(self):
        """Initialize the enhanced pipeline."""
        self.param_provider = get_parameter_provider()
        self.feature_aggregator = create_feature_aggregator()
        self.calibrators: Dict[str, NFLProbabilityCalibrator] = {}
        self._load_calibrators()

        # Log initialization
        logger.info("=" * 60)
        logger.info("ENHANCED PRODUCTION PIPELINE INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Features integrated:")
        logger.info(f"  - Defensive EPA matchups: {'✓' if self.feature_aggregator._defensive_epa is not None else '✗'}")
        logger.info(f"  - Weather/Environment: {'✓' if self.feature_aggregator._schedules is not None else '✗'}")
        logger.info(f"  - Rest/Travel: {'✓' if self.feature_aggregator._schedules is not None else '✗'}")
        logger.info(f"  - Snap Counts: {'✓' if self.feature_aggregator._snap_counts is not None else '✗'}")
        logger.info(f"  - Injuries: {'✓' if self.feature_aggregator._injuries is not None else '✗'}")
        logger.info(f"  - Target Shares: {'✓' if self.feature_aggregator._target_shares is not None else '✗'}")
        logger.info(f"  - Team Pace: {'✓' if self.feature_aggregator._team_pace is not None else '✗'}")
        logger.info("=" * 60)

    def _load_calibrators(self):
        """Load trained isotonic calibrators."""
        # Load overall calibrator
        overall_path = CALIBRATION_DIR / "overall_calibrator.json"
        if overall_path.exists():
            self.calibrators['overall'] = NFLProbabilityCalibrator()
            self.calibrators['overall'].load(str(overall_path))
            logger.info(f"Loaded overall calibrator")

        # Load position-specific calibrators
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_path = CALIBRATION_DIR / f"position_{pos}_calibrator.json"
            if pos_path.exists():
                cal = NFLProbabilityCalibrator()
                cal.load(str(pos_path))
                self.calibrators[f'position_{pos}'] = cal

        # Load market-specific calibrators
        markets = ['receptions', 'receiving_yards', 'rushing_yards', 'targets',
                   'carries', 'passing_yards', 'passing_tds', 'rushing_tds', 'receiving_tds']
        for market in markets:
            market_path = CALIBRATION_DIR / f"market_{market}_calibrator.json"
            if market_path.exists():
                cal = NFLProbabilityCalibrator()
                cal.load(str(market_path))
                self.calibrators[f'market_{market}'] = cal

    def get_enhanced_prediction(
        self,
        player_name: str,
        team: str,
        position: str,
        opponent: str,
        market: str,
        line: float,
        week: int = 11,
        season: int = 2025,
        market_prob_over: float = 0.5,
        n_simulations: int = 10000
    ) -> EnhancedPrediction:
        """
        Generate an enhanced prediction with all contextual features.

        Args:
            player_name: Player name (e.g., "T.Kelce")
            team: Team abbreviation
            position: Position (QB, RB, WR, TE)
            opponent: Opponent team
            market: Market type (e.g., "receiving_yards")
            line: Prop line
            week: Week number
            season: Season year
            market_prob_over: Market implied probability
            n_simulations: Monte Carlo simulations

        Returns:
            EnhancedPrediction with full feature visibility
        """
        # Step 1: Get base statistics from trailing data
        base_stats = self._get_base_statistics(player_name, team, position, week, season)

        if not base_stats:
            return self._get_default_prediction(
                player_name, team, position, opponent, market, line, week, season
            )

        base_mean, base_std = self._extract_market_stats(base_stats, market, position)

        if base_mean <= 0:
            return self._get_default_prediction(
                player_name, team, position, opponent, market, line, week, season
            )

        # Step 2: Aggregate all contextual features
        features = self.feature_aggregator.get_all_features(
            player_name=player_name,
            team=team,
            position=position,
            opponent=opponent,
            week=week,
            season=season,
            market=market
        )

        # Step 3: Apply adjustments to base projection
        adjusted_mean, adjusted_std = self.feature_aggregator.apply_all_adjustments(
            base_mean=base_mean,
            base_std=base_std,
            features=features,
            market=market
        )

        # Step 4: Apply additional variance inflation
        adjusted_std = self._adjust_variance(adjusted_mean, adjusted_std, market, position)

        # Step 5: Simulate distribution
        samples = np.random.normal(adjusted_mean, adjusted_std, n_simulations)
        samples = np.maximum(0, samples)

        # Step 6: Calculate raw probability
        raw_prob_over = float(np.mean(samples > line))

        # Step 7: Apply isotonic calibration
        calibrated_prob = self._calibrate_probability(raw_prob_over, position, market)

        # Step 8: Calculate edges
        edge_over = calibrated_prob - market_prob_over
        edge_under = (1 - calibrated_prob) - (1 - market_prob_over)

        # Step 9: Calculate feature contributions
        feature_contributions = self._calculate_feature_contributions(
            base_mean, adjusted_mean, features
        )

        # Step 10: Determine confidence tier
        confidence_tier = self._get_confidence_tier(
            edge_over, edge_under, len(base_stats.get('weeks', []))
        )

        # Step 11: Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(base_stats, features)

        return EnhancedPrediction(
            player_name=player_name,
            position=position,
            team=team,
            opponent=opponent,
            market=market,
            line=line,
            week=week,
            season=season,
            base_mean=base_mean,
            base_std=base_std,
            adjusted_mean=adjusted_mean,
            adjusted_std=adjusted_std,
            raw_prob_over=raw_prob_over,
            calibrated_prob_over=calibrated_prob,
            market_prob_over=market_prob_over,
            edge_over=edge_over,
            edge_under=edge_under,
            confidence_tier=confidence_tier,
            data_quality_score=data_quality_score,
            feature_contributions=feature_contributions,
            features=features,
            prediction_timestamp=datetime.now().isoformat(),
            model_version="v2.0-enhanced",
            calibrator_version="2025-walk-forward",
        )

    def _get_base_statistics(
        self,
        player_name: str,
        team: str,
        position: str,
        week: int,
        season: int,
        n_weeks: int = 4
    ) -> Dict[str, Any]:
        """Get player trailing stats with Bayesian shrinkage."""
        weekly_data = self.param_provider.weekly_data

        # Filter player data
        player_data = weekly_data[
            (weekly_data['player_name'] == player_name) &
            (weekly_data['recent_team'] == team)
        ]

        # Use only current season data if available
        if 'season' in weekly_data.columns:
            current_season_data = player_data[player_data['season'] == season]
            if len(current_season_data) >= 3:
                player_data = current_season_data

        # Filter to weeks before prediction week
        player_data = player_data[player_data['week'] < week]

        if len(player_data) == 0:
            return {}

        # Get recent games
        recent_data = player_data.nlargest(n_weeks, 'week')
        n_games = len(recent_data)

        # Get position averages for Bayesian shrinkage
        pos_data = weekly_data[weekly_data['position'] == position]
        pos_means = {}
        for col in ['targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards', 'passing_yards']:
            if col in pos_data.columns:
                median_val = pos_data[col].quantile(0.5)
                starter_data = pos_data[pos_data[col] >= median_val]
                pos_means[col] = starter_data[col].mean() if len(starter_data) > 0 else pos_data[col].mean()

        # Apply Bayesian shrinkage
        SHRINKAGE_STRENGTH = 3.0
        player_weight = n_games / (n_games + SHRINKAGE_STRENGTH)

        stats = {
            'weeks': list(recent_data['week'].values),
            'n_games': n_games,
            'player_weight': player_weight,
        }

        for col in ['targets', 'receptions', 'receiving_yards', 'receiving_tds',
                    'carries', 'rushing_yards', 'rushing_tds',
                    'passing_yards', 'passing_tds']:
            if col in recent_data.columns:
                raw_mean = recent_data[col].mean()
                pos_mean = pos_means.get(col, raw_mean)

                shrunk_mean = player_weight * raw_mean + (1 - player_weight) * pos_mean

                stats[f'mean_{col}'] = shrunk_mean
                stats[f'std_{col}'] = recent_data[col].std() if n_games > 1 else shrunk_mean * 0.5

        return stats

    def _extract_market_stats(
        self,
        stats: Dict[str, Any],
        market: str,
        position: str
    ) -> Tuple[float, float]:
        """Extract mean and std for specific market."""
        market_map = {
            'receptions': ('mean_receptions', 'std_receptions'),
            'receiving_yards': ('mean_receiving_yards', 'std_receiving_yards'),
            'rushing_yards': ('mean_rushing_yards', 'std_rushing_yards'),
            'targets': ('mean_targets', 'std_targets'),
            'carries': ('mean_carries', 'std_carries'),
            'passing_yards': ('mean_passing_yards', 'std_passing_yards'),
            'passing_tds': ('mean_passing_tds', 'std_passing_tds'),
            'rushing_tds': ('mean_rushing_tds', 'std_rushing_tds'),
            'receiving_tds': ('mean_receiving_tds', 'std_receiving_tds'),
        }

        mean_key, std_key = market_map.get(market, ('', ''))
        mean_val = stats.get(mean_key, 0.0)
        std_val = stats.get(std_key, mean_val * 0.5)

        return mean_val, std_val

    def _adjust_variance(
        self,
        mean_val: float,
        std_val: float,
        market: str,
        position: str
    ) -> float:
        """Adjust variance based on actual population CV."""
        if mean_val <= 0:
            return std_val

        current_cv = std_val / mean_val
        actual_cv = self.param_provider.get_league_avg_cv(market)

        if actual_cv < 0.3:
            actual_cv = 0.7

        if current_cv > 0:
            inflation = max(1.0, min(3.0, actual_cv / current_cv))
        else:
            inflation = 1.5

        if 'td' in market.lower():
            inflation = max(inflation, 1.8)

        return std_val * inflation

    def _calibrate_probability(
        self,
        raw_prob: float,
        position: str,
        market: str
    ) -> float:
        """Apply isotonic calibration."""
        market_key = f'market_{market}'
        if market_key in self.calibrators:
            return float(self.calibrators[market_key].transform(np.array([raw_prob]))[0])

        pos_key = f'position_{position}'
        if pos_key in self.calibrators:
            return float(self.calibrators[pos_key].transform(np.array([raw_prob]))[0])

        if 'overall' in self.calibrators:
            return float(self.calibrators['overall'].transform(np.array([raw_prob]))[0])

        return raw_prob

    def _calculate_feature_contributions(
        self,
        base_mean: float,
        adjusted_mean: float,
        features: AllFeatures
    ) -> Dict[str, float]:
        """Calculate how much each feature contributed to the adjustment."""
        contributions = {}

        # Calculate each feature's isolated contribution
        base = base_mean

        # Defensive matchup
        def_mult = features.defensive_matchup.matchup_multiplier
        contributions['defensive_matchup'] = (def_mult - 1.0) * 100  # As percentage

        # Weather
        weather_mult = features.weather.passing_epa_multiplier
        contributions['weather'] = (weather_mult - 1.0) * 100

        # Rest
        rest_mult = features.rest_travel.rest_epa_multiplier
        contributions['rest'] = (rest_mult - 1.0) * 100

        # Travel
        travel_mult = features.rest_travel.travel_epa_multiplier
        contributions['travel'] = (travel_mult - 1.0) * 100

        # Snap trend
        snap_mult = features.snap_counts.snap_share_multiplier
        contributions['snap_trend'] = (snap_mult - 1.0) * 100

        # Injury redistribution
        injury_mult = features.injury_impact.injury_redistribution_multiplier
        contributions['injury_redistribution'] = (injury_mult - 1.0) * 100

        # Team pace
        pace_mult = features.team_pace.pace_multiplier
        contributions['team_pace'] = (pace_mult - 1.0) * 100

        # NGS Skill
        ngs_mult = features.ngs.ngs_skill_multiplier
        contributions['ngs_skill'] = (ngs_mult - 1.0) * 100

        # QB Connection (for WR/TE)
        qb_conn_mult = features.qb_connection.qb_connection_multiplier
        contributions['qb_connection'] = (qb_conn_mult - 1.0) * 100

        # Historical matchup vs opponent
        hist_mult = features.historical_matchup.vs_opponent_multiplier
        contributions['historical_vs_opponent'] = (hist_mult - 1.0) * 100

        # Total adjustment
        total_pct = ((adjusted_mean / base_mean) - 1.0) * 100 if base_mean > 0 else 0.0
        contributions['total_adjustment_pct'] = total_pct

        return contributions

    def _get_confidence_tier(self, edge_over: float, edge_under: float, n_games: int) -> str:
        """Determine confidence tier based on edge and data quality."""
        if n_games < 3:
            return 'LOW_DATA'

        best_edge = max(abs(edge_over), abs(edge_under))

        if best_edge >= 0.15:
            return 'HIGH_EDGE'
        elif best_edge >= 0.10:
            return 'MEDIUM_HIGH_EDGE'
        elif best_edge >= 0.05:
            return 'MEDIUM_EDGE'
        elif best_edge >= 0.02:
            return 'LOW_EDGE'
        else:
            return 'NO_EDGE'

    def _calculate_data_quality_score(
        self,
        base_stats: Dict[str, Any],
        features: AllFeatures
    ) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 0.0

        # Games played (max 25 points)
        n_games = base_stats.get('n_games', 0)
        score += min(25, n_games * 6.25)

        # Features available (each worth 10 points, max 50)
        if features.defensive_matchup.opponent_def_epa != 0:
            score += 10
        if features.weather.wind_speed > 0 or features.weather.is_dome:
            score += 10
        if features.rest_travel.days_rest != 7:
            score += 10
        if features.snap_counts.avg_offense_pct > 0:
            score += 10
        if len(features.injury_impact.teammates_out) > 0 or features.injury_impact.player_injury_status != "healthy":
            score += 10

        # Normalize to 0-1
        return min(1.0, score / 100.0)

    def _get_default_prediction(
        self,
        player_name: str,
        team: str,
        position: str,
        opponent: str,
        market: str,
        line: float,
        week: int,
        season: int
    ) -> EnhancedPrediction:
        """Return default prediction when no data available."""
        return EnhancedPrediction(
            player_name=player_name,
            position=position,
            team=team,
            opponent=opponent,
            market=market,
            line=line,
            week=week,
            season=season,
            base_mean=0.0,
            base_std=0.0,
            adjusted_mean=0.0,
            adjusted_std=0.0,
            raw_prob_over=0.5,
            calibrated_prob_over=0.5,
            market_prob_over=0.5,
            edge_over=0.0,
            edge_under=0.0,
            confidence_tier='NO_DATA',
            data_quality_score=0.0,
            feature_contributions={},
            features=AllFeatures(),
            prediction_timestamp=datetime.now().isoformat(),
            model_version="v2.0-enhanced",
        )

    def get_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'features_loaded': {
                'defensive_epa': self.feature_aggregator._defensive_epa is not None,
                'schedules': self.feature_aggregator._schedules is not None,
                'snap_counts': self.feature_aggregator._snap_counts is not None,
                'injuries': self.feature_aggregator._injuries is not None,
                'target_shares': self.feature_aggregator._target_shares is not None,
                'team_pace': self.feature_aggregator._team_pace is not None,
            },
            'calibrators_loaded': len(self.calibrators),
        }

        # Check critical features
        n_features = sum(health['features_loaded'].values())
        if n_features < 3:
            health['status'] = 'degraded'
            health['warnings'].append(f"Only {n_features}/6 feature sources loaded")

        # Check calibrators
        if 'overall' not in self.calibrators:
            health['issues'].append("No overall calibrator loaded")
            health['status'] = 'error'

        # Check data freshness
        try:
            data_rows = len(self.param_provider.weekly_data)
            if data_rows < 1000:
                health['warnings'].append(f"Limited data: {data_rows} rows")

            seasons = self.param_provider.weekly_data['season'].unique()
            if 2025 not in seasons:
                health['warnings'].append("2025 season data not loaded")

        except Exception as e:
            health['issues'].append(f"Cannot access weekly data: {e}")
            health['status'] = 'critical'

        return health


def create_enhanced_pipeline() -> EnhancedProductionPipeline:
    """Factory function to create enhanced pipeline."""
    return EnhancedProductionPipeline()


def test_enhanced_pipeline():
    """Test the enhanced pipeline."""
    print("=" * 70)
    print("ENHANCED PRODUCTION PIPELINE TEST")
    print("=" * 70)

    pipeline = create_enhanced_pipeline()

    # Health check
    health = pipeline.get_system_health_check()
    print(f"\nSystem Health: {health['status']}")
    print(f"Features Loaded: {health['features_loaded']}")
    print(f"Calibrators: {health['calibrators_loaded']}")
    if health['warnings']:
        print(f"Warnings: {health['warnings']}")
    if health['issues']:
        print(f"Issues: {health['issues']}")

    # Test prediction
    print("\n" + "=" * 70)
    print("TEST PREDICTION: Travis Kelce Receiving Yards")
    print("=" * 70)

    pred = pipeline.get_enhanced_prediction(
        player_name='T.Kelce',
        team='KC',
        position='TE',
        opponent='BUF',
        market='receiving_yards',
        line=60.5,
        week=11,
        season=2025,
        market_prob_over=0.52
    )

    print(f"\nPlayer: {pred.player_name} ({pred.team})")
    print(f"Opponent: {pred.opponent}")
    print(f"Market: {pred.market} @ {pred.line}")
    print(f"\nBase Projection: {pred.base_mean:.1f} ± {pred.base_std:.1f}")
    print(f"Adjusted Projection: {pred.adjusted_mean:.1f} ± {pred.adjusted_std:.1f}")
    print(f"Total Adjustment: {pred.get_total_adjustment_multiplier():.3f}x")

    print(f"\nFeature Contributions:")
    for feature, contribution in pred.feature_contributions.items():
        if contribution != 0:
            print(f"  {feature}: {contribution:+.1f}%")

    print(f"\nProbabilities:")
    print(f"  Raw P(Over): {pred.raw_prob_over:.3f}")
    print(f"  Calibrated P(Over): {pred.calibrated_prob_over:.3f}")
    print(f"  Market P(Over): {pred.market_prob_over:.3f}")

    print(f"\nEdges:")
    print(f"  Edge Over: {pred.edge_over:+.3f} ({pred.edge_over*100:+.1f}%)")
    print(f"  Edge Under: {pred.edge_under:+.3f} ({pred.edge_under*100:+.1f}%)")

    print(f"\nConfidence: {pred.confidence_tier}")
    print(f"Data Quality Score: {pred.data_quality_score:.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_enhanced_pipeline()
