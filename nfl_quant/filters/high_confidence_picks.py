"""
High-Confidence Pick Filtering

Filters betting recommendations based on backtest-validated confidence levels.
Only returns picks with demonstrated predictive accuracy.

Based on V3 Backtest Results (Weeks 1-8, 2024):
- Receiving yards: 75.7% coverage (GOOD)
- Rushing yards: 35.4% coverage → recalibrated (MEDIUM)
- Passing yards: 11.0% coverage (AVOID until fixed)
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels based on backtest validation."""
    HIGH = "HIGH"           # 70%+ coverage, use with standard edge thresholds
    MEDIUM = "MEDIUM"       # 50-70% coverage, require higher edges
    LOW = "LOW"             # <50% coverage, avoid or require very high edges
    DISABLED = "DISABLED"   # Critical issues, do not use


class PickFilter:
    """
    Filters betting picks based on stat-type-specific confidence levels.

    Usage:
        filter = PickFilter()
        high_conf_picks = filter.apply(recommendations_df)
    """

    # Stat-type confidence levels (from backtest validation)
    STAT_CONFIDENCE = {
        # RECEIVING PROPS (Well-Calibrated - 75.7% coverage)
        'player_reception_yds': {
            'confidence': ConfidenceLevel.HIGH,
            'min_edge': 0.05,           # 5% edge minimum
            'min_model_prob': 0.58,     # 58% model probability
            'use_market_calibrator': True,
            'calibrator_path': 'configs/calibrator_player_reception_yds.json',
            'notes': 'Excellent calibration. Safe for production.'
        },
        'player_receptions': {
            'confidence': ConfidenceLevel.HIGH,
            'min_edge': 0.05,
            'min_model_prob': 0.58,
            'use_market_calibrator': True,
            'calibrator_path': 'configs/calibrator_player_receptions.json',
            'notes': 'Excellent calibration. Safe for production.'
        },

        # RUSHING PROPS (Recalibrated - 35.4% → target 90% coverage)
        'player_rush_yds': {
            'confidence': ConfidenceLevel.MEDIUM,
            'min_edge': 0.08,           # Higher edge required (8%)
            'min_model_prob': 0.62,     # Higher confidence required
            'use_market_calibrator': True,
            'calibrator_path': 'configs/calibrator_player_rush_yds.json',
            'notes': 'Recalibrated 2025-11-03. Requires week 9 validation.'
        },
        'player_rush_att': {
            'confidence': ConfidenceLevel.MEDIUM,
            'min_edge': 0.08,
            'min_model_prob': 0.62,
            'use_market_calibrator': False,
            'notes': 'Use rushing yards calibrator as proxy.'
        },

        # PASSING PROPS (Critical Issues - 11.0% coverage)
        'player_pass_yds': {
            'confidence': ConfidenceLevel.DISABLED,
            'min_edge': 0.15,           # Very high edge if enabled
            'min_model_prob': 0.70,
            'use_market_calibrator': False,
            'notes': 'DISABLED: Severe underdispersion. Fix model before use.'
        },
        'player_pass_completions': {
            'confidence': ConfidenceLevel.DISABLED,
            'min_edge': 0.15,
            'min_model_prob': 0.70,
            'use_market_calibrator': False,
            'notes': 'DISABLED: Same issues as passing yards.'
        },
        'player_pass_tds': {
            'confidence': ConfidenceLevel.LOW,  # TDs are different distribution
            'min_edge': 0.10,
            'min_model_prob': 0.65,
            'use_market_calibrator': False,
            'notes': 'Lower confidence due to passing issues. Use with caution.'
        },

        # TD PROPS (Moderate Confidence)
        'player_anytime_td': {
            'confidence': ConfidenceLevel.MEDIUM,
            'min_edge': 0.10,
            'min_model_prob': 0.60,
            'use_market_calibrator': False,
            'notes': 'TD predictions enhanced but require higher edges.'
        },
        'player_1st_td': {
            'confidence': ConfidenceLevel.LOW,
            'min_edge': 0.15,
            'min_model_prob': 0.65,
            'use_market_calibrator': False,
            'notes': 'Very rare event. Requires very high edges.'
        },
    }

    def __init__(self,
                 enable_medium_confidence: bool = True,
                 enable_low_confidence: bool = False,
                 custom_min_edge: Optional[float] = None):
        """
        Initialize pick filter.

        Args:
            enable_medium_confidence: Include MEDIUM confidence picks (default True)
            enable_low_confidence: Include LOW confidence picks (default False)
            custom_min_edge: Override all minimum edges with this value
        """
        self.enable_medium = enable_medium_confidence
        self.enable_low = enable_low_confidence
        self.custom_min_edge = custom_min_edge

    def apply(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Filter recommendations to high-confidence picks only.

        Args:
            recommendations: DataFrame with columns:
                - market: Market type (e.g., 'player_reception_yds')
                - model_prob: Model probability (0-1)
                - edge: Edge over market (model_prob - market_prob)
                - player: Player name
                - pick: Over/Under X.X

        Returns:
            Filtered DataFrame with only high-confidence picks
        """
        if recommendations.empty:
            return recommendations

        filtered_picks = []

        for _, row in recommendations.iterrows():
            market = row.get('market', '')
            model_prob = row.get('model_prob', 0)
            edge = row.get('edge', 0)

            # Get confidence config for this market
            config = self.STAT_CONFIDENCE.get(market)

            if config is None:
                # Unknown market - skip
                continue

            confidence = config['confidence']

            # Check if this confidence level is enabled
            if confidence == ConfidenceLevel.DISABLED:
                continue
            if confidence == ConfidenceLevel.LOW and not self.enable_low:
                continue
            if confidence == ConfidenceLevel.MEDIUM and not self.enable_medium:
                continue

            # Apply edge and probability thresholds
            min_edge = self.custom_min_edge if self.custom_min_edge else config['min_edge']
            min_prob = config['min_model_prob']

            if edge >= min_edge and (model_prob >= min_prob or model_prob <= (1 - min_prob)):
                # Add confidence metadata
                row_dict = row.to_dict()
                row_dict['confidence_level'] = confidence.value
                row_dict['min_edge_threshold'] = min_edge
                row_dict['min_prob_threshold'] = min_prob
                row_dict['filter_notes'] = config.get('notes', '')

                filtered_picks.append(row_dict)

        result = pd.DataFrame(filtered_picks)

        if not result.empty:
            # Sort by confidence level then edge
            confidence_order = {
                'HIGH': 0,
                'MEDIUM': 1,
                'LOW': 2
            }
            result['confidence_sort'] = result['confidence_level'].map(confidence_order)
            result = result.sort_values(['confidence_sort', 'edge'], ascending=[True, False])
            result = result.drop('confidence_sort', axis=1)

        return result

    def get_summary_stats(self, recommendations: pd.DataFrame) -> Dict:
        """
        Get summary statistics on filtered picks.

        Returns:
            Dictionary with counts by confidence level and market type
        """
        filtered = self.apply(recommendations)

        if filtered.empty:
            return {
                'total_picks': 0,
                'by_confidence': {},
                'by_market': {}
            }

        return {
            'total_picks': len(filtered),
            'by_confidence': filtered['confidence_level'].value_counts().to_dict(),
            'by_market': filtered['market'].value_counts().to_dict(),
            'avg_edge': filtered['edge'].mean(),
            'avg_model_prob': filtered['model_prob'].mean(),
        }

    def print_filter_summary(self, recommendations: pd.DataFrame):
        """Print human-readable summary of filtering results."""
        total_input = len(recommendations)
        filtered = self.apply(recommendations)
        total_output = len(filtered)

        print("\n" + "="*80)
        print("HIGH-CONFIDENCE PICK FILTER SUMMARY")
        print("="*80)

        print(f"\n📊 Filtering Results:")
        print(f"  Input Recommendations: {total_input:,}")
        print(f"  High-Confidence Picks: {total_output:,}")
        print(f"  Filter Rate: {((total_input - total_output) / total_input * 100):.1f}% removed")

        if total_output > 0:
            stats = self.get_summary_stats(recommendations)

            print(f"\n🎯 By Confidence Level:")
            for conf, count in sorted(stats['by_confidence'].items()):
                pct = count / total_output * 100
                print(f"  {conf:<10} {count:>4} picks ({pct:>5.1f}%)")

            print(f"\n📈 By Market Type:")
            for market, count in sorted(stats['by_market'].items(), key=lambda x: -x[1])[:10]:
                pct = count / total_output * 100
                print(f"  {market:<30} {count:>4} picks ({pct:>5.1f}%)")

            print(f"\n💰 Edge Statistics:")
            print(f"  Average Edge: {stats['avg_edge']:.1%}")
            print(f"  Average Model Prob: {stats['avg_model_prob']:.1%}")

        print("\n" + "="*80)


def filter_to_high_confidence(recommendations: pd.DataFrame,
                              enable_medium: bool = True,
                              enable_low: bool = False) -> pd.DataFrame:
    """
    Convenience function to filter recommendations.

    Args:
        recommendations: Input recommendations DataFrame
        enable_medium: Include MEDIUM confidence picks
        enable_low: Include LOW confidence picks

    Returns:
        Filtered DataFrame with high-confidence picks
    """
    filter = PickFilter(
        enable_medium_confidence=enable_medium,
        enable_low_confidence=enable_low
    )
    return filter.apply(recommendations)


# Example usage
if __name__ == '__main__':
    # Create sample recommendations
    sample_recs = pd.DataFrame([
        # HIGH confidence (receiving yards) - should pass
        {'market': 'player_reception_yds', 'player': 'CeeDee Lamb', 'pick': 'Over 84.5',
         'model_prob': 0.65, 'edge': 0.12, 'line': 84.5},

        # MEDIUM confidence (rushing yards) - should pass with higher edge
        {'market': 'player_rush_yds', 'player': 'Christian McCaffrey', 'pick': 'Over 89.5',
         'model_prob': 0.70, 'edge': 0.10, 'line': 89.5},

        # MEDIUM confidence but low edge - should FAIL
        {'market': 'player_rush_yds', 'player': 'Tony Pollard', 'pick': 'Over 54.5',
         'model_prob': 0.60, 'edge': 0.04, 'line': 54.5},

        # DISABLED (passing yards) - should FAIL
        {'market': 'player_pass_yds', 'player': 'Patrick Mahomes', 'pick': 'Over 274.5',
         'model_prob': 0.75, 'edge': 0.15, 'line': 274.5},

        # HIGH confidence (receptions) - should pass
        {'market': 'player_receptions', 'player': 'Tyreek Hill', 'pick': 'Over 6.5',
         'model_prob': 0.68, 'edge': 0.09, 'line': 6.5},
    ])

    # Apply filter
    filter = PickFilter(enable_medium_confidence=True, enable_low_confidence=False)

    print("SAMPLE RECOMMENDATIONS:")
    print("-" * 80)
    print(sample_recs[['player', 'market', 'pick', 'edge', 'model_prob']].to_string(index=False))

    # Filter and print results
    filter.print_filter_summary(sample_recs)

    filtered = filter.apply(sample_recs)
    print("\nFILTERED PICKS:")
    print("-" * 80)
    if not filtered.empty:
        print(filtered[['player', 'market', 'pick', 'edge', 'model_prob', 'confidence_level']].to_string(index=False))
    else:
        print("No picks passed filter")
