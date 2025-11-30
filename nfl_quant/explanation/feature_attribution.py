"""
Feature Attribution and Explanation System.

Provides transparent explanations of model predictions by attributing
prediction changes to input features.

Methods:
1. SHAP-like attribution (Shapley values approximation)
2. Perturbation-based importance (change in prediction when feature changes)
3. Linear approximation (gradient-based for continuous features)
4. Simple delta method (baseline vs adjusted)

Output: Top 5 feature contributions with direction and magnitude
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttributionMethod(Enum):
    """Attribution methods."""
    DELTA = "delta"  # Simple: Adjusted - Baseline
    PERTURBATION = "perturbation"  # Perturbation analysis
    LINEAR = "linear"  # Linear approximation (gradient)


@dataclass
class FeatureContribution:
    """Single feature contribution to prediction."""
    feature_name: str
    baseline_value: float
    adjusted_value: float
    contribution: float  # Impact on prediction
    contribution_pct: float  # % of total adjustment
    direction: str  # "positive" or "negative"
    description: str  # Human-readable explanation


class FeatureAttributor:
    """
    Explains model predictions via feature attribution.

    Identifies which input features drive prediction changes.
    """

    def __init__(self):
        """Initialize feature attributor."""
        pass

    def explain_prediction_delta(
        self,
        baseline_prediction: float,
        adjusted_prediction: float,
        baseline_features: Dict[str, float],
        adjusted_features: Dict[str, float],
        top_k: int = 5
    ) -> List[FeatureContribution]:
        """
        Explain prediction change using delta method.

        Compares baseline vs adjusted features, attributes change proportionally.

        Args:
            baseline_prediction: Baseline prediction value
            adjusted_prediction: Adjusted prediction value
            baseline_features: Baseline feature values
            adjusted_features: Adjusted feature values
            top_k: Number of top features to return

        Returns:
            List of FeatureContribution objects (top K by absolute contribution)
        """
        total_delta = adjusted_prediction - baseline_prediction

        if abs(total_delta) < 0.01:
            logger.info("Prediction change negligible (<0.01), no meaningful attributions")
            return []

        # Calculate per-feature deltas
        feature_deltas = {}
        for feature_name in baseline_features.keys():
            if feature_name in adjusted_features:
                baseline_val = baseline_features[feature_name]
                adjusted_val = adjusted_features[feature_name]
                delta = adjusted_val - baseline_val

                if abs(delta) > 0.001:  # Filter noise
                    feature_deltas[feature_name] = {
                        'baseline': baseline_val,
                        'adjusted': adjusted_val,
                        'delta': delta
                    }

        # Estimate contribution (simple heuristic: proportional to delta magnitude)
        # More sophisticated: use sensitivity/gradient
        contributions = []

        for feature_name, vals in feature_deltas.items():
            # Assume linear relationship: contribution = delta * sensitivity
            # For now, use normalized delta as proxy
            delta_normalized = vals['delta'] / (abs(vals['baseline']) + 0.01)  # Avoid div by zero

            # Contribution = total_delta * (this feature's relative change)
            # Simplified: use delta directly scaled
            contrib = total_delta * (delta_normalized / sum(abs(v['delta']) / (abs(v['baseline']) + 0.01) for v in feature_deltas.values()))

            contrib_pct = (contrib / total_delta * 100) if total_delta != 0 else 0

            direction = "positive" if contrib > 0 else "negative"

            description = self._generate_description(
                feature_name=feature_name,
                baseline=vals['baseline'],
                adjusted=vals['adjusted'],
                contribution=contrib
            )

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                baseline_value=vals['baseline'],
                adjusted_value=vals['adjusted'],
                contribution=contrib,
                contribution_pct=contrib_pct,
                direction=direction,
                description=description
            ))

        # Sort by absolute contribution, return top K
        contributions_sorted = sorted(contributions, key=lambda x: abs(x.contribution), reverse=True)

        return contributions_sorted[:top_k]

    def explain_game_total(
        self,
        baseline_total: float,
        adjusted_total: float,
        adjustments: Dict[str, Dict[str, float]]
    ) -> List[FeatureContribution]:
        """
        Explain game total prediction with structured adjustments.

        Args:
            baseline_total: Baseline total (e.g., from neutral model)
            adjusted_total: Adjusted total (e.g., after weather, rest, etc.)
            adjustments: Dictionary of adjustment categories:
                {
                    'weather': {'wind_impact': -1.2, 'temp_impact': -0.5},
                    'rest': {'home_rest_boost': 0.3, 'away_short_rest_penalty': -0.8},
                    'divisional': {'divisional_adjustment': -0.5},
                    ...
                }

        Returns:
            List of FeatureContribution objects
        """
        total_delta = adjusted_total - baseline_total

        contributions = []

        for category, sub_adjustments in adjustments.items():
            for feature_name, impact in sub_adjustments.items():
                if abs(impact) < 0.01:  # Skip negligible
                    continue

                contrib_pct = (impact / total_delta * 100) if total_delta != 0 else 0

                direction = "positive" if impact > 0 else "negative"

                description = self._generate_description_from_category(
                    category=category,
                    feature_name=feature_name,
                    impact=impact
                )

                contributions.append(FeatureContribution(
                    feature_name=f"{category}:{feature_name}",
                    baseline_value=baseline_total,
                    adjusted_value=baseline_total + impact,
                    contribution=impact,
                    contribution_pct=contrib_pct,
                    direction=direction,
                    description=description
                ))

        # Sort by absolute contribution
        contributions_sorted = sorted(contributions, key=lambda x: abs(x.contribution), reverse=True)

        return contributions_sorted[:5]  # Top 5

    def explain_player_prop(
        self,
        baseline_projection: float,
        adjusted_projection: float,
        feature_impacts: Dict[str, float],
        stat_type: str = 'yards'
    ) -> List[FeatureContribution]:
        """
        Explain player prop projection.

        Args:
            baseline_projection: Baseline projection (e.g., from model mean)
            adjusted_projection: Adjusted projection (e.g., after context)
            feature_impacts: Dictionary mapping feature name to impact
                {
                    'weather_adjustment': -3.5,
                    'opponent_defense_rank': -2.0,
                    'usage_boost': 4.0,
                    ...
                }
            stat_type: Type of stat ('yards', 'receptions', etc.)

        Returns:
            List of FeatureContribution objects
        """
        total_delta = adjusted_projection - baseline_projection

        contributions = []

        for feature_name, impact in feature_impacts.items():
            if abs(impact) < 0.01:
                continue

            contrib_pct = (impact / total_delta * 100) if total_delta != 0 else 0

            direction = "positive" if impact > 0 else "negative"

            description = self._generate_player_description(
                feature_name=feature_name,
                impact=impact,
                stat_type=stat_type
            )

            contributions.append(FeatureContribution(
                feature_name=feature_name,
                baseline_value=baseline_projection,
                adjusted_value=baseline_projection + impact,
                contribution=impact,
                contribution_pct=contrib_pct,
                direction=direction,
                description=description
            ))

        # Sort by absolute contribution
        contributions_sorted = sorted(contributions, key=lambda x: abs(x.contribution), reverse=True)

        return contributions_sorted[:5]

    def _generate_description(
        self,
        feature_name: str,
        baseline: float,
        adjusted: float,
        contribution: float
    ) -> str:
        """Generate human-readable description."""
        delta = adjusted - baseline
        direction = "increased" if delta > 0 else "decreased"

        return (
            f"{feature_name} {direction} from {baseline:.2f} to {adjusted:.2f}, "
            f"contributing {contribution:+.2f} to prediction"
        )

    def _generate_description_from_category(
        self,
        category: str,
        feature_name: str,
        impact: float
    ) -> str:
        """Generate description from category and feature."""
        direction = "increases" if impact > 0 else "decreases"
        abs_impact = abs(impact)

        category_descriptions = {
            'weather': f"Weather ({feature_name}) {direction} total by {abs_impact:.1f} points",
            'rest': f"Rest advantage ({feature_name}) {direction} total by {abs_impact:.1f} points",
            'divisional': f"Divisional matchup {direction} total by {abs_impact:.1f} points",
            'travel': f"Travel impact ({feature_name}) {direction} total by {abs_impact:.1f} points",
            'market': f"Market prior blend {direction} total by {abs_impact:.1f} points",
            'pace': f"Pace adjustment ({feature_name}) {direction} total by {abs_impact:.1f} points"
        }

        return category_descriptions.get(category, f"{feature_name} impact: {impact:+.1f}")

    def _generate_player_description(
        self,
        feature_name: str,
        impact: float,
        stat_type: str
    ) -> str:
        """Generate player prop description."""
        direction = "increases" if impact > 0 else "decreases"
        abs_impact = abs(impact)

        descriptions = {
            'weather_adjustment': f"Weather conditions {direction} {stat_type} by {abs_impact:.1f}",
            'opponent_defense_rank': f"Opponent defense {direction} {stat_type} by {abs_impact:.1f}",
            'usage_boost': f"Increased usage {direction} {stat_type} by {abs_impact:.1f}",
            'game_script': f"Game script {direction} {stat_type} by {abs_impact:.1f}",
            'injury_impact': f"Teammate injury {direction} {stat_type} by {abs_impact:.1f}",
            'matchup_history': f"Historical matchup {direction} {stat_type} by {abs_impact:.1f}",
            'pace_adjustment': f"Game pace {direction} {stat_type} by {abs_impact:.1f}"
        }

        return descriptions.get(feature_name, f"{feature_name}: {impact:+.1f} {stat_type}")

    def format_explanation(
        self,
        contributions: List[FeatureContribution],
        prediction_name: str = "Prediction"
    ) -> str:
        """
        Format feature contributions as readable text.

        Args:
            contributions: List of FeatureContribution objects
            prediction_name: Name of prediction (e.g., "Game Total", "Player Yards")

        Returns:
            Formatted explanation string
        """
        if not contributions:
            return f"{prediction_name} Explanation: No significant feature impacts identified."

        lines = [f"\n{prediction_name} Explanation (Top {len(contributions)} Features):"]
        lines.append("=" * 80)

        for i, contrib in enumerate(contributions, 1):
            sign = "+" if contrib.contribution > 0 else ""
            lines.append(
                f"{i}. {contrib.feature_name}: {sign}{contrib.contribution:.2f} "
                f"({sign}{contrib.contribution_pct:.1f}%)"
            )
            lines.append(f"   {contrib.description}")

        return "\n".join(lines)


def create_explanation_summary(
    prediction_type: str,
    baseline_value: float,
    final_value: float,
    feature_impacts: Dict[str, float],
    top_k: int = 5
) -> Dict[str, any]:
    """
    Create a comprehensive explanation summary.

    Args:
        prediction_type: Type of prediction (e.g., "Game Total", "Receiving Yards")
        baseline_value: Baseline prediction
        final_value: Final adjusted prediction
        feature_impacts: Dictionary of feature impacts
        top_k: Number of top features to include

    Returns:
        Dictionary with explanation summary
    """
    attributor = FeatureAttributor()

    contributions = attributor.explain_player_prop(
        baseline_projection=baseline_value,
        adjusted_projection=final_value,
        feature_impacts=feature_impacts,
        stat_type=prediction_type.split()[-1].lower()  # Extract stat type
    )

    formatted_explanation = attributor.format_explanation(
        contributions=contributions,
        prediction_name=prediction_type
    )

    return {
        'prediction_type': prediction_type,
        'baseline_value': baseline_value,
        'final_value': final_value,
        'total_adjustment': final_value - baseline_value,
        'top_contributions': [
            {
                'feature': c.feature_name,
                'impact': c.contribution,
                'pct': c.contribution_pct,
                'description': c.description
            }
            for c in contributions
        ],
        'formatted_explanation': formatted_explanation
    }


# Example usage and testing
if __name__ == '__main__':
    attributor = FeatureAttributor()

    print("=== Feature Attribution Examples ===\n")

    # Example 1: Game total explanation
    print("1. Game Total Explanation")
    baseline_total = 46.5
    adjusted_total = 43.2

    adjustments = {
        'weather': {
            'wind_impact': -1.8,
            'temp_impact': -0.5
        },
        'rest': {
            'home_coming_off_bye': 0.4,
            'away_short_rest': -0.7
        },
        'divisional': {
            'divisional_adjustment': -0.5
        },
        'pace': {
            'slow_pace_adjustment': -0.4
        }
    }

    game_contributions = attributor.explain_game_total(
        baseline_total=baseline_total,
        adjusted_total=adjusted_total,
        adjustments=adjustments
    )

    explanation_text = attributor.format_explanation(
        contributions=game_contributions,
        prediction_name="Game Total"
    )
    print(explanation_text)
    print()

    # Example 2: Player prop explanation
    print("\n2. Player Prop Explanation (WR Receiving Yards)")
    baseline_yards = 75.0
    adjusted_yards = 68.5

    feature_impacts = {
        'weather_adjustment': -4.5,
        'opponent_defense_rank': -3.0,
        'game_script': 1.5,
        'pace_adjustment': -0.5
    }

    player_contributions = attributor.explain_player_prop(
        baseline_projection=baseline_yards,
        adjusted_projection=adjusted_yards,
        feature_impacts=feature_impacts,
        stat_type='yards'
    )

    player_explanation = attributor.format_explanation(
        contributions=player_contributions,
        prediction_name="WR Receiving Yards"
    )
    print(player_explanation)
    print()

    # Example 3: Comprehensive summary
    print("\n3. Comprehensive Explanation Summary")
    summary = create_explanation_summary(
        prediction_type="RB Rushing Yards",
        baseline_value=85.0,
        final_value=92.0,
        feature_impacts={
            'weather_adjustment': 3.0,
            'opponent_defense_rank': 2.5,
            'usage_boost': 2.0,
            'game_script': -0.5
        },
        top_k=5
    )

    print(f"Prediction Type: {summary['prediction_type']}")
    print(f"Baseline: {summary['baseline_value']:.1f}")
    print(f"Final: {summary['final_value']:.1f}")
    print(f"Total Adjustment: {summary['total_adjustment']:+.1f}")
    print(f"\nTop Contributions:")
    for contrib in summary['top_contributions']:
        print(f"  - {contrib['feature']}: {contrib['impact']:+.1f} ({contrib['pct']:+.1f}%)")
        print(f"    {contrib['description']}")
