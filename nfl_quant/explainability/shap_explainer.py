"""
SHAP-based Model Explainability for NFL Prop Predictions.

Provides transparent feature contribution explanations for XGBoost predictions.
Shows exactly how each factor (snap_share, defense EPA, etc.) affects the projection.

Example output:
    📊 CALCULATION BREAKDOWN:
    Base prediction: 4.2 targets
    + trailing_attempts: +1.2 (strong history)
    + snap_share: +0.3 (70% snap share)
    - opp_pass_def_epa: -0.4 (tough defense)
    = Final projection: 5.3 targets
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Single feature's contribution to the prediction."""
    feature_name: str
    feature_value: float
    shap_value: float  # Contribution to prediction (in units of output)
    direction: str  # 'positive' or 'negative'

    @property
    def formatted(self) -> str:
        """Format as human-readable string."""
        sign = '+' if self.shap_value >= 0 else ''
        return f"{sign}{self.shap_value:.2f} ({self.feature_name}: {self.feature_value:.2f})"


@dataclass
class PredictionExplanation:
    """Full explanation of a single prediction."""
    base_value: float  # Expected value (average prediction)
    prediction: float  # Final prediction
    contributions: List[FeatureContribution]

    def get_top_factors(self, n: int = 5) -> List[FeatureContribution]:
        """Get top N factors by absolute contribution."""
        sorted_contribs = sorted(
            self.contributions,
            key=lambda x: abs(x.shap_value),
            reverse=True
        )
        return sorted_contribs[:n]

    def format_breakdown(self, top_n: int = 5) -> str:
        """
        Format as readable breakdown.

        Example:
            Base: 4.2 targets
            + trailing_attempts: +1.2
            + snap_share: +0.3
            - opp_pass_def_epa: -0.4
            = Projection: 5.3 targets
        """
        lines = [f"Base: {self.base_value:.1f}"]

        for contrib in self.get_top_factors(top_n):
            sign = '+' if contrib.shap_value >= 0 else ''
            lines.append(
                f"  {sign}{contrib.shap_value:.2f} ← {contrib.feature_name} ({contrib.feature_value:.2f})"
            )

        lines.append(f"= Projection: {self.prediction:.1f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        result = {
            'shap_base_value': self.base_value,
            'shap_prediction': self.prediction,
        }
        for i, contrib in enumerate(self.get_top_factors(5)):
            result[f'shap_factor_{i+1}_name'] = contrib.feature_name
            result[f'shap_factor_{i+1}_value'] = contrib.shap_value
            result[f'shap_factor_{i+1}_input'] = contrib.feature_value
        return result


class SHAPExplainer:
    """
    SHAP explainer for XGBoost-based predictions.

    Wraps the XGBoost model and provides per-prediction explanations
    showing how each feature contributes to the final output.
    """

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.

        Args:
            model: XGBoost model (or model object with .model attribute)
            feature_names: List of feature names (for human-readable output)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")

        # Handle wrapped models (UsagePredictor has .model attribute)
        if hasattr(model, 'model'):
            self._model = model.model
        else:
            self._model = model

        self.feature_names = feature_names or []
        self._explainer = None

    def _get_explainer(self, X: pd.DataFrame) -> shap.Explainer:
        """Get or create SHAP TreeExplainer."""
        if self._explainer is None:
            # Use TreeExplainer for XGBoost (fast, exact)
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer

    def explain(
        self,
        X: pd.DataFrame,
        output_index: int = 0
    ) -> List[PredictionExplanation]:
        """
        Generate SHAP explanations for predictions.

        Args:
            X: Feature dataframe (same format as model input)
            output_index: Which output to explain (0 for regression/binary)

        Returns:
            List of PredictionExplanation objects (one per row in X)
        """
        if len(X) == 0:
            return []

        explainer = self._get_explainer(X)

        # Get SHAP values
        shap_values = explainer.shap_values(X)
        base_value = explainer.expected_value

        # Handle multi-output models
        if isinstance(base_value, np.ndarray):
            base_value = base_value[output_index]
        if isinstance(shap_values, list):
            shap_values = shap_values[output_index]

        # Get feature names
        feature_names = (
            self.feature_names if self.feature_names
            else list(X.columns)
        )

        explanations = []
        for i in range(len(X)):
            row_values = X.iloc[i].values
            row_shap = shap_values[i] if len(shap_values.shape) > 1 else shap_values

            contributions = []
            for j, (feat_name, feat_val, shap_val) in enumerate(
                zip(feature_names, row_values, row_shap)
            ):
                contributions.append(FeatureContribution(
                    feature_name=feat_name,
                    feature_value=float(feat_val),
                    shap_value=float(shap_val),
                    direction='positive' if shap_val >= 0 else 'negative'
                ))

            prediction = base_value + sum(row_shap)
            explanations.append(PredictionExplanation(
                base_value=float(base_value),
                prediction=float(prediction),
                contributions=contributions
            ))

        return explanations

    def explain_single(self, X: pd.DataFrame) -> PredictionExplanation:
        """Explain a single prediction (convenience method)."""
        explanations = self.explain(X)
        if explanations:
            return explanations[0]
        return None


class ClassifierSHAPExplainer:
    """
    SHAP explainer specifically for the V23 prop classifier.

    Explains the XGBoost classifier predictions (P(under) probability)
    by showing which features push toward OVER vs UNDER.
    """

    def __init__(self, classifier_model):
        """
        Initialize with V23 classifier model.

        Args:
            classifier_model: XGBoost classifier from active_model.joblib
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Run: pip install shap")

        self._model = classifier_model
        self._explainer = None

    def _get_explainer(self) -> shap.Explainer:
        """Get or create TreeExplainer."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer

    def explain_prediction(
        self,
        features: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Explain a single classifier prediction.

        Args:
            features: Single-row DataFrame with model features
            feature_names: Optional feature names for display

        Returns:
            Dictionary with:
            - base_probability: Average model probability
            - final_probability: This prediction's probability
            - top_factors: List of (feature, contribution, direction)
            - formatted_explanation: Human-readable string
        """
        explainer = self._get_explainer()

        # Get SHAP values (for binary classifier, we want class 1 = UNDER)
        shap_values = explainer.shap_values(features)
        base_value = explainer.expected_value

        # Handle binary classifier output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 = UNDER
            base_value = base_value[1]

        row_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        feature_vals = features.iloc[0].values
        names = feature_names or list(features.columns)

        # Build factor list sorted by absolute contribution
        factors = []
        for name, val, shap_val in zip(names, feature_vals, row_shap):
            factors.append({
                'feature': name,
                'value': float(val),
                'contribution': float(shap_val),
                'direction': 'UNDER' if shap_val > 0 else 'OVER'
            })

        factors.sort(key=lambda x: abs(x['contribution']), reverse=True)

        # Calculate final probability (in log-odds space, then convert)
        log_odds_base = np.log(base_value / (1 - base_value)) if 0 < base_value < 1 else 0
        log_odds_final = log_odds_base + sum(row_shap)
        final_prob = 1 / (1 + np.exp(-log_odds_final))

        # Format explanation
        lines = [f"Base P(UNDER): {base_value:.1%}"]
        for f in factors[:5]:
            sign = '+' if f['contribution'] >= 0 else ''
            direction = '→UNDER' if f['contribution'] > 0 else '→OVER'
            lines.append(
                f"  {f['feature']}: {sign}{f['contribution']:.3f} {direction}"
            )
        lines.append(f"Final P(UNDER): {final_prob:.1%}")

        return {
            'base_probability': float(base_value),
            'final_probability': float(final_prob),
            'top_factors': factors[:5],
            'formatted_explanation': '\n'.join(lines),
            'all_factors': factors
        }


def get_shap_reasoning(
    explanation: Dict,
    line: float,
    trailing: float,
    projection: float
) -> str:
    """
    Generate model reasoning string with SHAP-based explanations.

    Produces output like:
    📊 CALCULATION:
    Base P(UNDER): 52%
    + line_vs_trailing: +0.15 → UNDER (line high vs avg)
    - snap_share: -0.08 → OVER (high usage = more volume)
    + opp_def_epa: +0.05 → UNDER (tough defense)
    = Final: 64% UNDER

    Line: 5.5 | Trailing: 4.2 | Projection: 4.8
    → Take UNDER (line 1.3 above projection)
    """
    parts = []

    # SHAP breakdown
    parts.append("📊 CLASSIFIER FACTORS:")
    parts.append(f"  Base P(UNDER): {explanation['base_probability']:.0%}")

    for f in explanation['top_factors'][:4]:
        sign = '+' if f['contribution'] >= 0 else ''
        arrow = '→UNDER' if f['contribution'] > 0 else '→OVER'
        parts.append(f"  {sign}{f['contribution']:.2f} {f['feature']} {arrow}")

    final_p = explanation['final_probability']
    parts.append(f"  = P(UNDER): {final_p:.0%}")

    # Projection context
    parts.append("")
    parts.append("📈 PROJECTION:")
    parts.append(f"  Line: {line:.1f} | Trailing Avg: {trailing:.1f} | Model: {projection:.1f}")

    gap = line - projection
    if gap > 0:
        parts.append(f"  Line is {gap:.1f} ABOVE projection → UNDER")
    else:
        parts.append(f"  Line is {abs(gap):.1f} BELOW projection → OVER")

    return " • ".join(parts)


# Singleton explainers (lazy-loaded)
_usage_explainer: Optional[SHAPExplainer] = None
_classifier_explainer: Optional[ClassifierSHAPExplainer] = None


def get_usage_explainer(usage_predictor) -> SHAPExplainer:
    """Get or create SHAP explainer for usage predictor."""
    global _usage_explainer
    if _usage_explainer is None:
        _usage_explainer = SHAPExplainer(usage_predictor)
        logger.info("Created SHAP explainer for usage predictor")
    return _usage_explainer


def get_classifier_explainer(classifier_model) -> ClassifierSHAPExplainer:
    """Get or create SHAP explainer for V23 classifier."""
    global _classifier_explainer
    if _classifier_explainer is None:
        _classifier_explainer = ClassifierSHAPExplainer(classifier_model)
        logger.info("Created SHAP explainer for classifier")
    return _classifier_explainer
