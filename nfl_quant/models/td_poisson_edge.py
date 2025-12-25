"""
TD Poisson Edge - Touchdown Prediction using Poisson Regression

Uses Poisson regression for TD count prediction instead of XGBoost.
TDs are count data (0, 1, 2, 3...) following a Poisson distribution,
which is fundamentally different from continuous stats like yards.

Key features:
- Predicts expected TD count using Poisson regression
- Converts to P(over) or P(under) using Poisson CDF
- Uses red zone features as primary signals

Usage:
    edge = TDPoissonEdge()
    edge.train(train_data, 'player_pass_tds')
    prob = edge.predict_over_probability(features, line=1.5)
"""
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# TD-specific features
TD_FEATURES = {
    'player_pass_tds': [
        'trailing_passing_tds',     # Historical TD rate
        'trailing_passing_yards',   # Volume indicator
        'rz_pass_attempts_share',   # Red zone opportunity
        'vegas_total',              # Game environment
        'vegas_spread',             # Game script
        'opponent_pass_td_allowed', # Opponent RZ pass TD rate allowed
    ],
    'player_rush_tds': [
        'trailing_rushing_tds',     # Historical TD rate
        'trailing_carries',         # Volume indicator
        'trailing_rz_rush_share',   # Red zone opportunity (critical!)
        'rz_td_per_carry',          # Player's RZ TD conversion rate
        'gl_carry_share',           # Goal-line role (yardline <= 5)
        'vegas_total',              # Game environment
        'vegas_spread',             # Game script
        'opponent_rush_td_allowed', # Opponent RZ rush TD rate allowed
        'opp_rz_td_allowed',        # Combined opponent RZ TD rate
    ],
    'player_rec_tds': [
        'trailing_receiving_tds',   # Historical TD rate
        'trailing_targets',         # Volume indicator
        'trailing_rz_target_share', # Red zone opportunity
        'rz_td_per_target',         # Player's RZ TD conversion rate
        'gl_target_share',          # Goal-line role for receivers
        'vegas_total',              # Game environment
        'opponent_pass_td_allowed', # Opponent RZ pass TD rate allowed
        'opp_rz_td_allowed',        # Combined opponent RZ TD rate
    ],
}

# TD markets
TD_MARKETS = ['player_pass_tds', 'player_rush_tds', 'player_rec_tds']


class TDPoissonEdge:
    """
    Poisson regression for TD count prediction.

    Uses Poisson distribution which is appropriate for count data.
    Converts predictions to over/under probabilities using Poisson CDF.

    Attributes:
        models: Dict mapping market -> PoissonRegressor
        scalers: Dict mapping market -> StandardScaler
        imputers: Dict mapping market -> SimpleImputer
        metrics: Training metrics per market
        version: Model version string
        trained_date: When model was trained
    """

    def __init__(self):
        """Initialize TD Poisson edge."""
        self.models: Dict[str, PoissonRegressor] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.metrics: Dict[str, Dict] = {}
        self.version: Optional[str] = None
        self.trained_date: Optional[datetime] = None

    def get_features(self, market: str) -> List[str]:
        """Get feature list for a market."""
        return TD_FEATURES.get(market, [])

    def predict_td_count(
        self,
        features: pd.DataFrame,
        market: str,
    ) -> np.ndarray:
        """
        Predict expected TD count.

        Args:
            features: DataFrame with TD features
            market: TD market type

        Returns:
            Array of expected TD counts (lambda parameter)
        """
        if market not in self.models:
            raise ValueError(f"No trained model for market: {market}")

        model = self.models[market]

        # Use the features that were actually used during training
        if market in self.metrics and 'features_used' in self.metrics[market]:
            feature_cols = self.metrics[market]['features_used']
        else:
            feature_cols = self.get_features(market)

        # Get available features (must match training features exactly)
        available = [f for f in feature_cols if f in features.columns]

        if len(available) == 0:
            raise ValueError(f"No matching features found for {market}")

        X = features[available].copy()

        # Fill missing values before preprocessing
        X = X.fillna(0)

        # Apply preprocessing
        if market in self.imputers:
            X = self.imputers[market].transform(X)
        if market in self.scalers:
            X = self.scalers[market].transform(X)

        # Poisson regression predicts log(lambda), so we need exp()
        return model.predict(X)

    def predict_over_probability(
        self,
        features: pd.DataFrame,
        market: str,
        line: float,
    ) -> np.ndarray:
        """
        Predict P(TDs > line) using Poisson CDF.

        For line=1.5, computes P(TDs >= 2) = 1 - P(TDs <= 1)

        Args:
            features: DataFrame with TD features
            market: TD market type
            line: Betting line (e.g., 1.5)

        Returns:
            Array of P(over) probabilities
        """
        expected_tds = self.predict_td_count(features, market)

        # P(X > line) = 1 - P(X <= floor(line))
        # For line=1.5, we want P(X >= 2) = 1 - P(X <= 1)
        threshold = int(np.floor(line))

        return 1 - poisson.cdf(threshold, expected_tds)

    def predict_under_probability(
        self,
        features: pd.DataFrame,
        market: str,
        line: float,
    ) -> np.ndarray:
        """
        Predict P(TDs < line) using Poisson CDF.

        For line=1.5, computes P(TDs <= 1)

        Args:
            features: DataFrame with TD features
            market: TD market type
            line: Betting line (e.g., 1.5)

        Returns:
            Array of P(under) probabilities
        """
        expected_tds = self.predict_td_count(features, market)

        # P(X < line) = P(X <= floor(line))
        # For line=1.5, we want P(X <= 1)
        threshold = int(np.floor(line))

        return poisson.cdf(threshold, expected_tds)

    def evaluate_bet(
        self,
        row: pd.Series,
        market: str,
        line: float,
        min_confidence: float = 0.58,
    ) -> Dict[str, Any]:
        """
        Evaluate a single TD bet opportunity.

        Args:
            row: Series with features for a single bet
            market: TD market type
            line: Betting line
            min_confidence: Minimum probability to recommend bet

        Returns:
            Dict with bet recommendation
        """
        features = pd.DataFrame([row])

        try:
            expected_tds = self.predict_td_count(features, market)[0]
            p_over = self.predict_over_probability(features, market, line)[0]
            p_under = 1 - p_over
        except Exception as e:
            return {
                'should_bet': False,
                'direction': None,
                'confidence': 0.0,
                'expected_tds': None,
                'error': str(e),
            }

        # Determine direction and confidence
        if p_over > min_confidence:
            direction = 'OVER'
            confidence = p_over
        elif p_under > min_confidence:
            direction = 'UNDER'
            confidence = p_under
        else:
            direction = None
            confidence = max(p_over, p_under)

        return {
            'should_bet': confidence >= min_confidence,
            'direction': direction,
            'confidence': confidence,
            'expected_tds': float(expected_tds),
            'p_over': float(p_over),
            'p_under': float(p_under),
            'line': line,
            'market': market,
        }

    def train(
        self,
        train_data: pd.DataFrame,
        market: str,
    ) -> Dict[str, Any]:
        """
        Train Poisson model for a TD market.

        Args:
            train_data: Historical data with TD outcomes
            market: TD market type

        Returns:
            Training metrics
        """
        feature_cols = self.get_features(market)

        # Get target column
        target_map = {
            'player_pass_tds': 'passing_tds',
            'player_rush_tds': 'rushing_tds',
            'player_rec_tds': 'receiving_tds',
        }
        target_col = target_map.get(market, 'actual')

        if target_col not in train_data.columns:
            # Try 'actual' as fallback
            if 'actual' in train_data.columns:
                target_col = 'actual'
            else:
                raise ValueError(f"No target column found for {market}")

        # Get available features
        available = [f for f in feature_cols if f in train_data.columns]

        if len(available) < 2:
            raise ValueError(
                f"Insufficient features for {market}: {available}"
            )

        # Prepare data
        train_data = train_data.dropna(subset=[target_col])
        X = train_data[available].copy()
        y = train_data[target_col].copy()

        # Ensure y is non-negative integer
        y = y.clip(lower=0).astype(int)

        if len(X) < 50:
            raise ValueError(f"Insufficient samples for {market}: {len(X)}")

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        self.imputers[market] = imputer
        self.scalers[market] = scaler

        # Train Poisson regression
        model = PoissonRegressor(alpha=0.1, max_iter=1000)
        model.fit(X_scaled, y)

        self.models[market] = model

        # Calculate metrics
        predictions = model.predict(X_scaled)
        mae = np.abs(predictions - y).mean()
        rmse = np.sqrt(np.mean((predictions - y) ** 2))

        # Compute pseudo R-squared for Poisson
        y_mean = y.mean()
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - predictions) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        metrics = {
            'market': market,
            'n_samples': len(X),
            'n_features': len(available),
            'features_used': available,
            'mae': float(mae),
            'rmse': float(rmse),
            'pseudo_r2': float(r2),
            'mean_actual': float(y.mean()),
            'mean_predicted': float(predictions.mean()),
        }

        self.metrics[market] = metrics
        self.trained_date = datetime.now()
        self.version = "td_poisson_v1"

        print(f"  Trained {market}:")
        print(f"    Samples: {len(X)}")
        print(f"    MAE: {mae:.3f}")
        print(f"    Mean actual TDs: {y.mean():.2f}")
        print(f"    Mean predicted TDs: {predictions.mean():.2f}")

        return metrics

    def save(self, path: Path = None) -> None:
        """Save TD Poisson edge to disk."""
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'td_poisson_edge.joblib'

        bundle = {
            'models': self.models,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'metrics': self.metrics,
            'version': self.version,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
        }
        joblib.dump(bundle, path)
        print(f"TD Poisson edge saved to: {path}")

    @classmethod
    def load(cls, path: Path = None) -> 'TDPoissonEdge':
        """Load TD Poisson edge from disk."""
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'td_poisson_edge.joblib'

        bundle = joblib.load(path)

        edge = cls()
        edge.models = bundle.get('models', {})
        edge.scalers = bundle.get('scalers', {})
        edge.imputers = bundle.get('imputers', {})
        edge.metrics = bundle.get('metrics', {})
        edge.version = bundle.get('version')
        edge.trained_date = (
            datetime.fromisoformat(bundle['trained_date'])
            if bundle.get('trained_date')
            else None
        )

        return edge

    def __repr__(self) -> str:
        return (
            f"TDPoissonEdge(version='{self.version}', "
            f"markets={list(self.models.keys())})"
        )
