"""
Anytime TD (ATTD) Ensemble Model

Combines Poisson-based P(TDs >= 1) with Logistic Regression for ATTD prediction.
Uses position-specific models with DraftKings rule compliance.

Architecture:
    ATTD Probability = w1 * P_poisson(TDs >= 1) + w2 * P_logistic(TD)

DraftKings ATTD Rules:
- QB: Rush + Rec TDs only (NOT passing TDs)
- RB: Rush + Rec TDs
- WR: Rec TDs only
- TE: Rec TDs only

Usage:
    ensemble = ATTDEnsemble()
    ensemble.train_all_positions(train_data)
    prob = ensemble.predict_attd_probability(features, position='RB')
"""

from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging

logger = logging.getLogger(__name__)


# Position-specific feature sets for ATTD
# Key insight: trailing_rz_carries/targets are most predictive
# - RBs with trailing_rz_carries >= 3.0: 53.9% TD rate (vs 24% base)
# - WRs with trailing_rz_targets >= 2.0: 48.8% TD rate (vs 18% base)
ATTD_FEATURES = {
    'QB': [
        'trailing_rz_carries',       # KEY: Trailing RZ rush opportunities
        'trailing_rushing_tds',      # Historical rushing TD rate (NOT passing)
        'rz_td_per_carry',           # RZ conversion efficiency
        'gl_carry_share',            # Goal-line role
        'snap_share',                # Playing time
        'opp_rz_td_allowed',         # Opponent RZ TD rate
        'vegas_total',               # Game environment
        'vegas_spread',              # Game script
    ],
    'RB': [
        'trailing_rz_carries',       # KEY: Trailing RZ rush opportunities (53.9% TD rate @ >=3)
        'trailing_rz_targets',       # RZ receiving opportunities
        'trailing_rz_touches',       # Combined RZ touches
        'trailing_rushing_tds',      # Historical rushing TD rate
        'trailing_receiving_tds',    # Historical receiving TD rate
        'rz_td_per_carry',           # RZ conversion efficiency
        'gl_carry_share',            # Goal-line role
        'snap_share',                # Playing time
        'opp_rz_td_allowed',         # Opponent RZ TD rate
        'vegas_total',               # Game environment
        'vegas_spread',              # Game script
    ],
    'WR': [
        'trailing_rz_targets',       # KEY: Trailing RZ target opportunities (48.8% TD rate @ >=2)
        'trailing_rz_touches',       # Combined RZ touches
        'trailing_receiving_tds',    # Historical receiving TD rate
        'rz_td_per_target',          # RZ conversion efficiency
        'gl_target_share',           # Goal-line role
        'snap_share',                # Playing time
        'opp_rz_td_allowed',         # Opponent RZ TD rate
        'vegas_total',               # Game environment
    ],
    'TE': [
        'trailing_rz_targets',       # KEY: Trailing RZ target opportunities
        'trailing_rz_touches',       # Combined RZ touches
        'trailing_receiving_tds',    # Historical receiving TD rate
        'rz_td_per_target',          # RZ conversion efficiency
        'gl_target_share',           # Goal-line role
        'snap_share',                # Playing time
        'opp_rz_td_allowed',         # Opponent RZ TD rate
        'vegas_total',               # Game environment
    ],
}

# TD types per position for Poisson component (DraftKings rules)
TD_TYPES_BY_POSITION = {
    'QB': ['rush', 'rec'],     # QBs can score ATTD via rush or rec, NOT pass
    'RB': ['rush', 'rec'],     # RBs can score ATTD via rush or rec
    'WR': ['rec'],             # WRs score ATTD via rec only
    'TE': ['rec'],             # TEs score ATTD via rec only
}


class ATTDEnsemble:
    """
    Ensemble model for Anytime TD prediction.

    Combines:
    - Poisson component: Predicts lambda per TD type, converts to P(any)
    - Logistic component: Binary classifier on scored_td (0/1)

    Ensemble weights are learned per position through walk-forward validation.
    """

    def __init__(self):
        """Initialize ATTD ensemble."""
        # Poisson models: position -> {td_type: model}
        self.poisson_models: Dict[str, Dict[str, PoissonRegressor]] = {}

        # Logistic models: position -> model
        self.logistic_models: Dict[str, LogisticRegression] = {}

        # Ensemble weights: position -> (w_poisson, w_logistic)
        self.ensemble_weights: Dict[str, Tuple[float, float]] = {
            'QB': (0.5, 0.5),
            'RB': (0.5, 0.5),
            'WR': (0.5, 0.5),
            'TE': (0.5, 0.5),
        }

        # Preprocessing: position -> {scaler, imputer}
        self.scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}

        # Metrics
        self.metrics: Dict[str, Dict] = {}
        self.version: Optional[str] = None
        self.trained_date: Optional[datetime] = None

    def get_features(self, position: str) -> List[str]:
        """Get feature list for a position."""
        return ATTD_FEATURES.get(position, ATTD_FEATURES['RB'])

    def train_position(
        self,
        train_data: pd.DataFrame,
        position: str,
    ) -> Dict[str, Any]:
        """
        Train both Poisson and Logistic components for a position.

        Args:
            train_data: Training data with TD outcomes and features
            position: Player position (QB, RB, WR, TE)

        Returns:
            Training metrics
        """
        logger.info(f"Training ATTD ensemble for {position}...")

        feature_cols = self.get_features(position)
        available = [f for f in feature_cols if f in train_data.columns]

        if len(available) < 2:
            raise ValueError(f"Insufficient features for {position}: {available}")

        # Create binary target: scored_td = 1 if any TD scored
        train_data = train_data.copy()
        train_data['scored_td'] = self._create_scored_td_target(train_data, position)

        # Filter to valid rows
        train_data = train_data.dropna(subset=['scored_td'])
        train_data = train_data[train_data['scored_td'].isin([0, 1])]

        if len(train_data) < 50:
            raise ValueError(f"Insufficient samples for {position}: {len(train_data)}")

        X = train_data[available].copy()
        y = train_data['scored_td'].astype(int)

        # Preprocessing
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        self.imputers[position] = imputer
        self.scalers[position] = scaler

        # Train Logistic component
        logistic_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',  # Handle imbalanced TD outcomes
        )
        logistic_model.fit(X_scaled, y)
        self.logistic_models[position] = logistic_model

        # Train Poisson component per TD type
        self.poisson_models[position] = {}
        td_types = TD_TYPES_BY_POSITION.get(position, ['rush', 'rec'])

        for td_type in td_types:
            target_col = f'{td_type}ing_tds' if td_type == 'rush' else f'{td_type}eiving_tds'
            if target_col not in train_data.columns:
                # Try alternate column names
                alt_col = f'trailing_{target_col}'
                if alt_col in train_data.columns:
                    # Use trailing as proxy for actual (will be less accurate)
                    y_poisson = train_data[alt_col].fillna(0).clip(lower=0).astype(int)
                else:
                    continue
            else:
                y_poisson = train_data[target_col].fillna(0).clip(lower=0).astype(int)

            poisson_model = PoissonRegressor(alpha=0.1, max_iter=1000)
            poisson_model.fit(X_scaled, y_poisson)
            self.poisson_models[position][td_type] = poisson_model

        # Calculate metrics
        y_pred_logistic = logistic_model.predict_proba(X_scaled)[:, 1]
        y_pred_ensemble = self.predict_attd_probability_internal(X_scaled, position)

        accuracy = ((y_pred_ensemble >= 0.5) == y).mean()
        brier = np.mean((y_pred_ensemble - y) ** 2)

        metrics = {
            'position': position,
            'n_samples': len(X),
            'n_features': len(available),
            'features_used': available,
            'td_rate': float(y.mean()),
            'accuracy': float(accuracy),
            'brier_score': float(brier),
        }

        self.metrics[position] = metrics
        self.trained_date = datetime.now()
        self.version = "attd_ensemble_v1"

        logger.info(f"  {position}: samples={len(X)}, TD_rate={y.mean():.1%}, acc={accuracy:.1%}")

        return metrics

    def _create_scored_td_target(self, df: pd.DataFrame, position: str) -> pd.Series:
        """Create binary scored_td target based on position's TD types."""
        td_types = TD_TYPES_BY_POSITION.get(position, ['rush', 'rec'])
        scored = pd.Series(0, index=df.index)

        for td_type in td_types:
            col = f'{td_type}ing_tds' if td_type == 'rush' else f'{td_type}eiving_tds'
            if col in df.columns:
                scored = scored | (df[col].fillna(0) >= 1)

        return scored.astype(int)

    def predict_attd_probability(
        self,
        features: pd.DataFrame,
        position: str,
    ) -> np.ndarray:
        """
        Predict P(any TD) using ensemble.

        Args:
            features: DataFrame with ATTD features
            position: Player position

        Returns:
            Array of P(any TD) probabilities
        """
        if position not in self.logistic_models:
            raise ValueError(f"No trained model for position: {position}")

        feature_cols = self.get_features(position)
        available = [f for f in feature_cols if f in features.columns]

        if len(available) == 0:
            raise ValueError(f"No matching features for {position}")

        X = features[available].copy()
        X = X.fillna(0)

        # Apply preprocessing
        if position in self.imputers:
            X = self.imputers[position].transform(X)
        if position in self.scalers:
            X = self.scalers[position].transform(X)

        return self.predict_attd_probability_internal(X, position)

    def predict_attd_probability_internal(
        self,
        X_scaled: np.ndarray,
        position: str,
    ) -> np.ndarray:
        """
        Internal prediction using preprocessed features.

        Combines Poisson and Logistic components with ensemble weights.
        """
        # Logistic component
        logistic_model = self.logistic_models.get(position)
        if logistic_model is not None:
            p_logistic = logistic_model.predict_proba(X_scaled)[:, 1]
        else:
            p_logistic = np.zeros(len(X_scaled))

        # Poisson component: P(any TD) = 1 - P(0 TDs for all types)
        p_no_td = np.ones(len(X_scaled))
        poisson_models = self.poisson_models.get(position, {})

        for td_type, model in poisson_models.items():
            lambda_vals = model.predict(X_scaled)
            # P(X = 0) = exp(-lambda)
            p_no_td *= np.exp(-lambda_vals)

        p_poisson = 1 - p_no_td

        # Ensemble
        w_poisson, w_logistic = self.ensemble_weights.get(position, (0.5, 0.5))
        p_ensemble = w_poisson * p_poisson + w_logistic * p_logistic

        # Clip to valid probability range
        return np.clip(p_ensemble, 0, 1)

    def tune_ensemble_weights(
        self,
        val_data: pd.DataFrame,
        position: str,
    ) -> Tuple[float, float]:
        """
        Tune ensemble weights using validation data.

        Grid search over weight combinations to minimize Brier score.

        Args:
            val_data: Validation data with TD outcomes
            position: Player position

        Returns:
            Optimal (w_poisson, w_logistic) weights
        """
        feature_cols = self.get_features(position)
        available = [f for f in feature_cols if f in val_data.columns]

        val_data = val_data.copy()
        val_data['scored_td'] = self._create_scored_td_target(val_data, position)
        val_data = val_data.dropna(subset=['scored_td'])

        if len(val_data) < 20:
            logger.warning(f"Insufficient validation data for {position}")
            return (0.5, 0.5)

        X = val_data[available].copy().fillna(0)
        y = val_data['scored_td'].astype(int)

        # Apply preprocessing
        if position in self.imputers:
            X = self.imputers[position].transform(X)
        if position in self.scalers:
            X = self.scalers[position].transform(X)

        # Get component predictions
        logistic_model = self.logistic_models.get(position)
        p_logistic = logistic_model.predict_proba(X)[:, 1] if logistic_model else np.zeros(len(X))

        p_no_td = np.ones(len(X))
        for td_type, model in self.poisson_models.get(position, {}).items():
            lambda_vals = model.predict(X)
            p_no_td *= np.exp(-lambda_vals)
        p_poisson = 1 - p_no_td

        # Grid search
        best_brier = float('inf')
        best_weights = (0.5, 0.5)

        for w_poisson in np.arange(0.0, 1.05, 0.1):
            w_logistic = 1.0 - w_poisson
            p_ensemble = w_poisson * p_poisson + w_logistic * p_logistic
            brier = np.mean((p_ensemble - y) ** 2)

            if brier < best_brier:
                best_brier = brier
                best_weights = (w_poisson, w_logistic)

        self.ensemble_weights[position] = best_weights
        logger.info(f"  {position} optimal weights: Poisson={best_weights[0]:.1f}, Logistic={best_weights[1]:.1f}")

        return best_weights

    def evaluate_bet(
        self,
        row: pd.Series,
        position: str,
        min_confidence: float = 0.55,
        implied_odds: float = 0.5,
        min_edge: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Evaluate a single ATTD bet opportunity.

        IMPORTANT: Only recommends YES bets (player WILL score).
        Never recommends NO bets on ATTD - the edge is in finding
        high-probability TD scorers, not betting against players.

        Args:
            row: Series with features for a single bet
            position: Player position
            min_confidence: Minimum P(TD) to recommend bet (default 0.55)
            implied_odds: Implied probability from sportsbook odds
            min_edge: Minimum edge vs implied odds (default 0.05 = 5%)

        Returns:
            Dict with bet recommendation (YES only)
        """
        features = pd.DataFrame([row])

        try:
            p_attd = self.predict_attd_probability(features, position)[0]
        except Exception as e:
            return {
                'should_bet': False,
                'bet_direction': None,
                'p_attd': None,
                'edge': None,
                'error': str(e),
            }

        edge = p_attd - implied_odds

        # ONLY bet YES if:
        # 1. P(TD) >= min_confidence threshold
        # 2. Edge vs implied odds >= min_edge
        should_bet_yes = (p_attd >= min_confidence) and (edge >= min_edge)

        return {
            'should_bet': should_bet_yes,
            'bet_direction': 'YES' if should_bet_yes else None,
            'p_attd': float(p_attd),
            'edge': float(edge),
            'position': position,
            'implied_odds': implied_odds,
            'confidence_met': p_attd >= min_confidence,
            'edge_met': edge >= min_edge,
        }

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        position: str,
        n_val_weeks: int = 10,
    ) -> pd.DataFrame:
        """
        Walk-forward validation for ATTD ensemble.

        Args:
            df: Full dataset with global_week column
            position: Player position
            n_val_weeks: Number of weeks for validation

        Returns:
            DataFrame with validation predictions and actuals
        """
        logger.info(f"Walk-forward validation for {position}...")

        weeks = sorted(df['global_week'].unique())
        if len(weeks) < 5:
            logger.warning("Not enough weeks for validation")
            return pd.DataFrame()

        val_weeks = weeks[-n_val_weeks:]
        all_preds = []

        for test_week in val_weeks:
            train_data = df[df['global_week'] < test_week - 1].copy()
            test_data = df[df['global_week'] == test_week].copy()

            if len(train_data) < 50 or len(test_data) == 0:
                continue

            try:
                self.train_position(train_data, position)
            except Exception as e:
                logger.warning(f"Week {test_week} training error: {e}")
                continue

            for idx, row in test_data.iterrows():
                try:
                    features = pd.DataFrame([row])
                    p_attd = self.predict_attd_probability(features, position)[0]
                    scored = self._create_scored_td_target(pd.DataFrame([row]), position).iloc[0]

                    all_preds.append({
                        'global_week': test_week,
                        'player': row.get('player', ''),
                        'position': position,
                        'p_attd': p_attd,
                        'scored_td': scored,
                    })
                except Exception:
                    continue

        if not all_preds:
            return pd.DataFrame()

        preds_df = pd.DataFrame(all_preds)

        # Print validation metrics
        logger.info(f"\n  Validation Results ({position}):")
        logger.info(f"    Total predictions: {len(preds_df)}")

        for thresh in [0.50, 0.55, 0.60, 0.65]:
            high_conf = preds_df[preds_df['p_attd'] >= thresh]
            if len(high_conf) >= 5:
                hits = high_conf['scored_td'].sum()
                total = len(high_conf)
                hit_rate = hits / total
                # Assume -110 odds (0.909 payout)
                roi = (hit_rate * 0.909) - (1 - hit_rate)
                logger.info(f"    @ {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

        return preds_df

    def save(self, path: Path = None) -> None:
        """Save ATTD ensemble to disk."""
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'attd_ensemble.joblib'

        bundle = {
            'poisson_models': self.poisson_models,
            'logistic_models': self.logistic_models,
            'ensemble_weights': self.ensemble_weights,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'metrics': self.metrics,
            'version': self.version,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
        }
        joblib.dump(bundle, path)
        logger.info(f"ATTD ensemble saved to: {path}")

    @classmethod
    def load(cls, path: Path = None) -> 'ATTDEnsemble':
        """Load ATTD ensemble from disk."""
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'attd_ensemble.joblib'

        bundle = joblib.load(path)

        ensemble = cls()
        ensemble.poisson_models = bundle.get('poisson_models', {})
        ensemble.logistic_models = bundle.get('logistic_models', {})
        ensemble.ensemble_weights = bundle.get('ensemble_weights', {})
        ensemble.scalers = bundle.get('scalers', {})
        ensemble.imputers = bundle.get('imputers', {})
        ensemble.metrics = bundle.get('metrics', {})
        ensemble.version = bundle.get('version')
        ensemble.trained_date = (
            datetime.fromisoformat(bundle['trained_date'])
            if bundle.get('trained_date')
            else None
        )

        return ensemble

    def __repr__(self) -> str:
        positions = list(self.logistic_models.keys())
        return f"ATTDEnsemble(version='{self.version}', positions={positions})"
