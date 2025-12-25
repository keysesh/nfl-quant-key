"""
LVT Edge Implementation

The LVT (Line vs Trailing) edge captures statistical reversion.
When Vegas lines diverge significantly from trailing performance,
regression to mean provides a predictable edge.

Key characteristics:
- Uses minimal features (7) to prevent overfitting
- High confidence threshold (~65-70%)
- Low volume, high conviction bets
- Target: 65-70% hit rate
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base_edge import BaseEdge
from .edge_calibrator import EdgeCalibrator
from configs.edge_config import (
    LVT_FEATURES,
    LVT_THRESHOLDS,
    LVT_MODEL_PARAMS,
    get_lvt_threshold,
)
from configs.model_config import (
    SWEET_SPOT_PARAMS,
    smooth_sweet_spot,
    TRAILING_DEFLATION_FACTORS,
)


class LVTEdge(BaseEdge):
    """
    LVT (Line vs Trailing) Edge Implementation.

    This edge triggers when:
    1. |line_vs_trailing| > min_lvt threshold
    2. Model P(UNDER) > confidence threshold

    Uses 12 features: 7 core (statistical reversion) + 5 V23/V28 (context).
    """

    def __init__(self):
        """Initialize LVT edge with configured features and thresholds."""
        super().__init__(
            name="lvt",
            features=LVT_FEATURES,
            thresholds=LVT_THRESHOLDS,
        )
        self.scalers: Dict[str, StandardScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.calibrator: EdgeCalibrator = EdgeCalibrator()

    def extract_features(
        self,
        df: pd.DataFrame,
        market: str,
        historical_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract LVT-specific features.

        Extracts 12 features for LVT edge:
        Core (7):
        - line_vs_trailing (core signal)
        - line_level
        - line_in_sweet_spot
        - LVT_in_sweet_spot
        - market_under_rate
        - vegas_spread
        - implied_team_total

        V23/V28 (5 - Dec 2025):
        - opp_def_epa
        - has_opponent_context
        - rest_days
        - elo_diff
        - lvt_x_defense

        Args:
            df: DataFrame with 'line', 'trailing_*', 'vegas_*' columns
            market: Market being processed
            historical_data: Not used for LVT (no player history needed)

        Returns:
            DataFrame with 12 LVT features
        """
        result = pd.DataFrame(index=df.index)

        # Get trailing column name for this market
        trailing_col = self._get_trailing_col(market)
        stat_col = trailing_col.replace('trailing_', '')

        # 1. line_vs_trailing - Core signal
        if trailing_col in df.columns and 'line' in df.columns:
            trailing = df[trailing_col].fillna(df['line'])

            # Apply deflation factor
            deflation = TRAILING_DEFLATION_FACTORS.get(market, 0.90)
            deflated_trailing = trailing * deflation

            # Calculate LVT as percentage difference
            result['line_vs_trailing'] = np.where(
                deflated_trailing > 0,
                (df['line'] - deflated_trailing) / deflated_trailing * 100,
                0
            )
        else:
            result['line_vs_trailing'] = 0

        # 2. line_level - Raw line value
        result['line_level'] = df.get('line', 0)

        # 3. line_in_sweet_spot - Gaussian decay
        if 'line' in df.columns:
            result['line_in_sweet_spot'] = df['line'].apply(
                lambda x: smooth_sweet_spot(x, market)
            )
        else:
            result['line_in_sweet_spot'] = 0.5

        # 4. LVT_in_sweet_spot - Interaction
        result['LVT_in_sweet_spot'] = (
            result['line_vs_trailing'] * result['line_in_sweet_spot']
        )

        # 5. market_under_rate - Market regime
        if 'market_under_rate' in df.columns:
            result['market_under_rate'] = df['market_under_rate']
        else:
            result['market_under_rate'] = 0.50  # Neutral default

        # 6. vegas_spread - Game context
        result['vegas_spread'] = df.get('vegas_spread', 0)

        # 7. implied_team_total - Scoring environment
        if 'implied_team_total' in df.columns:
            result['implied_team_total'] = df['implied_team_total']
        elif 'vegas_total' in df.columns and 'vegas_spread' in df.columns:
            # Calculate from vegas_total and spread
            result['implied_team_total'] = (
                df['vegas_total'] / 2 - df['vegas_spread'] / 2
            )
        else:
            result['implied_team_total'] = 24.0  # League average

        # =====================================================================
        # V23/V28 Features (Dec 2025 Enhancement)
        # =====================================================================

        # 8. opp_def_epa - Opponent defense EPA
        result['opp_def_epa'] = df.get('opp_def_epa', 0.0)
        if isinstance(result['opp_def_epa'], pd.Series):
            result['opp_def_epa'] = result['opp_def_epa'].fillna(0.0)

        # 9. has_opponent_context - Binary flag for data availability
        if 'has_opponent_context' in df.columns:
            result['has_opponent_context'] = df['has_opponent_context']
        else:
            # Infer from opp_def_epa: if non-zero, we have context
            result['has_opponent_context'] = (result['opp_def_epa'] != 0).astype(int)

        # 10. rest_days - Days since last game
        result['rest_days'] = df.get('rest_days', 7.0)
        if isinstance(result['rest_days'], pd.Series):
            result['rest_days'] = result['rest_days'].fillna(7.0)

        # 11. elo_diff - Home Elo minus Away Elo
        result['elo_diff'] = df.get('elo_diff', 0.0)
        if isinstance(result['elo_diff'], pd.Series):
            result['elo_diff'] = result['elo_diff'].fillna(0.0)

        # 12. lvt_x_defense - Interaction: LVT signal * defense quality
        result['lvt_x_defense'] = result['line_vs_trailing'] * result['opp_def_epa']

        return result[LVT_FEATURES]

    def should_trigger(
        self,
        row: pd.Series,
        market: str,
    ) -> bool:
        """
        Check if LVT edge triggers.

        Triggers when:
        1. |line_vs_trailing| > min_lvt for market
        2. Direction aligns (positive LVT suggests UNDER)

        Args:
            row: Series with LVT features
            market: Market being evaluated

        Returns:
            True if LVT edge conditions are met
        """
        threshold = get_lvt_threshold(market)

        # Check if LVT is strong enough
        lvt = row.get('line_vs_trailing', 0)
        if abs(lvt) < threshold.min_lvt:
            return False

        # LVT edge triggers
        return True

    def get_confidence(
        self,
        row: pd.Series,
        market: str,
    ) -> float:
        """
        Get calibrated P(UNDER) confidence.

        Args:
            row: Series with LVT features
            market: Market being evaluated

        Returns:
            Calibrated P(UNDER) from trained model, or heuristic if no model
        """
        if market not in self.models:
            # Heuristic fallback based on LVT magnitude
            lvt = row.get('line_vs_trailing', 0)
            # Positive LVT → UNDER is more likely
            # Sigmoid-like transformation
            return 1 / (1 + np.exp(-lvt / 5))

        # Use trained model
        model = self.models[market]
        features = row[LVT_FEATURES].values.reshape(1, -1)

        # Apply preprocessing
        if market in self.imputers:
            features = self.imputers[market].transform(features)
        if market in self.scalers:
            features = self.scalers[market].transform(features)

        probs = model.predict_proba(features)
        raw_prob = float(probs[0, 1])  # P(UNDER)

        # NOTE: Calibration disabled - IsotonicRegression was squashing all outputs
        # to the same value (~61.8%) regardless of input features.
        # Raw probabilities provide better differentiation between players.
        # TODO: Consider Platt scaling or re-training calibrator with more variance.
        return raw_prob

    def train(
        self,
        train_data: pd.DataFrame,
        market: str,
        validation_weeks: int = 10,
    ) -> Dict[str, Any]:
        """
        Train LVT edge model for a market.

        Key differences from unified model:
        1. Filter to high-LVT samples only
        2. Use only 7 features
        3. Conservative hyperparameters

        Args:
            train_data: Historical odds/actuals with trailing stats
            market: Market to train for
            validation_weeks: Weeks for walk-forward validation

        Returns:
            Training metrics
        """
        threshold = get_lvt_threshold(market)

        # Extract features
        features_df = self.extract_features(train_data, market)
        train_data = train_data.copy()
        for col in features_df.columns:
            train_data[col] = features_df[col].values

        # Filter to high-LVT samples (where this edge has signal)
        high_lvt_mask = abs(train_data['line_vs_trailing']) >= threshold.min_lvt
        filtered_data = train_data[high_lvt_mask].copy()

        if len(filtered_data) < 50:
            raise ValueError(
                f"Insufficient high-LVT samples for {market}: "
                f"{len(filtered_data)} (need 50+)"
            )

        # Prepare features and target
        X = filtered_data[LVT_FEATURES]
        y = filtered_data['under_hit']

        # Create preprocessing pipeline
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        self.imputers[market] = imputer
        self.scalers[market] = scaler

        # Split data: 80% train, 20% calibration using TEMPORAL split (no leakage)
        # FIXED: Use time-based split instead of random split to prevent future data leak
        if len(X_scaled) >= 100:
            # Ensure we have global_week for temporal ordering
            if 'global_week' not in filtered_data.columns:
                filtered_data['global_week'] = (
                    (filtered_data['season'] - 2023) * 18 + filtered_data['week']
                )

            # Sort by time and split temporally (first 80% train, last 20% calibration)
            sorted_indices = filtered_data.sort_values('global_week').index
            n_train = int(len(sorted_indices) * 0.8)

            train_indices = sorted_indices[:n_train]
            calib_indices = sorted_indices[n_train:]

            # Get position indices for X_scaled array
            train_pos = [filtered_data.index.get_loc(idx) for idx in train_indices]
            calib_pos = [filtered_data.index.get_loc(idx) for idx in calib_indices]

            X_train = X_scaled[train_pos]
            X_calib = X_scaled[calib_pos]
            y_train = y.iloc[train_pos].values
            y_calib = y.iloc[calib_pos].values
        else:
            # Small sample: use all for training, skip calibration
            X_train, X_calib, y_train, y_calib = X_scaled, None, y.values, None

        # Train with conservative parameters on TRAIN SPLIT ONLY
        model = xgb.XGBClassifier(**LVT_MODEL_PARAMS)
        model.fit(X_train, y_train)

        self.models[market] = model

        # Calculate metrics on TRAINING SET ONLY (not full data)
        train_preds = model.predict_proba(X_train)[:, 1]
        train_accuracy = ((train_preds > 0.5) == y_train).mean()

        # Calculate TEST accuracy on held-out calibration set (honest estimate)
        test_accuracy = None
        if X_calib is not None and len(X_calib) >= 20:
            test_preds = model.predict_proba(X_calib)[:, 1]
            test_accuracy = ((test_preds > 0.5) == y_calib).mean()
            print(f"  {market}: Train={train_accuracy:.1%} | Test={test_accuracy:.1%} (gap={train_accuracy - test_accuracy:.1%})")
        else:
            print(f"  {market}: Train={train_accuracy:.1%} | Test=N/A (insufficient holdout)")

        # Fit calibrator on HELD-OUT CALIBRATION SET (NOT training data)
        try:
            if X_calib is not None and len(X_calib) >= 50:
                calib_preds = model.predict_proba(X_calib)[:, 1]
                y_calib_arr = y_calib.values if hasattr(y_calib, 'values') else y_calib
                calib_metrics = self.calibrator.fit(market, y_calib_arr, calib_preds)
                print(f"  Calibration: ECE improved by {calib_metrics['ece_improvement']:.1f}% (on held-out set)")
            else:
                print(f"  Calibration: Skipped (insufficient samples for held-out set)")
        except Exception as e:
            print(f"  Warning: Calibration failed: {e}")

        metrics = {
            'market': market,
            'n_samples': len(filtered_data),
            'n_total': len(train_data),
            'filter_rate': len(filtered_data) / len(train_data),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,  # Honest OOS estimate
            'train_test_gap': (train_accuracy - test_accuracy) if test_accuracy else None,
            'feature_importance': dict(zip(LVT_FEATURES, model.feature_importances_)),
        }

        self.metrics[market] = metrics
        self.trained_date = datetime.now()
        self.version = "lvt_edge_v4"  # v4: Fixed temporal validation (no random split leakage)

        return metrics

    def _get_trailing_col(self, market: str) -> str:
        """Get trailing column name for a market."""
        mapping = {
            'player_receptions': 'trailing_receptions',
            'player_rush_yds': 'trailing_rushing_yards',
            'player_reception_yds': 'trailing_receiving_yards',
            'player_rush_attempts': 'trailing_carries',
            'player_pass_yds': 'trailing_passing_yards',
            'player_pass_attempts': 'trailing_attempts',
            'player_pass_completions': 'trailing_completions',
        }
        return mapping.get(market, f'trailing_{market}')

    @classmethod
    def load(cls, path: Path = None) -> 'LVTEdge':
        """
        Load LVT edge from disk.

        Args:
            path: Path to saved model. Defaults to standard location.

        Returns:
            Loaded LVTEdge instance
        """
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'lvt_edge_model.joblib'

        import joblib
        bundle = joblib.load(path)

        edge = cls()
        edge.models = bundle.get('models', {})
        edge.metrics = bundle.get('metrics', {})
        edge.scalers = bundle.get('scalers', {})
        edge.imputers = bundle.get('imputers', {})
        edge.version = bundle.get('version')
        edge.trained_date = (
            datetime.fromisoformat(bundle['trained_date'])
            if bundle.get('trained_date')
            else None
        )

        # Load calibrator if present
        if 'calibrator' in bundle:
            edge.calibrator = EdgeCalibrator()
            edge.calibrator.calibrators = bundle['calibrator'].get('calibrators', {})
            edge.calibrator.calibration_metrics = bundle['calibrator'].get('metrics', {})

        return edge

    def save(self, path: Path = None) -> None:
        """
        Save LVT edge to disk.

        Args:
            path: Path to save file. Defaults to standard location.
        """
        if path is None:
            from nfl_quant.config_paths import MODELS_DIR
            path = MODELS_DIR / 'lvt_edge_model.joblib'

        import joblib
        bundle = {
            'name': self.name,
            'features': self.features,
            'thresholds': {k: v.__dict__ for k, v in self.thresholds.items()},
            'models': self.models,
            'metrics': self.metrics,
            'scalers': self.scalers,
            'imputers': self.imputers,
            'version': self.version,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
            'calibrator': {
                'calibrators': self.calibrator.calibrators,
                'metrics': self.calibrator.calibration_metrics,
            },
        }
        joblib.dump(bundle, path)
        print(f"LVT edge saved to: {path}")
