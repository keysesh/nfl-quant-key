"""
Base Edge Class

Abstract base class for edge detection strategies.
All edge implementations (LVT, Player Bias) inherit from this.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


class BaseEdge(ABC):
    """
    Abstract base class for edge detection strategies.

    Each edge represents an independent betting signal with:
    - Specific features it uses
    - Confidence thresholds per market
    - Training/inference logic
    """

    def __init__(
        self,
        name: str,
        features: List[str],
        thresholds: Dict[str, Any],
    ):
        """
        Initialize edge.

        Args:
            name: Edge identifier (e.g., "lvt", "player_bias")
            features: List of feature column names this edge uses
            thresholds: Dict mapping market -> threshold config
        """
        self.name = name
        self.features = features
        self.thresholds = thresholds
        self.models: Dict[str, Any] = {}  # market -> trained model
        self.metrics: Dict[str, Dict] = {}  # market -> training metrics
        self.version: Optional[str] = None
        self.trained_date: Optional[datetime] = None

    @abstractmethod
    def extract_features(
        self,
        df: pd.DataFrame,
        market: str,
        historical_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract edge-specific features from data.

        Args:
            df: DataFrame with raw data (odds, stats)
            market: Market being processed (e.g., "player_receptions")
            historical_data: Optional historical data for player stats

        Returns:
            DataFrame with edge-specific features
        """
        pass

    @abstractmethod
    def should_trigger(
        self,
        row: pd.Series,
        market: str,
    ) -> bool:
        """
        Check if this edge triggers for a specific bet.

        Args:
            row: Series with features for a single bet
            market: Market being evaluated

        Returns:
            True if edge conditions are met
        """
        pass

    @abstractmethod
    def get_confidence(
        self,
        row: pd.Series,
        market: str,
    ) -> float:
        """
        Get model confidence (P(UNDER)) for this bet.

        Args:
            row: Series with features for a single bet
            market: Market being evaluated

        Returns:
            Probability between 0 and 1
        """
        pass

    def get_direction(
        self,
        row: pd.Series,
        market: str,
    ) -> str:
        """
        Get recommended direction based on confidence.

        Args:
            row: Series with features for a single bet
            market: Market being evaluated

        Returns:
            "UNDER" if P(UNDER) > 0.5, else "OVER"
        """
        confidence = self.get_confidence(row, market)
        return "UNDER" if confidence > 0.5 else "OVER"

    def train(
        self,
        train_data: pd.DataFrame,
        market: str,
        validation_weeks: int = 10,
    ) -> Dict[str, Any]:
        """
        Train the edge model for a specific market.

        Default implementation uses walk-forward validation.
        Override in subclasses for edge-specific training logic.

        Args:
            train_data: Historical odds/actuals data
            market: Market to train for
            validation_weeks: Number of weeks for validation

        Returns:
            Dict with training metrics
        """
        raise NotImplementedError("Subclasses must implement train()")

    def predict(
        self,
        df: pd.DataFrame,
        market: str,
    ) -> pd.Series:
        """
        Generate predictions for multiple bets.

        Args:
            df: DataFrame with features
            market: Market being predicted

        Returns:
            Series of P(UNDER) probabilities
        """
        if market not in self.models:
            raise ValueError(f"No trained model for market: {market}")

        model = self.models[market]
        features_df = df[self.features].copy()

        # Handle missing values (XGBoost handles NaN natively)
        probs = model.predict_proba(features_df)

        # Return P(UNDER) - assumes class 1 is UNDER
        return pd.Series(probs[:, 1], index=df.index)

    def evaluate_bet(
        self,
        row: pd.Series,
        market: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single bet opportunity.

        Args:
            row: Series with features for a single bet
            market: Market being evaluated

        Returns:
            Dict with trigger status, confidence, direction, etc.
        """
        triggers = self.should_trigger(row, market)
        confidence = self.get_confidence(row, market) if triggers else 0.0
        direction = self.get_direction(row, market) if triggers else None

        return {
            'edge': self.name,
            'triggers': triggers,
            'confidence': confidence,
            'direction': direction,
            'market': market,
        }

    def save(self, path: Path) -> None:
        """
        Save edge models and metadata to disk.

        Args:
            path: Path to save file (.joblib)
        """
        bundle = {
            'name': self.name,
            'features': self.features,
            'thresholds': self.thresholds,
            'models': self.models,
            'metrics': self.metrics,
            'version': self.version,
            'trained_date': self.trained_date.isoformat() if self.trained_date else None,
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: Path) -> 'BaseEdge':
        """
        Load edge from disk.

        Args:
            path: Path to saved file (.joblib)

        Returns:
            Loaded edge instance
        """
        bundle = joblib.load(path)

        # Create instance (subclass should override)
        edge = cls.__new__(cls)
        edge.name = bundle['name']
        edge.features = bundle['features']
        edge.thresholds = bundle['thresholds']
        edge.models = bundle['models']
        edge.metrics = bundle.get('metrics', {})
        edge.version = bundle.get('version')
        edge.trained_date = (
            datetime.fromisoformat(bundle['trained_date'])
            if bundle.get('trained_date')
            else None
        )

        return edge

    def get_feature_importance(self, market: str) -> Dict[str, float]:
        """
        Get feature importance for a market's model.

        Args:
            market: Market to get importance for

        Returns:
            Dict mapping feature name -> importance
        """
        if market not in self.models:
            return {}

        model = self.models[market]
        if hasattr(model, 'feature_importances_'):
            return dict(zip(self.features, model.feature_importances_))

        return {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"features={len(self.features)}, "
            f"markets={list(self.models.keys())}"
            f")"
        )
