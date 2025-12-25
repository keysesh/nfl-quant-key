"""
Edge Prediction Logger

Logs edge predictions for post-game analysis and ROI tracking.
Each prediction is logged with timestamp, market, player, line,
direction, confidence, and source (which edge triggered).

Usage:
    logger = EdgePredictionLogger()
    logger.log_prediction({
        'player': 'Patrick Mahomes',
        'market': 'player_pass_yds',
        'line': 285.5,
        'direction': 'UNDER',
        'confidence': 0.72,
        'source': 'BOTH',
        'units': 1.0,
    })
    logger.save()  # Flush to disk
"""
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import csv


class EdgePredictionLogger:
    """
    Logs edge predictions for tracking and analysis.

    Maintains a daily log file with all predictions.
    Supports matching predictions to actuals post-game.

    Attributes:
        log_dir: Directory for log files
        predictions: In-memory buffer of predictions
        current_date: Date of current session
    """

    def __init__(self, log_dir: Path = None):
        """
        Initialize prediction logger.

        Args:
            log_dir: Directory for log files. Defaults to data/logs/edge_predictions/
        """
        if log_dir is None:
            from nfl_quant.config_paths import DATA_DIR
            log_dir = DATA_DIR / 'logs' / 'edge_predictions'

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.predictions: List[Dict] = []
        self.current_date = date.today()
        self.session_start = datetime.now()

    def log_prediction(self, prediction: Dict[str, Any]) -> None:
        """
        Log a single prediction.

        Args:
            prediction: Dict with prediction details. Required keys:
                - player: Player name
                - market: Market type (e.g., 'player_receptions')
                - line: Betting line
                - direction: 'UNDER' or 'OVER'
                - confidence: P(direction) from model
                - source: Edge source ('LVT_ONLY', 'PLAYER_BIAS_ONLY', 'BOTH')

            Optional keys:
                - units: Bet size in units (default 1.0)
                - odds: American odds (default -110)
                - week: NFL week
                - season: NFL season
                - opponent: Opposing team
                - game_id: NFLverse game ID
        """
        # Add timestamp
        prediction['timestamp'] = datetime.now().isoformat()
        prediction['prediction_date'] = self.current_date.isoformat()

        # Add defaults
        prediction.setdefault('units', 1.0)
        prediction.setdefault('odds', -110)
        prediction.setdefault('actual', None)
        prediction.setdefault('hit', None)

        self.predictions.append(prediction)

    def log_batch(self, predictions: pd.DataFrame) -> None:
        """
        Log a batch of predictions from a DataFrame.

        Args:
            predictions: DataFrame with prediction columns
        """
        for _, row in predictions.iterrows():
            self.log_prediction(row.to_dict())

    def save(self, flush: bool = True) -> Path:
        """
        Save predictions to disk.

        Args:
            flush: Clear in-memory buffer after saving

        Returns:
            Path to saved file
        """
        if not self.predictions:
            return None

        # Create filename with date
        filename = f'predictions_{self.current_date.isoformat()}.csv'
        filepath = self.log_dir / filename

        # Load existing if present
        existing_df = None
        if filepath.exists():
            existing_df = pd.read_csv(filepath)

        # Convert new predictions to DataFrame
        new_df = pd.DataFrame(self.predictions)

        # Combine with existing
        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Save
        combined_df.to_csv(filepath, index=False)

        if flush:
            self.predictions = []

        return filepath

    def load_predictions(
        self,
        prediction_date: date = None,
        start_date: date = None,
        end_date: date = None,
    ) -> pd.DataFrame:
        """
        Load predictions from disk.

        Args:
            prediction_date: Load specific date
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame of predictions
        """
        if prediction_date:
            filename = f'predictions_{prediction_date.isoformat()}.csv'
            filepath = self.log_dir / filename
            if filepath.exists():
                return pd.read_csv(filepath)
            return pd.DataFrame()

        # Load all files in date range
        all_files = sorted(self.log_dir.glob('predictions_*.csv'))

        if not all_files:
            return pd.DataFrame()

        dfs = []
        for f in all_files:
            # Extract date from filename
            file_date_str = f.stem.replace('predictions_', '')
            try:
                file_date = date.fromisoformat(file_date_str)
            except ValueError:
                continue

            # Apply date filters
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            dfs.append(pd.read_csv(f))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def match_actuals(
        self,
        actuals: pd.DataFrame,
        prediction_date: date = None,
    ) -> pd.DataFrame:
        """
        Match predictions to actual outcomes.

        Args:
            actuals: DataFrame with 'player', 'market', 'line', 'actual' columns
            prediction_date: Date to match (defaults to today)

        Returns:
            DataFrame with matched predictions and outcomes
        """
        if prediction_date is None:
            prediction_date = self.current_date

        predictions = self.load_predictions(prediction_date=prediction_date)

        if predictions.empty or actuals.empty:
            return predictions

        # Normalize player names for matching
        predictions['player_lower'] = predictions['player'].str.lower().str.strip()
        actuals['player_lower'] = actuals['player'].str.lower().str.strip()

        # Merge on player + market + line
        matched = predictions.merge(
            actuals[['player_lower', 'market', 'line', 'actual']],
            on=['player_lower', 'market', 'line'],
            how='left',
            suffixes=('', '_actual'),
        )

        # Calculate hit
        if 'actual_actual' in matched.columns:
            matched['actual'] = matched['actual_actual']
            matched.drop(columns=['actual_actual'], inplace=True)

        matched['hit'] = matched.apply(
            lambda row: self._calculate_hit(row),
            axis=1,
        )

        # Clean up
        matched.drop(columns=['player_lower'], inplace=True, errors='ignore')

        return matched

    def _calculate_hit(self, row: pd.Series) -> Optional[bool]:
        """Calculate if prediction hit."""
        if pd.isna(row.get('actual')):
            return None

        actual = row['actual']
        line = row['line']
        direction = row['direction']

        if direction == 'UNDER':
            return actual < line
        else:
            return actual > line

    def calculate_roi(
        self,
        matched: pd.DataFrame,
        odds: int = -110,
    ) -> Dict[str, float]:
        """
        Calculate ROI from matched predictions.

        Args:
            matched: DataFrame with 'hit', 'units' columns
            odds: Default odds (American format)

        Returns:
            Dict with ROI metrics
        """
        valid = matched[matched['hit'].notna()].copy()

        if len(valid) == 0:
            return {'n_bets': 0, 'roi': 0.0, 'hit_rate': 0.0}

        valid['units'] = valid['units'].fillna(1.0)
        valid['odds'] = valid['odds'].fillna(odds)

        # Calculate profit per bet
        valid['profit'] = valid.apply(
            lambda row: self._calculate_profit(row['hit'], row['units'], row['odds']),
            axis=1,
        )

        total_wagered = valid['units'].sum()
        total_profit = valid['profit'].sum()

        return {
            'n_bets': len(valid),
            'total_wagered': float(total_wagered),
            'total_profit': float(total_profit),
            'roi': float(total_profit / total_wagered * 100) if total_wagered > 0 else 0.0,
            'hit_rate': float(valid['hit'].mean()),
            'hits': int(valid['hit'].sum()),
        }

    def _calculate_profit(self, hit: bool, units: float, odds: int) -> float:
        """Calculate profit for a single bet."""
        if hit:
            # Convert American odds to decimal payout
            if odds < 0:
                return units * (100 / abs(odds))
            else:
                return units * (odds / 100)
        else:
            return -units

    def get_daily_summary(
        self,
        prediction_date: date = None,
    ) -> Dict[str, Any]:
        """
        Get summary for a specific date.

        Args:
            prediction_date: Date to summarize (defaults to today)

        Returns:
            Dict with daily summary
        """
        if prediction_date is None:
            prediction_date = self.current_date

        predictions = self.load_predictions(prediction_date=prediction_date)

        if predictions.empty:
            return {'date': prediction_date.isoformat(), 'n_predictions': 0}

        # Group by source
        by_source = predictions.groupby('source').size().to_dict()

        # Group by market
        by_market = predictions.groupby('market').size().to_dict()

        return {
            'date': prediction_date.isoformat(),
            'n_predictions': len(predictions),
            'by_source': by_source,
            'by_market': by_market,
            'avg_confidence': float(predictions['confidence'].mean()),
            'total_units': float(predictions['units'].sum()),
        }

    def save_matched_results(
        self,
        matched: pd.DataFrame,
        prediction_date: date = None,
    ) -> Path:
        """
        Save matched results to disk.

        Args:
            matched: DataFrame with matched predictions
            prediction_date: Date of predictions

        Returns:
            Path to saved file
        """
        if prediction_date is None:
            prediction_date = self.current_date

        results_dir = self.log_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        filename = f'results_{prediction_date.isoformat()}.csv'
        filepath = results_dir / filename

        matched.to_csv(filepath, index=False)
        return filepath

    def __repr__(self) -> str:
        return (
            f"EdgePredictionLogger("
            f"log_dir='{self.log_dir}', "
            f"pending={len(self.predictions)}, "
            f"date='{self.current_date}'"
            f")"
        )
