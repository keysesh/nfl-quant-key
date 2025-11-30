#!/usr/bin/env python3
"""
Simple Prediction Validation for Weeks 5-11
============================================

Validates model predictions against actual outcomes without betting recommendations.
Calculates prediction accuracy metrics (MAE, RMSE, R²) for each stat type.

Usage:
    python scripts/backtest/simple_prediction_validation.py --start-week 5 --end-week 11
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class PredictionValidator:
    """Validates model predictions against actual outcomes."""

    def __init__(self, season: int = 2025):
        self.season = season
        self.results = []

    def load_actual_stats(self, week: int) -> pd.DataFrame:
        """Load actual player stats for a given week."""
        logger.info(f"Loading actual stats for Week {week}...")

        weekly_path = project_root / "data/nflverse/weekly_stats.parquet"
        df = pd.read_parquet(weekly_path)

        # Filter to specific week and season
        week_stats = df[
            (df['season'] == self.season) &
            (df['week'] == week)
        ].copy()

        logger.info(f"  ✓ Loaded {len(week_stats):,} player-weeks")
        return week_stats

    def load_predictions(self, week: int) -> pd.DataFrame:
        """Load model predictions for a given week."""
        pred_path = project_root / f"data/model_predictions_week{week}.csv"

        if not pred_path.exists():
            logger.warning(f"  ⚠️  Predictions not found: {pred_path}")
            return None

        preds = pd.read_csv(pred_path)
        logger.info(f"  ✓ Loaded {len(preds):,} predictions")
        return preds

    def normalize_name(self, name: str) -> str:
        """Normalize player name for matching."""
        if pd.isna(name):
            return ""
        return name.strip().lower().replace("'", "").replace(".", "").replace(" jr", "").replace(" sr", "")

    def validate_week(self, week: int) -> Dict:
        """Validate predictions for a single week."""
        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATING WEEK {week}")
        logger.info(f"{'='*80}\n")

        # Load data
        actual_stats = self.load_actual_stats(week)
        predictions = self.load_predictions(week)

        if predictions is None:
            logger.error(f"Cannot validate Week {week}: No predictions available")
            return None

        # Normalize player names for matching
        actual_stats['player_normalized'] = actual_stats['player_name'].apply(self.normalize_name)
        predictions['player_normalized'] = predictions['player_name'].apply(self.normalize_name)

        # Merge predictions with actuals
        merged = predictions.merge(
            actual_stats,
            left_on=['player_normalized', 'position'],
            right_on=['player_normalized', 'position'],
            suffixes=('_pred', '_actual'),
            how='inner'
        )

        logger.info(f"  ✓ Matched {len(merged):,} player predictions to actual stats")

        if len(merged) == 0:
            logger.warning(f"  ⚠️  No matches found for Week {week}")
            return None

        # Calculate metrics for each stat type
        metrics = {}

        # Rushing yards
        if 'rushing_yards_mean' in merged.columns and 'rushing_yards' in merged.columns:
            rush_data = merged[
                (merged['rushing_yards_mean'].notna()) &
                (merged['rushing_yards'].notna()) &
                (merged['rushing_yards'] > 0)  # Only players who actually rushed
            ]
            if len(rush_data) > 0:
                metrics['rushing_yards'] = self._calculate_metrics(
                    rush_data['rushing_yards'],
                    rush_data['rushing_yards_mean']
                )
                metrics['rushing_yards']['n'] = len(rush_data)

        # Receiving yards
        if 'receiving_yards_mean' in merged.columns and 'receiving_yards' in merged.columns:
            rec_data = merged[
                (merged['receiving_yards_mean'].notna()) &
                (merged['receiving_yards'].notna()) &
                (merged['receiving_yards'] > 0)
            ]
            if len(rec_data) > 0:
                metrics['receiving_yards'] = self._calculate_metrics(
                    rec_data['receiving_yards'],
                    rec_data['receiving_yards_mean']
                )
                metrics['receiving_yards']['n'] = len(rec_data)

        # Receptions
        if 'receptions_mean' in merged.columns and 'receptions' in merged.columns:
            rec_data = merged[
                (merged['receptions_mean'].notna()) &
                (merged['receptions'].notna())
            ]
            if len(rec_data) > 0:
                metrics['receptions'] = self._calculate_metrics(
                    rec_data['receptions'],
                    rec_data['receptions_mean']
                )
                metrics['receptions']['n'] = len(rec_data)

        # Passing yards
        if 'passing_yards_mean' in merged.columns and 'passing_yards' in merged.columns:
            pass_data = merged[
                (merged['passing_yards_mean'].notna()) &
                (merged['passing_yards'].notna()) &
                (merged['passing_yards'] > 0)
            ]
            if len(pass_data) > 0:
                metrics['passing_yards'] = self._calculate_metrics(
                    pass_data['passing_yards'],
                    pass_data['passing_yards_mean']
                )
                metrics['passing_yards']['n'] = len(pass_data)

        # Log results
        logger.info(f"\n📊 Week {week} Validation Results:")
        for stat_type, stat_metrics in metrics.items():
            logger.info(f"\n  {stat_type.upper()}:")
            logger.info(f"    Sample: {stat_metrics['n']} players")
            logger.info(f"    MAE: {stat_metrics['mae']:.2f}")
            logger.info(f"    RMSE: {stat_metrics['rmse']:.2f}")
            logger.info(f"    R²: {stat_metrics['r2']:.3f}")
            logger.info(f"    Mean Error: {stat_metrics['mean_error']:.2f}")

        # Store results
        week_result = {
            'week': week,
            'matched_players': len(merged),
            **{f"{stat}_{metric}": metrics.get(stat, {}).get(metric, None)
               for stat in ['rushing_yards', 'receiving_yards', 'receptions', 'passing_yards']
               for metric in ['mae', 'rmse', 'r2', 'n']}
        }
        self.results.append(week_result)

        return week_result

    def _calculate_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """Calculate prediction accuracy metrics."""
        errors = actual - predicted
        squared_errors = errors ** 2

        mae = np.abs(errors).mean()
        rmse = np.sqrt(squared_errors.mean())
        mean_error = errors.mean()

        # R²
        ss_res = squared_errors.sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_error': mean_error
        }

    def validate_season(self, start_week: int = 5, end_week: int = 11) -> pd.DataFrame:
        """Validate all weeks in range."""
        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATING 2025 SEASON: WEEKS {start_week}-{end_week}")
        logger.info(f"{'='*80}\n")

        for week in range(start_week, end_week + 1):
            self.validate_week(week)

        # Generate summary
        self.generate_summary()

        # Save results
        self.save_results()

        return pd.DataFrame(self.results)

    def generate_summary(self):
        """Generate summary of all validation results."""
        if len(self.results) == 0:
            logger.warning("No results to summarize")
            return

        results_df = pd.DataFrame(self.results)

        logger.info(f"\n{'='*80}")
        logger.info(f"SEASON SUMMARY (Weeks {results_df['week'].min()}-{results_df['week'].max()})")
        logger.info(f"{'='*80}\n")

        # Average metrics across all weeks
        logger.info("📊 Overall Prediction Accuracy:\n")

        for stat_type in ['rushing_yards', 'receiving_yards', 'receptions', 'passing_yards']:
            mae_col = f"{stat_type}_mae"
            rmse_col = f"{stat_type}_rmse"
            r2_col = f"{stat_type}_r2"
            n_col = f"{stat_type}_n"

            if mae_col in results_df.columns:
                avg_mae = results_df[mae_col].mean()
                avg_rmse = results_df[rmse_col].mean()
                avg_r2 = results_df[r2_col].mean()
                total_n = results_df[n_col].sum()

                logger.info(f"  {stat_type.upper().replace('_', ' ')}:")
                logger.info(f"    Total Predictions: {total_n:.0f}")
                logger.info(f"    Avg MAE: {avg_mae:.2f}")
                logger.info(f"    Avg RMSE: {avg_rmse:.2f}")
                logger.info(f"    Avg R²: {avg_r2:.3f}\n")

    def save_results(self):
        """Save validation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_df = pd.DataFrame(self.results)
        output_path = project_root / f"reports/prediction_validation_{timestamp}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n💾 Saved validation results: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate NFL QUANT predictions against actual outcomes"
    )
    parser.add_argument(
        '--start-week',
        type=int,
        default=5,
        help="First week to validate (default: 5)"
    )
    parser.add_argument(
        '--end-week',
        type=int,
        default=11,
        help="Last week to validate (default: 11)"
    )
    parser.add_argument(
        '--week',
        type=int,
        help="Validate single week only"
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help="Season year (default: 2025)"
    )

    args = parser.parse_args()

    # Create validator
    validator = PredictionValidator(season=args.season)

    # Run validation
    if args.week:
        validator.validate_week(args.week)
        validator.generate_summary()
        validator.save_results()
    else:
        validator.validate_season(args.start_week, args.end_week)


if __name__ == "__main__":
    main()
