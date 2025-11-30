#!/usr/bin/env python3
"""
Calibration Comparison Backtest (Nov 24, 2025)
================================================

Purpose: Compare calibration methods to determine best approach for Week 12 deployment.

Methods Compared:
1. Raw Model (no calibration)
2. 30% Shrinkage (current production)
3. 50% Shrinkage (more conservative)
4. Platt Scaling (logistic regression on holdout)

Temporal Split:
- Train/Calibration: Weeks 2-8 (7 weeks)
- Test: Weeks 9-11 (3 weeks)

Metrics:
- Brier Score (probability accuracy)
- Calibration Error (Mean Absolute Error of predicted vs actual hit rates)
- ROI (Return on Investment using Kelly staking)
- Hit Rate by confidence bucket

Usage:
    python scripts/backtest/calibration_comparison_backtest.py

Author: Claude Code (NFL QUANT Project)
Date: Nov 24, 2025
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationBacktester:
    """
    Compare calibration methods using temporal train/test split.
    """

    # Temporal split
    TRAIN_WEEKS = list(range(2, 9))  # Weeks 2-8
    TEST_WEEKS = list(range(9, 12))  # Weeks 9-11
    SEASON = 2025

    # Market mapping: props file column -> prediction file column
    MARKET_MAP = {
        'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std'),
        'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std'),
        'player_pass_yds': ('passing_yards_mean', 'passing_yards_std'),
        'player_receptions': ('receptions_mean', 'receptions_std'),
    }

    # Actual stat mapping
    ACTUAL_MAP = {
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
        'player_receptions': 'receptions',
    }

    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.platt_calibrators = {}  # Per-market calibrators

    def load_week_data(self, week: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load predictions, props, and actuals for a week."""

        # Predictions
        pred_path = project_root / f"data/model_predictions_week{week}.csv"
        if not pred_path.exists():
            logger.warning(f"Missing predictions: {pred_path}")
            return None, None, None
        predictions = pd.read_csv(pred_path)

        # Props (historical odds)
        props_path = project_root / f"data/backtest/historical_by_week/week_{week}_props.csv"
        if not props_path.exists():
            logger.warning(f"Missing props: {props_path}")
            return None, None, None
        props = pd.read_csv(props_path)

        # Actuals (weekly stats)
        actuals_path = project_root / "data/nflverse/weekly_stats.parquet"
        actuals = pd.read_parquet(actuals_path)
        actuals = actuals[(actuals['season'] == self.SEASON) & (actuals['week'] == week)]

        return predictions, props, actuals

    def normalize_name(self, name: str) -> str:
        """Normalize player name for matching."""
        if pd.isna(name):
            return ""
        return name.strip().lower().replace('.', '').replace("'", "")

    def calculate_over_probability(
        self,
        mean: float,
        std: float,
        line: float,
        shrinkage: float = 0.0
    ) -> float:
        """
        Calculate probability of going OVER the line.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            line: Prop line
            shrinkage: Shrinkage factor (0 = raw, 0.3 = 30% toward 0.5)
        """
        if std <= 0:
            std = 1.0  # Minimum std

        # Raw probability from normal CDF
        raw_prob = 1 - norm.cdf(line + 0.5, loc=mean, scale=std)  # +0.5 for half point

        # Apply shrinkage
        if shrinkage > 0:
            calibrated_prob = raw_prob + shrinkage * (0.5 - raw_prob)
        else:
            calibrated_prob = raw_prob

        return np.clip(calibrated_prob, 0.01, 0.99)

    def match_data_for_week(self, week: int) -> List[Dict]:
        """
        Match predictions to props and actuals for a week.
        Returns list of matched records.
        """
        predictions, props, actuals = self.load_week_data(week)

        if predictions is None or props is None or actuals is None:
            return []

        # Filter to OVER props only (we'll compute UNDER from OVER)
        props_over = props[props['prop_type'] == 'over'].copy()

        matched = []

        for _, prop in props_over.iterrows():
            market = prop['market']
            if market not in self.MARKET_MAP:
                continue

            player_name = self.normalize_name(prop['player'])
            line = prop['line']
            odds = prop['american_odds']

            mean_col, std_col = self.MARKET_MAP[market]
            actual_col = self.ACTUAL_MAP[market]

            # Find prediction
            pred_match = predictions[
                predictions['player_name'].apply(self.normalize_name) == player_name
            ]

            if len(pred_match) == 0:
                continue

            pred_row = pred_match.iloc[0]

            # Check columns exist
            if mean_col not in pred_row or std_col not in pred_row:
                continue

            mean_val = pred_row[mean_col]
            std_val = pred_row[std_col]

            if pd.isna(mean_val) or pd.isna(std_val):
                continue

            # Find actual - use player_display_name for full names
            actual_match = actuals[
                actuals['player_display_name'].apply(self.normalize_name) == player_name
            ]

            if len(actual_match) == 0:
                continue

            actual_val = actual_match[actual_col].iloc[0]

            if pd.isna(actual_val):
                continue

            # Determine if OVER hit
            over_hit = 1 if actual_val > line else 0

            matched.append({
                'week': week,
                'player': prop['player'],
                'market': market,
                'line': line,
                'odds': odds,
                'mean': mean_val,
                'std': std_val,
                'actual': actual_val,
                'over_hit': over_hit
            })

        return matched

    def load_all_data(self):
        """Load data for all weeks, split into train/test."""
        logger.info("Loading training data (Weeks 2-8)...")
        for week in self.TRAIN_WEEKS:
            week_data = self.match_data_for_week(week)
            self.train_data.extend(week_data)
            logger.info(f"  Week {week}: {len(week_data)} matched props")

        logger.info(f"  Total training samples: {len(self.train_data)}")

        logger.info("\nLoading test data (Weeks 9-11)...")
        for week in self.TEST_WEEKS:
            week_data = self.match_data_for_week(week)
            self.test_data.extend(week_data)
            logger.info(f"  Week {week}: {len(week_data)} matched props")

        logger.info(f"  Total test samples: {len(self.test_data)}")

    def train_platt_calibrators(self):
        """Train Platt scaling calibrators on training data."""
        logger.info("\nTraining Platt scaling calibrators...")

        train_df = pd.DataFrame(self.train_data)

        for market in self.MARKET_MAP.keys():
            market_data = train_df[train_df['market'] == market]

            if len(market_data) < 20:
                logger.warning(f"  {market}: Insufficient data ({len(market_data)}), skipping")
                continue

            # Calculate raw probabilities
            X = np.array([
                self.calculate_over_probability(row['mean'], row['std'], row['line'], shrinkage=0)
                for _, row in market_data.iterrows()
            ]).reshape(-1, 1)

            y = market_data['over_hit'].values

            # Fit logistic regression
            calibrator = LogisticRegression(solver='lbfgs')
            calibrator.fit(X, y)

            self.platt_calibrators[market] = calibrator
            logger.info(f"  {market}: Trained on {len(market_data)} samples")

    def apply_platt_calibration(self, raw_prob: float, market: str) -> float:
        """Apply Platt scaling to a raw probability."""
        if market not in self.platt_calibrators:
            return raw_prob  # Fallback to raw

        calibrated = self.platt_calibrators[market].predict_proba([[raw_prob]])[0, 1]
        return np.clip(calibrated, 0.01, 0.99)

    def calculate_kelly_stake(self, prob: float, odds: float, fraction: float = 0.25) -> float:
        """Calculate Kelly stake (quarter Kelly by default)."""
        # Convert American odds to decimal
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))

        # Kelly formula
        b = decimal_odds - 1
        q = 1 - prob
        kelly = (prob * b - q) / b

        if kelly <= 0:
            return 0

        return fraction * kelly

    def evaluate_method(
        self,
        data: List[Dict],
        method: str,
        shrinkage: float = 0.0
    ) -> Dict:
        """
        Evaluate a calibration method on data.

        Args:
            data: List of matched prop records
            method: 'raw', 'shrinkage_30', 'shrinkage_50', 'platt'
            shrinkage: Shrinkage factor for shrinkage methods

        Returns:
            Dictionary of metrics
        """
        results = []

        for record in data:
            raw_prob = self.calculate_over_probability(
                record['mean'], record['std'], record['line'], shrinkage=0
            )

            if method == 'raw':
                prob = raw_prob
            elif method.startswith('shrinkage'):
                prob = self.calculate_over_probability(
                    record['mean'], record['std'], record['line'], shrinkage=shrinkage
                )
            elif method == 'platt':
                prob = self.apply_platt_calibration(raw_prob, record['market'])
            else:
                prob = raw_prob

            # Determine bet side and result FIRST, then calculate stake
            if prob > 0.5:
                # Bet OVER
                bet_side = 'OVER'
                hit = record['over_hit'] == 1
                bet_prob = prob  # Use OVER probability
            else:
                # Bet UNDER
                bet_side = 'UNDER'
                hit = record['over_hit'] == 0
                bet_prob = 1 - prob  # Flip to UNDER probability

            # Calculate stake with the CORRECT bet probability
            stake = self.calculate_kelly_stake(bet_prob, record['odds'])
            prob = bet_prob  # Update prob for reporting

            # Calculate profit
            if stake > 0 and hit:
                odds = record['odds']
                if odds > 0:
                    profit = stake * (odds / 100)
                else:
                    profit = stake * (100 / abs(odds))
            elif stake > 0:
                profit = -stake
            else:
                profit = 0

            results.append({
                'week': record['week'],
                'player': record['player'],
                'market': record['market'],
                'prob': prob,
                'actual_hit': hit,
                'stake': stake,
                'profit': profit,
                'bet_side': bet_side
            })

        # Calculate metrics
        results_df = pd.DataFrame(results)

        # Filter to actual bets (stake > 0)
        bets = results_df[results_df['stake'] > 0]

        # Brier score (all predictions)
        all_probs = results_df['prob'].values
        all_hits = results_df['actual_hit'].astype(int).values
        brier_score = np.mean((all_probs - all_hits) ** 2)

        # Calibration error (bucketed) - on ALL predictions
        calibration_error = self.calculate_calibration_error(results_df)

        # ROI (Kelly-filtered bets only)
        total_staked = bets['stake'].sum()
        total_profit = bets['profit'].sum()
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        # Hit rate - BOTH filtered and unfiltered
        hit_rate_filtered = bets['actual_hit'].mean() * 100 if len(bets) > 0 else 0
        hit_rate_all = results_df['actual_hit'].mean() * 100  # TRUE model quality

        # By confidence bucket
        bucket_results = self.calculate_bucket_results(results_df)

        # Bet side breakdown (OVER vs UNDER)
        over_bets = bets[bets['bet_side'] == 'OVER']
        under_bets = bets[bets['bet_side'] == 'UNDER']
        over_hit_rate = over_bets['actual_hit'].mean() * 100 if len(over_bets) > 0 else 0
        under_hit_rate = under_bets['actual_hit'].mean() * 100 if len(under_bets) > 0 else 0

        return {
            'method': method,
            'total_predictions': len(results_df),
            'total_bets': len(bets),
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'roi': roi,
            'hit_rate': hit_rate_filtered,  # Kelly-filtered (for betting)
            'hit_rate_all': hit_rate_all,   # All predictions (true model quality)
            'total_profit': total_profit,
            'total_staked': total_staked,
            'bucket_results': bucket_results,
            'over_bets': len(over_bets),
            'over_hit_rate': over_hit_rate,
            'under_bets': len(under_bets),
            'under_hit_rate': under_hit_rate
        }

    def calculate_calibration_error(self, results_df: pd.DataFrame) -> float:
        """
        Calculate calibration error using probability buckets.

        Uses bucket MIDPOINT as expected value (not mean of predictions).
        This measures how well predicted probabilities match actual outcomes.

        Example: If we predict 50-55% (midpoint 52.5%) and actual hit rate is 61%,
        calibration error for that bucket is |0.525 - 0.61| = 0.085 (8.5%)
        """
        buckets = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        weighted_errors = []
        total_n = 0

        for i in range(len(buckets) - 1):
            low, high = buckets[i], buckets[i+1]
            bucket_data = results_df[(results_df['prob'] >= low) & (results_df['prob'] < high)]

            if len(bucket_data) >= 5:  # Minimum sample
                # FIX: Use bucket midpoint as expected, not mean of predictions
                expected = (low + high) / 2  # Bucket midpoint
                actual = bucket_data['actual_hit'].mean()
                n = len(bucket_data)
                weighted_errors.append(abs(expected - actual) * n)
                total_n += n

        # Return weighted average calibration error
        return sum(weighted_errors) / total_n if total_n > 0 else 0.0

    def calculate_bucket_results(self, results_df: pd.DataFrame) -> Dict:
        """Calculate results by confidence bucket."""
        buckets = {
            '50-55%': (0.50, 0.55),
            '55-60%': (0.55, 0.60),
            '60-65%': (0.60, 0.65),
            '65-70%': (0.65, 0.70),
            '70-75%': (0.70, 0.75),
            '75-80%': (0.75, 0.80),
            '80%+': (0.80, 1.01),
        }

        bucket_results = {}

        for name, (low, high) in buckets.items():
            bucket_data = results_df[
                (results_df['prob'] >= low) &
                (results_df['prob'] < high) &
                (results_df['stake'] > 0)
            ]

            if len(bucket_data) > 0:
                hit_rate = bucket_data['actual_hit'].mean() * 100
                roi = (bucket_data['profit'].sum() / bucket_data['stake'].sum() * 100)
                bucket_results[name] = {
                    'n': len(bucket_data),
                    'hit_rate': hit_rate,
                    'roi': roi,
                    'expected_hit_rate': (low + high) / 2 * 100
                }

        return bucket_results

    def run_comparison(self):
        """Run full calibration comparison."""
        logger.info("=" * 80)
        logger.info("CALIBRATION COMPARISON BACKTEST")
        logger.info(f"Train: Weeks {self.TRAIN_WEEKS[0]}-{self.TRAIN_WEEKS[-1]}")
        logger.info(f"Test: Weeks {self.TEST_WEEKS[0]}-{self.TEST_WEEKS[-1]}")
        logger.info("=" * 80)

        # Load data
        self.load_all_data()

        if len(self.train_data) == 0 or len(self.test_data) == 0:
            logger.error("Insufficient data for comparison")
            return None

        # Train Platt calibrators
        self.train_platt_calibrators()

        # Evaluate methods on TEST data
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON TEST SET (Weeks 9-11)")
        logger.info("=" * 80)

        methods = [
            ('raw', 0.0),
            ('shrinkage_30', 0.30),
            ('shrinkage_50', 0.50),
            ('platt', 0.0)
        ]

        results = []

        for method, shrinkage in methods:
            logger.info(f"\nEvaluating: {method.upper()}")
            result = self.evaluate_method(self.test_data, method, shrinkage)
            results.append(result)

            logger.info(f"  Brier Score: {result['brier_score']:.4f}")
            logger.info(f"  Calibration Error: {result['calibration_error']:.3f} ({result['calibration_error']*100:.1f}%)")
            logger.info(f"  ROI: {result['roi']:.1f}%")
            logger.info(f"  Hit Rate (Kelly-filtered): {result['hit_rate']:.1f}% ({result['total_bets']} bets)")
            logger.info(f"  Hit Rate (ALL predictions): {result['hit_rate_all']:.1f}% ({result['total_predictions']} preds)")
            logger.info(f"  OVER bets: {result['over_bets']} @ {result['over_hit_rate']:.1f}% hit")
            logger.info(f"  UNDER bets: {result['under_bets']} @ {result['under_hit_rate']:.1f}% hit")
            logger.info(f"  Total P&L: ${result['total_profit']:.2f}")

        # Generate summary report
        self.generate_report(results)

        return results

    def generate_report(self, results: List[Dict]):
        """Generate final comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("FINAL COMPARISON SUMMARY")
        logger.info("=" * 80)

        # Summary table - main metrics
        logger.info("\n{:<15} {:>8} {:>10} {:>8} {:>10} {:>10} {:>10}".format(
            "Method", "Brier", "Cal.Err%", "ROI%", "Hit%(bet)", "Hit%(all)", "Profit"
        ))
        logger.info("-" * 80)

        best_brier = min(r['brier_score'] for r in results)
        best_roi = max(r['roi'] for r in results)
        best_cal = min(r['calibration_error'] for r in results)

        for r in results:
            brier_mark = "*" if r['brier_score'] == best_brier else ""
            roi_mark = "*" if r['roi'] == best_roi else ""
            cal_mark = "*" if r['calibration_error'] == best_cal else ""

            logger.info("{:<15} {:>7.4f}{:<1} {:>8.1f}%{:<1} {:>7.1f}%{:<1} {:>9.1f}% {:>9.1f}% ${:>9.2f}".format(
                r['method'],
                r['brier_score'], brier_mark,
                r['calibration_error'] * 100, cal_mark,
                r['roi'], roi_mark,
                r['hit_rate'],
                r['hit_rate_all'],
                r['total_profit']
            ))

        logger.info("\n* = best in category")
        logger.info("Hit%(bet) = Kelly-filtered bets only, Hit%(all) = TRUE model quality")

        # OVER vs UNDER breakdown
        logger.info("\n" + "-" * 80)
        logger.info("OVER vs UNDER BREAKDOWN (reveals systematic bias)")
        logger.info("-" * 80)
        logger.info("{:<15} {:>10} {:>10} {:>10} {:>10}".format(
            "Method", "OVER_N", "OVER_Hit%", "UNDER_N", "UNDER_Hit%"
        ))
        logger.info("-" * 55)

        for r in results:
            over_bias = "OVER BIAS!" if r['over_hit_rate'] < 50 and r['under_hit_rate'] > 50 else ""
            logger.info("{:<15} {:>10} {:>9.1f}% {:>10} {:>9.1f}%  {}".format(
                r['method'],
                r['over_bets'],
                r['over_hit_rate'],
                r['under_bets'],
                r['under_hit_rate'],
                over_bias
            ))

        # Bucket analysis for best method
        best_method = min(results, key=lambda x: x['calibration_error'])

        logger.info(f"\nConfidence Bucket Analysis ({best_method['method'].upper()}):")
        logger.info("{:<12} {:>6} {:>12} {:>12} {:>10}".format(
            "Bucket", "N", "Hit Rate", "Expected", "ROI"
        ))
        logger.info("-" * 55)

        for bucket, data in best_method['bucket_results'].items():
            logger.info("{:<12} {:>6} {:>10.1f}% {:>10.1f}% {:>8.1f}%".format(
                bucket,
                data['n'],
                data['hit_rate'],
                data['expected_hit_rate'],
                data['roi']
            ))

        # Recommendation
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATION FOR WEEK 12")
        logger.info("=" * 80)

        # Decision criteria
        raw = next(r for r in results if r['method'] == 'raw')
        shrink30 = next(r for r in results if r['method'] == 'shrinkage_30')
        shrink50 = next(r for r in results if r['method'] == 'shrinkage_50')
        platt = next(r for r in results if r['method'] == 'platt')

        # Check if Platt is significantly better
        platt_brier_improvement = (raw['brier_score'] - platt['brier_score']) / raw['brier_score'] * 100

        if platt_brier_improvement > 10 and platt['calibration_error'] < 0.10 and platt['roi'] > 0:
            recommendation = "PLATT SCALING"
            reason = f"Brier improvement: {platt_brier_improvement:.1f}%, positive ROI on test set"
        elif shrink30['roi'] > shrink50['roi'] and shrink30['calibration_error'] < 0.12:
            recommendation = "30% SHRINKAGE (current)"
            reason = "Better ROI than 50% shrinkage, acceptable calibration"
        elif shrink50['roi'] > 0 and shrink50['calibration_error'] < shrink30['calibration_error']:
            recommendation = "50% SHRINKAGE"
            reason = "Better calibration, positive ROI"
        else:
            recommendation = "30% SHRINKAGE (conservative)"
            reason = "Default fallback - more conservative approach"

        logger.info(f"\nRecommended Method: {recommendation}")
        logger.info(f"Reason: {reason}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df = pd.DataFrame(results)
        results_path = project_root / f"reports/calibration_comparison_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"\nResults saved to: {results_path}")


def main():
    backtester = CalibrationBacktester()
    backtester.run_comparison()


if __name__ == "__main__":
    main()
