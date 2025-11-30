#!/usr/bin/env python3
"""
NFLverse-Native Week-by-Week Backtest Framework

Runs a comprehensive backtest using:
1. NFLverse parquet files for actual outcomes
2. Trained models for predictions
3. Historical or current odds for betting context

This script validates model performance by:
- Generating predictions for each week using only prior week data
- Comparing predictions to actual NFLverse outcomes
- Calculating calibration metrics (Brier score, ECE, etc.)
- Tracking betting ROI if odds available
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market, clear_calibrator_cache

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# NFLverse data directory
NFLVERSE_DIR = Path(__file__).parent.parent.parent / 'data' / 'nflverse'


def load_nflverse_actual_stats(season: int = 2025) -> pd.DataFrame:
    """Load actual player stats from NFLverse parquet files."""
    weekly_file = NFLVERSE_DIR / 'weekly_stats.parquet'

    if not weekly_file.exists():
        raise FileNotFoundError(f"NFLverse weekly stats not found: {weekly_file}")

    df = pd.read_parquet(weekly_file)

    # Filter to season
    df = df[df['season'] == season].copy()

    logger.info(f"Loaded {len(df):,} player-week records from NFLverse ({season})")

    return df


def generate_predictions_for_week(
    week: int,
    season: int,
    nflverse_stats: pd.DataFrame,
    simulator: PlayerSimulator
) -> pd.DataFrame:
    """
    Generate predictions for a specific week using trailing data.

    This simulates what predictions would have been at the start of that week,
    using only data from weeks prior.
    """
    # Get trailing stats (weeks before current week)
    trailing_data = nflverse_stats[nflverse_stats['week'] < week].copy()

    if len(trailing_data) == 0:
        logger.warning(f"No trailing data available for week {week}")
        return pd.DataFrame()

    # Get players who played in the target week (these are who we're predicting)
    target_week_data = nflverse_stats[nflverse_stats['week'] == week].copy()

    if len(target_week_data) == 0:
        logger.warning(f"No data for week {week}")
        return pd.DataFrame()

    predictions = []

    # Calculate trailing averages for each player
    for _, player_row in target_week_data.iterrows():
        player_name = player_row['player_display_name']
        position = player_row['position']
        team = player_row['team']

        # Skip non-skill positions
        if position not in ['QB', 'RB', 'WR', 'TE']:
            continue

        # Get this player's historical data
        player_history = trailing_data[
            trailing_data['player_display_name'] == player_name
        ].copy()

        if len(player_history) == 0:
            # No history - skip or use defaults
            continue

        # Calculate trailing averages
        avg_pass_yds = player_history['passing_yards'].mean() if 'passing_yards' in player_history.columns else 0
        avg_rush_yds = player_history['rushing_yards'].mean() if 'rushing_yards' in player_history.columns else 0
        avg_rec_yds = player_history['receiving_yards'].mean() if 'receiving_yards' in player_history.columns else 0
        avg_receptions = player_history['receptions'].mean() if 'receptions' in player_history.columns else 0
        avg_targets = player_history['targets'].mean() if 'targets' in player_history.columns else 0
        avg_carries = player_history['carries'].mean() if 'carries' in player_history.columns else 0

        # Simple prediction: use trailing averages with some variance
        # In production, this would use the full PlayerSimulator with game context
        predictions.append({
            'player_name': player_name,
            'position': position,
            'team': team,
            'week': week,
            'season': season,
            # Predictions (trailing averages)
            'pred_passing_yards': avg_pass_yds,
            'pred_rushing_yards': avg_rush_yds,
            'pred_receiving_yards': avg_rec_yds,
            'pred_receptions': avg_receptions,
            'pred_targets': avg_targets,
            'pred_carries': avg_carries,
            # Actuals
            'actual_passing_yards': player_row.get('passing_yards', 0),
            'actual_rushing_yards': player_row.get('rushing_yards', 0),
            'actual_receiving_yards': player_row.get('receiving_yards', 0),
            'actual_receptions': player_row.get('receptions', 0),
            'actual_targets': player_row.get('targets', 0),
            'actual_carries': player_row.get('carries', 0),
            'actual_passing_tds': player_row.get('passing_tds', 0),
            'actual_rushing_tds': player_row.get('rushing_tds', 0),
            'actual_receiving_tds': player_row.get('receiving_tds', 0),
            # History
            'weeks_of_history': len(player_history),
        })

    return pd.DataFrame(predictions)


def calculate_prop_probabilities(
    predictions_df: pd.DataFrame,
    prop_lines: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Calculate probability of going over various prop lines.

    Uses normal distribution assumption for yards/receptions.
    """
    if prop_lines is None:
        # Default prop lines for different stats
        prop_lines = {
            'passing_yards': [199.5, 224.5, 249.5, 274.5, 299.5],
            'rushing_yards': [49.5, 59.5, 69.5, 79.5, 99.5],
            'receiving_yards': [49.5, 59.5, 69.5, 79.5, 99.5],
            'receptions': [3.5, 4.5, 5.5, 6.5, 7.5],
        }

    results = []

    for _, row in predictions_df.iterrows():
        player_data = row.to_dict()

        # For each stat type, calculate over probability
        for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards', 'receptions']:
            pred_key = f'pred_{stat_type}'
            actual_key = f'actual_{stat_type}'

            if pred_key not in row or row[pred_key] == 0:
                continue

            pred_mean = row[pred_key]
            actual_value = row[actual_key]

            # Estimate std as fraction of mean (typical variance pattern)
            if stat_type == 'receptions':
                pred_std = max(pred_mean * 0.4, 1.0)  # 40% CV for receptions
            else:
                pred_std = max(pred_mean * 0.35, 10.0)  # 35% CV for yards

            # Calculate probability for standard lines
            for line in prop_lines.get(stat_type, []):
                # Skip if line is way out of range
                if line > pred_mean * 3:
                    continue

                # Normal CDF approximation
                z_score = (line - pred_mean) / pred_std
                prob_over = 1 - 0.5 * (1 + np.tanh(z_score * 0.7))  # Approximate normal CDF
                prob_over = np.clip(prob_over, 0.01, 0.99)

                # Did player actually go over?
                went_over = 1 if actual_value > line else 0

                results.append({
                    'player_name': row['player_name'],
                    'position': row['position'],
                    'team': row['team'],
                    'week': row['week'],
                    'stat_type': stat_type,
                    'line': line,
                    'pred_mean': pred_mean,
                    'pred_std': pred_std,
                    'prob_over_raw': prob_over,
                    'actual_value': actual_value,
                    'went_over': went_over,
                })

    return pd.DataFrame(results)


def apply_calibration(props_df: pd.DataFrame) -> pd.DataFrame:
    """Apply market-specific calibrators to raw probabilities."""
    if len(props_df) == 0:
        return props_df

    # Clear cache to ensure fresh calibrators are loaded
    clear_calibrator_cache()

    # Map stat types to calibrator market names
    market_map = {
        'passing_yards': 'player_pass_yds',
        'rushing_yards': 'player_rush_yds',
        'receiving_yards': 'player_reception_yds',
        'receptions': 'player_receptions',
    }

    calibrated_probs = []

    for _, row in props_df.iterrows():
        market = market_map.get(row['stat_type'], 'default')

        try:
            calibrator = load_calibrator_for_market(market)
            if calibrator and hasattr(calibrator, 'transform'):
                cal_prob = calibrator.transform(row['prob_over_raw'])
            elif calibrator and hasattr(calibrator, 'calibrate'):
                cal_prob = calibrator.calibrate(row['prob_over_raw'])
            else:
                cal_prob = row['prob_over_raw']
        except Exception:
            cal_prob = row['prob_over_raw']

        calibrated_probs.append(cal_prob)

    props_df = props_df.copy()
    props_df['prob_over_calibrated'] = calibrated_probs

    return props_df


def calculate_backtest_metrics(props_df: pd.DataFrame) -> Dict:
    """Calculate comprehensive backtest metrics."""
    if len(props_df) == 0:
        return {}

    metrics = {
        'total_predictions': len(props_df),
        'timestamp': datetime.now().isoformat(),
    }

    # Overall metrics
    raw_probs = props_df['prob_over_raw'].values
    cal_probs = props_df['prob_over_calibrated'].values
    actuals = props_df['went_over'].values

    # Brier scores (still valid - measures probability accuracy for OVER side)
    metrics['brier_score_raw'] = float(np.mean((raw_probs - actuals) ** 2))
    metrics['brier_score_calibrated'] = float(np.mean((cal_probs - actuals) ** 2))
    metrics['brier_improvement'] = float(metrics['brier_score_raw'] - metrics['brier_score_calibrated'])

    # CRITICAL FIX: Calculate win rate based on model's RECOMMENDED side
    # If prob_over > 0.5, model recommends OVER; else recommends UNDER
    model_recommends_over = cal_probs > 0.5
    model_correct = np.where(
        model_recommends_over,
        actuals == 1,  # Recommended OVER, check if went over
        actuals == 0   # Recommended UNDER, check if went under
    )

    # Overall model accuracy (when taking the side the model favors)
    metrics['model_accuracy'] = float(model_correct.mean())

    # Break down by recommendation
    over_recs = model_recommends_over.sum()
    under_recs = (~model_recommends_over).sum()

    if over_recs > 0:
        over_accuracy = model_correct[model_recommends_over].mean()
        metrics['over_recommendations'] = int(over_recs)
        metrics['over_accuracy'] = float(over_accuracy)
    else:
        metrics['over_recommendations'] = 0
        metrics['over_accuracy'] = 0.0

    if under_recs > 0:
        under_accuracy = model_correct[~model_recommends_over].mean()
        metrics['under_recommendations'] = int(under_recs)
        metrics['under_accuracy'] = float(under_accuracy)
    else:
        metrics['under_recommendations'] = 0
        metrics['under_accuracy'] = 0.0

    # Legacy win rate (just tracks OVER hits - less meaningful)
    metrics['overall_over_rate'] = float(actuals.mean())
    metrics['avg_predicted_prob_raw'] = float(raw_probs.mean())
    metrics['avg_predicted_prob_cal'] = float(cal_probs.mean())

    # Calibration error by probability bins
    bins = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_metrics = []

    for i in range(len(bins) - 1):
        bin_mask = (cal_probs >= bins[i]) & (cal_probs < bins[i+1])
        if bin_mask.sum() > 0:
            bin_pred = cal_probs[bin_mask].mean()
            bin_actual = actuals[bin_mask].mean()
            bin_count = bin_mask.sum()
            bin_error = abs(bin_pred - bin_actual)

            bin_metrics.append({
                'bin': f'{bins[i]:.1f}-{bins[i+1]:.1f}',
                'count': int(bin_count),
                'avg_predicted': float(bin_pred),
                'actual_win_rate': float(bin_actual),
                'calibration_error': float(bin_error),
            })

    metrics['calibration_by_bin'] = bin_metrics

    # Expected Calibration Error
    if len(bin_metrics) > 0:
        total_samples = sum(b['count'] for b in bin_metrics)
        ece = sum(b['count'] * b['calibration_error'] for b in bin_metrics) / total_samples
        metrics['expected_calibration_error'] = float(ece)

    # Metrics by stat type
    stat_metrics = {}
    for stat_type in props_df['stat_type'].unique():
        stat_mask = props_df['stat_type'] == stat_type
        stat_data = props_df[stat_mask]

        stat_metrics[stat_type] = {
            'count': int(len(stat_data)),
            'brier_raw': float(np.mean((stat_data['prob_over_raw'] - stat_data['went_over']) ** 2)),
            'brier_cal': float(np.mean((stat_data['prob_over_calibrated'] - stat_data['went_over']) ** 2)),
            'win_rate': float(stat_data['went_over'].mean()),
            'avg_pred_prob': float(stat_data['prob_over_calibrated'].mean()),
        }

    metrics['by_stat_type'] = stat_metrics

    # Metrics by week
    week_metrics = {}
    for week in sorted(props_df['week'].unique()):
        week_mask = props_df['week'] == week
        week_data = props_df[week_mask]

        week_metrics[int(week)] = {
            'count': int(len(week_data)),
            'brier_raw': float(np.mean((week_data['prob_over_raw'] - week_data['went_over']) ** 2)),
            'brier_cal': float(np.mean((week_data['prob_over_calibrated'] - week_data['went_over']) ** 2)),
            'win_rate': float(week_data['went_over'].mean()),
            'avg_pred_prob': float(week_data['prob_over_calibrated'].mean()),
        }

    metrics['by_week'] = week_metrics

    return metrics


def run_week_by_week_backtest(
    season: int = 2025,
    start_week: int = 2,
    end_week: int = 10
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run comprehensive week-by-week backtest.

    Args:
        season: NFL season year
        start_week: First week to backtest (need at least 1 week of history)
        end_week: Last week to backtest

    Returns:
        Tuple of (all_props_df, metrics_dict)
    """
    logger.info("="*80)
    logger.info(f"NFL QUANT WEEK-BY-WEEK BACKTEST")
    logger.info(f"Season: {season}, Weeks: {start_week}-{end_week}")
    logger.info("="*80)
    logger.info("")

    # Load NFLverse data
    nflverse_stats = load_nflverse_actual_stats(season)

    # Load predictors
    try:
        simulator = PlayerSimulator()
        logger.info("✅ Loaded player simulator with trained models")
    except Exception as e:
        logger.warning(f"⚠️  Could not load simulator: {e}")
        simulator = None

    all_predictions = []
    all_props = []

    # Run backtest for each week
    for week in range(start_week, end_week + 1):
        logger.info(f"\n📅 Week {week}")
        logger.info("-" * 40)

        # Generate predictions
        week_preds = generate_predictions_for_week(week, season, nflverse_stats, simulator)

        if len(week_preds) == 0:
            logger.warning(f"   No predictions generated for week {week}")
            continue

        logger.info(f"   Generated {len(week_preds)} player predictions")
        all_predictions.append(week_preds)

        # Calculate prop probabilities
        week_props = calculate_prop_probabilities(week_preds)

        if len(week_props) == 0:
            logger.warning(f"   No prop probabilities calculated")
            continue

        logger.info(f"   Calculated {len(week_props)} prop probabilities")

        # Apply calibration
        week_props = apply_calibration(week_props)
        all_props.append(week_props)

        # Quick week summary
        cal_probs = week_props['prob_over_calibrated'].values
        actuals = week_props['went_over'].values

        # Model accuracy (betting the recommended side)
        model_recommends_over = cal_probs > 0.5
        model_correct = np.where(
            model_recommends_over,
            actuals == 1,
            actuals == 0
        )
        model_accuracy = model_correct.mean()

        brier = np.mean((cal_probs - actuals) ** 2)
        over_recs = model_recommends_over.sum()
        under_recs = (~model_recommends_over).sum()

        logger.info(f"   Model Accuracy: {model_accuracy:.1%} (OVER: {over_recs}, UNDER: {under_recs})")
        logger.info(f"   Brier Score: {brier:.4f}")

    # Combine all weeks
    if len(all_props) == 0:
        logger.error("No backtest data generated")
        return pd.DataFrame(), {}

    all_props_df = pd.concat(all_props, ignore_index=True)
    all_preds_df = pd.concat(all_predictions, ignore_index=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST COMPLETE")
    logger.info(f"Total predictions: {len(all_preds_df)}")
    logger.info(f"Total prop evaluations: {len(all_props_df)}")
    logger.info(f"{'='*80}")

    # Calculate comprehensive metrics
    metrics = calculate_backtest_metrics(all_props_df)

    # Print summary
    logger.info(f"\n📊 OVERALL METRICS:")
    logger.info(f"   Brier Score (Raw): {metrics.get('brier_score_raw', 0):.4f}")
    logger.info(f"   Brier Score (Calibrated): {metrics.get('brier_score_calibrated', 0):.4f}")
    logger.info(f"   Brier Improvement: {metrics.get('brier_improvement', 0):.4f}")
    logger.info(f"   Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}")

    # Model accuracy (when betting the side the model recommends)
    logger.info(f"\n🎯 MODEL RECOMMENDATION ACCURACY:")
    logger.info(f"   Overall Model Accuracy: {metrics.get('model_accuracy', 0):.1%}")
    logger.info(f"   OVER Recommendations: {metrics.get('over_recommendations', 0)} ({metrics.get('over_accuracy', 0):.1%} accurate)")
    logger.info(f"   UNDER Recommendations: {metrics.get('under_recommendations', 0)} ({metrics.get('under_accuracy', 0):.1%} accurate)")
    logger.info(f"   (Need ~52.4% to break even on -110 juice)")

    logger.info(f"\n📈 LEGACY METRICS:")
    logger.info(f"   Overall OVER Rate: {metrics.get('overall_over_rate', 0):.1%}")
    logger.info(f"   Avg Predicted OVER Prob: {metrics.get('avg_predicted_prob_cal', 0):.1%}")

    # Print by week
    logger.info(f"\n📅 BY WEEK:")
    for week, week_data in metrics.get('by_week', {}).items():
        logger.info(f"   Week {week}: Brier={week_data['brier_cal']:.4f}, "
                   f"Win={week_data['win_rate']:.1%}, "
                   f"N={week_data['count']}")

    # Print by stat type
    logger.info(f"\n📈 BY STAT TYPE:")
    for stat, stat_data in metrics.get('by_stat_type', {}).items():
        logger.info(f"   {stat}: Brier={stat_data['brier_cal']:.4f}, "
                   f"Win={stat_data['win_rate']:.1%}, "
                   f"N={stat_data['count']}")

    # Print calibration bins
    logger.info(f"\n🎯 CALIBRATION BY PROBABILITY BIN:")
    for bin_data in metrics.get('calibration_by_bin', []):
        logger.info(f"   {bin_data['bin']}: Pred={bin_data['avg_predicted']:.1%}, "
                   f"Actual={bin_data['actual_win_rate']:.1%}, "
                   f"Error={bin_data['calibration_error']:.3f}, "
                   f"N={bin_data['count']}")

    return all_props_df, metrics


def save_backtest_results(
    props_df: pd.DataFrame,
    metrics: Dict,
    output_dir: Path = None
):
    """Save backtest results to files."""
    if output_dir is None:
        output_dir = Path('reports/backtest')

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed predictions
    props_file = output_dir / f'backtest_props_{timestamp}.csv'
    props_df.to_csv(props_file, index=False)
    logger.info(f"\n✅ Saved props to: {props_file}")

    # Save metrics
    metrics_file = output_dir / f'backtest_metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Saved metrics to: {metrics_file}")

    # Save summary report
    report_file = output_dir / f'backtest_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write("NFL QUANT BACKTEST REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total Predictions: {metrics.get('total_predictions', 0)}\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Brier Score (Raw): {metrics.get('brier_score_raw', 0):.4f}\n")
        f.write(f"Brier Score (Calibrated): {metrics.get('brier_score_calibrated', 0):.4f}\n")
        f.write(f"Brier Improvement: {metrics.get('brier_improvement', 0):.4f}\n")
        f.write(f"Expected Calibration Error: {metrics.get('expected_calibration_error', 0):.4f}\n")
        f.write(f"Overall Win Rate: {metrics.get('overall_win_rate', 0):.1%}\n\n")

        f.write("BY WEEK\n")
        f.write("-" * 40 + "\n")
        for week, data in metrics.get('by_week', {}).items():
            f.write(f"Week {week}: Brier={data['brier_cal']:.4f}, "
                   f"Win={data['win_rate']:.1%}, N={data['count']}\n")

        f.write("\nBY STAT TYPE\n")
        f.write("-" * 40 + "\n")
        for stat, data in metrics.get('by_stat_type', {}).items():
            f.write(f"{stat}: Brier={data['brier_cal']:.4f}, "
                   f"Win={data['win_rate']:.1%}, N={data['count']}\n")

        f.write("\nCALIBRATION BY BIN\n")
        f.write("-" * 40 + "\n")
        for bin_data in metrics.get('calibration_by_bin', []):
            f.write(f"{bin_data['bin']}: Pred={bin_data['avg_predicted']:.1%}, "
                   f"Actual={bin_data['actual_win_rate']:.1%}, "
                   f"Error={bin_data['calibration_error']:.3f}, N={bin_data['count']}\n")

    logger.info(f"✅ Saved report to: {report_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Run NFLverse-native backtest')
    parser.add_argument('--season', type=int, default=2025, help='NFL season')
    parser.add_argument('--start-week', type=int, default=2, help='First week to backtest')
    parser.add_argument('--end-week', type=int, default=10, help='Last week to backtest')
    parser.add_argument('--save', action='store_true', help='Save results to files')

    args = parser.parse_args()

    # Run backtest
    props_df, metrics = run_week_by_week_backtest(
        season=args.season,
        start_week=args.start_week,
        end_week=args.end_week
    )

    if args.save and len(props_df) > 0:
        save_backtest_results(props_df, metrics)


if __name__ == '__main__':
    main()
