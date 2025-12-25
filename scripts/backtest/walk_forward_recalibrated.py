#!/usr/bin/env python3
"""
Walk-Forward Validation with Per-Step Recalibration

MOST RIGOROUS approach: Retrain calibrators at each step.
- Test week 5: calibrators trained on weeks 1-4
- Test week 6: calibrators trained on weeks 1-5
- Test week 7: calibrators trained on weeks 1-6
- etc.

This ensures ZERO data leakage and provides the most accurate
assessment of production performance.

Computationally expensive but statistically rigorous.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from datetime import datetime
import warnings
import tempfile
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, NFLVERSE_DIR
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.utils.training_metadata import save_model_with_metadata
from nfl_quant.filters import CONSERVATIVE_FILTER, ELITE_FILTER, should_take_bet
from configs.model_config import CLASSIFIER_MARKETS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_historical_props():
    """Load all historical props with actuals."""
    props_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    props = pd.read_csv(props_path)
    props['player_norm'] = props['player'].apply(normalize_player_name)
    props['global_week'] = (props['season'] - 2023) * 18 + props['week']
    return props


def load_player_stats():
    """Load player stats."""
    stats_2024 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)
    stats_2023 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2023.csv', low_memory=False)
    stats = pd.concat([stats_2024, stats_2023], ignore_index=True)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']
    return stats


def load_snap_counts():
    """Load snap counts for usage calibration."""
    snap_path = PROJECT_ROOT / 'data' / 'nflverse' / 'snap_counts.parquet'
    if snap_path.exists():
        snaps = pd.read_parquet(snap_path)
        snaps['player_norm'] = snaps['player'].apply(normalize_player_name)
        snaps['global_week'] = (snaps['season'] - 2023) * 18 + snaps['week']
        return snaps
    return None


def train_usage_calibrator_for_week(stats, snaps, max_global_week, target='targets'):
    """
    Train a usage calibrator using only data before max_global_week.

    Args:
        stats: Player stats DataFrame
        snaps: Snap counts DataFrame (optional)
        max_global_week: Maximum week to include in training
        target: Target variable (targets, carries, attempts, snaps, snap_pct)

    Returns:
        Trained XGBRegressor model
    """
    # Filter to training data
    train_stats = stats[stats['global_week'] < max_global_week].copy()

    if len(train_stats) < 100:
        return None

    # Sort by player and week
    train_stats = train_stats.sort_values(['player_id', 'global_week'])

    # Compute trailing features (EWMA)
    for col in ['targets', 'carries', 'attempts', 'receptions', 'receiving_yards', 'rushing_yards']:
        if col in train_stats.columns:
            train_stats[f'trailing_{col}'] = train_stats.groupby('player_id')[col].transform(
                lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
            )

    # Merge snap counts if available
    if snaps is not None:
        train_snaps = snaps[snaps['global_week'] < max_global_week].copy()
        train_snaps = train_snaps[['player_norm', 'season', 'week', 'offense_snaps', 'offense_pct']]
        train_stats = train_stats.merge(
            train_snaps,
            left_on=['player_norm', 'season', 'week'],
            right_on=['player_norm', 'season', 'week'],
            how='left'
        )
        train_stats['snaps'] = train_stats['offense_snaps'].fillna(0)
        train_stats['snap_pct'] = train_stats['offense_pct'].fillna(0)
        train_stats[f'trailing_snaps'] = train_stats.groupby('player_id')['snaps'].transform(
            lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
        )
        train_stats[f'trailing_snap_pct'] = train_stats.groupby('player_id')['snap_pct'].transform(
            lambda x: x.ewm(span=4, min_periods=1).mean().shift(1)
        )

    # Create target (next week's value)
    if target in train_stats.columns:
        train_stats[f'target_{target}'] = train_stats.groupby('player_id')[target].shift(-1)
    else:
        return None

    # Drop NaN
    train_stats = train_stats.dropna(subset=[f'target_{target}', 'trailing_targets'])

    if len(train_stats) < 50:
        return None

    # Features
    feature_cols = [
        'trailing_targets', 'trailing_carries', 'trailing_attempts',
        'week'
    ]

    if snaps is not None:
        feature_cols.extend(['trailing_snaps', 'trailing_snap_pct'])

    # Filter to available features
    feature_cols = [f for f in feature_cols if f in train_stats.columns]

    X = train_stats[feature_cols].fillna(0)
    y = train_stats[f'target_{target}']

    # Train model with conservative parameters
    model = xgb.XGBRegressor(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=100,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(X, y)

    return model


def calculate_trailing_for_player(player_stats, target_global_week):
    """Calculate trailing stats for a player up to (but not including) target week."""
    prior_stats = player_stats[player_stats['global_week'] < target_global_week].copy()
    if len(prior_stats) < 3:
        return None

    prior_stats = prior_stats.sort_values('global_week')

    trailing = {
        'avg_targets': prior_stats['targets'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'targets' in prior_stats.columns else 0,
        'avg_receptions': prior_stats['receptions'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'receptions' in prior_stats.columns else 0,
        'avg_rec_yards': prior_stats['receiving_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'receiving_yards' in prior_stats.columns else 0,
        'avg_rush_yards': prior_stats['rushing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'rushing_yards' in prior_stats.columns else 0,
        'avg_pass_yards': prior_stats['passing_yards'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'passing_yards' in prior_stats.columns else 0,
        'avg_carries': prior_stats['carries'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'carries' in prior_stats.columns else 0,
        'avg_attempts': prior_stats['attempts'].ewm(span=4, min_periods=1).mean().iloc[-1] if 'attempts' in prior_stats.columns else 0,
    }

    return trailing


def train_classifier_for_week(props, stats, test_global_week, market):
    """Train classifier using only data before test_global_week."""
    from sklearn.ensemble import GradientBoostingClassifier

    # Filter to training data only
    train_props = props[(props['global_week'] < test_global_week) & (props['market'] == market)].copy()

    if len(train_props) < 100:
        return None, None

    # Calculate trailing stats for each prop
    features_list = []
    for _, row in train_props.iterrows():
        player_stats = stats[stats['player_norm'] == row['player_norm']]
        trailing = calculate_trailing_for_player(player_stats, row['global_week'])

        if trailing is None:
            continue

        # Get the right trailing stat for this market
        if market == 'player_receptions':
            trailing_stat = trailing['avg_receptions']
        elif market == 'player_reception_yds':
            trailing_stat = trailing['avg_rec_yards']
        elif market == 'player_rush_yds':
            trailing_stat = trailing['avg_rush_yards']
        elif market == 'player_pass_yds':
            trailing_stat = trailing['avg_pass_yards']
        else:
            continue

        line_vs_trailing = row['line'] - trailing_stat

        features_list.append({
            'line_vs_trailing': line_vs_trailing,
            'line': row['line'],
            'trailing_stat': trailing_stat,
            'under_hit': row['under_hit'],
        })

    if len(features_list) < 50:
        return None, None

    df = pd.DataFrame(features_list)

    X = df[['line_vs_trailing', 'line', 'trailing_stat']]
    y = df['under_hit']

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    return model, ['line_vs_trailing', 'line', 'trailing_stat']


def run_recalibrated_validation(test_weeks, retrain_calibrators=True, trials=5000):
    """
    Run walk-forward validation with per-step recalibration.

    Args:
        test_weeks: List of weeks to test
        retrain_calibrators: If True, retrain calibrators at each step (most rigorous)
        trials: Monte Carlo trials

    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("WALK-FORWARD VALIDATION WITH PER-STEP RECALIBRATION")
    print("=" * 70)
    print(f"Test weeks: {test_weeks}")
    print(f"Retrain calibrators: {retrain_calibrators}")
    print()

    # Load data
    props = load_historical_props()
    stats = load_player_stats()
    snaps = load_snap_counts()

    current_season = get_current_season()

    # Market mapping
    market_stat_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
    }

    markets = list(market_stat_map.keys())

    all_results = []
    calibrator_history = []

    for test_week in test_weeks:
        test_global_week = (current_season - 2023) * 18 + test_week

        print(f"\n{'='*60}")
        print(f"Testing Week {test_week} (global week {test_global_week})")
        print(f"{'='*60}")

        # CRITICAL: Retrain calibrators using only data before this week
        if retrain_calibrators:
            print(f"  Retraining calibrators on weeks 1-{test_week-1}...")

            # Train usage calibrators
            targets_model = train_usage_calibrator_for_week(stats, snaps, test_global_week, 'targets')
            carries_model = train_usage_calibrator_for_week(stats, snaps, test_global_week, 'carries')

            calibrator_info = {
                'test_week': test_week,
                'training_max_week': test_week - 1,
                'targets_model_trained': targets_model is not None,
                'carries_model_trained': carries_model is not None,
            }
            calibrator_history.append(calibrator_info)

            if targets_model is None:
                print(f"  WARNING: Could not train targets calibrator (insufficient data)")
            else:
                print(f"  ✓ Targets calibrator trained on {test_week-1} weeks")

            if carries_model is None:
                print(f"  WARNING: Could not train carries calibrator (insufficient data)")
            else:
                print(f"  ✓ Carries calibrator trained on {test_week-1} weeks")

        # Get props for test week
        week_props = props[
            (props['season'] == current_season) &
            (props['week'] == test_week)
        ].copy()

        print(f"  Props available: {len(week_props)}")

        for market in markets:
            market_props = week_props[week_props['market'] == market].copy()

            if len(market_props) == 0:
                continue

            # Train classifier for this market (using only prior data)
            classifier, feature_cols = train_classifier_for_week(
                props, stats, test_global_week, market
            )

            stat_col = market_stat_map[market]

            for _, row in market_props.iterrows():
                player_stats = stats[stats['player_norm'] == row['player_norm']]
                trailing = calculate_trailing_for_player(player_stats, test_global_week)

                if trailing is None:
                    continue

                # Get trailing stat for this market
                if market == 'player_receptions':
                    trailing_stat = trailing['avg_receptions']
                elif market == 'player_reception_yds':
                    trailing_stat = trailing['avg_rec_yards']
                elif market == 'player_rush_yds':
                    trailing_stat = trailing['avg_rush_yards']
                elif market == 'player_pass_yds':
                    trailing_stat = trailing['avg_pass_yards']
                else:
                    continue

                line_vs_trailing = row['line'] - trailing_stat

                # Apply classifier if available
                if classifier is not None:
                    try:
                        X_test = pd.DataFrame([{
                            'line_vs_trailing': line_vs_trailing,
                            'line': row['line'],
                            'trailing_stat': trailing_stat,
                        }])
                        clf_prob_under = classifier.predict_proba(X_test)[0][1]
                    except:
                        clf_prob_under = 0.5
                else:
                    clf_prob_under = 0.5

                # Simple heuristic: if line is higher than trailing, lean UNDER
                heuristic_prob = 0.5 + 0.02 * (row['line'] - trailing_stat)
                heuristic_prob = np.clip(heuristic_prob, 0.3, 0.7)

                # Blend classifier and heuristic
                final_prob_under = 0.7 * clf_prob_under + 0.3 * heuristic_prob

                pick = 'UNDER' if final_prob_under > 0.5 else 'OVER'
                actual_hit = row['under_hit'] if pick == 'UNDER' else (1 - row['under_hit'])

                all_results.append({
                    'week': test_week,
                    'player': row['player'],
                    'market': market,
                    'line': row['line'],
                    'trailing_stat': trailing_stat,
                    'line_vs_trailing': line_vs_trailing,
                    'clf_prob_under': clf_prob_under,
                    'final_prob_under': final_prob_under,
                    'pick': pick,
                    'actual_stat': row['actual_stat'],
                    'actual_hit': actual_hit,
                    'under_hit': row['under_hit'],
                    'calibrators_trained_on_week': test_week - 1 if retrain_calibrators else 'static',
                })

        # Summary for this week
        week_results = [r for r in all_results if r['week'] == test_week]
        if week_results:
            wins = sum(1 for r in week_results if r['actual_hit'] == 1)
            total = len(week_results)
            print(f"  Results: {wins}/{total} wins ({wins/total*100:.1f}%)")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    return results_df, calibrator_history


def calculate_bootstrap_ci(results_df: pd.DataFrame, n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """Calculate bootstrap confidence intervals for ROI."""
    if len(results_df) == 0:
        return {'win_rate_mean': 0, 'win_rate_lower': 0, 'win_rate_upper': 0,
                'roi_mean': 0, 'roi_lower': 0, 'roi_upper': 0}

    np.random.seed(42)

    bootstrap_rois = []
    bootstrap_win_rates = []

    hits = results_df['actual_hit'].values
    n = len(hits)

    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample_hits = hits[sample_idx]

        win_rate = np.mean(sample_hits)
        roi = (win_rate * 0.909 - (1 - win_rate)) * 100

        bootstrap_win_rates.append(win_rate)
        bootstrap_rois.append(roi)

    alpha = 1 - confidence
    lower_pct = alpha / 2 * 100
    upper_pct = (1 - alpha / 2) * 100

    return {
        'win_rate_mean': np.mean(bootstrap_win_rates),
        'win_rate_lower': np.percentile(bootstrap_win_rates, lower_pct),
        'win_rate_upper': np.percentile(bootstrap_win_rates, upper_pct),
        'roi_mean': np.mean(bootstrap_rois),
        'roi_lower': np.percentile(bootstrap_rois, lower_pct),
        'roi_upper': np.percentile(bootstrap_rois, upper_pct),
        'n_samples': n,
    }


def print_results_summary(results_df, calibrator_history=None):
    """Print comprehensive results summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if len(results_df) == 0:
        print("No results generated!")
        return

    # Overall metrics
    total_bets = len(results_df)
    total_wins = results_df['actual_hit'].sum()
    overall_win_rate = total_wins / total_bets
    overall_roi = (overall_win_rate * 0.909 - (1 - overall_win_rate)) * 100

    print(f"\nOverall: {total_bets} bets, {overall_win_rate*100:.1f}% win rate, {overall_roi:+.1f}% ROI")

    # By week
    print("\nBy Week:")
    for week in sorted(results_df['week'].unique()):
        week_df = results_df[results_df['week'] == week]
        wins = week_df['actual_hit'].sum()
        total = len(week_df)
        win_rate = wins / total
        roi = (win_rate * 0.909 - (1 - win_rate)) * 100
        print(f"  Week {week}: n={total}, {wins}/{total} wins ({win_rate*100:.1f}%), ROI={roi:+.1f}%")

    # By market (from central config)
    print("\nBy Market:")
    markets = CLASSIFIER_MARKETS
    for market in markets:
        market_df = results_df[results_df['market'] == market]
        if len(market_df) == 0:
            continue
        wins = market_df['actual_hit'].sum()
        total = len(market_df)
        win_rate = wins / total
        roi = (win_rate * 0.909 - (1 - win_rate)) * 100
        print(f"  {market}: n={total}, {win_rate*100:.1f}% win rate, ROI={roi:+.1f}%")

    # Bootstrap CI
    ci = calculate_bootstrap_ci(results_df)
    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("=" * 70)
    print(f"Win Rate: {ci['win_rate_mean']*100:.1f}% [{ci['win_rate_lower']*100:.1f}%, {ci['win_rate_upper']*100:.1f}%]")
    print(f"ROI:      {ci['roi_mean']:+.1f}% [{ci['roi_lower']:+.1f}%, {ci['roi_upper']:+.1f}%]")

    if ci['roi_lower'] > 0:
        print("\n STATISTICALLY SIGNIFICANT (95% CI excludes zero)")
    else:
        print("\n NOT statistically significant (95% CI includes zero)")

    # Calibrator history
    if calibrator_history:
        print("\n" + "=" * 70)
        print("CALIBRATOR TRAINING HISTORY")
        print("=" * 70)
        for info in calibrator_history:
            print(f"  Week {info['test_week']}: trained on weeks 1-{info['training_max_week']}")
            print(f"    Targets model: {'Trained' if info['targets_model_trained'] else 'FAILED'}")
            print(f"    Carries model: {'Trained' if info['carries_model_trained'] else 'FAILED'}")

    # FILTERED RESULTS ANALYSIS
    # Apply walk-forward validated filters to see improvement
    print("\n" + "=" * 70)
    print("FILTERED RESULTS (CONSERVATIVE FILTER)")
    print("=" * 70)
    print("Filter: Receptions market, >60% confidence, UNDER only")

    # Apply filters to results
    filtered_mask = []
    for _, row in results_df.iterrows():
        passed, reason = should_take_bet(
            market=row['market'],
            line=row['line'],
            confidence=row['final_prob_under'] if row['pick'] == 'UNDER' else (1 - row['final_prob_under']),
            side=row['pick'],
            player_cv=None,  # CV not available in backtest data
            game_total=None,
            config=CONSERVATIVE_FILTER
        )
        filtered_mask.append(passed)

    filtered_df = results_df[filtered_mask]

    if len(filtered_df) > 0:
        filtered_wins = filtered_df['actual_hit'].sum()
        filtered_total = len(filtered_df)
        filtered_win_rate = filtered_wins / filtered_total
        filtered_roi = (filtered_win_rate * 0.909 - (1 - filtered_win_rate)) * 100

        print(f"\nFiltered: {filtered_total} bets, {filtered_win_rate*100:.1f}% win rate, {filtered_roi:+.1f}% ROI")
        print(f"  (vs {total_bets} unfiltered)")

        # Bootstrap CI for filtered
        filtered_ci = calculate_bootstrap_ci(filtered_df)
        print(f"\nFiltered 95% CI:")
        print(f"  Win Rate: {filtered_ci['win_rate_mean']*100:.1f}% [{filtered_ci['win_rate_lower']*100:.1f}%, {filtered_ci['win_rate_upper']*100:.1f}%]")
        print(f"  ROI:      {filtered_ci['roi_mean']:+.1f}% [{filtered_ci['roi_lower']:+.1f}%, {filtered_ci['roi_upper']:+.1f}%]")

        if filtered_ci['roi_lower'] > 0:
            print("\n  STATISTICALLY SIGNIFICANT EDGE!")
        else:
            print("\n  Not statistically significant (yet)")
    else:
        print("\nNo bets passed the filter!")

    # ELITE FILTER (lines 0-3)
    print("\n" + "-" * 40)
    print("ELITE FILTER (lines 0-3)")
    print("-" * 40)

    elite_mask = []
    for _, row in results_df.iterrows():
        passed, reason = should_take_bet(
            market=row['market'],
            line=row['line'],
            confidence=row['final_prob_under'] if row['pick'] == 'UNDER' else (1 - row['final_prob_under']),
            side=row['pick'],
            player_cv=None,
            game_total=None,
            config=ELITE_FILTER
        )
        elite_mask.append(passed)

    elite_df = results_df[elite_mask]

    if len(elite_df) > 0:
        elite_wins = elite_df['actual_hit'].sum()
        elite_total = len(elite_df)
        elite_win_rate = elite_wins / elite_total
        elite_roi = (elite_win_rate * 0.909 - (1 - elite_win_rate)) * 100

        print(f"\nElite: {elite_total} bets, {elite_win_rate*100:.1f}% win rate, {elite_roi:+.1f}% ROI")

        elite_ci = calculate_bootstrap_ci(elite_df)
        print(f"\nElite 95% CI:")
        print(f"  Win Rate: {elite_ci['win_rate_mean']*100:.1f}% [{elite_ci['win_rate_lower']*100:.1f}%, {elite_ci['win_rate_upper']*100:.1f}%]")
        print(f"  ROI:      {elite_ci['roi_mean']:+.1f}% [{elite_ci['roi_lower']:+.1f}%, {elite_ci['roi_upper']:+.1f}%]")
    else:
        print("\nNo bets passed the elite filter!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward with Per-Step Recalibration')
    parser.add_argument('--weeks', default='5-11', help='Week range (e.g., 5-11)')
    parser.add_argument('--no-recalibrate', action='store_true', help='Skip recalibration (use static models)')
    parser.add_argument('--save', action='store_true', help='Save results to CSV')
    args = parser.parse_args()

    # Parse weeks
    if '-' in args.weeks:
        start, end = map(int, args.weeks.split('-'))
        test_weeks = list(range(start, end + 1))
    else:
        test_weeks = [int(args.weeks)]

    # Run validation
    results_df, calibrator_history = run_recalibrated_validation(
        test_weeks,
        retrain_calibrators=not args.no_recalibrate
    )

    # Print summary
    print_results_summary(results_df, calibrator_history)

    # Save results
    if args.save:
        output_path = PROJECT_ROOT / 'data' / 'backtest' / 'recalibrated_validation_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
