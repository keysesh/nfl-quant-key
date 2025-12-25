#!/usr/bin/env python3
"""
Train ATTD Ensemble Model

Trains the Anytime TD ensemble model with position-specific sub-models.
Combines Poisson-based P(TDs >= 1) with Logistic Regression.

Key Features:
- Position-specific models (QB, RB, WR, TE)
- DraftKings ATTD rules compliance
- Walk-forward validation
- Ensemble weight optimization

Usage:
    python scripts/train/train_attd_ensemble.py
    python scripts/train/train_attd_ensemble.py --position RB
    python scripts/train/train_attd_ensemble.py --validate-only
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging

from nfl_quant.models.attd_ensemble import ATTDEnsemble, ATTD_FEATURES
from nfl_quant.features.rz_td_conversion import load_and_compute_rz_td_rates, merge_rz_td_rates
from nfl_quant.features.goal_line_detector import load_and_compute_goal_line_roles, merge_goal_line_features
from nfl_quant.features.rz_opportunity import compute_rz_opportunity_features, merge_rz_opportunity
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """Load training data from weekly stats (construct ATTD outcomes)."""
    print("Loading training data...")

    # Load weekly stats (has actual TD counts)
    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'

    if stats_path.exists():
        print(f"  Loading from: {stats_path}")
        df = pd.read_parquet(stats_path)
    else:
        # Fallback to CSV
        csv_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'
        print(f"  Loading from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)

    print(f"  Total rows: {len(df):,}")

    # Filter to relevant positions
    positions = ['QB', 'RB', 'WR', 'TE']
    df = df[df['position'].isin(positions)].copy()

    print(f"  Rows for {positions}: {len(df):,}")
    print(f"  Position distribution: {df['position'].value_counts().to_dict()}")

    return df


def compute_trailing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing TD features with EWMA."""
    print("Computing trailing features...")

    df = df.copy()
    df = df.sort_values(['player_id', 'season', 'week'])

    ewma_span = 4

    # Trailing TD rates
    td_cols = ['rushing_tds', 'receiving_tds']
    for col in td_cols:
        if col in df.columns:
            trailing_col = f'trailing_{col}'
            df[trailing_col] = (
                df.groupby('player_id')[col]
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )
            df[trailing_col] = df[trailing_col].fillna(0)

    # Volume indicators
    volume_cols = ['targets', 'carries', 'receptions']
    for col in volume_cols:
        if col in df.columns:
            trailing_col = f'trailing_{col}'
            df[trailing_col] = (
                df.groupby('player_id')[col]
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )
            df[trailing_col] = df[trailing_col].fillna(0)

    # Snap share
    if 'offense_pct' in df.columns:
        df['snap_share'] = df['offense_pct'] / 100.0
    elif 'snap_share' not in df.columns:
        df['snap_share'] = 0.5  # Default

    print(f"  Added trailing features")

    return df


def add_rz_and_gl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add red zone and goal-line features."""
    print("Adding RZ and GL features...")

    seasons = df['season'].unique()

    for season in seasons:
        try:
            # RZ Opportunity features (KEY: trailing_rz_carries, trailing_rz_targets)
            # These are the most predictive features for ATTD
            rz_opps = compute_rz_opportunity_features(int(season))
            if not rz_opps.empty:
                df = merge_rz_opportunity(df, rz_opps)
                print(f"  Season {season}: merged RZ opportunity features for {rz_opps['player_id'].nunique()} players")

            # RZ TD rates (efficiency metrics)
            rz_rates = load_and_compute_rz_td_rates(int(season))
            if not rz_rates.empty:
                df = merge_rz_td_rates(df, rz_rates, player_id_col='player_id')

            # Goal-line roles
            gl_roles = load_and_compute_goal_line_roles(int(season))
            if not gl_roles.empty:
                df = merge_goal_line_features(df, gl_roles, player_id_col='player_id')

        except Exception as e:
            print(f"  Season {season} RZ/GL features error: {e}")

    # Fill defaults for RZ opportunity features
    rz_opp_cols = ['trailing_rz_carries', 'trailing_rz_targets', 'trailing_rz_touches']
    for col in rz_opp_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Fill defaults for RZ TD rates
    rz_cols = ['rz_td_per_carry', 'rz_td_per_target', 'rz_td_per_snap']
    for col in rz_cols:
        if col not in df.columns:
            df[col] = 0.10

    gl_cols = ['gl_carry_share', 'gl_target_share', 'gl_opportunity_share']
    for col in gl_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Log feature coverage
    for col in rz_opp_cols:
        non_zero = (df[col] > 0).sum()
        print(f"  {col}: {non_zero}/{len(df)} rows with data ({non_zero/len(df)*100:.1f}%)")

    print(f"  Added RZ and GL features")

    return df


def add_vegas_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Vegas game environment features."""
    # Try to merge from schedules or use defaults
    if 'vegas_total' not in df.columns:
        df['vegas_total'] = 45.0
    else:
        df['vegas_total'] = df['vegas_total'].fillna(45.0)

    if 'vegas_spread' not in df.columns:
        df['vegas_spread'] = 0.0
    else:
        df['vegas_spread'] = df['vegas_spread'].fillna(0.0)

    return df


def add_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add opponent RZ defense features."""
    if 'opp_rz_td_allowed' not in df.columns:
        df['opp_rz_td_allowed'] = 0.20  # League average

    return df


def add_global_week(df: pd.DataFrame) -> pd.DataFrame:
    """Add global_week for ordering."""
    if 'global_week' not in df.columns:
        df['global_week'] = (df['season'] - 2023) * 18 + df['week']
    return df


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full training data preparation pipeline."""
    df = compute_trailing_features(df)
    df = add_rz_and_gl_features(df)
    df = add_vegas_features(df)
    df = add_opponent_features(df)
    df = add_global_week(df)

    # Create scored_td column (any TD scored)
    df['rushing_tds'] = df.get('rushing_tds', 0).fillna(0)
    df['receiving_tds'] = df.get('receiving_tds', 0).fillna(0)
    df['scored_td'] = ((df['rushing_tds'] >= 1) | (df['receiving_tds'] >= 1)).astype(int)

    print(f"\nData prepared: {len(df):,} rows")
    print(f"  TD rate: {df['scored_td'].mean():.1%}")

    return df


def train_position(
    ensemble: ATTDEnsemble,
    df: pd.DataFrame,
    position: str,
) -> dict:
    """Train ATTD ensemble for a single position."""
    print(f"\n{'='*60}")
    print(f"Training ATTD Ensemble for: {position}")
    print(f"{'='*60}")

    # Filter to position
    pos_df = df[df['position'] == position].copy()
    print(f"  Rows: {len(pos_df):,}")

    if len(pos_df) < 100:
        print(f"  Insufficient data: {len(pos_df)} samples")
        return None

    # Train model
    try:
        metrics = ensemble.train_position(pos_df, position)
    except Exception as e:
        print(f"  Training error: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Tune ensemble weights using validation split
    val_split = int(len(pos_df) * 0.2)
    val_data = pos_df.tail(val_split)
    if len(val_data) >= 20:
        ensemble.tune_ensemble_weights(val_data, position)

    return metrics


def validate_position(
    ensemble: ATTDEnsemble,
    df: pd.DataFrame,
    position: str,
    n_val_weeks: int = 10,
) -> pd.DataFrame:
    """Walk-forward validation for a position."""
    pos_df = df[df['position'] == position].copy()
    return ensemble.walk_forward_validate(pos_df, position, n_val_weeks)


def main():
    parser = argparse.ArgumentParser(description="Train ATTD Ensemble Model")
    parser.add_argument('--position', type=str, help='Train single position only')
    parser.add_argument('--validate-only', action='store_true', help='Run validation only')
    args = parser.parse_args()

    print("="*80)
    print("ATTD ENSEMBLE TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load and prepare data
    df = load_training_data()

    if len(df) == 0:
        print("No training data found")
        return

    df = prepare_training_data(df)

    # Initialize ensemble
    ensemble = ATTDEnsemble()

    # Determine positions to train
    if args.position:
        positions = [args.position]
    else:
        positions = ['QB', 'RB', 'WR', 'TE']

    print(f"\nPositions to train: {positions}")

    # Train each position
    all_metrics = {}
    all_validation = {}

    for position in positions:
        if not args.validate_only:
            metrics = train_position(ensemble, df, position)
            if metrics:
                all_metrics[position] = metrics

        # Validate
        val_results = validate_position(ensemble, df, position)
        if len(val_results) > 0:
            all_validation[position] = val_results

    # Save model
    if all_metrics and not args.validate_only:
        ensemble.save()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Trained positions: {list(all_metrics.keys())}")

        # Summary table
        print("\n" + "-"*70)
        print(f"{'Position':<10} {'Samples':<10} {'TD Rate':<10} {'Accuracy':<10} {'Brier':<10}")
        print("-"*70)
        for position, metrics in all_metrics.items():
            print(
                f"{position:<10} "
                f"{metrics['n_samples']:<10} "
                f"{metrics['td_rate']:.1%}     "
                f"{metrics['accuracy']:.1%}     "
                f"{metrics['brier_score']:.3f}"
            )

        # Print ensemble weights
        print("\nEnsemble Weights:")
        for position in all_metrics.keys():
            w_poisson, w_logistic = ensemble.ensemble_weights.get(position, (0.5, 0.5))
            print(f"  {position}: Poisson={w_poisson:.1f}, Logistic={w_logistic:.1f}")

    # Print validation summary
    if all_validation:
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        for position, val_df in all_validation.items():
            if len(val_df) == 0:
                continue

            print(f"\n{position}:")
            for thresh in [0.50, 0.55, 0.60]:
                high_conf = val_df[val_df['p_attd'] >= thresh]
                if len(high_conf) >= 5:
                    hits = high_conf['scored_td'].sum()
                    total = len(high_conf)
                    hit_rate = hits / total
                    roi = (hit_rate * 0.909) - (1 - hit_rate)
                    print(f"  @ {thresh:.0%}: N={total:4d}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

    print("\nDone.")


if __name__ == '__main__':
    main()
