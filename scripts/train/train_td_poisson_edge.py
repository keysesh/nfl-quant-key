#!/usr/bin/env python3
"""
Train TD Poisson Edge Model

Trains Poisson regression models for TD props using the TDPoissonEdge class.
This is separate from XGBoost-based edges because TDs are count data.

Key Features:
- Poisson regression (appropriate for count data: 0, 1, 2, 3...)
- Red zone features (85% of rushing TDs come from red zone)
- Walk-forward validation (no leakage)
- Separate models per TD market

Markets:
- player_pass_tds: QB passing touchdowns
- player_rush_tds: Running back rushing touchdowns
- player_rec_tds: Receiver touchdowns

Usage:
    python scripts/train/train_td_poisson_edge.py
    python scripts/train/train_td_poisson_edge.py --market player_pass_tds
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
from scipy.stats import poisson

from nfl_quant.models.td_poisson_edge import TDPoissonEdge, TD_MARKETS, TD_FEATURES
from nfl_quant.features.red_zone_features import (
    compute_red_zone_features,
    merge_red_zone_features,
)
from nfl_quant.features.rz_td_conversion import load_and_compute_rz_td_rates, merge_rz_td_rates
from nfl_quant.features.goal_line_detector import load_and_compute_goal_line_roles, merge_goal_line_features
from nfl_quant.features.defensive_metrics import DefensiveMetricsExtractor
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import DATA_DIR, MODELS_DIR


def load_training_data() -> pd.DataFrame:
    """Load historical odds/actuals data for TD markets."""
    print("Loading training data...")

    # Try enriched data first
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'

    if enriched_path.exists():
        print(f"  Loading from: {enriched_path}")
        df = pd.read_csv(enriched_path, low_memory=False)
    else:
        # Fallback
        combined_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
        print(f"  Loading from: {combined_path}")
        df = pd.read_csv(combined_path, low_memory=False)

    print(f"  Total rows: {len(df):,}")

    # Filter to TD markets
    td_markets = ['player_pass_tds', 'player_rush_tds', 'player_rec_tds']
    df_td = df[df['market'].isin(td_markets)].copy()

    print(f"  TD market rows: {len(df_td):,}")
    print(f"  Markets: {df_td['market'].value_counts().to_dict()}")

    return df_td


def load_player_stats() -> pd.DataFrame:
    """Load player stats for feature extraction."""
    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'

    if stats_path.exists():
        stats = pd.read_parquet(stats_path)
    else:
        csv_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'
        stats = pd.read_csv(csv_path, low_memory=False)

    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    return stats


def compute_td_trailing_features(
    df: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute trailing TD-specific features.

    Features:
    - trailing_passing_tds: EWMA of passing TDs
    - trailing_rushing_tds: EWMA of rushing TDs
    - trailing_receiving_tds: EWMA of receiving TDs
    - trailing_passing_yards: Volume indicator
    - trailing_carries: Volume indicator
    - trailing_targets: Volume indicator
    """
    print("Computing trailing TD features...")

    df = df.copy()
    stats = stats.copy()

    # Normalize player names
    if 'player_norm' not in df.columns:
        df['player_norm'] = df['player'].apply(normalize_player_name)

    # Sort stats
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    # Compute trailing stats (EWMA with shift to prevent leakage)
    ewma_span = 4  # 4-game EWMA

    # Passing TDs (for QBs)
    if 'passing_tds' in stats.columns:
        stats['trailing_passing_tds'] = (
            stats.groupby('player_norm')['passing_tds']
            .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
        )

    # Rushing TDs
    if 'rushing_tds' in stats.columns:
        stats['trailing_rushing_tds'] = (
            stats.groupby('player_norm')['rushing_tds']
            .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
        )

    # Receiving TDs
    if 'receiving_tds' in stats.columns:
        stats['trailing_receiving_tds'] = (
            stats.groupby('player_norm')['receiving_tds']
            .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
        )

    # Volume indicators
    volume_cols = {
        'passing_yards': 'trailing_passing_yards',
        'carries': 'trailing_carries',
        'targets': 'trailing_targets',
    }

    for src, dst in volume_cols.items():
        if src in stats.columns:
            stats[dst] = (
                stats.groupby('player_norm')[src]
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean())
            )

    # Merge trailing features into main df
    trailing_cols = [
        'trailing_passing_tds', 'trailing_rushing_tds', 'trailing_receiving_tds',
        'trailing_passing_yards', 'trailing_carries', 'trailing_targets',
    ]
    available_cols = ['player_norm', 'season', 'week'] + [c for c in trailing_cols if c in stats.columns]

    df = df.merge(
        stats[available_cols].drop_duplicates(),
        on=['player_norm', 'season', 'week'],
        how='left',
        suffixes=('', '_stats')
    )

    # Handle duplicate columns
    for col in trailing_cols:
        if f'{col}_stats' in df.columns:
            df[col] = df[col].fillna(df[f'{col}_stats'])
            df.drop(columns=[f'{col}_stats'], inplace=True)

    print(f"  Added {len([c for c in trailing_cols if c in df.columns])} trailing features")

    return df


def add_vegas_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Vegas game environment features."""
    df = df.copy()

    # Ensure we have vegas features
    if 'vegas_total' not in df.columns:
        df['vegas_total'] = 45.0  # Default
    else:
        df['vegas_total'] = df['vegas_total'].fillna(45.0)

    if 'vegas_spread' not in df.columns:
        df['vegas_spread'] = 0.0  # Default (pick-em)
    else:
        df['vegas_spread'] = df['vegas_spread'].fillna(0.0)

    return df


def compute_opponent_td_defense(
    df: pd.DataFrame,
    stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute opponent's TD defense (TDs allowed per game).

    Features:
    - opponent_pass_td_allowed: RZ pass TD rate allowed
    - opponent_rush_td_allowed: RZ rush TD rate allowed
    - opp_rz_td_allowed: Combined RZ TD rate allowed
    """
    print("Computing opponent TD defense features...")

    # Initialize columns with actual league averages from PBP data (2023-2024)
    # Rush: 17.2%, Pass: 21.4% (old defaults were incorrectly 12% and 8%)
    df['opponent_pass_td_allowed'] = 0.214  # Actual RZ pass TD rate (was 0.08)
    df['opponent_rush_td_allowed'] = 0.172  # Actual RZ rush TD rate (was 0.12)
    df['opp_rz_td_allowed'] = 0.20  # League avg total RZ TD rate

    # Check if opponent column exists
    if 'opponent' not in df.columns:
        print("  No opponent column found, using league averages")
        return df

    # Try to load defensive metrics extractor
    try:
        # Load by season to get proper PBP data
        seasons = df['season'].unique()
        for season in seasons:
            season_mask = df['season'] == season
            season_df = df[season_mask]

            try:
                extractor = DefensiveMetricsExtractor(season=int(season))

                for idx, row in season_df.iterrows():
                    opponent = row.get('opponent')
                    week = row.get('week')

                    if pd.isna(opponent) or pd.isna(week):
                        continue

                    try:
                        rz_defense = extractor.get_rz_td_defense(
                            defense_team=str(opponent),
                            current_week=int(week),
                            trailing_weeks=4
                        )

                        df.loc[idx, 'opponent_pass_td_allowed'] = rz_defense['rz_pass_td_rate']
                        df.loc[idx, 'opponent_rush_td_allowed'] = rz_defense['rz_rush_td_rate']
                        df.loc[idx, 'opp_rz_td_allowed'] = rz_defense['rz_total_td_rate']
                    except Exception:
                        # Keep default values for this row
                        pass

            except Exception as e:
                print(f"  Could not load defensive metrics for {season}: {e}")
                continue

        # Count how many got actual values vs defaults
        non_default = (df['opp_rz_td_allowed'] != 0.20).sum()
        print(f"  Computed actual opponent RZ defense for {non_default}/{len(df)} rows")

    except Exception as e:
        print(f"  Using league averages due to error: {e}")

    return df


def prepare_td_training_data(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    market: str,
) -> pd.DataFrame:
    """
    Prepare training data for a specific TD market.

    Args:
        df: Historical odds data with actuals
        stats: Player stats
        market: TD market type

    Returns:
        Prepared DataFrame with features and target
    """
    print(f"\nPreparing training data for {market}...")

    # Filter to market
    market_df = df[df['market'] == market].copy()
    print(f"  Initial rows: {len(market_df)}")

    if len(market_df) == 0:
        return pd.DataFrame()

    # Add trailing features
    market_df = compute_td_trailing_features(market_df, stats)

    # Add Vegas features
    market_df = add_vegas_features(market_df)

    # Add opponent defense features
    market_df = compute_opponent_td_defense(market_df, stats)

    # Add RZ TD conversion rates and goal-line features
    try:
        for season in market_df['season'].unique():
            # RZ TD rates
            rz_rates = load_and_compute_rz_td_rates(int(season))
            if not rz_rates.empty:
                market_df = merge_rz_td_rates(market_df, rz_rates, player_id_col='player_id')

            # Goal-line roles
            gl_roles = load_and_compute_goal_line_roles(int(season))
            if not gl_roles.empty:
                market_df = merge_goal_line_features(market_df, gl_roles, player_id_col='player_id')
    except Exception as e:
        print(f"  RZ/GL features unavailable: {e}")

    # Compute target variable
    target_col_map = {
        'player_pass_tds': 'passing_tds',
        'player_rush_tds': 'rushing_tds',
        'player_rec_tds': 'receiving_tds',
    }

    target_col = target_col_map.get(market)

    # If we have 'actual' column, use that
    if 'actual' in market_df.columns:
        market_df['td_count'] = market_df['actual'].fillna(0).astype(int)
    elif target_col and target_col in market_df.columns:
        market_df['td_count'] = market_df[target_col].fillna(0).astype(int)
    else:
        # Try to merge from stats
        if target_col and target_col in stats.columns:
            market_df = market_df.merge(
                stats[['player_norm', 'season', 'week', target_col]].drop_duplicates(),
                on=['player_norm', 'season', 'week'],
                how='left'
            )
            market_df['td_count'] = market_df[target_col].fillna(0).astype(int)
        else:
            print(f"  Warning: No target column found for {market}")
            return pd.DataFrame()

    # Ensure td_count is non-negative
    market_df['td_count'] = market_df['td_count'].clip(lower=0)

    # Add global_week for ordering
    if 'global_week' not in market_df.columns:
        market_df['global_week'] = (market_df['season'] - 2023) * 18 + market_df['week']

    # Drop rows without target
    market_df = market_df.dropna(subset=['td_count'])

    print(f"  Final rows: {len(market_df)}")
    print(f"  TD distribution: {market_df['td_count'].value_counts().sort_index().to_dict()}")

    return market_df


def validate_td_model(
    edge: TDPoissonEdge,
    df: pd.DataFrame,
    market: str,
    n_val_weeks: int = 10,
) -> pd.DataFrame:
    """
    Walk-forward validation of TD model.

    Args:
        edge: Trained TDPoissonEdge
        df: Full dataset
        market: Market type
        n_val_weeks: Number of weeks for validation

    Returns:
        DataFrame with validation results
    """
    print(f"\nValidating {market} model...")

    weeks = sorted(df['global_week'].unique())

    if len(weeks) < 5:
        print("  Not enough weeks for validation")
        return pd.DataFrame()

    val_weeks = weeks[-n_val_weeks:]
    all_preds = []

    for test_week in val_weeks:
        train_data = df[df['global_week'] < test_week - 1].copy()
        test_data = df[df['global_week'] == test_week].copy()

        if len(train_data) < 50 or len(test_data) == 0:
            continue

        # Train on historical data
        try:
            edge.train(train_data, market)
        except Exception as e:
            print(f"  Week {test_week} training error: {e}")
            continue

        # Predict on test week
        for idx, row in test_data.iterrows():
            features = pd.DataFrame([row])

            try:
                expected_tds = edge.predict_td_count(features, market)[0]
                line = row.get('line', 1.5)
                p_over = edge.predict_over_probability(features, market, line)[0]
                p_under = 1 - p_over

                all_preds.append({
                    'global_week': test_week,
                    'player': row.get('player', ''),
                    'line': line,
                    'actual': row['td_count'],
                    'expected_tds': expected_tds,
                    'p_over': p_over,
                    'p_under': p_under,
                    'over_hit': 1 if row['td_count'] > line else 0,
                    'under_hit': 1 if row['td_count'] < line else 0,
                })
            except Exception as e:
                continue

    if not all_preds:
        return pd.DataFrame()

    preds_df = pd.DataFrame(all_preds)

    # Print validation metrics
    print(f"\n  Validation Results ({market}):")
    print(f"    Total predictions: {len(preds_df)}")

    for thresh in [0.55, 0.58, 0.60, 0.65]:
        # OVER bets
        over_bets = preds_df[preds_df['p_over'] >= thresh]
        if len(over_bets) >= 5:
            hits = over_bets['over_hit'].sum()
            total = len(over_bets)
            hit_rate = hits / total
            roi = (hit_rate * 0.909) - (1 - hit_rate)
            print(f"    OVER @ {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

        # UNDER bets
        under_bets = preds_df[preds_df['p_under'] >= thresh]
        if len(under_bets) >= 5:
            hits = under_bets['under_hit'].sum()
            total = len(under_bets)
            hit_rate = hits / total
            roi = (hit_rate * 0.909) - (1 - hit_rate)
            print(f"    UNDER @ {thresh:.0%}: N={total}, Hit={hit_rate:.1%}, ROI={roi:+.1%}")

    return preds_df


def train_market(
    edge: TDPoissonEdge,
    df: pd.DataFrame,
    stats: pd.DataFrame,
    market: str,
) -> dict:
    """
    Train TD Poisson edge for a single market.

    Returns training metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training TD Poisson Edge for: {market}")
    print(f"{'='*60}")

    # Prepare data
    market_df = prepare_td_training_data(df, stats, market)

    if len(market_df) < 100:
        print(f"  Insufficient data: {len(market_df)} samples")
        return None

    # Train model
    try:
        metrics = edge.train(market_df, market)
    except Exception as e:
        print(f"  Training error: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Validate
    val_results = validate_td_model(edge, market_df, market)

    if len(val_results) > 0:
        metrics['validation'] = val_results

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TD Poisson Edge Model")
    parser.add_argument('--market', type=str, help='Train single market only')
    args = parser.parse_args()

    print("="*80)
    print("TD POISSON EDGE TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Load data
    df = load_training_data()
    stats = load_player_stats()

    if len(df) == 0:
        print("No TD market data found")
        return

    # Normalize player names
    if 'player_norm' not in df.columns:
        df['player_norm'] = df['player'].apply(normalize_player_name)

    # Try to compute red zone features
    print("\nComputing red zone features...")
    try:
        for season in df['season'].unique():
            rz_features = compute_red_zone_features(int(season))
            if not rz_features.empty:
                df = merge_red_zone_features(df, rz_features)
    except Exception as e:
        print(f"  Red zone features unavailable: {e}")

    # Initialize edge
    edge = TDPoissonEdge()

    # Determine markets to train
    if args.market:
        markets = [args.market]
    else:
        # Use available markets in data
        available_markets = df['market'].unique().tolist()
        markets = [m for m in TD_MARKETS if m in available_markets]

    print(f"\nMarkets to train: {markets}")

    # Train each market
    all_metrics = {}
    for market in markets:
        metrics = train_market(edge, df, stats, market)
        if metrics:
            all_metrics[market] = metrics

    # Save model
    if all_metrics:
        edge.save()

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Trained markets: {list(all_metrics.keys())}")

        # Summary table
        print("\n" + "-"*70)
        print(f"{'Market':<20} {'Samples':<10} {'MAE':<10} {'Mean Actual':<12} {'Mean Pred':<10}")
        print("-"*70)
        for market, metrics in all_metrics.items():
            print(
                f"{market:<20} "
                f"{metrics['n_samples']:<10} "
                f"{metrics['mae']:<10.3f} "
                f"{metrics['mean_actual']:<12.2f} "
                f"{metrics['mean_predicted']:<10.2f}"
            )
    else:
        print("\nNo markets trained successfully")


if __name__ == '__main__':
    main()
