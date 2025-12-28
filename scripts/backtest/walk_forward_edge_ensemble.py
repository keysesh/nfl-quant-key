#!/usr/bin/env python3
"""
Walk-Forward Backtest for Edge Ensemble (LVT + Player Bias)

Proper walk-forward validation comparable to XGBoost unified.
Trains edge models each week using only historical data.

Usage:
    python scripts/backtest/walk_forward_edge_ensemble.py
"""

import pandas as pd
import numpy as np
import logging
import joblib
import warnings
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, BACKTEST_DIR
from nfl_quant.utils.player_names import normalize_player_name
from configs.edge_config import LVT_THRESHOLDS, PLAYER_BIAS_THRESHOLDS

# Use same markets as XGBoost for fair comparison
EDGE_MARKETS = [
    'player_receptions',
    'player_rush_yds',
    'player_reception_yds',
    'player_pass_attempts',
    'player_pass_completions',
    'player_rush_attempts',
]
from configs.model_config import EWMA_SPAN, MARKET_DIRECTION_CONSTRAINTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Market to stat column mapping (for trailing stats)
MARKET_STAT_MAP = {
    'player_receptions': 'receptions',
    'player_reception_yds': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_rush_attempts': 'carries',
    'player_pass_yds': 'passing_yards',
    'player_pass_completions': 'completions',
    'player_pass_attempts': 'attempts',
}

# Trailing column names for each market
MARKET_TRAILING_MAP = {
    'player_receptions': 'trailing_receptions',
    'player_reception_yds': 'trailing_receiving_yards',
    'player_rush_yds': 'trailing_rushing_yards',
    'player_rush_attempts': 'trailing_carries',
    'player_pass_completions': 'trailing_completions',
    'player_pass_attempts': 'trailing_attempts',
}


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load odds and stats data."""
    logger.info("Loading data...")

    # Load enriched odds/actuals
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    odds = pd.read_csv(enriched_path, low_memory=False)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    # Exclude 2023
    odds = odds[odds['season'] >= 2024].copy()
    odds['global_week'] = (odds['season'] - 2023) * 18 + odds['week']

    # Deduplicate
    odds['group_key'] = (
        odds['player_norm'] + '_' +
        odds['season'].astype(str) + '_' +
        odds['week'].astype(str) + '_' +
        odds['market']
    )
    market_medians = odds.groupby('market')['line'].median()

    def get_primary_line(group):
        if len(group) == 1:
            return group
        market = group['market'].iloc[0]
        median = market_medians.get(market, group['line'].median())
        group = group.copy()
        group['dist_from_median'] = abs(group['line'] - median)
        return group.nsmallest(1, 'dist_from_median')

    odds = odds.groupby('group_key', group_keys=False).apply(get_primary_line)
    odds = odds.drop(columns=['dist_from_median', 'group_key'], errors='ignore')

    # Load stats
    stats_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'
    stats = pd.read_csv(stats_path, low_memory=False)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    logger.info(f"  Loaded {len(odds):,} odds, {len(stats):,} stats")
    return odds, stats


def compute_trailing_stats(stats: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """Compute trailing stats using only data before max_global_week."""
    stats = stats[stats['global_week'] < max_global_week].copy()
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    # Basic stat columns
    stat_cols = ['receptions', 'receiving_yards', 'rushing_yards', 'carries',
                 'completions', 'attempts', 'passing_yards']

    for col in stat_cols:
        if col in stats.columns:
            stats[f'trailing_{col}'] = stats.groupby('player_norm')[col].transform(
                lambda x: x.ewm(span=EWMA_SPAN, min_periods=1).mean().shift(1)
            )

    # ENHANCED FEATURES: Add high-value features from XGBoost
    enhanced_cols = ['target_share', 'air_yards_share', 'wopr']

    for col in enhanced_cols:
        if col in stats.columns:
            stats[f'trailing_{col}'] = stats.groupby('player_norm')[col].transform(
                lambda x: x.ewm(span=EWMA_SPAN, min_periods=1).mean().shift(1)
            )

    # Also compute trailing coefficient of variation (stability indicator)
    for col in stat_cols[:4]:  # receptions, rec_yds, rush_yds, carries
        if col in stats.columns:
            mean = stats.groupby('player_norm')[col].transform(
                lambda x: x.ewm(span=EWMA_SPAN, min_periods=2).mean().shift(1)
            )
            std = stats.groupby('player_norm')[col].transform(
                lambda x: x.ewm(span=EWMA_SPAN, min_periods=2).std().shift(1)
            )
            stats[f'trailing_cv_{col}'] = (std / (mean + 0.01)).clip(0, 2)

    return stats


def compute_player_bias(odds: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """Compute player bias features using only historical data."""
    hist = odds[odds['global_week'] < max_global_week].copy()
    hist = hist.sort_values(['player_norm', 'global_week'])

    # Player under rate (rolling 10 games, shift to avoid leakage)
    hist['player_under_rate'] = hist.groupby('player_norm')['under_hit'].transform(
        lambda x: x.rolling(10, min_periods=3).mean().shift(1)
    )

    # Player bet count
    hist['player_bet_count'] = hist.groupby('player_norm').cumcount()

    # Player bias (deviation from 0.5)
    hist['player_bias'] = hist['player_under_rate'] - 0.5

    return hist[['player_norm', 'season', 'week', 'market',
                 'player_under_rate', 'player_bet_count', 'player_bias']].drop_duplicates()


def prepare_week_data(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    test_global_week: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train and test data for a specific week."""

    # Training data: weeks < test_week - 1 (1 week gap)
    train_odds = odds[odds['global_week'] < test_global_week - 1].copy()
    test_odds = odds[odds['global_week'] == test_global_week].copy()

    # Compute trailing stats from all data before test week
    trailing = compute_trailing_stats(stats, test_global_week)

    # For TRAIN: merge on exact season/week (trailing from that point in time)
    trailing_cols = [c for c in trailing.columns if c.startswith('trailing_')]
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols
    trailing_merge = trailing[merge_cols].drop_duplicates(['player_norm', 'season', 'week'])
    train_odds = train_odds.merge(trailing_merge, on=['player_norm', 'season', 'week'], how='left')

    # For TEST: use LATEST trailing value per player (most recent game before test week)
    latest_trailing = trailing.sort_values('global_week').groupby('player_norm').last().reset_index()
    latest_cols = ['player_norm'] + trailing_cols
    test_odds = test_odds.merge(latest_trailing[latest_cols], on='player_norm', how='left')

    # Compute player bias features
    player_bias = compute_player_bias(odds, test_global_week)

    # For TRAIN: merge on exact season/week/market
    train_odds = train_odds.merge(
        player_bias[['player_norm', 'season', 'week', 'market', 'player_under_rate', 'player_bet_count', 'player_bias']],
        on=['player_norm', 'season', 'week', 'market'],
        how='left'
    )

    # For TEST: use LATEST player bias per player/market
    player_bias['global_week'] = (player_bias['season'] - 2023) * 18 + player_bias['week']
    latest_bias = player_bias.sort_values('global_week').groupby(['player_norm', 'market']).last().reset_index()
    test_odds = test_odds.merge(
        latest_bias[['player_norm', 'market', 'player_under_rate', 'player_bet_count', 'player_bias']],
        on=['player_norm', 'market'],
        how='left'
    )

    # Compute line_vs_trailing
    for df in [train_odds, test_odds]:
        df['line_vs_trailing'] = 0.0
        for market, stat_col in MARKET_STAT_MAP.items():
            trailing_col = f'trailing_{stat_col}'
            if trailing_col in df.columns:
                mask = (df['market'] == market) & (df[trailing_col] > 0)
                df.loc[mask, 'line_vs_trailing'] = (
                    (df.loc[mask, 'line'] - df.loc[mask, trailing_col]) /
                    df.loc[mask, trailing_col] * 100
                )

    # Add required features with defaults
    for df in [train_odds, test_odds]:
        df['line_level'] = df['line']
        df['market_under_rate'] = df.groupby('market')['under_hit'].transform(
            lambda x: x.expanding().mean().shift(1)
        ).fillna(0.5)

        # Only fill NaN values for trailing features that should exist
        # (they come from stats merge which may have missing rows)
        for col in ['trailing_target_share', 'trailing_air_yards_share', 'trailing_wopr',
                    'trailing_cv_receptions', 'trailing_cv_receiving_yards']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Vegas features from enriched data - fill with market-level median
        for col in ['vegas_spread', 'vegas_total']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

    return train_odds, test_odds


def train_simple_edge_model(train_df: pd.DataFrame, market: str, edge_type: str) -> Optional[object]:
    """Train a simple XGBoost edge model with enhanced features."""
    import xgboost as xgb

    market_train = train_df[train_df['market'] == market].copy()

    # Require minimum samples
    if len(market_train) < 100:
        return None

    # ENHANCED FEATURES: Add target_share, air_yards, CV, vegas features
    if edge_type == 'lvt':
        features = [
            'line_vs_trailing',
            'line_level',
            'market_under_rate',
            'vegas_spread',
            'vegas_total',
            'trailing_target_share',    # #1 predictor in XGBoost (17%)
            'trailing_air_yards_share', # Deep threat indicator
            'trailing_wopr',            # Weighted Opportunity Rating
            'trailing_cv_receptions',   # Consistency/stability
            'trailing_cv_receiving_yards',
        ]
    else:  # player_bias
        features = [
            'player_under_rate',
            'player_bias',
            'player_bet_count',
            'line_level',
            'vegas_spread',
            'trailing_target_share',
            'trailing_wopr',
        ]

    # Filter to available features
    available = [f for f in features if f in market_train.columns]
    if len(available) < 2:
        return None

    # Prepare X, y
    X = market_train[available].fillna(0)
    y = market_train['under_hit']

    # Drop rows with NaN target
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]

    if len(X) < 50:
        return None

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    try:
        model.fit(X, y)
        model.feature_names = available
        return model
    except:
        return None


def predict_with_model(model, df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Generate predictions with a trained model."""
    if model is None:
        return pd.DataFrame()

    market_df = df[df['market'] == market].copy().reset_index(drop=True)
    if len(market_df) == 0:
        return pd.DataFrame()

    features = model.feature_names
    X = market_df[features].fillna(0)

    try:
        probs = model.predict_proba(X)[:, 1]  # P(UNDER)
        market_df['p_under'] = probs
        return market_df
    except:
        return pd.DataFrame()


def run_walk_forward():
    """Run walk-forward backtest for edge ensemble."""
    logger.info("="*60)
    logger.info("WALK-FORWARD EDGE ENSEMBLE BACKTEST")
    logger.info("="*60)

    start_time = time.time()

    # Load data
    odds, stats = load_all_data()

    # Filter to edge markets
    edge_markets = list(EDGE_MARKETS)
    odds = odds[odds['market'].isin(edge_markets)].copy()

    logger.info(f"Data: {len(odds):,} rows, {len(edge_markets)} markets")

    # Week range
    min_week = odds['global_week'].min()
    max_week = odds['global_week'].max()
    start_week = max(min_week + 5, 23)  # 2024 week 5, need history

    all_results = []

    for test_week in range(start_week, max_week + 1):
        train_df, test_df = prepare_week_data(odds, stats, test_week)

        if len(test_df) == 0:
            continue

        test_season = test_df['season'].iloc[0]
        test_week_num = test_df['week'].iloc[0]

        logger.info(f"\n{'='*40}")
        logger.info(f"Testing {test_season} Week {test_week_num} (global: {test_week})")
        logger.info(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")

        week_results = []

        for market in edge_markets:
            # Train both edge models
            lvt_model = train_simple_edge_model(train_df, market, 'lvt')
            pb_model = train_simple_edge_model(train_df, market, 'player_bias')

            if lvt_model is None and pb_model is None:
                continue

            # Get market test data
            market_test = test_df[test_df['market'] == market].copy().reset_index(drop=True)
            if len(market_test) == 0:
                continue

            # Get LVT predictions
            lvt_probs = None
            if lvt_model is not None:
                features = lvt_model.feature_names
                X = market_test[features].fillna(0)
                try:
                    lvt_probs = lvt_model.predict_proba(X)[:, 1]
                except:
                    pass

            # Get Player Bias predictions
            pb_probs = None
            if pb_model is not None:
                features = pb_model.feature_names
                X = market_test[features].fillna(0)
                try:
                    pb_probs = pb_model.predict_proba(X)[:, 1]
                except:
                    pass

            # Process each row
            for i, row in market_test.iterrows():
                sources = []
                probs = []

                # Get LVT prediction (use 50% threshold like XGBoost)
                if lvt_probs is not None:
                    lvt_p = lvt_probs[i]
                    probs.append(lvt_p)
                    if lvt_p >= 0.50 or lvt_p <= 0.50:
                        sources.append('LVT')

                # Get PB prediction
                if pb_probs is not None:
                    pb_p = pb_probs[i]
                    probs.append(pb_p)
                    if pb_p >= 0.50 or pb_p <= 0.50:
                        sources.append('PB')

                if len(probs) == 0:
                    continue

                ensemble_prob = np.mean(probs)
                source = '+'.join(sources) if len(sources) > 0 else 'WEAK'

                # Apply direction constraints (use 50% threshold like XGBoost)
                direction_constraint = MARKET_DIRECTION_CONSTRAINTS.get(market)

                if direction_constraint == 'UNDER_ONLY':
                    if ensemble_prob < 0.50:
                        continue
                    direction = 'UNDER'
                    prob = ensemble_prob
                elif ensemble_prob >= 0.50:
                    direction = 'UNDER'
                    prob = ensemble_prob
                else:
                    direction = 'OVER'
                    prob = 1 - ensemble_prob

                # Determine hit
                actual_stat = row.get('actual_stat', np.nan)
                under_hit = row.get('under_hit', np.nan)

                if pd.isna(actual_stat) or pd.isna(under_hit):
                    continue

                hit = int(under_hit) if direction == 'UNDER' else int(not under_hit)

                result = {
                    'season': test_season,
                    'week': test_week_num,
                    'global_week': test_week,
                    'player': row['player'],
                    'market': market,
                    'line': row['line'],
                    'direction': direction,
                    'prob': prob,
                    'source': source,
                    'actual_stat': actual_stat,
                    'hit': hit,
                    'roi_contribution': 0.909 if hit else -1.0,
                }
                week_results.append(result)

        all_results.extend(week_results)

        if week_results:
            hits = sum(r['hit'] for r in week_results)
            total = len(week_results)
            logger.info(f"  Week: {hits}/{total} ({100*hits/total:.1f}%)")

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = BACKTEST_DIR / 'walk_forward_edge_ensemble.csv'
    results_df.to_csv(output_path, index=False)

    # Calculate summary
    total = len(results_df)
    hits = results_df['hit'].sum()
    win_rate = hits / total if total > 0 else 0
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100

    # By source
    source_stats = {}
    for source in results_df['source'].unique():
        s_df = results_df[results_df['source'] == source]
        s_hits = s_df['hit'].sum()
        s_wr = s_hits / len(s_df)
        s_roi = (s_wr * 0.909 - (1 - s_wr)) * 100
        source_stats[source] = {
            'bets': len(s_df),
            'win_rate': round(s_wr * 100, 1),
            'roi': round(s_roi, 1),
        }

    # By market
    market_stats = {}
    for market in edge_markets:
        m_df = results_df[results_df['market'] == market]
        if len(m_df) > 0:
            m_hits = m_df['hit'].sum()
            m_wr = m_hits / len(m_df)
            m_roi = (m_wr * 0.909 - (1 - m_wr)) * 100
            market_stats[market] = {
                'bets': len(m_df),
                'win_rate': round(m_wr * 100, 1),
                'roi': round(m_roi, 1),
            }

    summary = {
        'model': 'edge_ensemble_walk_forward',
        'total_bets': total,
        'win_rate': round(win_rate * 100, 2),
        'roi': round(roi, 2),
        'by_source': source_stats,
        'by_market': market_stats,
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(time.time() - start_time, 1),
    }

    summary_path = BACKTEST_DIR / 'walk_forward_edge_ensemble_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("EDGE ENSEMBLE WALK-FORWARD RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {total:,} bets, {win_rate*100:.1f}% WR, {roi:.1f}% ROI")
    logger.info(f"\nBy Source:")
    for source, s in source_stats.items():
        logger.info(f"  {source}: {s['bets']} bets, {s['win_rate']}% WR, {s['roi']}% ROI")
    logger.info(f"\nBy Market:")
    for market, m in market_stats.items():
        logger.info(f"  {market}: {m['bets']} bets, {m['win_rate']}% WR, {m['roi']}% ROI")
    logger.info(f"\nSaved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_walk_forward()
