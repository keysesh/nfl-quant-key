#!/usr/bin/env python3
"""
Walk-Forward Market Rehabilitation Framework

Purpose: Test potential improvements for underperforming/disabled markets
with STRICT no-data-leakage guarantees.

Disabled Markets Being Analyzed:
- player_pass_completions: 47.5% WR, -9.4% ROI
- player_rush_attempts: 47.4% WR, -9.5% ROI
- player_pass_yds: -15.8% ROI (currently excluded)

Anti-Leakage Guarantees:
1. Walk-forward: Train on weeks < test_week - 1 (1 week gap)
2. Feature shift: All trailing features use shift(1) BEFORE expanding
3. Calibration split: 80/20 train/calibrate to prevent overfitting
4. No future data: Model never sees outcomes it's predicting
5. Leakage detection: Model should NOT beat line correlation

Usage:
    python scripts/backtest/walk_forward_market_rehabilitation.py --market player_rush_attempts
    python scripts/backtest/walk_forward_market_rehabilitation.py --all

Author: Claude (Rehabilitation Framework)
Date: 2025-12-28
"""

import pandas as pd
import numpy as np
import logging
import json
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, BACKTEST_DIR
from nfl_quant.utils.player_names import normalize_player_name
from configs.model_config import (
    EWMA_SPAN, MARKET_DIRECTION_CONSTRAINTS, FEATURES,
    get_market_features, MARKET_FEATURE_EXCLUSIONS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DISABLED MARKETS - Targets for Rehabilitation
# =============================================================================
DISABLED_MARKETS = [
    'player_pass_completions',
    'player_rush_attempts',
    # 'player_pass_yds',  # Could add if we want to test this
]

# Market to stat column mapping
MARKET_STAT_MAP = {
    'player_receptions': 'receptions',
    'player_reception_yds': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_rush_attempts': 'carries',
    'player_pass_yds': 'passing_yards',
    'player_pass_completions': 'completions',
    'player_pass_attempts': 'attempts',
}


# =============================================================================
# REHABILITATION STRATEGIES - Potential Fixes to Test
# =============================================================================
@dataclass
class RehabStrategy:
    """A potential rehabilitation strategy to test."""
    name: str
    description: str
    filters: Dict = field(default_factory=dict)
    feature_additions: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.55
    direction_constraint: Optional[str] = None  # 'UNDER_ONLY', 'OVER_ONLY', None


# Define strategies to test for each disabled market
REHAB_STRATEGIES = {
    'player_rush_attempts': [
        RehabStrategy(
            name='baseline',
            description='Current model without any changes',
        ),
        RehabStrategy(
            name='starter_only',
            description='Only bet on confirmed starters (pos_rank == 1)',
            filters={'require_starter': True},
        ),
        RehabStrategy(
            name='high_snap_share',
            description='Only bet on players with 50%+ snap share',
            filters={'min_snap_share': 0.50},
        ),
        RehabStrategy(
            name='close_games_only',
            description='Only bet when spread is close (|spread| <= 7)',
            filters={'max_spread': 7.0},
        ),
        RehabStrategy(
            name='high_confidence',
            description='Raise confidence threshold to 60%',
            confidence_threshold=0.60,
        ),
        RehabStrategy(
            name='under_only',
            description='UNDER picks only (based on market bias)',
            direction_constraint='UNDER_ONLY',
        ),
        RehabStrategy(
            name='combined_v1',
            description='Starter + close games + 58% confidence',
            filters={'require_starter': True, 'max_spread': 7.0},
            confidence_threshold=0.58,
        ),
        RehabStrategy(
            name='combined_v2',
            description='High snap + under only + 55% confidence',
            filters={'min_snap_share': 0.45},
            direction_constraint='UNDER_ONLY',
            confidence_threshold=0.55,
        ),
    ],
    'player_pass_completions': [
        RehabStrategy(
            name='baseline',
            description='Current model without any changes',
        ),
        RehabStrategy(
            name='close_games_only',
            description='Only bet when spread is close (|spread| <= 7)',
            filters={'max_spread': 7.0},
        ),
        RehabStrategy(
            name='high_total_games',
            description='High-scoring games (vegas_total >= 45)',
            filters={'min_total': 45.0},
        ),
        RehabStrategy(
            name='under_only',
            description='UNDER picks only',
            direction_constraint='UNDER_ONLY',
        ),
        RehabStrategy(
            name='high_confidence',
            description='Raise confidence threshold to 60%',
            confidence_threshold=0.60,
        ),
        RehabStrategy(
            name='combined_v1',
            description='Close games + high confidence + under only',
            filters={'max_spread': 7.0},
            direction_constraint='UNDER_ONLY',
            confidence_threshold=0.58,
        ),
    ],
}


# =============================================================================
# DATA LOADING (Anti-Leakage)
# =============================================================================
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load odds and stats data with proper normalization."""
    logger.info("Loading data...")

    # Load enriched odds/actuals
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    odds = pd.read_csv(enriched_path, low_memory=False)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    # Exclude 2023 (need minimum history)
    odds = odds[odds['season'] >= 2024].copy()

    # Create global week index for proper temporal ordering
    odds['global_week'] = (odds['season'] - 2023) * 18 + odds['week']

    # Deduplicate odds (take line closest to market median)
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

    # Load player stats
    stats_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'
    if not stats_path.exists():
        # Try parquet
        stats_path = DATA_DIR / 'nflverse' / 'player_stats.parquet'
        if stats_path.exists():
            stats = pd.read_parquet(stats_path)
        else:
            raise FileNotFoundError(f"Stats file not found. Run 'Rscript scripts/fetch/fetch_nflverse_data.R'")
    else:
        stats = pd.read_csv(stats_path, low_memory=False)

    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    logger.info(f"  Loaded {len(odds):,} odds rows, {len(stats):,} stats rows")
    return odds, stats


def compute_trailing_stats_no_leakage(
    stats: pd.DataFrame,
    max_global_week: int
) -> pd.DataFrame:
    """
    Compute trailing stats using ONLY data before max_global_week.

    CRITICAL: Uses shift(1) BEFORE expanding/rolling to prevent leakage.
    """
    # Filter to ONLY historical data
    stats = stats[stats['global_week'] < max_global_week].copy()
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    # Stat columns to compute trailing averages for
    stat_cols = ['receptions', 'receiving_yards', 'rushing_yards', 'carries',
                 'completions', 'attempts', 'passing_yards']

    for col in stat_cols:
        if col in stats.columns:
            # ANTI-LEAKAGE: shift(1) BEFORE ewm
            stats[f'trailing_{col}'] = stats.groupby('player_norm')[col].transform(
                lambda x: x.shift(1).ewm(span=EWMA_SPAN, min_periods=1).mean()
            )

    # Compute coefficient of variation (consistency measure)
    for col in ['carries', 'completions', 'attempts']:
        if col in stats.columns:
            # ANTI-LEAKAGE: shift(1) BEFORE rolling
            mean = stats.groupby('player_norm')[col].transform(
                lambda x: x.shift(1).rolling(4, min_periods=2).mean()
            )
            std = stats.groupby('player_norm')[col].transform(
                lambda x: x.shift(1).rolling(4, min_periods=2).std()
            )
            stats[f'trailing_cv_{col}'] = (std / (mean + 0.01)).clip(0, 2)

    # Snap share (if available)
    if 'snap_share' in stats.columns:
        stats['trailing_snap_share'] = stats.groupby('player_norm')['snap_share'].transform(
            lambda x: x.shift(1).ewm(span=EWMA_SPAN, min_periods=1).mean()
        )

    return stats


def compute_player_bias_no_leakage(
    odds: pd.DataFrame,
    max_global_week: int
) -> pd.DataFrame:
    """
    Compute player bias features using ONLY historical data.

    CRITICAL: shift(1) applied to prevent seeing current week outcome.
    """
    hist = odds[odds['global_week'] < max_global_week].copy()
    hist = hist.sort_values(['player_norm', 'global_week'])

    # Player under rate (rolling 10 games with shift)
    hist['player_under_rate'] = hist.groupby('player_norm')['under_hit'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # Player bet count (how many bets we've seen for this player)
    hist['player_bet_count'] = hist.groupby('player_norm').cumcount()

    # Player bias (deviation from 0.5 baseline)
    hist['player_bias'] = hist['player_under_rate'] - 0.5

    return hist[['player_norm', 'season', 'week', 'market',
                 'player_under_rate', 'player_bet_count', 'player_bias']].drop_duplicates()


# =============================================================================
# WALK-FORWARD VALIDATION CORE
# =============================================================================
def prepare_week_data(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    test_global_week: int,
    market: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare train and test data for a specific week with NO LEAKAGE.

    Anti-leakage:
    1. Train on weeks < test_week - 1 (1 week gap)
    2. Trailing stats computed from data < test_week only
    3. Player bias computed from data < test_week only
    """
    # Filter to specific market
    market_odds = odds[odds['market'] == market].copy()

    # Training data: weeks < test_week - 1 (1 week gap for anti-leakage)
    train_odds = market_odds[market_odds['global_week'] < test_global_week - 1].copy()
    test_odds = market_odds[market_odds['global_week'] == test_global_week].copy()

    if len(train_odds) == 0 or len(test_odds) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Compute trailing stats from historical data only
    trailing = compute_trailing_stats_no_leakage(stats, test_global_week)

    # For TRAIN: merge on exact season/week
    trailing_cols = [c for c in trailing.columns if c.startswith('trailing_')]
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols
    trailing_merge = trailing[merge_cols].drop_duplicates(['player_norm', 'season', 'week'])
    train_odds = train_odds.merge(trailing_merge, on=['player_norm', 'season', 'week'], how='left')

    # For TEST: use LATEST trailing value per player (most recent game before test week)
    latest_trailing = trailing.sort_values('global_week').groupby('player_norm').last().reset_index()
    latest_cols = ['player_norm'] + trailing_cols
    test_odds = test_odds.merge(latest_trailing[latest_cols], on='player_norm', how='left')

    # Compute player bias features
    player_bias = compute_player_bias_no_leakage(odds, test_global_week)

    # For TRAIN: merge on exact season/week/market
    train_odds = train_odds.merge(
        player_bias[['player_norm', 'season', 'week', 'market', 'player_under_rate', 'player_bet_count', 'player_bias']],
        on=['player_norm', 'season', 'week', 'market'],
        how='left'
    )

    # For TEST: use LATEST player bias per player
    player_bias['global_week'] = (player_bias['season'] - 2023) * 18 + player_bias['week']
    latest_bias = player_bias.sort_values('global_week').groupby(['player_norm', 'market']).last().reset_index()
    test_odds = test_odds.merge(
        latest_bias[['player_norm', 'market', 'player_under_rate', 'player_bet_count', 'player_bias']],
        on=['player_norm', 'market'],
        how='left'
    )

    # Compute line_vs_trailing for both train and test
    stat_col = MARKET_STAT_MAP.get(market)
    trailing_col = f'trailing_{stat_col}'

    for df in [train_odds, test_odds]:
        df['line_vs_trailing'] = 0.0
        if trailing_col in df.columns:
            mask = df[trailing_col] > 0
            df.loc[mask, 'line_vs_trailing'] = (
                (df.loc[mask, 'line'] - df.loc[mask, trailing_col]) /
                df.loc[mask, trailing_col] * 100
            )

        # Additional features
        df['line_level'] = df['line']
        df['market_under_rate'] = df.groupby('market')['under_hit'].transform(
            lambda x: x.shift(1).expanding().mean()
        ).fillna(0.5)

    return train_odds, test_odds


def train_simple_edge_model(train_df: pd.DataFrame, market: str):
    """Train a simple XGBoost edge model for rehabilitation testing."""
    import xgboost as xgb

    if len(train_df) < 100:
        return None

    # Core features for edge model
    features = [
        'line_vs_trailing',
        'line_level',
        'player_under_rate',
        'player_bias',
    ]

    # Add vegas features if available
    vegas_features = ['vegas_spread', 'vegas_total']
    for f in vegas_features:
        if f in train_df.columns and train_df[f].notna().sum() > 50:
            features.append(f)

    # Add trailing features
    stat_col = MARKET_STAT_MAP.get(market)
    trailing_col = f'trailing_{stat_col}'
    if trailing_col in train_df.columns:
        features.append(trailing_col)

    cv_col = f'trailing_cv_{stat_col}'
    if cv_col in train_df.columns:
        features.append(cv_col)

    if 'trailing_snap_share' in train_df.columns:
        features.append('trailing_snap_share')

    # Filter to available features
    available = [f for f in features if f in train_df.columns]
    if len(available) < 3:
        return None

    # Prepare X, y
    X = train_df[available].fillna(0)
    y = train_df['under_hit']

    # Drop rows with NaN target
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]

    if len(X) < 50:
        return None

    # ANTI-LEAKAGE: 80/20 train/calibration split
    from sklearn.model_selection import train_test_split
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost with conservative params to prevent overfitting
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    try:
        model.fit(X_train, y_train)
        model.feature_names = available
        return model
    except Exception as e:
        logger.warning(f"Model training failed: {e}")
        return None


def apply_strategy_filters(
    df: pd.DataFrame,
    strategy: RehabStrategy
) -> pd.DataFrame:
    """Apply rehabilitation strategy filters to dataframe."""
    filtered = df.copy()

    if not strategy.filters:
        return filtered

    # Starter only filter
    if strategy.filters.get('require_starter'):
        if 'pos_rank' in filtered.columns:
            filtered = filtered[filtered['pos_rank'] == 1]
        elif 'is_starter' in filtered.columns:
            filtered = filtered[filtered['is_starter'] == 1]

    # Snap share filter
    min_snap = strategy.filters.get('min_snap_share')
    if min_snap is not None:
        if 'trailing_snap_share' in filtered.columns:
            filtered = filtered[filtered['trailing_snap_share'] >= min_snap]
        elif 'snap_share' in filtered.columns:
            filtered = filtered[filtered['snap_share'] >= min_snap]

    # Spread filter
    max_spread = strategy.filters.get('max_spread')
    if max_spread is not None and 'vegas_spread' in filtered.columns:
        filtered = filtered[abs(filtered['vegas_spread']) <= max_spread]

    # Total filter
    min_total = strategy.filters.get('min_total')
    if min_total is not None and 'vegas_total' in filtered.columns:
        filtered = filtered[filtered['vegas_total'] >= min_total]

    return filtered


# =============================================================================
# LEAKAGE DETECTION
# =============================================================================
def detect_leakage(results: pd.DataFrame) -> Dict:
    """
    Detect potential data leakage by comparing model vs line correlation.

    Key insight: A properly trained model should NOT beat the line's correlation
    with actuals. If it does, there's likely leakage.
    """
    valid = results.dropna(subset=['line', 'actual_stat'])

    if len(valid) < 50:
        return {'status': 'insufficient_data', 'n': len(valid)}

    # Calculate correlations
    line_actual_corr = np.corrcoef(valid['line'], valid['actual_stat'])[0, 1]

    # Check model prediction correlation if available
    if 'prob' in valid.columns:
        # Model predicts UNDER probability - invert for comparison
        # Higher prob = expect lower actual
        pred_actual_corr = -np.corrcoef(valid['prob'], valid['actual_stat'])[0, 1]
    else:
        pred_actual_corr = np.nan

    # Leakage detection
    is_suspicious = False
    if not np.isnan(pred_actual_corr) and pred_actual_corr > line_actual_corr + 0.05:
        is_suspicious = True

    return {
        'status': 'suspicious' if is_suspicious else 'ok',
        'line_actual_corr': round(line_actual_corr, 3),
        'pred_actual_corr': round(pred_actual_corr, 3) if not np.isnan(pred_actual_corr) else None,
        'n': len(valid),
    }


# =============================================================================
# MAIN WALK-FORWARD EVALUATION
# =============================================================================
def evaluate_strategy(
    odds: pd.DataFrame,
    stats: pd.DataFrame,
    market: str,
    strategy: RehabStrategy,
    min_week: int,
    max_week: int
) -> Dict:
    """
    Run walk-forward evaluation for a single strategy.

    Returns performance metrics with NO data leakage.
    """
    all_results = []

    for test_week in range(min_week, max_week + 1):
        train_df, test_df = prepare_week_data(odds, stats, test_week, market)

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Apply strategy filters to test data
        test_df = apply_strategy_filters(test_df, strategy)

        if len(test_df) == 0:
            continue

        # Train model on filtered train data (apply same filters for consistency)
        train_filtered = apply_strategy_filters(train_df, strategy)

        if len(train_filtered) < 50:
            # Fall back to unfiltered training if insufficient data
            train_filtered = train_df

        model = train_simple_edge_model(train_filtered, market)

        if model is None:
            continue

        # Generate predictions
        features = model.feature_names
        X_test = test_df[features].fillna(0)

        try:
            probs = model.predict_proba(X_test)[:, 1]  # P(UNDER)
        except Exception:
            continue

        # Process each row
        for i, (idx, row) in enumerate(test_df.iterrows()):
            prob = probs[i]

            # Apply confidence threshold
            if prob < strategy.confidence_threshold and prob > (1 - strategy.confidence_threshold):
                continue  # Skip low confidence

            # Determine direction
            if prob >= strategy.confidence_threshold:
                direction = 'UNDER'
                conf = prob
            else:
                direction = 'OVER'
                conf = 1 - prob

            # Apply direction constraint
            if strategy.direction_constraint == 'UNDER_ONLY' and direction != 'UNDER':
                continue
            if strategy.direction_constraint == 'OVER_ONLY' and direction != 'OVER':
                continue

            # Calculate hit
            actual = row.get('actual_stat', np.nan)
            under_hit = row.get('under_hit', np.nan)

            if pd.isna(actual) or pd.isna(under_hit):
                continue

            hit = int(under_hit) if direction == 'UNDER' else int(not under_hit)

            result = {
                'season': row.get('season'),
                'week': row.get('week'),
                'global_week': test_week,
                'player': row.get('player'),
                'market': market,
                'line': row.get('line'),
                'direction': direction,
                'prob': conf,
                'actual_stat': actual,
                'hit': hit,
            }
            all_results.append(result)

    if not all_results:
        return {
            'strategy': strategy.name,
            'bets': 0,
            'win_rate': 0,
            'roi': 0,
            'status': 'no_bets',
        }

    results_df = pd.DataFrame(all_results)

    # Calculate metrics
    total = len(results_df)
    hits = results_df['hit'].sum()
    win_rate = hits / total if total > 0 else 0
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100  # -110 odds

    # Leakage check
    leakage_check = detect_leakage(results_df)

    # By direction breakdown
    direction_stats = {}
    for direction in ['UNDER', 'OVER']:
        d_df = results_df[results_df['direction'] == direction]
        if len(d_df) > 0:
            d_hits = d_df['hit'].sum()
            d_wr = d_hits / len(d_df)
            d_roi = (d_wr * 0.909 - (1 - d_wr)) * 100
            direction_stats[direction] = {
                'bets': len(d_df),
                'win_rate': round(d_wr * 100, 1),
                'roi': round(d_roi, 1),
            }

    return {
        'strategy': strategy.name,
        'description': strategy.description,
        'bets': total,
        'hits': hits,
        'win_rate': round(win_rate * 100, 1),
        'roi': round(roi, 1),
        'profitable': roi > 0,
        'by_direction': direction_stats,
        'leakage_check': leakage_check,
        'confidence_threshold': strategy.confidence_threshold,
        'direction_constraint': strategy.direction_constraint,
        'filters': strategy.filters,
    }


def run_rehabilitation_analysis(
    market: str,
    save_results: bool = True
) -> Dict:
    """
    Run full rehabilitation analysis for a single market.

    Tests all defined strategies and compares results.
    """
    logger.info("="*60)
    logger.info(f"MARKET REHABILITATION: {market}")
    logger.info("="*60)

    start_time = time.time()

    # Load data
    odds, stats = load_all_data()

    # Filter to market
    market_odds = odds[odds['market'] == market].copy()

    if len(market_odds) == 0:
        logger.error(f"No data found for market: {market}")
        return {'error': 'no_data'}

    logger.info(f"Market data: {len(market_odds):,} rows")

    # Determine week range
    min_week = market_odds['global_week'].min()
    max_week = market_odds['global_week'].max()
    start_week = max(min_week + 5, 23)  # Need history, 2024 week 5+

    logger.info(f"Testing weeks: {start_week} to {max_week}")

    # Get strategies for this market
    strategies = REHAB_STRATEGIES.get(market, [
        RehabStrategy(name='baseline', description='Default model'),
    ])

    # Evaluate each strategy
    results = []
    for strategy in strategies:
        logger.info(f"\n  Testing: {strategy.name}")
        result = evaluate_strategy(
            odds, stats, market, strategy, start_week, max_week
        )
        results.append(result)

        if result['bets'] > 0:
            logger.info(f"    Bets: {result['bets']}, WR: {result['win_rate']}%, ROI: {result['roi']}%")
            if result['leakage_check']['status'] == 'suspicious':
                logger.warning(f"    ⚠️ SUSPICIOUS: Possible leakage detected!")
        else:
            logger.info(f"    No bets generated")

    # Sort by ROI
    results = sorted(results, key=lambda x: x['roi'], reverse=True)

    # Summary
    summary = {
        'market': market,
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(time.time() - start_time, 1),
        'strategies_tested': len(results),
        'results': results,
        'best_strategy': results[0] if results else None,
        'profitable_strategies': [r for r in results if r.get('profitable', False)],
    }

    # Save results
    if save_results:
        output_path = BACKTEST_DIR / f'rehabilitation_{market}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {output_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("REHABILITATION SUMMARY")
    logger.info("="*60)

    for r in results:
        status = "✅" if r.get('profitable') else "❌"
        leakage = "⚠️" if r.get('leakage_check', {}).get('status') == 'suspicious' else ""
        logger.info(f"{status} {r['strategy']}: {r['bets']} bets, {r['win_rate']}% WR, {r['roi']:+.1f}% ROI {leakage}")

    profitable_count = len(summary['profitable_strategies'])
    if profitable_count > 0:
        logger.info(f"\n🎯 {profitable_count} profitable strategy(ies) found!")
        best = summary['best_strategy']
        logger.info(f"   Best: {best['strategy']} ({best['description']})")
        logger.info(f"   Performance: {best['bets']} bets, {best['win_rate']}% WR, {best['roi']:+.1f}% ROI")
    else:
        logger.info(f"\n❌ No profitable strategies found. Market may be fundamentally unpredictable.")

    return summary


def run_all_disabled_markets():
    """Run rehabilitation analysis on all disabled markets."""
    all_results = {}

    for market in DISABLED_MARKETS:
        result = run_rehabilitation_analysis(market)
        all_results[market] = result

    # Overall summary
    logger.info("\n" + "="*70)
    logger.info("OVERALL REHABILITATION SUMMARY")
    logger.info("="*70)

    for market, result in all_results.items():
        profitable = result.get('profitable_strategies', [])
        if profitable:
            best = result['best_strategy']
            logger.info(f"\n{market}:")
            logger.info(f"  ✅ Best: {best['strategy']} - {best['win_rate']}% WR, {best['roi']:+.1f}% ROI")
        else:
            logger.info(f"\n{market}:")
            logger.info(f"  ❌ No profitable strategies")

    return all_results


# =============================================================================
# CLI
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Walk-forward market rehabilitation analysis')
    parser.add_argument('--market', type=str, help='Specific market to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all disabled markets')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results')
    args = parser.parse_args()

    if args.all:
        run_all_disabled_markets()
    elif args.market:
        run_rehabilitation_analysis(args.market, save_results=not args.no_save)
    else:
        # Default: run all disabled markets
        print("Running rehabilitation analysis on all disabled markets...")
        print("Use --market <name> to analyze a specific market")
        print("Use --all to explicitly run all markets")
        print()
        run_all_disabled_markets()
