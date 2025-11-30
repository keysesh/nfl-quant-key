#!/usr/bin/env python3
"""
Walk-Forward Validation for V2 Feature Engine

Compares:
- V12 Baseline (12 features)
- V12 + Snap Counts (14 features)
- V12 + NGS Features (market-specific)
- V12 + ALL V2 Features (full feature set)

Tests the marginal impact of new V2 features on prediction accuracy and ROI.

Usage:
    python scripts/backtest/backtest_v2_features.py
    python scripts/backtest/backtest_v2_features.py --market player_receptions
    python scripts/backtest/backtest_v2_features.py --quick  # Fast test with fewer configs
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import argparse
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.features.core import FeatureEngine, get_feature_engine
from nfl_quant.utils.player_names import normalize_player_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE CONFIGURATIONS
# =============================================================================

# Baseline V12 features
V12_FEATURES = [
    'line_vs_trailing', 'line_level', 'line_in_sweet_spot',
    'player_under_rate', 'player_bias', 'market_under_rate',
    'LVT_x_player_tendency', 'LVT_x_player_bias', 'LVT_x_regime',
    'LVT_in_sweet_spot', 'market_bias_strength', 'player_market_aligned'
]

# V2 feature groups
V2_SNAP = ['snap_share', 'snap_trend']
V2_NGS_REC = ['avg_separation', 'avg_cushion', 'yac_above_expectation']
V2_NGS_RUSH = ['eight_box_rate', 'rush_efficiency', 'opp_eight_box_rate']
V2_OPPORTUNITY = ['trailing_wopr', 'trailing_racr']
V2_EPA_REC = ['trailing_receiving_epa']
V2_EPA_RUSH = ['trailing_rushing_epa']

# Market-specific feature configurations
MARKET_FEATURE_CONFIGS = {
    'player_receptions': {
        'V12_baseline': V12_FEATURES,
        'V12_snap': V12_FEATURES + V2_SNAP,
        'V12_ngs': V12_FEATURES + V2_NGS_REC,
        'V12_opportunity': V12_FEATURES + V2_OPPORTUNITY + V2_EPA_REC,
        'V12_all_v2': V12_FEATURES + V2_SNAP + V2_NGS_REC + V2_OPPORTUNITY + V2_EPA_REC,
    },
    'player_reception_yds': {
        'V12_baseline': V12_FEATURES,
        'V12_snap': V12_FEATURES + V2_SNAP,
        'V12_ngs': V12_FEATURES + V2_NGS_REC,
        'V12_opportunity': V12_FEATURES + V2_OPPORTUNITY + V2_EPA_REC,
        'V12_all_v2': V12_FEATURES + V2_SNAP + V2_NGS_REC + V2_OPPORTUNITY + V2_EPA_REC,
    },
    'player_rush_yds': {
        'V12_baseline': V12_FEATURES,
        'V12_snap': V12_FEATURES + V2_SNAP,
        'V12_ngs': V12_FEATURES + V2_NGS_RUSH,
        'V12_epa': V12_FEATURES + V2_EPA_RUSH,
        'V12_all_v2': V12_FEATURES + V2_SNAP + V2_NGS_RUSH + V2_EPA_RUSH,
    },
    'player_pass_yds': {
        'V12_baseline': V12_FEATURES,
        'V12_snap': V12_FEATURES + V2_SNAP,
        'V12_all_v2': V12_FEATURES + V2_SNAP,  # Limited V2 features for passing
    },
}

# Stat column mapping
MARKET_STAT_COL = {
    'player_receptions': 'receptions',
    'player_reception_yds': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_pass_yds': 'passing_yards',
}

# NOTE: V12 model parameters are now centralized in FeatureEngine.build_v12_model_params()
# This ensures consistency between training, backtest, and prediction scripts.


# =============================================================================
# PLAYER ID MAPPING
# =============================================================================

class PlayerIDMapper:
    """Map player names to GSIS IDs for V2 feature lookup."""

    def __init__(self):
        self.weekly_stats = None
        self.name_to_id: Dict[str, str] = {}
        self.name_to_position: Dict[str, str] = {}
        self.name_to_team: Dict[str, str] = {}
        self._load_mappings()

    def _load_mappings(self):
        """Load player mappings from weekly stats."""
        path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
        if not path.exists():
            logger.warning("Weekly stats not found for player ID mapping")
            return

        self.weekly_stats = pd.read_parquet(path)

        # Build name -> ID mapping (use most recent data for each player)
        for _, row in self.weekly_stats.sort_values(['season', 'week'], ascending=False).iterrows():
            player_name = row.get('player_display_name')
            if player_name is None or pd.isna(player_name):
                continue

            name_lower = str(player_name).lower().strip()
            if name_lower and name_lower not in self.name_to_id:
                self.name_to_id[name_lower] = row['player_id']
                self.name_to_position[name_lower] = row.get('position', 'WR')
                self.name_to_team[name_lower] = row.get('team', '')

        logger.info(f"Loaded {len(self.name_to_id)} player ID mappings")

    def get_player_id(self, player_name: str) -> Optional[str]:
        """Get player ID from name."""
        name_lower = player_name.lower().strip()
        return self.name_to_id.get(name_lower)

    def get_position(self, player_name: str, market: str) -> str:
        """Get player position from name or infer from market."""
        name_lower = player_name.lower().strip()
        if name_lower in self.name_to_position:
            return self.name_to_position[name_lower]

        # Infer from market
        if market == 'player_pass_yds':
            return 'QB'
        elif market == 'player_rush_yds':
            return 'RB'
        else:
            return 'WR'

    def get_team(self, player_name: str) -> str:
        """Get player team from name."""
        name_lower = player_name.lower().strip()
        return self.name_to_team.get(name_lower, '')


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_historical_odds() -> pd.DataFrame:
    """Load and prepare historical odds data."""
    path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'

    if not path.exists():
        raise FileNotFoundError(f"Historical odds not found: {path}")

    odds = pd.read_csv(path)

    # Add global week for temporal ordering
    odds['global_week'] = (odds['season'] - 2023) * 18 + odds['week']

    # Normalize player names
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    logger.info(f"Loaded {len(odds)} historical odds records")
    logger.info(f"Seasons: {sorted(odds['season'].unique())}")
    logger.info(f"Markets: {odds['market'].unique()}")

    return odds


def load_player_stats() -> pd.DataFrame:
    """Load player stats for trailing calculations."""
    # Try multiple sources
    stats_list = []

    # 2023 stats
    path_2023 = PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2023.csv'
    if path_2023.exists():
        stats_2023 = pd.read_csv(path_2023)
        if 'player_display_name' in stats_2023.columns:
            stats_2023['player_norm'] = stats_2023['player_display_name'].apply(normalize_player_name)
        stats_list.append(stats_2023)

    # 2024-2025 stats
    path_2024_2025 = PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv'
    if path_2024_2025.exists():
        stats_2024 = pd.read_csv(path_2024_2025)
        if 'player_display_name' in stats_2024.columns:
            stats_2024['player_norm'] = stats_2024['player_display_name'].apply(normalize_player_name)
        stats_list.append(stats_2024)

    if not stats_list:
        # Fall back to weekly stats
        weekly = pd.read_parquet(PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet')
        weekly['player_norm'] = weekly['player_display_name'].apply(normalize_player_name)
        return weekly

    stats = pd.concat(stats_list, ignore_index=True)
    logger.info(f"Loaded {len(stats)} player stat records")

    return stats


def prepare_odds_with_trailing(odds: pd.DataFrame, stats: pd.DataFrame,
                                engine: FeatureEngine) -> pd.DataFrame:
    """Merge odds with trailing stats."""
    result = odds.copy()

    # Calculate trailing stats for each stat type
    for market, stat_col in MARKET_STAT_COL.items():
        if stat_col not in stats.columns:
            continue

        trailing_col = f'trailing_{stat_col}'

        # Calculate trailing stat with NO LEAKAGE
        stats_sorted = stats.sort_values(['player_norm', 'season', 'week'])
        stats_sorted[trailing_col] = engine.calculate_trailing_stat(
            df=stats_sorted,
            stat_col=stat_col,
            player_col='player_norm',
            span=4,
            min_periods=1,
            no_leakage=True
        )

        # Merge trailing stats to odds
        trailing_lookup = stats_sorted[['player_norm', 'season', 'week', trailing_col]].drop_duplicates()
        result = result.merge(
            trailing_lookup,
            on=['player_norm', 'season', 'week'],
            how='left',
            suffixes=('', '_dup')
        )

    return result


# =============================================================================
# V2 FEATURE EXTRACTION - Using core.py as single source of truth
# =============================================================================

# NOTE: All feature extraction is now done via engine methods from core.py:
# - engine.extract_v12_features_for_week() for V12 features
# - engine.precompute_v2_features() for V2 features
# - engine.add_v2_features_to_df() to merge V2 features


def precompute_v2_features(odds_merged: pd.DataFrame, engine: FeatureEngine,
                           player_mapper: PlayerIDMapper, market: str) -> pd.DataFrame:
    """
    Thin wrapper around engine.precompute_v2_features() from core.py.
    """
    logger.info(f"    Pre-computing V2 features for {market}...")

    # Build player ID mapping
    player_id_map = player_mapper.name_to_id

    # Use core.py implementation
    result = engine.precompute_v2_features(odds_merged, market, player_id_map)

    logger.info(f"    Pre-computed V2 features for {len(result)} rows")
    return result


def add_v2_features_to_df(df: pd.DataFrame, engine: FeatureEngine,
                          player_mapper: PlayerIDMapper, market: str,
                          precomputed_v2: pd.DataFrame = None) -> pd.DataFrame:
    """
    Thin wrapper around engine.add_v2_features_to_df() from core.py.
    """
    if precomputed_v2 is not None:
        return engine.add_v2_features_to_df(df, precomputed_v2, market)
    return df


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def calculate_roi(actuals: np.ndarray, preds: np.ndarray, threshold: float) -> Dict:
    """Calculate ROI and related metrics at a given threshold."""
    mask = preds >= threshold
    n_bets = mask.sum()

    if n_bets == 0:
        return {'n_bets': 0, 'hit_rate': 0.0, 'roi': 0.0}

    hits = actuals[mask]
    hit_rate = hits.mean()

    # ROI calculation: -110 odds = 0.909 profit on win, -1.0 loss on loss
    profits = np.where(hits == 1, 0.909, -1.0)
    roi = profits.mean() * 100

    return {
        'n_bets': int(n_bets),
        'hit_rate': float(hit_rate),
        'roi': float(roi),
        'total_profit': float(profits.sum()),
        'wins': int(hits.sum()),
        'losses': int(n_bets - hits.sum()),
    }


def bootstrap_roi_ci(actuals: np.ndarray, preds: np.ndarray, threshold: float,
                      n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for ROI."""
    mask = preds >= threshold
    n_bets = mask.sum()

    if n_bets < 10:
        return (np.nan, np.nan)

    hits = actuals[mask]
    rois = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_idx = np.random.choice(len(hits), size=len(hits), replace=True)
        sample_hits = hits[sample_idx]

        profits = np.where(sample_hits == 1, 0.909, -1.0)
        rois.append(profits.mean() * 100)

    alpha = (1 - ci) / 2
    lower = np.percentile(rois, alpha * 100)
    upper = np.percentile(rois, (1 - alpha) * 100)

    return (lower, upper)


def walk_forward_validate(market: str, feature_config_name: str,
                          feature_cols: List[str], odds_merged: pd.DataFrame,
                          engine: FeatureEngine, player_mapper: PlayerIDMapper,
                          precomputed_v2: pd.DataFrame = None,
                          window_weeks: int = 20,
                          test_start_week: int = 37) -> Dict:
    """
    Walk-forward validation for a single market and feature configuration.

    Args:
        market: Market name (e.g., 'player_receptions')
        feature_config_name: Name of feature configuration
        feature_cols: List of feature columns to use
        odds_merged: Odds data with trailing stats
        engine: FeatureEngine instance
        player_mapper: Player ID mapper
        window_weeks: Number of weeks for training window
        test_start_week: First global week to test (default: 37 = 2025 week 1)

    Returns:
        Dictionary with validation results
    """
    market_odds = odds_merged[odds_merged['market'] == market].copy()

    # Get test weeks (2025 weeks 1-12)
    test_weeks = sorted(market_odds[
        (market_odds['global_week'] >= test_start_week) &
        (market_odds['global_week'] <= test_start_week + 11)  # 12 weeks
    ]['global_week'].unique())

    if len(test_weeks) == 0:
        logger.warning(f"No test weeks found for {market}")
        return {'error': 'No test weeks'}

    all_preds = []
    all_actuals = []
    weekly_results = []
    feature_importances = []

    for test_week in test_weeks:
        # Get training weeks (prior weeks, up to window_weeks)
        train_weeks = sorted([
            w for w in market_odds['global_week'].unique()
            if w < test_week
        ])[-window_weeks:]

        if len(train_weeks) < 5:
            continue

        # Check if we need V2 features
        all_v2_features = (V2_SNAP + V2_NGS_REC + V2_NGS_RUSH +
                          V2_OPPORTUNITY + V2_EPA_REC + V2_EPA_RUSH)
        needs_v2 = any(f in feature_cols for f in all_v2_features)

        # Extract training features using core.py method
        train_data_list = []
        for train_week in train_weeks:
            week_features = engine.extract_v12_features_for_week(
                odds_merged, odds_merged, train_week, market
            )
            if len(week_features) > 0:
                # Add V2 features only if needed
                if needs_v2:
                    week_features = add_v2_features_to_df(
                        week_features, engine, player_mapper, market,
                        precomputed_v2=precomputed_v2
                    )
                train_data_list.append(week_features)

        if len(train_data_list) == 0:
            continue

        train_data = pd.concat(train_data_list, ignore_index=True)

        # Extract test features using core.py method
        test_data = engine.extract_v12_features_for_week(
            odds_merged, odds_merged, test_week, market
        )

        if len(test_data) == 0:
            continue

        # Add V2 features to test data only if needed
        if needs_v2:
            test_data = add_v2_features_to_df(
                test_data, engine, player_mapper, market,
                precomputed_v2=precomputed_v2
            )

        # Get available features
        available_features = [
            f for f in feature_cols
            if f in train_data.columns and f in test_data.columns
        ]

        if len(available_features) < 3:
            continue

        # Clean data
        train_clean = train_data[available_features + ['under_hit']].dropna()
        test_clean = test_data[available_features + ['under_hit']].dropna()

        if len(train_clean) < 50 or len(test_clean) == 0:
            continue

        # Prepare data
        X_train = train_clean[available_features]
        y_train = train_clean['under_hit']
        X_test = test_clean[available_features]
        y_test = test_clean['under_hit']

        # Get V12 model params from centralized FeatureEngine
        params = engine.build_v12_model_params(available_features)

        # Train XGBoost with V12 architecture
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params, dtrain, num_boost_round=200,
            evals=[(dtest, 'test')], early_stopping_rounds=30,
            verbose_eval=False
        )

        # Predict
        preds = model.predict(dtest)

        all_preds.extend(preds)
        all_actuals.extend(y_test.values)

        # Track weekly results
        weekly_results.append({
            'global_week': test_week,
            'n_bets': len(preds),
            'hit_rate': y_test.mean(),
            'avg_pred': preds.mean(),
        })

        # Track feature importance
        importance = model.get_score(importance_type='gain')
        feature_importances.append(importance)

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    if len(all_preds) == 0:
        return {'error': 'No predictions generated'}

    # Calculate results by threshold
    results_by_threshold = {}
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        metrics = calculate_roi(all_actuals, all_preds, threshold)

        # Add confidence interval
        if metrics['n_bets'] >= 10:
            ci_lower, ci_upper = bootstrap_roi_ci(all_preds, all_actuals, threshold)
            metrics['roi_ci_lower'] = ci_lower
            metrics['roi_ci_upper'] = ci_upper
        else:
            metrics['roi_ci_lower'] = np.nan
            metrics['roi_ci_upper'] = np.nan

        results_by_threshold[threshold] = metrics

    # Aggregate feature importance
    avg_importance = {}
    for imp_dict in feature_importances:
        for feat, score in imp_dict.items():
            if feat not in avg_importance:
                avg_importance[feat] = []
            avg_importance[feat].append(score)

    avg_importance = {
        feat: {'mean': np.mean(scores), 'std': np.std(scores)}
        for feat, scores in avg_importance.items()
    }

    return {
        'market': market,
        'config': feature_config_name,
        'n_features': len(available_features) if 'available_features' in dir() else 0,
        'n_total_predictions': len(all_preds),
        'n_weeks_tested': len(weekly_results),
        'results_by_threshold': results_by_threshold,
        'weekly_results': weekly_results,
        'feature_importance': avg_importance,
    }


# =============================================================================
# STATISTICAL COMPARISON
# =============================================================================

def compare_configs(baseline_results: Dict, test_results: Dict,
                    threshold: float = 0.65) -> Dict:
    """Compare two configurations using statistical tests."""
    baseline = baseline_results['results_by_threshold'].get(threshold, {})
    test = test_results['results_by_threshold'].get(threshold, {})

    if not baseline or not test:
        return {'error': 'Missing threshold data'}

    # ROI difference
    roi_diff = test.get('roi', 0) - baseline.get('roi', 0)

    # Simple significance estimate based on sample sizes and hit rates
    n1 = baseline.get('n_bets', 0)
    n2 = test.get('n_bets', 0)

    if n1 < 10 or n2 < 10:
        p_value = np.nan
    else:
        # Use proportion test (simplified)
        p1 = baseline.get('hit_rate', 0.5)
        p2 = test.get('hit_rate', 0.5)

        pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))

        if se > 0:
            z = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            p_value = np.nan

    return {
        'baseline_roi': baseline.get('roi', 0),
        'test_roi': test.get('roi', 0),
        'roi_diff': roi_diff,
        'baseline_hit_rate': baseline.get('hit_rate', 0),
        'test_hit_rate': test.get('hit_rate', 0),
        'baseline_n': n1,
        'test_n': n2,
        'p_value': p_value,
        'significant': p_value < 0.05 if pd.notna(p_value) else False,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_backtest(markets: List[str] = None, quick: bool = False):
    """Run full walk-forward validation backtest."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 60)
    logger.info("V2 Feature Walk-Forward Validation Backtest")
    logger.info("=" * 60)

    # Load data
    logger.info("\n[1/4] Loading data...")
    odds = load_historical_odds()
    stats = load_player_stats()

    # Initialize engine and mapper
    engine = get_feature_engine()
    player_mapper = PlayerIDMapper()

    # Prepare odds with trailing stats
    logger.info("\n[2/4] Calculating trailing stats...")
    odds_merged = prepare_odds_with_trailing(odds, stats, engine)

    # Determine markets to test
    if markets is None:
        markets = ['player_receptions', 'player_reception_yds', 'player_rush_yds', 'player_pass_yds']

    # Run validation
    logger.info("\n[3/4] Running walk-forward validation...")

    all_results = []
    comparison_results = []

    for market in markets:
        logger.info(f"\n--- Market: {market} ---")

        configs = MARKET_FEATURE_CONFIGS.get(market, {})

        if quick:
            # Only test baseline and all_v2
            configs = {k: v for k, v in configs.items() if k in ['V12_baseline', 'V12_all_v2']}

        # Pre-compute V2 features ONCE for this market
        precomputed_v2 = None
        all_v2_features = (V2_SNAP + V2_NGS_REC + V2_NGS_RUSH +
                          V2_OPPORTUNITY + V2_EPA_REC + V2_EPA_RUSH)
        needs_v2 = any(any(f in v for f in all_v2_features) for v in configs.values())

        if needs_v2:
            precomputed_v2 = precompute_v2_features(odds_merged, engine, player_mapper, market)

        baseline_result = None

        for config_name, feature_cols in configs.items():
            logger.info(f"  Testing {config_name} ({len(feature_cols)} features)...")

            result = walk_forward_validate(
                market=market,
                feature_config_name=config_name,
                feature_cols=feature_cols,
                odds_merged=odds_merged,
                engine=engine,
                player_mapper=player_mapper,
                precomputed_v2=precomputed_v2,
            )

            if 'error' not in result:
                all_results.append(result)

                # Store baseline for comparison
                if config_name == 'V12_baseline':
                    baseline_result = result
                elif baseline_result is not None:
                    # Compare to baseline
                    comparison = compare_configs(baseline_result, result)
                    comparison['market'] = market
                    comparison['config'] = config_name
                    comparison_results.append(comparison)

                # Print summary
                best_threshold = max(
                    result['results_by_threshold'].keys(),
                    key=lambda t: result['results_by_threshold'][t].get('roi', -999)
                )
                best_metrics = result['results_by_threshold'][best_threshold]

                logger.info(f"    Best @ {best_threshold:.0%}: "
                           f"N={best_metrics['n_bets']}, "
                           f"Hit={best_metrics['hit_rate']:.1%}, "
                           f"ROI={best_metrics['roi']:+.1f}%")

    # Generate reports
    logger.info("\n[4/4] Generating reports...")

    reports_dir = PROJECT_ROOT / 'reports'
    reports_dir.mkdir(exist_ok=True)

    # Results summary
    print_results_summary(all_results, comparison_results)

    # Save detailed results
    save_results_csv(all_results, reports_dir / f'v2_backtest_results_{timestamp}.csv')
    save_feature_importance(all_results, reports_dir / f'v2_feature_importance_{timestamp}.csv')

    logger.info(f"\nResults saved to reports/v2_backtest_results_{timestamp}.csv")

    return all_results, comparison_results


def print_results_summary(all_results: List[Dict], comparisons: List[Dict]):
    """Print formatted results summary."""

    print("\n" + "=" * 90)
    print("V2 FEATURE WALK-FORWARD VALIDATION RESULTS")
    print("=" * 90)
    print(f"Test Period: 2025 Weeks 1-12 | Training Window: 20 weeks rolling")
    print("=" * 90)

    # Group by market
    markets = set(r['market'] for r in all_results)

    for market in sorted(markets):
        market_results = [r for r in all_results if r['market'] == market]

        print(f"\n{'=' * 90}")
        print(f"Market: {market}")
        print(f"{'=' * 90}")
        print(f"{'Config':<20} {'Thresh':>8} {'N Bets':>8} {'Hit%':>8} {'ROI':>10} {'95% CI':>20} {'p-val':>8}")
        print("-" * 90)

        for result in market_results:
            config = result['config']

            # Find best threshold
            best_thresh = 0.65
            best_metrics = result['results_by_threshold'].get(best_thresh, {})

            if best_metrics.get('n_bets', 0) < 10:
                # Try lower thresholds
                for t in [0.60, 0.55, 0.50]:
                    if result['results_by_threshold'].get(t, {}).get('n_bets', 0) >= 10:
                        best_thresh = t
                        best_metrics = result['results_by_threshold'][t]
                        break

            n_bets = best_metrics.get('n_bets', 0)
            hit_rate = best_metrics.get('hit_rate', 0) * 100
            roi = best_metrics.get('roi', 0)
            ci_lower = best_metrics.get('roi_ci_lower', np.nan)
            ci_upper = best_metrics.get('roi_ci_upper', np.nan)

            # Get p-value from comparisons
            comp = next((c for c in comparisons if c['market'] == market and c['config'] == config), None)
            p_val = comp['p_value'] if comp else np.nan

            ci_str = f"[{ci_lower:+.1f}%, {ci_upper:+.1f}%]" if pd.notna(ci_lower) else "N/A"
            p_str = f"{p_val:.3f}" if pd.notna(p_val) else "-"

            print(f"{config:<20} {best_thresh:>7.0%} {n_bets:>8} {hit_rate:>7.1f}% {roi:>+9.1f}% {ci_str:>20} {p_str:>8}")

        # Feature importance for best config
        all_v2_result = next((r for r in market_results if 'all_v2' in r['config']), None)
        if all_v2_result and all_v2_result.get('feature_importance'):
            print(f"\nTop Features (V12_all_v2):")
            importance = all_v2_result['feature_importance']
            sorted_features = sorted(importance.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]
            for feat, scores in sorted_features:
                v2_marker = " <- V2" if feat in (V2_SNAP + V2_NGS_REC + V2_NGS_RUSH +
                                                   V2_OPPORTUNITY + V2_EPA_REC + V2_EPA_RUSH) else ""
                print(f"  {feat:<30} {scores['mean']:.3f} (±{scores['std']:.3f}){v2_marker}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Overall comparison
    for market in sorted(markets):
        baseline = next((r for r in all_results if r['market'] == market and r['config'] == 'V12_baseline'), None)
        all_v2 = next((r for r in all_results if r['market'] == market and 'all_v2' in r['config']), None)

        if baseline and all_v2:
            b_roi = baseline['results_by_threshold'].get(0.65, {}).get('roi', 0)
            v2_roi = all_v2['results_by_threshold'].get(0.65, {}).get('roi', 0)
            diff = v2_roi - b_roi

            comp = next((c for c in comparisons if c['market'] == market and 'all_v2' in c['config']), None)
            sig = " *" if comp and comp.get('significant') else ""

            print(f"{market:<25} V12: {b_roi:+.1f}% -> V2: {v2_roi:+.1f}% (diff: {diff:+.1f}%){sig}")

    print("\n* = statistically significant at p < 0.05")


def save_results_csv(all_results: List[Dict], path: Path):
    """Save results to CSV."""
    rows = []

    for result in all_results:
        for threshold, metrics in result['results_by_threshold'].items():
            rows.append({
                'market': result['market'],
                'config': result['config'],
                'threshold': threshold,
                'n_bets': metrics.get('n_bets', 0),
                'hit_rate': metrics.get('hit_rate', 0),
                'roi': metrics.get('roi', 0),
                'roi_ci_lower': metrics.get('roi_ci_lower', np.nan),
                'roi_ci_upper': metrics.get('roi_ci_upper', np.nan),
                'n_weeks': result.get('n_weeks_tested', 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def save_feature_importance(all_results: List[Dict], path: Path):
    """Save feature importance to CSV."""
    rows = []

    for result in all_results:
        importance = result.get('feature_importance', {})
        for feat, scores in importance.items():
            rows.append({
                'market': result['market'],
                'config': result['config'],
                'feature': feat,
                'importance_mean': scores['mean'],
                'importance_std': scores['std'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 Feature Walk-Forward Validation')
    parser.add_argument('--market', type=str, help='Single market to test')
    parser.add_argument('--quick', action='store_true', help='Quick test (baseline vs all_v2 only)')

    args = parser.parse_args()

    markets = [args.market] if args.market else None

    results, comparisons = run_backtest(markets=markets, quick=args.quick)
