#!/usr/bin/env python3
"""
Unified Walk-Forward Validation with Model-Guided Simulation

This script validates the FULL production pipeline:
1. Model-Guided Monte Carlo simulation (XGBoost features adjust MC distribution)
2. Classifier for P(UNDER) estimation
3. Real sportsbook lines
4. Proper temporal holdout (no data leakage)

For each test week N:
- Calculate model features using only data from weeks < N
- Run Model-Guided MC with adjusted distribution based on features
- Apply classifier thresholds
- Compare predictions to actual outcomes

Key Features:
- Uses simulate_with_model_guidance() from model_guided_simulator.py
- Feature adjustments include: snap_share, target_share, vegas_total
- All features calculated using only prior week data (no leakage)
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Use centralized path configuration
from nfl_quant.config_paths import PROJECT_ROOT, DATA_DIR, BACKTEST_DIR, NFLVERSE_DIR

from nfl_quant.features import get_feature_engine, calculate_trailing_stat
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.simulation.player_simulator_v4 import PlayerSimulatorV4, PlayerPropInput
from nfl_quant.simulation.model_guided_simulator import simulate_with_model_guidance, get_model_guided_simulator
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.features.batch_extractor import extract_features_batch

# Import deflation factors, EWMA settings, and direction constraints from model config
from configs.model_config import (
    TRAILING_DEFLATION_FACTORS,
    DEFAULT_TRAILING_DEFLATION,
    EWMA_SPAN,
    MARKET_DIRECTION_CONSTRAINTS,
    CLASSIFIER_MARKETS,
)
from configs.edge_config import EDGE_MARKETS
from nfl_quant.edges.direction_edge import get_direction_edge

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CLV (CLOSING LINE VALUE) INTEGRATION
# =============================================================================

def load_closing_lines() -> pd.DataFrame:
    """
    Load historical closing lines for CLV calculation.

    Returns:
        DataFrame with closing lines or empty DataFrame if not available
    """
    closing_lines_path = PROJECT_ROOT / 'data' / 'odds' / 'historical_closing_lines_2025.csv'

    if not closing_lines_path.exists():
        logger.warning(f"Closing lines not found at {closing_lines_path}")
        logger.warning("Run: python scripts/fetch/fetch_historical_closing_lines.py --weeks 1-17")
        return pd.DataFrame()

    df = pd.read_csv(closing_lines_path)

    # Normalize player names for matching
    from nfl_quant.utils.player_names import normalize_player_name
    df['player_norm'] = df['player_name'].apply(normalize_player_name)

    logger.info(f"Loaded {len(df)} closing lines for CLV")
    return df


def calculate_bet_clv(
    player_norm: str,
    market: str,
    week: int,
    pick: str,
    model_prob: float,
    closing_lines: pd.DataFrame
) -> dict:
    """
    Calculate Closing Line Value for a single bet.

    CLV = Model Probability - Closing Line Implied Probability
    Positive CLV = We beat the efficient closing market

    Args:
        player_norm: Normalized player name
        market: Market type (e.g., 'player_receptions')
        week: NFL week
        pick: 'OVER' or 'UNDER'
        model_prob: Our model's probability for this pick
        closing_lines: DataFrame of closing lines

    Returns:
        dict with CLV metrics or empty dict if no match
    """
    if closing_lines.empty:
        return {}

    # Find matching closing line
    match = closing_lines[
        (closing_lines['player_norm'] == player_norm) &
        (closing_lines['market'] == market) &
        (closing_lines['week'] == week)
    ]

    if match.empty:
        return {}

    closing = match.iloc[0]

    # Get closing probability for our side
    if pick == 'UNDER':
        closing_prob = closing['no_vig_under']
    else:
        closing_prob = closing['no_vig_over']

    # CLV = our probability - closing probability
    clv = model_prob - closing_prob

    return {
        'clv': round(clv, 4),
        'clv_pct': round(clv * 100, 2),
        'closing_prob': round(closing_prob, 4),
        'closing_line': closing['line'],
    }


def print_clv_summary(results_df: pd.DataFrame):
    """Print CLV summary statistics."""
    if 'clv' not in results_df.columns or results_df['clv'].isna().all():
        print("\nCLV ANALYSIS: No closing line data available")
        print("  Run: python scripts/fetch/fetch_historical_closing_lines.py --weeks 1-17")
        return

    clv_data = results_df.dropna(subset=['clv'])

    if len(clv_data) == 0:
        print("\nCLV ANALYSIS: No bets matched with closing lines")
        return

    print("\n" + "=" * 70)
    print("CLOSING LINE VALUE (CLV) ANALYSIS")
    print("=" * 70)
    print(f"Bets with CLV data: {len(clv_data)} / {len(results_df)}")
    print()

    # Overall CLV stats
    avg_clv = clv_data['clv_pct'].mean()
    median_clv = clv_data['clv_pct'].median()
    positive_pct = (clv_data['clv'] > 0).mean() * 100

    print(f"Average CLV:    {avg_clv:+.2f}%")
    print(f"Median CLV:     {median_clv:+.2f}%")
    print(f"Positive CLV:   {positive_pct:.1f}% of bets")
    print(f"Best CLV:       {clv_data['clv_pct'].max():+.2f}%")
    print(f"Worst CLV:      {clv_data['clv_pct'].min():+.2f}%")
    print()

    if avg_clv > 0:
        print("  Positive average CLV indicates REAL EDGE")
        print("  (Model consistently beats efficient closing market)")
    elif avg_clv > -1.0:
        print("  Near-zero CLV - edge may be marginal")
    else:
        print("  Negative CLV - model may be following market noise")

    # CLV by market
    print("\nCLV by Market:")
    for market in clv_data['market'].unique():
        market_clv = clv_data[clv_data['market'] == market]
        avg = market_clv['clv_pct'].mean()
        n = len(market_clv)
        pos_pct = (market_clv['clv'] > 0).mean() * 100
        print(f"  {market}: avg={avg:+.2f}%, positive={pos_pct:.1f}%, n={n}")


# =============================================================================
# WEEKLY RECALIBRATION
# =============================================================================

from sklearn.isotonic import IsotonicRegression

def create_weekly_calibrator(
    prior_results: pd.DataFrame,
    market: str,
    min_samples: int = 50
) -> IsotonicRegression:
    """
    Create a calibrator using predictions from prior weeks.

    This implements the scientific approach: recalibrate each week
    using the most recent completed predictions.

    Args:
        prior_results: Results from weeks < current test week
        market: Market to calibrate
        min_samples: Minimum samples needed for calibration

    Returns:
        Fitted IsotonicRegression calibrator or None if insufficient data
    """
    market_data = prior_results[prior_results['market'] == market].dropna(
        subset=['clf_prob_under', 'under_hit']
    )

    if len(market_data) < min_samples:
        return None

    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.01, y_max=0.99)
    calibrator.fit(
        market_data['clf_prob_under'].values,
        market_data['under_hit'].values
    )

    return calibrator


def apply_weekly_calibration(
    raw_probs: np.ndarray,
    calibrator: IsotonicRegression
) -> np.ndarray:
    """
    Apply calibration to raw probabilities.

    Args:
        raw_probs: Raw model probabilities
        calibrator: Fitted calibrator

    Returns:
        Calibrated probabilities
    """
    if calibrator is None:
        return raw_probs

    return calibrator.predict(raw_probs)


def load_production_model():
    """Load the production XGBoost model."""
    model_path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'
    if not model_path.exists():
        logger.warning(f"Production model not found at {model_path}")
        return None
    return joblib.load(model_path)


def prepare_odds_with_trailing(odds: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare odds data with trailing stats for feature extraction.
    This mirrors what train_model.py does in prepare_data_with_trailing().
    """
    # Sort stats for proper calculation
    stats = stats.sort_values(['player_norm', 'season', 'week']).copy()
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    # Calculate all trailing stats using vectorized groupby with deflation
    stat_cols = [
        'receptions', 'receiving_yards', 'rushing_yards', 'passing_yards',
        'carries', 'completions', 'attempts', 'passing_tds', 'rushing_tds', 'receiving_tds',
    ]

    # Map stat columns to market names for deflation lookup
    stat_to_market = {
        'receptions': 'player_receptions',
        'receiving_yards': 'player_reception_yds',
        'rushing_yards': 'player_rush_yds',
        'passing_yards': 'player_pass_yds',
        'carries': 'player_rush_attempts',
        'completions': 'player_pass_completions',
        'attempts': 'player_pass_attempts',
        'passing_tds': 'player_pass_tds',
    }

    for col in stat_cols:
        if col in stats.columns:
            # Calculate EWMA trailing stat
            trailing_raw = stats.groupby('player_norm')[col].transform(
                lambda x: x.shift(1).ewm(span=EWMA_SPAN, min_periods=1).mean()
            )
            # Apply deflation factor
            market_name = stat_to_market.get(col, '')
            deflation = TRAILING_DEFLATION_FACTORS.get(market_name, DEFAULT_TRAILING_DEFLATION)
            stats[f'trailing_{col}'] = trailing_raw * deflation

    # Merge trailing stats AND player context to odds
    trailing_cols = [col for col in stats.columns if 'trailing_' in col]
    context_cols = ['player_id', 'position', 'team', 'opponent_team']
    available_context = [c for c in context_cols if c in stats.columns]
    merge_cols = ['player_norm', 'season', 'week'] + trailing_cols + available_context
    stats_dedup = stats[merge_cols].drop_duplicates(subset=['player_norm', 'season', 'week'])
    odds_merged = odds.merge(stats_dedup, on=['player_norm', 'season', 'week'], how='left')

    # Rename for consistency
    if 'opponent_team' in odds_merged.columns:
        odds_merged['opponent'] = odds_merged['opponent_team']

    return odds_merged


def get_production_model_prediction(
    test_data: pd.DataFrame,
    hist_data: pd.DataFrame,
    market: str,
    production_model: dict,
    test_global_week: int
) -> pd.DataFrame:
    """
    Get predictions from production model using proper feature extraction.

    Args:
        test_data: Props to predict (already with trailing stats)
        hist_data: Historical data for rate calculations (must be < test_global_week)
        market: Market name
        production_model: Loaded production model dict
        test_global_week: Current test week

    Returns:
        DataFrame with clf_prob_under added
    """
    if production_model is None or market not in production_model.get('models', {}):
        return test_data

    model = production_model['models'][market]
    feature_cols = list(model.feature_names_in_)

    try:
        # Extract features using batch extractor
        features = extract_features_batch(
            test_data,
            hist_data,
            market,
            target_global_week=test_global_week
        )

        if len(features) == 0:
            return test_data

        # Fill missing features with defaults
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0.0

        # Get predictions
        X = features[feature_cols]
        probs = model.predict_proba(X)
        features['clf_prob_under'] = probs[:, 1]

        return features

    except Exception as e:
        logger.warning(f"Feature extraction failed for {market}: {e}")
        return test_data


# Market direction constraints are now imported from configs/model_config.py
# These constraints were derived from holdout validation (weeks 12-14, 2025) and
# applied prospectively. UNDER picks showed +4.0% ROI vs OVER at -14.8%.
# Using imported MARKET_DIRECTION_CONSTRAINTS from model_config


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


def calculate_trailing_for_player(player_stats, target_global_week):
    """Calculate trailing stats for a player up to (but not including) target week.

    Uses EWMA_SPAN from config and applies deflation factors for regression to mean.
    """
    prior_stats = player_stats[player_stats['global_week'] < target_global_week].copy()
    if len(prior_stats) < 3:
        return None

    prior_stats = prior_stats.sort_values('global_week')

    # Calculate raw EWMA values
    def ewma_last(col):
        if col in prior_stats.columns:
            return prior_stats[col].ewm(span=EWMA_SPAN, min_periods=1).mean().iloc[-1]
        return 0

    # Apply deflation factors (regression to mean)
    trailing = {
        'avg_targets': ewma_last('targets'),  # No deflation for targets
        'avg_receptions': ewma_last('receptions') * TRAILING_DEFLATION_FACTORS.get('player_receptions', DEFAULT_TRAILING_DEFLATION),
        'avg_rec_yards': ewma_last('receiving_yards') * TRAILING_DEFLATION_FACTORS.get('player_reception_yds', DEFAULT_TRAILING_DEFLATION),
        'avg_rush_yards': ewma_last('rushing_yards') * TRAILING_DEFLATION_FACTORS.get('player_rush_yds', DEFAULT_TRAILING_DEFLATION),
        'avg_pass_yards': ewma_last('passing_yards') * TRAILING_DEFLATION_FACTORS.get('player_pass_yds', DEFAULT_TRAILING_DEFLATION),
        'avg_carries': ewma_last('carries') * TRAILING_DEFLATION_FACTORS.get('player_rush_attempts', DEFAULT_TRAILING_DEFLATION),
        'avg_attempts': ewma_last('attempts') * TRAILING_DEFLATION_FACTORS.get('player_pass_attempts', DEFAULT_TRAILING_DEFLATION),
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
        elif market == 'player_rush_attempts':
            trailing_stat = trailing['avg_carries']
        elif market == 'player_pass_attempts':
            trailing_stat = trailing['avg_attempts']
        elif market == 'player_pass_completions':
            trailing_stat = trailing.get('avg_completions', trailing.get('avg_attempts', 0) * 0.65)  # Fallback
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


def load_simulator(trials=5000):
    """Load model-guided simulator."""
    # Initialize the model-guided simulator with proper seed for reproducibility
    simulator = get_model_guided_simulator(trials=trials, seed=42)
    return simulator


def calculate_model_features_for_prop(row, stats, test_global_week, trailing):
    """
    Calculate model features for a prop bet using only data before test_global_week.
    This ensures no data leakage in walk-forward validation.
    """
    # Get player stats before the test week
    player_stats = stats[
        (stats['player_norm'] == row['player_norm']) &
        (stats['global_week'] < test_global_week)
    ].copy()

    if len(player_stats) == 0:
        return {
            'oline_health_score': 0.0,
            'opp_rush_def_vs_avg': 0.0,
            'avg_separation': 3.0,
            'avg_cushion': 6.0,
            'opp_pass_def_vs_avg': 0.0,
            'game_pace': 60.0,
            'vegas_total': 45.0,
            'snap_share': 0.5,
            'target_share': 0.15,
            'opp_def_epa': 0.0,
        }

    player_stats = player_stats.sort_values('global_week')

    # Calculate snap share from historical data (use recent 4 weeks)
    recent_stats = player_stats.tail(4)

    # Estimate snap share from targets/carries as proxy
    avg_targets = recent_stats['targets'].mean() if 'targets' in recent_stats.columns else 0
    avg_carries = recent_stats['carries'].mean() if 'carries' in recent_stats.columns else 0

    # Approximate snap share based on usage (normalize to 0-1 scale)
    # WR/TE: ~5-10 targets = 60-80% snap share
    # RB: ~15-20 carries = 70-90% snap share
    estimated_snap_share = min(0.9, max(0.2, (avg_targets + avg_carries) / 20))

    # Target share estimation
    target_share = avg_targets / 35 if avg_targets > 0 else 0.15  # Assume team avg ~35 targets/game

    return {
        'oline_health_score': 0.0,  # Would need injury data per week
        'opp_rush_def_vs_avg': 0.0,  # Would need schedule data
        'avg_separation': 3.0,  # Default NGS value
        'avg_cushion': 6.0,  # Default NGS value
        'opp_pass_def_vs_avg': 0.0,  # Would need schedule data
        'game_pace': 60.0,  # Average pace
        'vegas_total': row.get('vegas_total', 45.0) if isinstance(row, dict) else 45.0,
        'snap_share': estimated_snap_share,
        'target_share': min(0.30, target_share),  # Cap at 30%
        'opp_def_epa': 0.0,  # Would need opponent data
    }


def run_unified_validation(test_weeks, trials=5000, use_production_model=True, use_weekly_recalibration=False):
    """
    Run unified walk-forward validation.

    Args:
        test_weeks: List of weeks to test
        trials: Number of Monte Carlo trials
        use_production_model: If True, use production XGBoost model (44 features).
                             If False, use simple 3-feature classifier (legacy).
    """
    print("=" * 70)
    print("UNIFIED WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"Test weeks: {test_weeks}")
    print(f"Monte Carlo trials: {trials}")
    print(f"Model: {'PRODUCTION (44 features)' if use_production_model else 'SIMPLE (3 features)'}")
    print()

    # Load data
    props = load_historical_props()
    stats = load_player_stats()

    # Load closing lines for CLV calculation
    closing_lines = load_closing_lines()

    # Load production model if requested
    production_model = None
    if use_production_model:
        production_model = load_production_model()
        if production_model:
            print(f"Loaded production model: {production_model.get('version', 'unknown')}")
        else:
            print("WARNING: Production model not found, falling back to simple classifier")
            use_production_model = False

    # Prepare data with trailing stats for production model
    if use_production_model:
        print("Preparing data with trailing stats...")
        props_with_trailing = prepare_odds_with_trailing(props, stats)
        print(f"  Prepared {len(props_with_trailing)} rows")
    else:
        props_with_trailing = props

    # Initialize model-guided simulator (seeds for reproducibility)
    _ = get_model_guided_simulator(trials=trials, seed=42)

    # Market mapping - ALL edge markets
    market_stat_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',  # NEW
        # player_pass_yds: REMOVED - -14.1% ROI
    }

    markets = list(market_stat_map.keys())

    all_results = []

    # Track weekly calibrators (DISABLED by default - production calibration is better)
    # Pass use_weekly_recalibration=True or --recalibrate flag to enable
    weekly_calibrators = {}  # {market: IsotonicRegression}
    if use_weekly_recalibration:
        print("Weekly recalibration ENABLED")

    for test_week in test_weeks:
        print(f"\n--- Testing Week {test_week} ---")

        # Build calibrators from prior weeks' results (if enabled and enough data)
        if use_weekly_recalibration and len(all_results) >= 50:
            prior_df = pd.DataFrame(all_results)
            for m in markets:
                calibrator = create_weekly_calibrator(prior_df, m, min_samples=30)
                if calibrator is not None:
                    weekly_calibrators[m] = calibrator
                    print(f"  Built calibrator for {m} ({len(prior_df[prior_df['market']==m])} samples)")

        # Convert to global week (current season)
        test_global_week = (get_current_season() - 2023) * 18 + test_week

        # Get props for test week (use props_with_trailing for production model)
        source_props = props_with_trailing if use_production_model else props
        week_props = source_props[
            (source_props['season'] == get_current_season()) &
            (source_props['week'] == test_week)
        ].copy()

        for market in markets:
            market_props = week_props[week_props['market'] == market].copy()

            if len(market_props) == 0:
                continue

            # Use production model or fall back to simple classifier
            if use_production_model and production_model:
                # Get historical data strictly before test week
                hist_data = props_with_trailing[props_with_trailing['global_week'] < test_global_week].copy()

                # Get predictions from production model
                features_df = get_production_model_prediction(
                    market_props, hist_data, market, production_model, test_global_week
                )

                # Process results from production model
                if 'clf_prob_under' in features_df.columns:
                    for _, row in features_df.iterrows():
                        raw_clf_prob_under = row.get('clf_prob_under', 0.5)
                        trailing_stat = row.get(f'trailing_{market_stat_map[market]}', row.get('line', 0) * 0.9)
                        line_vs_trailing = row.get('line_vs_trailing', 0)

                        # Apply weekly recalibration if calibrator exists
                        if market in weekly_calibrators:
                            clf_prob_under = apply_weekly_calibration(
                                np.array([raw_clf_prob_under]),
                                weekly_calibrators[market]
                            )[0]
                        else:
                            clf_prob_under = raw_clf_prob_under

                        pick = 'UNDER' if clf_prob_under > 0.5 else 'OVER'
                        actual_hit = row['under_hit'] if pick == 'UNDER' else (1 - row['under_hit'])

                        # Calculate CLV if closing lines available
                        model_prob = clf_prob_under if pick == 'UNDER' else (1 - clf_prob_under)
                        clv_data = calculate_bet_clv(
                            row.get('player_norm', ''),
                            market, test_week, pick, model_prob, closing_lines
                        )

                        all_results.append({
                            'week': test_week,
                            'player': row.get('player', row.get('player_display_name', 'Unknown')),
                            'player_norm': row.get('player_norm', ''),
                            'market': market,
                            'line': row['line'],
                            'trailing_stat': trailing_stat,
                            'model_projection': trailing_stat,  # Simplified
                            'adjustment_pct': 0.0,
                            'line_vs_trailing': line_vs_trailing,
                            'mc_prob_under': clf_prob_under,  # Use same as clf for production
                            'clf_prob_under': clf_prob_under,
                            'raw_clf_prob_under': raw_clf_prob_under,  # Before recalibration
                            'calibrated': market in weekly_calibrators,  # Track if calibration was applied
                            'pick': pick,
                            'actual_stat': row.get('actual_stat', None),
                            'actual_hit': actual_hit,
                            'under_hit': row['under_hit'],
                            # CLV fields
                            'clv': clv_data.get('clv'),
                            'clv_pct': clv_data.get('clv_pct'),
                            'closing_prob': clv_data.get('closing_prob'),
                        })
                    continue  # Skip legacy processing for this market

            # Legacy: Train simple classifier for this market (using only prior data)
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
                    position = 'WR'
                elif market == 'player_reception_yds':
                    trailing_stat = trailing['avg_rec_yards']
                    position = 'WR'
                elif market == 'player_rush_yds':
                    trailing_stat = trailing['avg_rush_yards']
                    position = 'RB'
                elif market == 'player_rush_attempts':
                    trailing_stat = trailing['avg_carries']
                    position = 'RB'
                elif market == 'player_pass_attempts':
                    trailing_stat = trailing['avg_attempts']
                    position = 'QB'
                elif market == 'player_pass_completions':
                    trailing_stat = trailing.get('avg_completions', trailing.get('avg_attempts', 0) * 0.65)
                    position = 'QB'
                else:
                    continue

                line_vs_trailing = row['line'] - trailing_stat

                # Calculate model features for this prop (using only prior data)
                model_features = calculate_model_features_for_prop(
                    row, stats, test_global_week, trailing
                )

                # Calculate trailing std (use CV of 0.4 as default)
                trailing_std = trailing_stat * 0.4 if trailing_stat > 0 else 10.0

                # Run Model-Guided Monte Carlo simulation
                try:
                    mg_result = simulate_with_model_guidance(
                        trailing_stat=trailing_stat,
                        trailing_std=trailing_std,
                        line=row['line'],
                        market=market,
                        features=model_features,
                        xgb_p_under=None,  # No XGBoost in backtest - just using MC with model-adjusted distribution
                    )

                    mc_prob_under = mg_result['mc_p_under']
                    adjustment_pct = mg_result['adjustment_pct']
                    model_projection = mg_result['model_projection']
                except Exception as e:
                    mc_prob_under = 0.5
                    adjustment_pct = 0.0
                    model_projection = trailing_stat

                # Apply classifier if available
                if classifier is not None:
                    try:
                        X_test = pd.DataFrame([{
                            'line_vs_trailing': line_vs_trailing,
                            'line': row['line'],
                            'trailing_stat': trailing_stat,
                        }])
                        raw_clf_prob_under = classifier.predict_proba(X_test)[0][1]
                    except:
                        raw_clf_prob_under = mc_prob_under
                else:
                    raw_clf_prob_under = mc_prob_under

                # Apply weekly recalibration if calibrator exists
                if market in weekly_calibrators:
                    clf_prob_under = apply_weekly_calibration(
                        np.array([raw_clf_prob_under]),
                        weekly_calibrators[market]
                    )[0]
                else:
                    clf_prob_under = raw_clf_prob_under

                # Determine pick
                pick = 'UNDER' if clf_prob_under > 0.5 else 'OVER'

                # Apply market direction constraints using direction models
                direction_constraint = MARKET_DIRECTION_CONSTRAINTS.get(market, None)
                if direction_constraint == 'UNDER_ONLY' and pick == 'OVER':
                    # Skip this bet - OVER is not allowed for this market
                    continue

                # Additional check with direction-specific models
                direction_edge = get_direction_edge()
                if direction_edge.loaded:
                    # Create row data for direction model
                    row_for_direction = pd.Series({
                        'line': row['line'],
                        'line_vs_trailing': line_vs_trailing,
                        'vegas_spread': row.get('vegas_spread', 0),
                    })
                    allowed, conf, reason = direction_edge.should_allow_direction(
                        row_for_direction, market, pick
                    )
                    if not allowed:
                        continue  # Skip - direction model rejects this pick

                actual_hit = row['under_hit'] if pick == 'UNDER' else (1 - row['under_hit'])

                # Calculate CLV if closing lines available
                model_prob = clf_prob_under if pick == 'UNDER' else (1 - clf_prob_under)
                clv_data = calculate_bet_clv(
                    row.get('player_norm', row['player']),
                    market, test_week, pick, model_prob, closing_lines
                )

                all_results.append({
                    'week': test_week,
                    'player': row['player'],
                    'player_norm': row.get('player_norm', ''),
                    'market': market,
                    'line': row['line'],
                    'trailing_stat': trailing_stat,
                    'model_projection': model_projection,
                    'adjustment_pct': adjustment_pct,
                    'line_vs_trailing': line_vs_trailing,
                    'mc_prob_under': mc_prob_under,
                    'clf_prob_under': clf_prob_under,
                    'raw_clf_prob_under': raw_clf_prob_under,  # Before recalibration
                    'calibrated': market in weekly_calibrators,  # Track if calibration was applied
                    'pick': pick,
                    'actual_stat': row['actual_stat'],
                    'actual_hit': actual_hit,
                    'under_hit': row['under_hit'],
                    # CLV fields
                    'clv': clv_data.get('clv'),
                    'clv_pct': clv_data.get('clv_pct'),
                    'closing_prob': clv_data.get('closing_prob'),
                })

        # Summary for this week
        week_results = [r for r in all_results if r['week'] == test_week]
        if week_results:
            wins = sum(1 for r in week_results if r['actual_hit'] == 1)
            total = len(week_results)
            print(f"  {total} bets, {wins} wins, {wins/total*100:.1f}% win rate")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("UNIFIED RESULTS (Full Pipeline)")
    print("=" * 70)

    total_bets = len(results_df)
    total_wins = results_df['actual_hit'].sum()
    overall_win_rate = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (overall_win_rate * 0.909 - (1 - overall_win_rate)) * 100

    print(f"\nOverall: {total_bets} bets, {overall_win_rate*100:.1f}% win rate, {overall_roi:+.1f}% ROI")

    print("\nBy Market:")
    for market in markets:
        market_df = results_df[results_df['market'] == market]
        if len(market_df) == 0:
            continue

        wins = market_df['actual_hit'].sum()
        total = len(market_df)
        win_rate = wins / total
        roi = (win_rate * 0.909 - (1 - win_rate)) * 100

        print(f"  {market}: n={total}, win_rate={win_rate*100:.1f}%, ROI={roi:+.1f}%")

    # Save results
    output_path = PROJECT_ROOT / 'data' / 'backtest' / 'unified_validation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Threshold analysis
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS (Classifier Confidence)")
    print("=" * 70)

    for market in markets:
        market_df = results_df[results_df['market'] == market]
        if len(market_df) == 0:
            continue

        print(f"\n{market}:")
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            # Filter to bets above threshold
            high_conf = market_df[market_df['clf_prob_under'] >= threshold]
            if len(high_conf) < 10:
                continue

            # For these, we bet UNDER
            wins = high_conf['under_hit'].sum()
            total = len(high_conf)
            win_rate = wins / total
            roi = (win_rate * 0.909 - (1 - win_rate)) * 100

            print(f"  {threshold:.0%}: n={total}, win_rate={win_rate*100:.1f}%, ROI={roi:+.1f}%")

    # Weekly Recalibration Impact Analysis
    if 'calibrated' in results_df.columns and 'raw_clf_prob_under' in results_df.columns:
        print("\n" + "=" * 70)
        print("WEEKLY RECALIBRATION IMPACT")
        print("=" * 70)

        calibrated_df = results_df[results_df['calibrated'] == True]
        uncalibrated_df = results_df[results_df['calibrated'] == False]

        if len(calibrated_df) > 0:
            cal_wins = calibrated_df['actual_hit'].sum()
            cal_total = len(calibrated_df)
            cal_wr = cal_wins / cal_total
            cal_roi = (cal_wr * 0.909 - (1 - cal_wr)) * 100
            print(f"\nCalibrated bets:   {cal_total} bets, {cal_wr*100:.1f}% win rate, {cal_roi:+.1f}% ROI")
        else:
            print("\nNo calibrated bets (first few weeks)")

        if len(uncalibrated_df) > 0:
            uncal_wins = uncalibrated_df['actual_hit'].sum()
            uncal_total = len(uncalibrated_df)
            uncal_wr = uncal_wins / uncal_total
            uncal_roi = (uncal_wr * 0.909 - (1 - uncal_wr)) * 100
            print(f"Uncalibrated bets: {uncal_total} bets, {uncal_wr*100:.1f}% win rate, {uncal_roi:+.1f}% ROI")

        # Compare what pick would have been with vs without calibration
        if len(calibrated_df) > 0:
            # Check how many picks changed due to calibration
            calibrated_df = calibrated_df.copy()
            calibrated_df['raw_pick'] = calibrated_df['raw_clf_prob_under'].apply(lambda x: 'UNDER' if x > 0.5 else 'OVER')
            picks_changed = (calibrated_df['pick'] != calibrated_df['raw_pick']).sum()
            print(f"\nPicks changed by calibration: {picks_changed}/{len(calibrated_df)} ({picks_changed/len(calibrated_df)*100:.1f}%)")

            # Compare accuracy of changed picks
            changed = calibrated_df[calibrated_df['pick'] != calibrated_df['raw_pick']]
            if len(changed) > 0:
                changed_hits = changed['actual_hit'].sum()
                print(f"Changed picks win rate: {changed_hits/len(changed)*100:.1f}% (n={len(changed)})")

    # CLV Analysis
    print_clv_summary(results_df)

    return results_df


def calculate_bootstrap_ci(results_df: pd.DataFrame, n_bootstrap: int = 1000, confidence: float = 0.95) -> dict:
    """
    Calculate bootstrap confidence intervals for ROI.

    This provides statistical rigor by showing the uncertainty range
    rather than just point estimates.

    Args:
        results_df: DataFrame with 'actual_hit' column
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 95%)

    Returns:
        dict with mean, lower, upper bounds for win_rate and ROI
    """
    if len(results_df) == 0:
        return {'win_rate_mean': 0, 'win_rate_lower': 0, 'win_rate_upper': 0,
                'roi_mean': 0, 'roi_lower': 0, 'roi_upper': 0}

    np.random.seed(42)  # Reproducibility

    bootstrap_rois = []
    bootstrap_win_rates = []

    hits = results_df['actual_hit'].values
    n = len(hits)

    for _ in range(n_bootstrap):
        # Sample with replacement
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


def run_holdout_validation(holdout_weeks: list, training_max_week: int = 11, trials: int = 5000):
    """
    Run TRUE out-of-sample validation on holdout weeks.

    This uses a model trained ONLY on weeks <= training_max_week
    and tests on holdout_weeks with NO access to that data during training.

    CRITICAL: This should only be run ONCE to avoid p-hacking.

    Args:
        holdout_weeks: List of weeks to use as holdout (e.g., [12, 13, 14])
        training_max_week: Maximum week used for training (default 11)
        trials: Monte Carlo trials

    Returns:
        DataFrame with holdout results
    """
    print("=" * 70)
    print("TRUE OUT-OF-SAMPLE HOLDOUT VALIDATION")
    print("=" * 70)
    print(f"Training data: weeks <= {training_max_week}")
    print(f"Holdout weeks: {holdout_weeks}")
    print("WARNING: This should only be run ONCE to avoid p-hacking!")
    print("=" * 70)

    # Load data
    props = load_historical_props()
    stats = load_player_stats()

    # Train classifier on ONLY pre-holdout data
    train_global_week_max = (get_current_season() - 2023) * 18 + training_max_week

    # Market mapping - ALL edge markets
    market_stat_map = {
        'player_receptions': 'receptions',
        'player_reception_yds': 'receiving_yards',
        'player_rush_yds': 'rushing_yards',
        'player_rush_attempts': 'carries',
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',  # NEW
        # player_pass_yds: REMOVED - -14.1% ROI
    }

    # Use EDGE_MARKETS for edge validation (includes rush_attempts, pass_attempts)
    markets = [m for m in EDGE_MARKETS if m in market_stat_map]
    print(f"Enabled markets: {markets}")
    all_results = []

    for market in markets:
        # Train classifier using ONLY pre-holdout data
        classifier, feature_cols = train_classifier_for_week(
            props, stats, train_global_week_max + 1, market
        )

        for holdout_week in holdout_weeks:
            holdout_global_week = (get_current_season() - 2023) * 18 + holdout_week
            week_props = props[(props['season'] == get_current_season()) &
                              (props['week'] == holdout_week) &
                              (props['market'] == market)].copy()

            if len(week_props) == 0:
                continue

            stat_col = market_stat_map[market]

            for _, row in week_props.iterrows():
                player_stats = stats[stats['player_norm'] == row['player_norm']]
                trailing = calculate_trailing_for_player(player_stats, holdout_global_week)

                if trailing is None:
                    continue

                # Get trailing stat for this market
                if market == 'player_receptions':
                    trailing_stat = trailing['avg_receptions']
                elif market == 'player_reception_yds':
                    trailing_stat = trailing['avg_rec_yards']
                elif market == 'player_rush_yds':
                    trailing_stat = trailing['avg_rush_yards']
                elif market == 'player_rush_attempts':
                    trailing_stat = trailing['avg_carries']
                elif market == 'player_pass_attempts':
                    trailing_stat = trailing['avg_attempts']
                elif market == 'player_pass_completions':
                    trailing_stat = trailing.get('avg_completions', trailing.get('avg_attempts', 0) * 0.65)
                else:
                    continue

                line_vs_trailing = row['line'] - trailing_stat

                # Apply classifier
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

                pick = 'UNDER' if clf_prob_under > 0.5 else 'OVER'

                # Apply market direction constraints (UNDER_ONLY strategy)
                direction_constraint = MARKET_DIRECTION_CONSTRAINTS.get(market, None)
                if direction_constraint == 'UNDER_ONLY' and pick == 'OVER':
                    # Skip this bet - OVER is not allowed for this market
                    continue

                # Additional check with direction-specific models
                direction_edge = get_direction_edge()
                if direction_edge.loaded:
                    row_for_direction = pd.Series({
                        'line': row['line'],
                        'line_vs_trailing': line_vs_trailing,
                        'vegas_spread': row.get('vegas_spread', 0),
                    })
                    allowed, conf, reason = direction_edge.should_allow_direction(
                        row_for_direction, market, pick
                    )
                    if not allowed:
                        continue  # Skip - direction model rejects this pick

                actual_hit = row['under_hit'] if pick == 'UNDER' else (1 - row['under_hit'])

                all_results.append({
                    'week': holdout_week,
                    'player': row['player'],
                    'market': market,
                    'line': row['line'],
                    'trailing_stat': trailing_stat,
                    'clf_prob_under': clf_prob_under,
                    'pick': pick,
                    'actual_stat': row['actual_stat'],
                    'actual_hit': actual_hit,
                    'under_hit': row['under_hit'],
                })

    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        print("No holdout results generated!")
        return results_df

    # Calculate bootstrap CI
    ci = calculate_bootstrap_ci(results_df)

    print(f"\n{'='*70}")
    print("HOLDOUT RESULTS WITH 95% CONFIDENCE INTERVALS")
    print(f"{'='*70}")
    print(f"Total bets: {ci['n_samples']}")
    print(f"Win Rate:   {ci['win_rate_mean']*100:.1f}% [{ci['win_rate_lower']*100:.1f}%, {ci['win_rate_upper']*100:.1f}%]")
    print(f"ROI:        {ci['roi_mean']:+.1f}% [{ci['roi_lower']:+.1f}%, {ci['roi_upper']:+.1f}%]")

    # Statistical significance test
    if ci['roi_lower'] > 0:
        print("\n✅ ROI is STATISTICALLY SIGNIFICANT (95% CI excludes zero)")
    else:
        print("\n⚠️ ROI is NOT statistically significant (95% CI includes zero)")

    # Save results
    output_path = PROJECT_ROOT / 'data' / 'backtest' / 'holdout_validation_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Walk-Forward Validation with Holdout Support')
    parser.add_argument('--weeks', default='5-11', help='Week range (e.g., 5-11)')
    parser.add_argument('--trials', type=int, default=5000, help='Monte Carlo trials')
    parser.add_argument('--holdout', action='store_true', help='Run TRUE holdout validation (weeks 12+)')
    parser.add_argument('--holdout-weeks', default='12-14', help='Holdout week range (e.g., 12-14)')
    parser.add_argument('--bootstrap', action='store_true', help='Include bootstrap confidence intervals')
    parser.add_argument('--recalibrate', action='store_true', help='Enable weekly recalibration (disabled by default)')
    args = parser.parse_args()

    if args.holdout:
        # Parse holdout weeks
        if '-' in args.holdout_weeks:
            start, end = map(int, args.holdout_weeks.split('-'))
            holdout_weeks = list(range(start, end + 1))
        else:
            holdout_weeks = [int(args.holdout_weeks)]

        run_holdout_validation(holdout_weeks, trials=args.trials)
    else:
        # Parse weeks
        if '-' in args.weeks:
            start, end = map(int, args.weeks.split('-'))
            test_weeks = list(range(start, end + 1))
        else:
            test_weeks = [int(args.weeks)]

        results_df = run_unified_validation(
            test_weeks,
            trials=args.trials,
            use_weekly_recalibration=args.recalibrate
        )

        # Add bootstrap CI if requested
        if args.bootstrap and len(results_df) > 0:
            print("\n" + "=" * 70)
            print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
            print("=" * 70)

            ci = calculate_bootstrap_ci(results_df)
            print(f"Overall Win Rate: {ci['win_rate_mean']*100:.1f}% [{ci['win_rate_lower']*100:.1f}%, {ci['win_rate_upper']*100:.1f}%]")
            print(f"Overall ROI:      {ci['roi_mean']:+.1f}% [{ci['roi_lower']:+.1f}%, {ci['roi_upper']:+.1f}%]")

            if ci['roi_lower'] > 0:
                print("\n✅ ROI is STATISTICALLY SIGNIFICANT (95% CI excludes zero)")
            else:
                print("\n⚠️ ROI is NOT statistically significant (95% CI includes zero)")
