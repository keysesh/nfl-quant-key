#!/usr/bin/env python3
"""
Enhanced TD Model with Walk-Forward Validation

Features:
- Trailing TDs, targets, carries, target_share (from weekly stats)
- Red zone features: rz_target_share, rz_carry_share, gl_carry_share (from PBP)
- Opponent defense: opp_rz_td_allowed, opp_pass_td_allowed (from PBP)
- Team efficiency: team_rz_td_rate (from PBP)
- Context: vegas_total, position, is_home

All features use shift(1) or historical data to prevent leakage.

Usage:
    python scripts/backtest/walk_forward_td_enhanced.py
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.stats import poisson
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import IsotonicRegression

from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# RED ZONE FEATURE EXTRACTION (from PBP)
# ============================================================================

def compute_player_rz_features(pbp: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """
    Compute player red zone features using only data before max_global_week.

    Features:
    - rz_targets_per_game: Red zone targets per game
    - rz_carries_per_game: Red zone carries per game
    - gl_carries_per_game: Goal line (<=5 yds) carries per game
    - rz_td_rate: TD conversion rate in red zone
    """
    # Filter to historical data only
    pbp = pbp[pbp['global_week'] < max_global_week].copy()

    if len(pbp) == 0:
        return pd.DataFrame()

    # Red zone plays (within 20 yards)
    rz = pbp[pbp['yardline_100'] <= 20].copy()

    # Goal line plays (within 5 yards)
    gl = pbp[pbp['yardline_100'] <= 5].copy()

    # Count games per player
    player_games = pbp.groupby('player_norm')['game_id'].nunique().reset_index()
    player_games.columns = ['player_norm', 'games_played']

    # RZ targets per player
    rz_pass = rz[rz['play_type'] == 'pass']
    rz_targets = rz_pass.groupby('receiver_norm').agg({
        'play_id': 'count',
        'pass_touchdown': 'sum'
    }).reset_index()
    rz_targets.columns = ['player_norm', 'rz_targets', 'rz_pass_tds']

    # RZ carries per player
    rz_rush = rz[rz['play_type'] == 'run']
    rz_carries = rz_rush.groupby('rusher_norm').agg({
        'play_id': 'count',
        'rush_touchdown': 'sum'
    }).reset_index()
    rz_carries.columns = ['player_norm', 'rz_carries', 'rz_rush_tds']

    # Goal line carries
    gl_rush = gl[gl['play_type'] == 'run']
    gl_carries = gl_rush.groupby('rusher_norm').size().reset_index(name='gl_carries')
    gl_carries.columns = ['player_norm', 'gl_carries']

    # Merge all
    features = player_games.copy()
    features = features.merge(rz_targets, on='player_norm', how='left')
    features = features.merge(rz_carries, on='player_norm', how='left')
    features = features.merge(gl_carries, on='player_norm', how='left')

    # Fill NaN with 0
    for col in ['rz_targets', 'rz_pass_tds', 'rz_carries', 'rz_rush_tds', 'gl_carries']:
        features[col] = features[col].fillna(0)

    # Compute per-game rates
    features['rz_targets_per_game'] = features['rz_targets'] / features['games_played']
    features['rz_carries_per_game'] = features['rz_carries'] / features['games_played']
    features['gl_carries_per_game'] = features['gl_carries'] / features['games_played']

    # RZ TD rate (combined)
    features['rz_opportunities'] = features['rz_targets'] + features['rz_carries']
    features['rz_tds'] = features['rz_pass_tds'] + features['rz_rush_tds']
    features['rz_td_rate'] = np.where(
        features['rz_opportunities'] > 0,
        features['rz_tds'] / features['rz_opportunities'],
        0.0
    )

    return features[['player_norm', 'games_played', 'rz_targets_per_game',
                     'rz_carries_per_game', 'gl_carries_per_game', 'rz_td_rate']]


def compute_opponent_defense(pbp: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """
    Compute opponent defensive features using only data before max_global_week.

    Features:
    - opp_tds_allowed_per_game: Total TDs allowed per game
    - opp_rz_td_rate: Red zone TD rate allowed
    - opp_pass_tds_allowed: Pass TDs allowed per game
    - opp_rush_tds_allowed: Rush TDs allowed per game
    """
    pbp = pbp[pbp['global_week'] < max_global_week].copy()

    if len(pbp) == 0:
        return pd.DataFrame()

    # Games per defense
    def_games = pbp.groupby('defteam')['game_id'].nunique().reset_index()
    def_games.columns = ['team', 'games']

    # TDs allowed
    tds_allowed = pbp.groupby('defteam').agg({
        'touchdown': 'sum',
        'pass_touchdown': 'sum',
        'rush_touchdown': 'sum'
    }).reset_index()
    tds_allowed.columns = ['team', 'tds_allowed', 'pass_tds_allowed', 'rush_tds_allowed']

    # Red zone defense
    rz = pbp[pbp['yardline_100'] <= 20]
    rz_def = rz.groupby('defteam').agg({
        'play_id': 'count',
        'touchdown': 'sum'
    }).reset_index()
    rz_def.columns = ['team', 'rz_plays_faced', 'rz_tds_allowed']

    # Merge
    defense = def_games.merge(tds_allowed, on='team', how='left')
    defense = defense.merge(rz_def, on='team', how='left')

    # Compute rates
    defense['opp_tds_allowed_per_game'] = defense['tds_allowed'] / defense['games']
    defense['opp_pass_tds_allowed'] = defense['pass_tds_allowed'] / defense['games']
    defense['opp_rush_tds_allowed'] = defense['rush_tds_allowed'] / defense['games']
    defense['opp_rz_td_rate'] = np.where(
        defense['rz_plays_faced'] > 0,
        defense['rz_tds_allowed'] / defense['rz_plays_faced'],
        0.13  # League average
    )

    return defense[['team', 'opp_tds_allowed_per_game', 'opp_pass_tds_allowed',
                    'opp_rush_tds_allowed', 'opp_rz_td_rate']]


def compute_team_rz_efficiency(pbp: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """
    Compute team red zone efficiency using only data before max_global_week.
    """
    pbp = pbp[pbp['global_week'] < max_global_week].copy()

    if len(pbp) == 0:
        return pd.DataFrame()

    rz = pbp[pbp['yardline_100'] <= 20]

    team_rz = rz.groupby('posteam').agg({
        'play_id': 'count',
        'touchdown': 'sum'
    }).reset_index()
    team_rz.columns = ['team', 'rz_plays', 'rz_tds']
    team_rz['team_rz_td_rate'] = team_rz['rz_tds'] / team_rz['rz_plays']

    return team_rz[['team', 'team_rz_td_rate']]


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    """Load all required data sources."""
    logger.info("Loading data...")

    # Weekly stats
    stats = pd.read_parquet(DATA_DIR / 'nflverse' / 'weekly_stats.parquet')
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']
    stats['total_tds'] = stats['receiving_tds'].fillna(0) + stats['rushing_tds'].fillna(0)
    stats['scored_td'] = (stats['total_tds'] >= 1).astype(int)

    # Filter to skill positions
    stats = stats[stats['position'].isin(['WR', 'RB', 'TE'])].copy()

    logger.info(f"  Stats: {len(stats):,} rows")

    # PBP for red zone features
    pbp = pd.read_parquet(DATA_DIR / 'nflverse' / 'pbp.parquet')
    pbp['global_week'] = (pbp['season'] - 2023) * 18 + pbp['week']

    # Normalize player names in PBP
    pbp['receiver_norm'] = pbp['receiver_player_name'].apply(
        lambda x: normalize_player_name(x) if pd.notna(x) else None
    )
    pbp['rusher_norm'] = pbp['rusher_player_name'].apply(
        lambda x: normalize_player_name(x) if pd.notna(x) else None
    )

    # Create player_norm for game counting
    pbp['player_norm'] = pbp['receiver_norm'].fillna(pbp['rusher_norm'])

    logger.info(f"  PBP: {len(pbp):,} rows")

    return stats, pbp


def compute_trailing_stats(stats: pd.DataFrame, max_global_week: int) -> pd.DataFrame:
    """Compute trailing player stats using only data before max_global_week."""
    hist = stats[stats['global_week'] < max_global_week].copy()
    hist = hist.sort_values(['player_norm', 'season', 'week'])

    # EWMA with shift(1) to prevent leakage
    ewma_span = 4

    trailing_cols = {
        'total_tds': 'trailing_tds',
        'targets': 'trailing_targets',
        'carries': 'trailing_carries',
        'target_share': 'trailing_target_share',
        'receiving_yards': 'trailing_rec_yds',
        'rushing_yards': 'trailing_rush_yds',
    }

    for src, dst in trailing_cols.items():
        if src in hist.columns:
            hist[dst] = hist.groupby('player_norm')[src].transform(
                lambda x: x.shift(1).ewm(span=ewma_span, min_periods=1).mean()
            )

    # Get latest trailing value per player
    latest = hist.sort_values('global_week').groupby('player_norm').last().reset_index()

    return latest[['player_norm'] + list(trailing_cols.values())]


def prepare_features(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
    test_global_week: int
) -> pd.DataFrame:
    """
    Prepare all features for test week using only historical data.
    """
    # Get test week data
    test_data = stats[stats['global_week'] == test_global_week].copy()

    if len(test_data) == 0:
        return pd.DataFrame()

    # 1. Trailing stats (from weekly stats)
    trailing = compute_trailing_stats(stats, test_global_week)
    test_data = test_data.merge(trailing, on='player_norm', how='left')

    # 2. Red zone features (from PBP)
    rz_features = compute_player_rz_features(pbp, test_global_week)
    if len(rz_features) > 0:
        test_data = test_data.merge(rz_features, on='player_norm', how='left')

    # 3. Opponent defense (from PBP)
    opp_defense = compute_opponent_defense(pbp, test_global_week)
    if len(opp_defense) > 0 and 'opponent_team' in test_data.columns:
        test_data = test_data.merge(
            opp_defense,
            left_on='opponent_team',
            right_on='team',
            how='left'
        )

    # 4. Team RZ efficiency (from PBP)
    team_rz = compute_team_rz_efficiency(pbp, test_global_week)
    if len(team_rz) > 0 and 'recent_team' in test_data.columns:
        test_data = test_data.merge(
            team_rz,
            left_on='recent_team',
            right_on='team',
            how='left',
            suffixes=('', '_team')
        )

    # 5. Position encoding
    test_data['is_rb'] = (test_data['position'] == 'RB').astype(int)
    test_data['is_wr'] = (test_data['position'] == 'WR').astype(int)
    test_data['is_te'] = (test_data['position'] == 'TE').astype(int)

    # Fill NaN with sensible defaults (league averages)
    fill_values = {
        'trailing_tds': 0.2,
        'trailing_targets': 4.0,
        'trailing_carries': 5.0,
        'trailing_target_share': 0.10,
        'trailing_rec_yds': 30.0,
        'trailing_rush_yds': 25.0,
        'rz_targets_per_game': 0.5,
        'rz_carries_per_game': 0.5,
        'gl_carries_per_game': 0.2,
        'rz_td_rate': 0.25,
        'opp_tds_allowed_per_game': 2.5,
        'opp_pass_tds_allowed': 1.5,
        'opp_rush_tds_allowed': 0.8,
        'opp_rz_td_rate': 0.13,
        'team_rz_td_rate': 0.13,
    }

    for col, val in fill_values.items():
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(val)
        else:
            test_data[col] = val

    return test_data


def prepare_training_data(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
    max_global_week: int
) -> pd.DataFrame:
    """
    Prepare training data with all features.
    Uses data from global_week < max_global_week - 1 (1 week gap).
    """
    # Training weeks
    train_weeks = sorted(stats[stats['global_week'] < max_global_week - 1]['global_week'].unique())

    if len(train_weeks) < 3:
        return pd.DataFrame()

    all_train = []

    for week in train_weeks[-20:]:  # Use last 20 weeks for efficiency
        week_features = prepare_features(stats, pbp, week)
        if len(week_features) > 0:
            all_train.append(week_features)

    if not all_train:
        return pd.DataFrame()

    train_df = pd.concat(all_train, ignore_index=True)

    return train_df


# ============================================================================
# MODEL TRAINING AND PREDICTION
# ============================================================================

FEATURE_COLS = [
    'trailing_tds',
    'trailing_targets',
    'trailing_carries',
    'trailing_target_share',
    'trailing_rec_yds',
    'trailing_rush_yds',
    'rz_targets_per_game',
    'rz_carries_per_game',
    'gl_carries_per_game',
    'rz_td_rate',
    'opp_tds_allowed_per_game',
    'opp_pass_tds_allowed',
    'opp_rush_tds_allowed',
    'opp_rz_td_rate',
    'team_rz_td_rate',
    'is_rb',
    'is_wr',
    'is_te',
]


def train_td_model(train_df: pd.DataFrame):
    """
    Train XGBoost model to predict P(score TD).
    Returns model and calibrator.
    """
    # Prepare features
    available_features = [f for f in FEATURE_COLS if f in train_df.columns]

    X = train_df[available_features].fillna(0)
    y = train_df['scored_td']

    # Drop rows with NaN target
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]

    if len(X) < 100:
        return None, None, available_features

    # Train/calibration split (80/20)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Calibrate on held-out data
    calib_probs = model.predict_proba(X_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(calib_probs, y_calib)

    return model, calibrator, available_features


def predict_td_probability(
    model,
    calibrator,
    features: list,
    test_df: pd.DataFrame
) -> np.ndarray:
    """Predict calibrated P(score TD)."""
    X = test_df[features].fillna(0)

    raw_probs = model.predict_proba(X)[:, 1]

    if calibrator is not None:
        calibrated_probs = calibrator.predict(raw_probs)
    else:
        calibrated_probs = raw_probs

    return calibrated_probs


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================

def run_walk_forward():
    """Run walk-forward validation for enhanced TD model."""
    logger.info("=" * 70)
    logger.info("ENHANCED TD MODEL - WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    # Load data
    stats, pbp = load_data()

    # Get test weeks (start from week 24 = 2024 week 6, need history)
    all_weeks = sorted(stats['global_week'].unique())
    test_weeks = [w for w in all_weeks if w >= 28]  # 2024 week 10+

    logger.info(f"Testing {len(test_weeks)} weeks")

    all_results = []

    for test_week in test_weeks:
        # Prepare training data (weeks < test_week - 1)
        train_df = prepare_training_data(stats, pbp, test_week)

        if len(train_df) < 200:
            continue

        # Train model
        model, calibrator, features = train_td_model(train_df)

        if model is None:
            continue

        # Prepare test data
        test_df = prepare_features(stats, pbp, test_week)

        if len(test_df) == 0:
            continue

        # Predict
        probs = predict_td_probability(model, calibrator, features, test_df)
        test_df['p_score_td'] = probs

        # Record results
        season = 2024 if test_week <= 36 else 2025
        week_num = test_week - 18 if test_week <= 36 else test_week - 36

        for idx, row in test_df.iterrows():
            all_results.append({
                'season': season,
                'week': week_num,
                'global_week': test_week,
                'player': row.get('player_display_name', row['player_norm']),
                'position': row['position'],
                'team': row.get('recent_team', ''),
                'opponent': row.get('opponent_team', ''),
                'trailing_tds': row.get('trailing_tds', 0),
                'rz_targets_pg': row.get('rz_targets_per_game', 0),
                'gl_carries_pg': row.get('gl_carries_per_game', 0),
                'p_score_td': row['p_score_td'],
                'actual_tds': row['total_tds'],
                'scored_td': row['scored_td'],
            })

        # Week summary
        week_df = test_df.copy()

        # High confidence bets (p >= 0.40 since base rate ~20%)
        high_conf = week_df[week_df['p_score_td'] >= 0.40]
        if len(high_conf) > 0:
            hits = high_conf['scored_td'].sum()
            total = len(high_conf)
            wr = hits / total
            logger.info(f"Week {test_week}: {hits}/{total} ({wr*100:.1f}%) @ 40%+ conf")

    if not all_results:
        logger.error("No results generated")
        return

    results_df = pd.DataFrame(all_results)

    # Save results
    output_path = DATA_DIR / 'backtest' / 'walk_forward_td_enhanced.csv'
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(results_df):,} predictions to {output_path}")

    # ========================================================================
    # ANALYSIS BY CONFIDENCE THRESHOLD
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS BY CONFIDENCE THRESHOLD")
    logger.info("=" * 70)
    logger.info(f"{'Threshold':>10} {'N':>8} {'Hits':>8} {'WR':>10} {'ROI':>10}")
    logger.info("-" * 70)

    # Base TD rate
    base_rate = results_df['scored_td'].mean()
    logger.info(f"{'Base rate':>10} {len(results_df):>8} {results_df['scored_td'].sum():>8} {base_rate*100:>9.1f}% {'N/A':>10}")

    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        subset = results_df[results_df['p_score_td'] >= thresh]
        if len(subset) >= 20:
            hits = subset['scored_td'].sum()
            n = len(subset)
            wr = hits / n
            # ROI at typical anytime TD odds (-120 = 0.833 payout)
            roi = (wr * 0.833 - (1 - wr)) * 100
            logger.info(f"{thresh*100:>9.0f}% {n:>8} {hits:>8} {wr*100:>9.1f}% {roi:>+9.1f}%")

    # ========================================================================
    # RESULTS BY POSITION
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS BY POSITION @ 40%+ Confidence")
    logger.info("=" * 70)

    high_conf = results_df[results_df['p_score_td'] >= 0.40]
    for pos in ['RB', 'WR', 'TE']:
        pos_df = high_conf[high_conf['position'] == pos]
        if len(pos_df) >= 20:
            hits = pos_df['scored_td'].sum()
            n = len(pos_df)
            wr = hits / n
            roi = (wr * 0.833 - (1 - wr)) * 100
            logger.info(f"{pos}: {n} bets, {wr*100:.1f}% WR, {roi:+.1f}% ROI")

    # ========================================================================
    # FEATURE IMPORTANCE (from last model)
    # ========================================================================
    if model is not None:
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE IMPORTANCE")
        logger.info("=" * 70)

        importance = dict(zip(features, model.feature_importances_))
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {feat:<30}: {imp:.3f}")


if __name__ == '__main__':
    run_walk_forward()
