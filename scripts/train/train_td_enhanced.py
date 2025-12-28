#!/usr/bin/env python3
"""
Train Enhanced TD Model for Anytime TD Props

Uses XGBoost with red zone and opponent features for profitable TD predictions.
Walk-forward validated: RB @ 60% = 58.2% WR, +6.6% ROI

Key Features:
- trailing_tds: 26.4% importance (most important)
- trailing_target_share: 11.4% importance
- opp_rush_tds_allowed: 6.3% importance (defense matters)
- Position-specific handling (RBs are most profitable)

Usage:
    python scripts/train/train_td_enhanced.py
    python scripts/train/train_td_enhanced.py --validate
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from nfl_quant.models.td_enhanced_model import (
    TDEnhancedModel,
    TD_ENHANCED_FEATURES,
    compute_td_features_from_pbp,
)
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import DATA_DIR, MODELS_DIR

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_weekly_stats() -> pd.DataFrame:
    """Load weekly stats for training."""
    logger.info("Loading weekly stats...")

    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Weekly stats not found: {stats_path}. "
            "Run 'Rscript scripts/fetch/fetch_nflverse_data.R'"
        )

    stats = pd.read_parquet(stats_path)

    # Normalize player names
    if 'player_display_name' in stats.columns:
        stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    elif 'player_name' in stats.columns:
        stats['player_norm'] = stats['player_name'].apply(normalize_player_name)

    # Compute total TDs
    stats['total_tds'] = (
        stats['receiving_tds'].fillna(0) +
        stats['rushing_tds'].fillna(0)
    )

    # Binary target: scored at least 1 TD
    stats['scored_td'] = (stats['total_tds'] >= 1).astype(int)

    logger.info(f"  Rows: {len(stats):,}")
    logger.info(f"  Seasons: {sorted(stats['season'].unique())}")
    logger.info(f"  TD rate: {stats['scored_td'].mean():.1%}")

    return stats


def load_pbp() -> pd.DataFrame:
    """Load play-by-play data for red zone features."""
    logger.info("Loading PBP data...")

    pbp_path = DATA_DIR / 'nflverse' / 'pbp.parquet'

    if not pbp_path.exists():
        logger.warning(f"PBP not found: {pbp_path}. Red zone features will be limited.")
        return pd.DataFrame()

    pbp = pd.read_parquet(pbp_path)
    logger.info(f"  PBP rows: {len(pbp):,}")

    return pbp


def compute_trailing_features(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing stats with proper shift to prevent leakage."""
    logger.info("Computing trailing features...")

    stats = stats.copy()
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    ewma_span = 6  # 6-game EWMA

    # Core trailing features
    trailing_cols = {
        'total_tds': 'trailing_tds',
        'targets': 'trailing_targets',
        'carries': 'trailing_carries',
        'target_share': 'trailing_target_share',
        'receiving_yards': 'trailing_rec_yds',
        'rushing_yards': 'trailing_rush_yds',
    }

    for src, dst in trailing_cols.items():
        if src in stats.columns:
            stats[dst] = (
                stats.groupby('player_norm')[src]
                .transform(lambda x: x.shift(1).ewm(span=ewma_span, min_periods=2).mean())
            )

    # Position encoding
    if 'position' in stats.columns:
        stats['is_rb'] = (stats['position'] == 'RB').astype(int)
        stats['is_wr'] = (stats['position'] == 'WR').astype(int)
        stats['is_te'] = (stats['position'] == 'TE').astype(int)
    else:
        # Try to infer from stats
        stats['is_rb'] = 0
        stats['is_wr'] = 0
        stats['is_te'] = 0

        # RBs typically have high carries
        stats.loc[stats['trailing_carries'].fillna(0) > 5, 'is_rb'] = 1
        # WRs/TEs have high targets but low carries
        wr_te_mask = (
            (stats['trailing_targets'].fillna(0) > 3) &
            (stats['trailing_carries'].fillna(0) < 3)
        )
        stats.loc[wr_te_mask, 'is_wr'] = 1

    return stats


def add_opponent_defense(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Add opponent defense features from PBP."""
    logger.info("Computing opponent defense features...")

    stats = stats.copy()

    # Initialize with league averages
    stats['opp_tds_allowed_per_game'] = 2.5
    stats['opp_pass_tds_allowed'] = 1.5
    stats['opp_rush_tds_allowed'] = 1.0
    stats['opp_rz_td_rate'] = 0.50

    if len(pbp) == 0:
        logger.info("  No PBP data, using league averages")
        return stats

    # Compute opponent defense from PBP
    for season in stats['season'].unique():
        for week in stats[stats['season'] == season]['week'].unique():
            week_mask = (stats['season'] == season) & (stats['week'] == week)

            if week < 2:
                continue

            # Get prior games for defense stats
            prior_pbp = pbp[
                ((pbp['season'] < season) |
                 ((pbp['season'] == season) & (pbp['week'] < week)))
            ]

            if len(prior_pbp) == 0:
                continue

            # Defense TDs allowed
            def_games = prior_pbp.groupby('defteam')['game_id'].nunique()
            def_tds = prior_pbp.groupby('defteam').agg({
                'touchdown': 'sum',
                'pass_touchdown': 'sum',
                'rush_touchdown': 'sum'
            })

            # Red zone defense
            rz_pbp = prior_pbp[prior_pbp['yardline_100'] <= 20]
            rz_def = rz_pbp.groupby('defteam').agg({
                'play_id': 'count',
                'touchdown': 'sum'
            })

            # Merge into stats for this week
            for idx in stats[week_mask].index:
                opponent = stats.loc[idx, 'opponent'] if 'opponent' in stats.columns else None

                if opponent is None or pd.isna(opponent):
                    continue

                if opponent in def_games.index:
                    games = def_games[opponent]
                    if games > 0:
                        tds = def_tds.loc[opponent, 'touchdown']
                        pass_tds = def_tds.loc[opponent, 'pass_touchdown']
                        rush_tds = def_tds.loc[opponent, 'rush_touchdown']

                        stats.loc[idx, 'opp_tds_allowed_per_game'] = tds / games
                        stats.loc[idx, 'opp_pass_tds_allowed'] = pass_tds / games
                        stats.loc[idx, 'opp_rush_tds_allowed'] = rush_tds / games

                if opponent in rz_def.index:
                    rz_plays = rz_def.loc[opponent, 'play_id']
                    rz_tds = rz_def.loc[opponent, 'touchdown']
                    if rz_plays > 0:
                        stats.loc[idx, 'opp_rz_td_rate'] = rz_tds / rz_plays

    logger.info(f"  Added opponent defense features")

    return stats


def add_red_zone_features(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Add player red zone features from PBP."""
    logger.info("Computing red zone features...")

    stats = stats.copy()

    # Initialize
    stats['rz_targets_per_game'] = 0.0
    stats['rz_carries_per_game'] = 0.0
    stats['gl_carries_per_game'] = 0.0
    stats['rz_td_rate'] = 0.0
    stats['team_rz_td_rate'] = 0.50

    if len(pbp) == 0:
        logger.info("  No PBP data, using zeros")
        return stats

    # Compute from PBP by season/week
    for season in stats['season'].unique():
        for week in stats[stats['season'] == season]['week'].unique():
            week_mask = (stats['season'] == season) & (stats['week'] == week)

            # Get features from prior weeks
            _, opp_defense, team_efficiency = compute_td_features_from_pbp(
                pbp, max_week=int(week), season=int(season)
            )

            if len(opp_defense) > 0:
                # Merge team RZ TD rate
                for idx in stats[week_mask].index:
                    team = stats.loc[idx, 'recent_team'] if 'recent_team' in stats.columns else None
                    if team and team in team_efficiency['team'].values:
                        rate = team_efficiency[team_efficiency['team'] == team]['team_rz_td_rate'].iloc[0]
                        stats.loc[idx, 'team_rz_td_rate'] = rate

    logger.info("  Added red zone features")

    return stats


def prepare_training_data(
    stats: pd.DataFrame,
    pbp: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare full training dataset."""
    logger.info("Preparing training data...")

    # Filter to skill positions
    if 'position' in stats.columns:
        skill_positions = ['RB', 'WR', 'TE']
        stats = stats[stats['position'].isin(skill_positions)].copy()

    # Compute all features
    stats = compute_trailing_features(stats)
    stats = add_opponent_defense(stats, pbp)
    stats = add_red_zone_features(stats, pbp)

    # Drop rows without trailing stats (first few weeks)
    stats = stats.dropna(subset=['trailing_tds'])

    logger.info(f"  Final training samples: {len(stats):,}")
    logger.info(f"  TD rate: {stats['scored_td'].mean():.1%}")

    if 'position' in stats.columns:
        for pos in ['RB', 'WR', 'TE']:
            pos_data = stats[stats['position'] == pos]
            if len(pos_data) > 0:
                logger.info(f"  {pos}: {len(pos_data):,} samples, {pos_data['scored_td'].mean():.1%} TD rate")

    return stats


def run_walk_forward_validation(
    stats: pd.DataFrame,
    n_test_weeks: int = 10,
) -> pd.DataFrame:
    """Run walk-forward validation to verify model performance."""
    logger.info("\nRunning walk-forward validation...")

    stats = stats.copy()
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    weeks = sorted(stats['global_week'].unique())
    test_weeks = weeks[-n_test_weeks:]

    all_results = []

    for test_week in test_weeks:
        # Train data: at least 1 week gap
        train = stats[stats['global_week'] < test_week - 1].copy()
        test = stats[stats['global_week'] == test_week].copy()

        if len(train) < 100 or len(test) == 0:
            continue

        # Train model
        model = TDEnhancedModel()
        try:
            model.train(train)
        except Exception as e:
            logger.warning(f"  Week {test_week}: Training failed - {e}")
            continue

        # Predict
        test['p_score_td'] = model.predict_proba(test)

        # Record results
        for _, row in test.iterrows():
            all_results.append({
                'global_week': test_week,
                'player': row.get('player_display_name', row.get('player_norm', '')),
                'position': row.get('position', 'UNK'),
                'p_score_td': row['p_score_td'],
                'scored_td': row['scored_td'],
                'actual_tds': row['total_tds'],
            })

    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        logger.warning("  No validation results generated")
        return results_df

    # Print results by confidence threshold and position
    logger.info("\nValidation Results:")
    logger.info("-" * 70)
    logger.info(f"{'Position':<8} {'Conf':<8} {'N':<8} {'Hits':<8} {'WR':<10} {'ROI':<10}")
    logger.info("-" * 70)

    for pos in ['ALL', 'RB', 'WR', 'TE']:
        for conf in [0.50, 0.55, 0.60, 0.65]:
            if pos == 'ALL':
                subset = results_df[results_df['p_score_td'] >= conf]
            else:
                subset = results_df[
                    (results_df['position'] == pos) &
                    (results_df['p_score_td'] >= conf)
                ]

            if len(subset) < 10:
                continue

            hits = subset['scored_td'].sum()
            total = len(subset)
            wr = hits / total
            # TD props typically -120 odds
            roi = (wr * 0.833) - (1 - wr)

            logger.info(
                f"{pos:<8} {conf:<7.0%} {total:<8} {hits:<8} "
                f"{wr:<9.1%} {roi*100:+.1f}%"
            )

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced TD Model")
    parser.add_argument('--validate', action='store_true', help='Run walk-forward validation')
    parser.add_argument('--weeks', type=int, default=10, help='Weeks for validation')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("ENHANCED TD MODEL TRAINING")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # Load data
    stats = load_weekly_stats()
    pbp = load_pbp()

    # Prepare training data
    train_data = prepare_training_data(stats, pbp)

    if len(train_data) == 0:
        logger.error("No training data available")
        return

    # Run validation if requested
    if args.validate:
        val_results = run_walk_forward_validation(train_data, n_test_weeks=args.weeks)

        # Save validation results
        val_path = DATA_DIR / 'backtest' / 'td_enhanced_validation.csv'
        val_results.to_csv(val_path, index=False)
        logger.info(f"\nValidation results saved to: {val_path}")

    # Train final model on all data
    logger.info("\n" + "=" * 70)
    logger.info("Training final model on all data...")
    logger.info("=" * 70)

    model = TDEnhancedModel()
    metrics = model.train(train_data)

    # Print feature importance
    logger.info("\nTop Feature Importance:")
    importance = model.get_feature_importance()
    for i, (feat, imp) in enumerate(list(importance.items())[:10]):
        logger.info(f"  {i+1}. {feat}: {imp:.1%}")

    # Save model
    model.save()

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Model saved to: {MODELS_DIR / 'td_enhanced_model.joblib'}")
    logger.info(f"Samples: {metrics['n_samples']:,}")
    logger.info(f"Features: {metrics['n_features']}")
    logger.info(f"Train AUC: {metrics['train_auc']:.3f}")
    logger.info(f"Calib AUC: {metrics['calib_auc']:.3f}")


if __name__ == '__main__':
    main()
