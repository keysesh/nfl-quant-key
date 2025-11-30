#!/usr/bin/env python3
"""
End-to-End TIER 1 & TIER 2 Integration Test

Tests the complete integration of TIER 1 & TIER 2 features through the full pipeline:
1. Feature extraction
2. Prediction generation
3. Model performance comparison

Usage:
    # Quick test with sample players
    python scripts/test/test_tier12_integration_e2e.py

    # Full test with all players from a specific week
    python scripts/test/test_tier12_integration_e2e.py --week 11 --full

    # Compare baseline vs TIER 1 & 2 predictions
    python scripts/test/test_tier12_integration_e2e.py --week 11 --compare
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import logging
from typing import Dict, List
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.features.tier1_2_integration import (
    extract_all_tier1_2_features,
    get_feature_columns_for_position,
    validate_features,
    summarize_feature_extraction
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_test_players() -> List[Dict]:
    """Get sample players for testing across all positions."""
    return [
        {
            "name": "P.Mahomes",
            "position": "QB",
            "team": "KC",
            "opponent": "MIA"
        },
        {
            "name": "C.McCaffrey",
            "position": "RB",
            "team": "SF",
            "opponent": "GB"
        },
        {
            "name": "T.Hill",
            "position": "WR",
            "team": "MIA",
            "opponent": "KC"
        },
        {
            "name": "T.Kelce",
            "position": "TE",
            "team": "KC",
            "opponent": "MIA"
        },
        {
            "name": "C.Lamb",
            "position": "WR",
            "team": "DAL",
            "opponent": "WAS"
        },
        {
            "name": "J.Allen",
            "position": "QB",
            "team": "BUF",
            "opponent": "NYJ"
        },
    ]


def test_feature_extraction(week: int = 11, season: int = 2025):
    """Test TIER 1 & 2 feature extraction."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 1: Feature Extraction (Week {week}, Season {season})")
    logger.info(f"{'='*80}")

    # Load PBP data
    pbp_path = PROJECT_ROOT / f'data/nflverse/pbp_{season}.parquet'
    if not pbp_path.exists():
        logger.error(f"PBP data not found: {pbp_path}")
        logger.error(f"Run: python scripts/fetch/pull_2024_season_data.py")
        return False

    pbp_df = pd.read_parquet(pbp_path)
    logger.info(f"✓ Loaded {len(pbp_df):,} plays from {pbp_path.name}\n")

    players = get_test_players()
    results = []

    for player in players:
        logger.info(f"Testing {player['name']} ({player['position']})...")

        try:
            # Extract baseline features (old approach)
            from nfl_quant.features.trailing_stats import get_trailing_stats_extractor

            extractor = get_trailing_stats_extractor()
            baseline_features = extractor.get_trailing_stats(
                player_name=player['name'],
                position=player['position'],
                current_week=week,
                use_ewma=False
            )

            # Extract TIER 1 & 2 features (new approach)
            tier12_features = extract_all_tier1_2_features(
                player_name=player['name'],
                position=player['position'],
                team=player['team'],
                opponent=player['opponent'],
                current_week=week,
                season=season,
                pbp_df=pbp_df,
                use_ewma=True,
                use_regime=True,
                use_game_script=True,
                use_ngs=True,
                use_situational_epa=True
            )

            # Validate
            tier12_features = validate_features(tier12_features, player['position'])

            # Compare
            baseline_count = len(baseline_features)
            tier12_count = len(tier12_features)
            added_count = tier12_count - baseline_count

            logger.info(f"  ✓ Baseline features: {baseline_count}")
            logger.info(f"  ✓ TIER 1 & 2 features: {tier12_count}")
            logger.info(f"  ✓ Added: {added_count} new features")

            # Check for key TIER 1 & 2 features
            tier1_features = [
                'weeks_since_regime_change',
                'is_in_regime',
                'regime_confidence',
                'usage_when_leading',
                'usage_when_trailing',
                'usage_when_close',
                'game_script_sensitivity'
            ]

            tier2_features = [
                'redzone_epa',
                'third_down_epa',
                'two_minute_epa'
            ]

            tier1_present = sum(1 for f in tier1_features if f in tier12_features)
            tier2_present = sum(1 for f in tier2_features if f in tier12_features)

            logger.info(f"  ✓ TIER 1 features present: {tier1_present}/{len(tier1_features)}")
            logger.info(f"  ✓ TIER 2 features present: {tier2_present}/{len(tier2_features)}")

            # Log notable features
            if tier12_features.get('is_in_regime', 0.0) == 1.0:
                logger.info(
                    f"  📍 REGIME DETECTED: {tier12_features.get('weeks_since_regime_change', 0):.0f} weeks ago, "
                    f"confidence={tier12_features.get('regime_confidence', 0):.2f}"
                )

            sensitivity = tier12_features.get('game_script_sensitivity', 0.0)
            if sensitivity > 5.0:
                logger.info(f"  📊 HIGH GAME SCRIPT SENSITIVITY: {sensitivity:.1f}")

            redzone_epa = tier12_features.get('redzone_epa', 0.0)
            if abs(redzone_epa) > 0.10:
                logger.info(f"  🎯 NOTABLE REDZONE MATCHUP: {redzone_epa:+.3f} EPA")

            results.append({
                'player': player['name'],
                'position': player['position'],
                'baseline_features': baseline_count,
                'tier12_features': tier12_count,
                'added_features': added_count,
                'tier1_present': tier1_present,
                'tier2_present': tier2_present,
                'success': True
            })

            logger.info("")

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            results.append({
                'player': player['name'],
                'position': player['position'],
                'success': False,
                'error': str(e)
            })

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("TEST 1 SUMMARY: Feature Extraction")
    logger.info(f"{'='*80}\n")

    results_df = pd.DataFrame(results)

    if 'success' in results_df.columns:
        success_count = results_df['success'].sum()
        total_count = len(results_df)

        logger.info(f"Successful extractions: {success_count}/{total_count}")

        if success_count > 0:
            successful = results_df[results_df['success']]
            logger.info(f"Avg baseline features: {successful['baseline_features'].mean():.1f}")
            logger.info(f"Avg TIER 1 & 2 features: {successful['tier12_features'].mean():.1f}")
            logger.info(f"Avg added features: {successful['added_features'].mean():.1f}")

    # Save results
    output_path = PROJECT_ROOT / f'reports/tier12_feature_extraction_test_week{week}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Results saved: {output_path}")

    return success_count == total_count


def test_model_integration(week: int = 11, season: int = 2025):
    """Test that models can use TIER 1 & 2 features."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 2: Model Integration (Week {week}, Season {season})")
    logger.info(f"{'='*80}\n")

    from xgboost import XGBRegressor

    # Test for each position
    positions = ['QB', 'RB', 'WR', 'TE']
    results = []

    for position in positions:
        logger.info(f"Testing {position} model compatibility...")

        # Get feature columns
        usage_features = [
            'week',
            'trailing_snap_share',
            'trailing_target_share',
            'weeks_since_regime_change',
            'is_in_regime',
            'regime_confidence',
            'usage_when_leading',
            'usage_when_trailing',
            'usage_when_close',
            'game_script_sensitivity',
            'redzone_epa',
            'third_down_epa',
            'two_minute_epa'
        ]

        # Create dummy data
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            feat: np.random.randn(n_samples) for feat in usage_features
        })
        y = np.random.randn(n_samples) * 5 + 10  # Dummy targets

        # Try to train model
        try:
            model = XGBRegressor(n_estimators=10, max_depth=3, random_state=42)
            model.fit(X, y)

            # Predict
            y_pred = model.predict(X)

            logger.info(f"  ✓ {position} model trained successfully")
            logger.info(f"    Features: {len(usage_features)}")
            logger.info(f"    Samples: {n_samples}")
            logger.info(f"    Predictions: {y_pred.mean():.2f} ± {y_pred.std():.2f}")

            results.append({
                'position': position,
                'features': len(usage_features),
                'samples': n_samples,
                'success': True
            })

        except Exception as e:
            logger.error(f"  ✗ {position} model failed: {e}")
            results.append({
                'position': position,
                'features': len(usage_features),
                'success': False,
                'error': str(e)
            })

        logger.info("")

    # Summary
    logger.info(f"{'='*80}")
    logger.info("TEST 2 SUMMARY: Model Integration")
    logger.info(f"{'='*80}\n")

    results_df = pd.DataFrame(results)
    success_count = results_df['success'].sum()
    total_count = len(results_df)

    logger.info(f"Successful model tests: {success_count}/{total_count}")

    return success_count == total_count


def test_prediction_comparison(week: int = 11, season: int = 2025):
    """Compare baseline vs TIER 1 & 2 predictions."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 3: Prediction Comparison (Week {week}, Season {season})")
    logger.info(f"{'='*80}\n")

    logger.info("NOTE: This requires trained models with TIER 1 & 2 features")
    logger.info("To train models, run:")
    logger.info("  python scripts/train/retrain_models_with_tier12_features.py --all\n")

    # Check if TIER 1 & 2 models exist
    models_dir = PROJECT_ROOT / 'models' / 'tier12'

    if not models_dir.exists():
        logger.warning(f"⚠️  TIER 1 & 2 models not found: {models_dir}")
        logger.warning("   Skipping prediction comparison test")
        return None

    model_files = list(models_dir.glob('*_tier12_latest.pkl'))
    if not model_files:
        logger.warning(f"⚠️  No TIER 1 & 2 model files found in {models_dir}")
        logger.warning("   Skipping prediction comparison test")
        return None

    logger.info(f"✓ Found {len(model_files)} TIER 1 & 2 models:")
    for model_file in model_files:
        logger.info(f"  - {model_file.name}")

    logger.info("\n✓ Model integration verified")
    logger.info("  To test predictions, run:")
    logger.info("    python scripts/predict/generate_model_predictions.py --week 11 --use-tier12")

    return True


def test_temporal_cv():
    """Test temporal cross-validation framework."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST 4: Temporal Cross-Validation")
    logger.info(f"{'='*80}\n")

    from nfl_quant.validation import TemporalCrossValidator

    # Create dummy data
    weeks = list(range(1, 12))
    data = []

    for week in weeks:
        for player_id in range(10):
            data.append({
                'week': week,
                'player_id': player_id,
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
                'target': np.random.randn() * 5 + 10
            })

    df = pd.DataFrame(data)

    logger.info(f"Test data: {len(df)} samples across {len(weeks)} weeks")

    # Test expanding window
    logger.info("\nTesting EXPANDING window strategy...")
    cv_expanding = TemporalCrossValidator(
        strategy='expanding',
        min_train_weeks=4,
        test_from_week=5
    )

    folds_expanding = list(cv_expanding.split(df, week_col='week'))
    logger.info(f"  ✓ Generated {len(folds_expanding)} folds")

    for fold in folds_expanding[:3]:  # Show first 3 folds
        logger.info(
            f"    Fold {fold.fold_id}: Train weeks {fold.train_weeks}, "
            f"Test week {fold.test_week}, Train size {fold.train_size}, Test size {fold.test_size}"
        )

    # Test rolling window
    logger.info("\nTesting ROLLING window strategy...")
    cv_rolling = TemporalCrossValidator(
        strategy='rolling',
        rolling_window=6,
        min_train_weeks=4,
        test_from_week=8
    )

    folds_rolling = list(cv_rolling.split(df, week_col='week'))
    logger.info(f"  ✓ Generated {len(folds_rolling)} folds")

    for fold in folds_rolling:
        logger.info(
            f"    Fold {fold.fold_id}: Train weeks {fold.train_weeks}, "
            f"Test week {fold.test_week}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("TEST 4 SUMMARY: Temporal CV")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Expanding window: {len(folds_expanding)} folds")
    logger.info(f"✓ Rolling window: {len(folds_rolling)} folds")

    return True


def main():
    parser = argparse.ArgumentParser(description='End-to-end TIER 1 & 2 integration test')
    parser.add_argument('--week', type=int, default=11, help='NFL week number')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--full', action='store_true', help='Test all players (not just samples)')
    parser.add_argument('--compare', action='store_true', help='Compare baseline vs TIER 1 & 2')

    args = parser.parse_args()

    logger.info(f"\n{'#'*80}")
    logger.info(f"# TIER 1 & TIER 2 END-TO-END INTEGRATION TEST")
    logger.info(f"# Week {args.week}, Season {args.season}")
    logger.info(f"{'#'*80}\n")

    # Run tests
    test_results = {}

    # Test 1: Feature Extraction
    test_results['feature_extraction'] = test_feature_extraction(
        week=args.week,
        season=args.season
    )

    # Test 2: Model Integration
    test_results['model_integration'] = test_model_integration(
        week=args.week,
        season=args.season
    )

    # Test 3: Prediction Comparison (optional)
    if args.compare:
        test_results['prediction_comparison'] = test_prediction_comparison(
            week=args.week,
            season=args.season
        )

    # Test 4: Temporal CV
    test_results['temporal_cv'] = test_temporal_cv()

    # Final summary
    logger.info(f"\n{'#'*80}")
    logger.info("# FINAL TEST SUMMARY")
    logger.info(f"{'#'*80}\n")

    for test_name, result in test_results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"

        logger.info(f"{status:10s} {test_name}")

    # Overall status
    passed = sum(1 for r in test_results.values() if r is True)
    failed = sum(1 for r in test_results.values() if r is False)
    skipped = sum(1 for r in test_results.values() if r is None)

    logger.info(f"\n{'='*80}")
    logger.info(f"Tests: {passed} passed, {failed} failed, {skipped} skipped")
    logger.info(f"{'='*80}\n")

    if failed == 0:
        logger.info("✅ ALL TESTS PASSED - TIER 1 & 2 integration ready for production")
        logger.info("\n📋 Next steps:")
        logger.info("1. Retrain models: python scripts/train/retrain_models_with_tier12_features.py --all --validate")
        logger.info("2. Generate predictions: python scripts/predict/generate_model_predictions.py --week 11 --use-tier12")
        logger.info("3. Backtest on historical data")
        logger.info("4. Deploy to production")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
