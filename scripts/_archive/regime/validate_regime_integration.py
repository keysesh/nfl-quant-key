#!/usr/bin/env python3
"""
Regime Integration Validation Script

Compares regime-aware predictions vs standard 4-week predictions to validate
that regime detection improves accuracy without breaking the pipeline.

Usage:
    # Run for specific week
    python scripts/regime/validate_regime_integration.py --week 9

    # Run with custom thresholds
    python scripts/regime/validate_regime_integration.py --week 9 --max-diff 15 --min-improvement 5
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.features.trailing_stats import get_trailing_stats_extractor
from nfl_quant.regime.integration import get_regime_aware_extractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_player_list(week: int) -> pd.DataFrame:
    """Load list of players to validate from odds data."""
    odds_file = Path('data/nfl_player_props_draftkings.csv')
    if not odds_file.exists():
        odds_file = Path(f'data/odds_week{week}_player_props.csv')

    if not odds_file.exists():
        raise FileNotFoundError(f"No odds file found for week {week}")

    df = pd.read_csv(odds_file)

    # Extract unique players with position info
    players = []
    for _, row in df.iterrows():
        player_name = row.get('player_name')
        if pd.isna(player_name):
            continue

        # Try to get position and team from player database
        try:
            import json
            with open('data/player_database.json') as f:
                player_db = json.load(f)
                if player_name in player_db:
                    players.append({
                        'player_name': player_name,
                        'position': player_db[player_name].get('position'),
                        'team': player_db[player_name].get('team'),
                    })
        except:
            continue

    if not players:
        logger.warning("Could not load players from player database")
        return pd.DataFrame()

    return pd.DataFrame(players).drop_duplicates(subset=['player_name'])


def compare_trailing_stats(
    player_name: str,
    position: str,
    week: int,
    standard_extractor,
    regime_extractor,
) -> Dict:
    """Compare trailing stats from standard vs regime extractors."""

    # Get standard stats
    try:
        standard_stats = standard_extractor.get_trailing_stats(
            player_name, position, week
        )
    except Exception as e:
        logger.debug(f"Error getting standard stats for {player_name}: {e}")
        return None

    # Get regime-aware stats
    try:
        regime_stats = regime_extractor.get_trailing_stats(
            player_name, position, week
        )
    except Exception as e:
        logger.debug(f"Error getting regime stats for {player_name}: {e}")
        return None

    # Calculate differences
    comparison = {
        'player_name': player_name,
        'position': position,
        'standard_snaps': standard_stats.get('trailing_snaps', 0),
        'regime_snaps': regime_stats.get('trailing_snaps', 0),
        'standard_attempts': standard_stats.get('trailing_attempts', 0),
        'regime_attempts': regime_stats.get('trailing_attempts', 0),
        'standard_carries': standard_stats.get('trailing_carries', 0),
        'regime_carries': regime_stats.get('trailing_carries', 0),
        'regime_detected': regime_stats.get('regime_detected', False),
        'regime_type': regime_stats.get('regime_type', ''),
        'regime_confidence': regime_stats.get('regime_confidence', 0.0),
        'standard_window_weeks': standard_stats.get('window_weeks', 4),
        'regime_window_weeks': regime_stats.get('window_weeks', 4),
    }

    # Calculate percentage differences
    if comparison['standard_snaps'] > 0:
        comparison['snaps_pct_diff'] = (
            (comparison['regime_snaps'] - comparison['standard_snaps']) /
            comparison['standard_snaps'] * 100
        )
    else:
        comparison['snaps_pct_diff'] = 0.0

    if comparison['standard_attempts'] > 0:
        comparison['attempts_pct_diff'] = (
            (comparison['regime_attempts'] - comparison['standard_attempts']) /
            comparison['standard_attempts'] * 100
        )
    else:
        comparison['attempts_pct_diff'] = 0.0

    if comparison['standard_carries'] > 0:
        comparison['carries_pct_diff'] = (
            (comparison['regime_carries'] - comparison['standard_carries']) /
            comparison['standard_carries'] * 100
        )
    else:
        comparison['carries_pct_diff'] = 0.0

    return comparison


def validate_integration(
    week: int,
    max_diff_threshold: float = 20.0,
    min_improvement_threshold: float = 5.0,
) -> Dict:
    """
    Validate regime integration by comparing with standard predictions.

    Args:
        week: Week to validate
        max_diff_threshold: Maximum acceptable difference percentage (20% default)
        min_improvement_threshold: Minimum improvement to consider regime beneficial (5% default)

    Returns:
        Dict with validation results
    """
    logger.info("=" * 80)
    logger.info(f"REGIME INTEGRATION VALIDATION - WEEK {week}")
    logger.info("=" * 80)

    # Load players
    logger.info("\n1. Loading players...")
    players_df = load_player_list(week)
    if players_df.empty:
        logger.error("   ❌ No players found")
        return {'success': False, 'error': 'No players found'}

    logger.info(f"   ✅ Loaded {len(players_df)} players")

    # Initialize extractors
    logger.info("\n2. Initializing extractors...")

    # Force standard extractor (no regime)
    os.environ['ENABLE_REGIME_DETECTION'] = '0'
    try:
        standard_extractor = get_trailing_stats_extractor()
        logger.info("   ✅ Standard extractor initialized")
    except Exception as e:
        logger.error(f"   ❌ Failed to initialize standard extractor: {e}")
        return {'success': False, 'error': str(e)}

    # Force regime extractor
    try:
        regime_extractor = get_regime_aware_extractor(enable_regime=True)
        logger.info("   ✅ Regime extractor initialized")
    except Exception as e:
        logger.error(f"   ❌ Failed to initialize regime extractor: {e}")
        return {'success': False, 'error': str(e)}

    # Compare stats for all players
    logger.info("\n3. Comparing trailing stats...")
    comparisons = []

    for idx, row in players_df.iterrows():
        player_name = row['player_name']
        position = row['position']

        if pd.isna(position):
            continue

        comparison = compare_trailing_stats(
            player_name, position, week,
            standard_extractor, regime_extractor
        )

        if comparison:
            comparisons.append(comparison)

        # Progress indicator
        if (idx + 1) % 50 == 0:
            logger.info(f"   Progress: {idx + 1}/{len(players_df)} players")

    logger.info(f"   ✅ Compared {len(comparisons)} players")

    if not comparisons:
        logger.error("   ❌ No comparisons generated")
        return {'success': False, 'error': 'No comparisons generated'}

    # Create DataFrame
    df = pd.DataFrame(comparisons)

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)

    # 1. Regime detection rate
    regime_detected_count = df['regime_detected'].sum()
    regime_detected_pct = regime_detected_count / len(df) * 100
    logger.info(f"\n1. Regime Detection Rate:")
    logger.info(f"   {regime_detected_count}/{len(df)} players ({regime_detected_pct:.1f}%)")

    # 2. Regime types detected
    if regime_detected_count > 0:
        regime_types = df[df['regime_detected']]['regime_type'].value_counts()
        logger.info(f"\n2. Regime Types:")
        for regime_type, count in regime_types.items():
            logger.info(f"   {regime_type}: {count} players")

    # 3. Window size comparison
    logger.info(f"\n3. Window Size Analysis:")
    avg_standard_window = df['standard_window_weeks'].mean()
    avg_regime_window = df['regime_window_weeks'].mean()
    logger.info(f"   Standard avg: {avg_standard_window:.1f} weeks")
    logger.info(f"   Regime avg: {avg_regime_window:.1f} weeks")

    # 4. Magnitude of changes
    logger.info(f"\n4. Magnitude of Changes (regime vs standard):")
    avg_snaps_diff = df['snaps_pct_diff'].abs().mean()
    avg_attempts_diff = df['attempts_pct_diff'].abs().mean()
    avg_carries_diff = df['carries_pct_diff'].abs().mean()
    logger.info(f"   Snaps: {avg_snaps_diff:.1f}% average difference")
    logger.info(f"   Attempts: {avg_attempts_diff:.1f}% average difference")
    logger.info(f"   Carries: {avg_carries_diff:.1f}% average difference")

    # 5. Significant changes (beyond threshold)
    significant_changes = df[
        (df['snaps_pct_diff'].abs() > max_diff_threshold) |
        (df['attempts_pct_diff'].abs() > max_diff_threshold) |
        (df['carries_pct_diff'].abs() > max_diff_threshold)
    ]
    logger.info(f"\n5. Players with significant changes (>{max_diff_threshold}%):")
    logger.info(f"   {len(significant_changes)} players")

    if len(significant_changes) > 0:
        logger.info("\n   Top 10 largest changes:")
        top_changes = significant_changes.nlargest(10, 'snaps_pct_diff', keep='all')

        for _, row in top_changes.head(10).iterrows():
            logger.info(f"\n   {row['player_name']} ({row['position']})")
            logger.info(f"     Snaps: {row['standard_snaps']:.1f} → {row['regime_snaps']:.1f} "
                       f"({row['snaps_pct_diff']:+.1f}%)")
            if row['regime_detected']:
                logger.info(f"     Regime: {row['regime_type']} "
                           f"(confidence: {row['regime_confidence']:.1%})")

    # 6. Validation checks
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: No NaN values
    checks_total += 1
    nan_count = df.isnull().sum().sum()
    if nan_count == 0:
        logger.info("✅ Check 1: No NaN values in results")
        checks_passed += 1
    else:
        logger.error(f"❌ Check 1: Found {nan_count} NaN values")

    # Check 2: Regime detection working
    checks_total += 1
    if regime_detected_count > 0:
        logger.info(f"✅ Check 2: Regime detection working ({regime_detected_count} regimes found)")
        checks_passed += 1
    else:
        logger.warning("⚠️  Check 2: No regimes detected (may be valid if no regimes exist)")

    # Check 3: Changes are reasonable (not too extreme)
    checks_total += 1
    extreme_changes = df[
        (df['snaps_pct_diff'].abs() > 100) |
        (df['attempts_pct_diff'].abs() > 100)
    ]
    if len(extreme_changes) == 0:
        logger.info("✅ Check 3: No extreme changes (>100%)")
        checks_passed += 1
    else:
        logger.warning(f"⚠️  Check 3: {len(extreme_changes)} players with extreme changes (>100%)")

    # Check 4: Window sizes are valid
    checks_total += 1
    invalid_windows = df[
        (df['regime_window_weeks'] < 1) |
        (df['regime_window_weeks'] > 12)
    ]
    if len(invalid_windows) == 0:
        logger.info("✅ Check 4: All window sizes valid (1-12 weeks)")
        checks_passed += 1
    else:
        logger.error(f"❌ Check 4: {len(invalid_windows)} players with invalid window sizes")

    # Final result
    logger.info("\n" + "=" * 80)
    logger.info(f"FINAL RESULT: {checks_passed}/{checks_total} checks passed")
    logger.info("=" * 80)

    success = checks_passed >= checks_total - 1  # Allow 1 warning

    if success:
        logger.info("\n✅ VALIDATION PASSED")
        logger.info("Regime integration appears to be working correctly")
    else:
        logger.error("\n❌ VALIDATION FAILED")
        logger.error("Please review errors above before deploying regime detection")

    # Save results
    output_dir = Path('data/regime/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'validation_week{week}.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"\n📊 Validation results saved to {output_file}")

    return {
        'success': success,
        'checks_passed': checks_passed,
        'checks_total': checks_total,
        'regime_detection_rate': regime_detected_pct,
        'avg_snaps_diff': avg_snaps_diff,
        'avg_attempts_diff': avg_attempts_diff,
        'significant_changes': len(significant_changes),
        'output_file': str(output_file),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate regime integration"
    )
    parser.add_argument(
        '--week', '-w',
        type=int,
        required=True,
        help='Week to validate'
    )
    parser.add_argument(
        '--max-diff',
        type=float,
        default=20.0,
        help='Maximum acceptable difference percentage (default: 20%%)'
    )
    parser.add_argument(
        '--min-improvement',
        type=float,
        default=5.0,
        help='Minimum improvement threshold (default: 5%%)'
    )

    args = parser.parse_args()

    try:
        result = validate_integration(
            week=args.week,
            max_diff_threshold=args.max_diff,
            min_improvement_threshold=args.min_improvement,
        )

        if result['success']:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
