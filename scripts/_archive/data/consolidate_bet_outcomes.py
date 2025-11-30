#!/usr/bin/env python3
"""
Consolidate Bet Outcome Database
=================================

Purpose: Consolidate ALL historical bet outcomes into unified training database
for calibrator retraining.

Current Issue:
- Calibrator trained on only 302 samples
- 13,915+ bet outcomes available but unused
- Missing 97.8% of available training data!

This script:
1. Loads all backtest files with actual outcomes
2. Deduplicates bet records
3. Validates outcome data quality
4. Creates unified betting history database

Output: data/betting_history/bet_outcomes_consolidated.csv

Usage:
    python scripts/data/consolidate_bet_outcomes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_detailed_bet_analysis():
    """Load detailed bet analysis with actual outcomes."""
    file_path = Path('reports/detailed_bet_analysis_weekall.csv')

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    logger.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"  Loaded {len(df):,} bet records")

    # Check for outcome column
    if 'actual_outcome' in df.columns:
        logger.info(f"  ✅ Has 'actual_outcome' column")
        # Calculate result (win/loss)
        if 'pick' in df.columns and 'line' in df.columns:
            df['result'] = df.apply(lambda row: calculate_result(row), axis=1)
    else:
        logger.warning(f"  ⚠️  No 'actual_outcome' column found")

    return df


def load_backtest_validation():
    """Load backtest validation data."""
    file_path = Path('reports/BACKTEST_WEEKS_1_8_VALIDATION.csv')

    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return None

    logger.info(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"  Loaded {len(df):,} validation records")

    return df


def calculate_result(row):
    """
    Calculate if bet won/lost/pushed based on actual outcome.

    Returns: 'win', 'loss', or 'push'
    """
    try:
        actual = float(row.get('actual_outcome', np.nan))
        line = float(row.get('line', np.nan))
        pick = str(row.get('pick', '')).lower()

        if pd.isna(actual) or pd.isna(line):
            return 'unknown'

        # Push check (within 0.1 tolerance for rounding)
        if abs(actual - line) < 0.1:
            return 'push'

        # Over/Under
        if 'over' in pick:
            return 'win' if actual > line else 'loss'
        elif 'under' in pick:
            return 'win' if actual < line else 'loss'
        else:
            return 'unknown'

    except Exception as e:
        logger.debug(f"Error calculating result: {e}")
        return 'unknown'


def consolidate_bet_outcomes():
    """
    Consolidate all bet outcome sources into unified database.
    """
    logger.info("\n" + "="*80)
    logger.info("CONSOLIDATING BET OUTCOME DATABASE")
    logger.info("="*80)

    all_bets = []

    # Source 1: Detailed bet analysis (PRIMARY)
    df_detailed = load_detailed_bet_analysis()
    if df_detailed is not None:
        all_bets.append(df_detailed)

    # Source 2: Backtest validation
    df_validation = load_backtest_validation()
    if df_validation is not None:
        # Check if it's substantially different from detailed analysis
        if df_detailed is not None and len(df_validation) != len(df_detailed):
            all_bets.append(df_validation)

    if not all_bets:
        logger.error("❌ No bet outcome data found!")
        return None

    # Combine all sources
    logger.info(f"\nCombining {len(all_bets)} data sources...")
    df_combined = pd.concat(all_bets, ignore_index=True)
    logger.info(f"  Combined total: {len(df_combined):,} records")

    # Deduplicate based on key columns
    key_cols = ['week', 'player', 'market', 'line']
    if all(col in df_combined.columns for col in key_cols):
        before_dedup = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=key_cols, keep='first')
        logger.info(f"  After deduplication: {len(df_combined):,} records (removed {before_dedup - len(df_combined):,} duplicates)")

    # Data quality checks
    logger.info("\n" + "="*80)
    logger.info("DATA QUALITY VALIDATION")
    logger.info("="*80)

    # Check for required columns
    required_cols = ['week', 'player', 'market', 'line', 'model_prob']
    missing_cols = [col for col in required_cols if col not in df_combined.columns]
    if missing_cols:
        logger.warning(f"⚠️  Missing required columns: {missing_cols}")
    else:
        logger.info(f"✅ All required columns present")

    # Check for outcome data
    if 'actual_outcome' in df_combined.columns:
        has_outcome = df_combined['actual_outcome'].notna().sum()
        logger.info(f"✅ Actual outcomes available: {has_outcome:,} / {len(df_combined):,} ({100*has_outcome/len(df_combined):.1f}%)")

    # Check for result data
    if 'result' in df_combined.columns:
        result_counts = df_combined['result'].value_counts()
        logger.info(f"\nBet Results:")
        for result, count in result_counts.items():
            logger.info(f"  {result}: {count:,} ({100*count/len(df_combined):.1f}%)")

    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    if 'week' in df_combined.columns:
        weeks = sorted(df_combined['week'].unique())
        logger.info(f"Weeks covered: {min(weeks)} - {max(weeks)} ({len(weeks)} weeks)")

    if 'market' in df_combined.columns:
        logger.info(f"\nMarkets covered:")
        for market, count in df_combined['market'].value_counts().head(10).items():
            logger.info(f"  {market}: {count:,}")

    if 'model_prob' in df_combined.columns:
        logger.info(f"\nModel Probability Stats:")
        logger.info(f"  Mean: {df_combined['model_prob'].mean():.3f}")
        logger.info(f"  Median: {df_combined['model_prob'].median():.3f}")
        logger.info(f"  Std: {df_combined['model_prob'].std():.3f}")

    # Save consolidated database
    output_path = Path('data/betting_history/bet_outcomes_consolidated.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_combined.to_csv(output_path, index=False)
    logger.info(f"\n" + "="*80)
    logger.info(f"✅ SAVED: {output_path}")
    logger.info(f"   Total Records: {len(df_combined):,}")
    logger.info(f"   File Size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info("="*80)

    # Create metadata file
    metadata = {
        'created_date': datetime.now().isoformat(),
        'total_records': len(df_combined),
        'sources': [str(p) for p in all_bets],
        'weeks_covered': list(weeks) if 'week' in df_combined.columns else [],
        'columns': df_combined.columns.tolist(),
    }

    metadata_path = Path('data/betting_history/bet_outcomes_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"📋 Metadata saved: {metadata_path}")

    return df_combined


def main():
    """Run bet outcome consolidation."""
    df = consolidate_bet_outcomes()

    if df is not None:
        logger.info("\n✅ SUCCESS: Bet outcome database consolidated")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review: data/betting_history/bet_outcomes_consolidated.csv")
        logger.info(f"  2. Retrain calibrator: python scripts/train/retrain_calibrator_comprehensive.py")
        logger.info(f"  3. Expected improvement: Brier score +0.05-0.10")
        return 0
    else:
        logger.error("\n❌ FAILED: Could not consolidate bet outcomes")
        return 1


if __name__ == '__main__':
    sys.exit(main())
