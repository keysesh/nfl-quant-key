#!/usr/bin/env python3
"""
QB TD Rate Model Diagnostic Script
===================================

Purpose: Investigate why the QB passing TD rate model predicts constant values
regardless of input features.

This script:
1. Loads training data used for QB TD rate model
2. Analyzes target variable (QB_td_rate_pass) distribution and variance
3. Checks feature correlation with target
4. Identifies data quality issues
5. Generates diagnostic report with recommendations

Usage:
    python scripts/diagnostics/diagnose_qb_td_model.py

Output:
    - Console diagnostic report
    - data/diagnostics/qb_td_model_diagnosis.json (detailed findings)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_qb_training_data(pbp_path: Path) -> pd.DataFrame:
    """
    Load and prepare QB training data.

    Args:
        pbp_path: Path to play-by-play data

    Returns:
        DataFrame with QB efficiency features and target
    """
    logger.info("Loading play-by-play data...")
    pbp_df = pd.read_parquet(pbp_path)

    logger.info("Preparing QB passing TD rate training data...")

    # Get passing plays
    pass_plays = pbp_df[
        (pbp_df['play_type'] == 'pass') &
        (pbp_df['passer_player_id'].notna())
    ].copy()

    # Group by QB and game
    qb_games = (
        pass_plays.groupby(['passer_player_id', 'passer_player_name', 'game_id', 'week'])
        .agg({
            'complete_pass': 'sum',
            'pass_attempt': 'sum',
            'pass_touchdown': 'sum',
            'yards_gained': 'sum',
            'epa': 'mean',
        })
        .reset_index()
    )

    # Calculate TD rate
    qb_games['td_rate_pass'] = (
        qb_games['pass_touchdown'] / qb_games['pass_attempt']
    ).fillna(0)

    # Calculate completion pct
    qb_games['comp_pct'] = (
        qb_games['complete_pass'] / qb_games['pass_attempt']
    ).fillna(0)

    # Calculate yards per attempt
    qb_games['yards_per_attempt'] = (
        qb_games['yards_gained'] / qb_games['pass_attempt']
    ).fillna(0)

    logger.info(f"Found {len(qb_games)} QB-game samples")

    return qb_games


def analyze_target_distribution(df: pd.DataFrame) -> dict:
    """
    Analyze distribution and variance of QB_td_rate_pass target variable.

    Args:
        df: DataFrame with td_rate_pass column

    Returns:
        Dictionary of diagnostic results
    """
    logger.info("\n" + "="*70)
    logger.info("TARGET VARIABLE ANALYSIS: QB_td_rate_pass")
    logger.info("="*70)

    td_rate = df['td_rate_pass']

    results = {
        'count': int(len(td_rate)),
        'mean': float(td_rate.mean()),
        'std': float(td_rate.std()),
        'min': float(td_rate.min()),
        'max': float(td_rate.max()),
        'median': float(td_rate.median()),
        'q25': float(td_rate.quantile(0.25)),
        'q75': float(td_rate.quantile(0.75)),
        'variance': float(td_rate.var()),
        'coefficient_of_variation': float(td_rate.std() / td_rate.mean()) if td_rate.mean() > 0 else 0,
    }

    # Count frequency of exact values
    value_counts = td_rate.value_counts()
    results['unique_values'] = int(len(value_counts))
    results['most_common_value'] = float(value_counts.index[0])
    results['most_common_frequency'] = int(value_counts.iloc[0])
    results['most_common_pct'] = float(value_counts.iloc[0] / len(td_rate) * 100)

    # Check for zero variance issue
    results['has_variance'] = results['std'] > 0.001
    results['has_sufficient_variance'] = results['std'] > 0.01

    # Distribution shape
    results['skewness'] = float(stats.skew(td_rate))
    results['kurtosis'] = float(stats.kurtosis(td_rate))

    # Print summary
    logger.info(f"Sample Count: {results['count']}")
    logger.info(f"Mean TD Rate: {results['mean']:.4f} ({results['mean']*100:.2f}%)")
    logger.info(f"Std Dev: {results['std']:.4f}")
    logger.info(f"Coefficient of Variation: {results['coefficient_of_variation']:.4f}")
    logger.info(f"Min: {results['min']:.4f}, Max: {results['max']:.4f}")
    logger.info(f"Median: {results['median']:.4f}")
    logger.info(f"Unique Values: {results['unique_values']}")
    logger.info(f"Most Common Value: {results['most_common_value']:.4f} "
                f"({results['most_common_pct']:.1f}% of samples)")

    # Diagnosis
    if not results['has_variance']:
        logger.error("❌ CRITICAL: Target has near-zero variance!")
        logger.error("   → Model will predict constant value (mean)")
    elif not results['has_sufficient_variance']:
        logger.warning("⚠️  WARNING: Target has low variance")
        logger.warning("   → Model may struggle to learn meaningful patterns")
    else:
        logger.info("✅ Target has sufficient variance for model training")

    return results


def analyze_feature_correlations(df: pd.DataFrame) -> dict:
    """
    Analyze correlation between features and target.

    Args:
        df: DataFrame with features and td_rate_pass

    Returns:
        Dictionary of correlation analysis
    """
    logger.info("\n" + "="*70)
    logger.info("FEATURE CORRELATION ANALYSIS")
    logger.info("="*70)

    features = ['comp_pct', 'yards_per_attempt', 'pass_attempt', 'epa']
    target = 'td_rate_pass'

    correlations = {}
    for feat in features:
        if feat in df.columns:
            corr = df[feat].corr(df[target])
            correlations[feat] = float(corr)

            if abs(corr) > 0.3:
                logger.info(f"✅ {feat}: {corr:.4f} (moderate-strong correlation)")
            elif abs(corr) > 0.1:
                logger.info(f"⚠️  {feat}: {corr:.4f} (weak correlation)")
            else:
                logger.info(f"❌ {feat}: {corr:.4f} (very weak correlation)")

    results = {
        'correlations': correlations,
        'max_correlation': max(abs(c) for c in correlations.values()),
        'has_predictive_features': max(abs(c) for c in correlations.values()) > 0.1,
    }

    if not results['has_predictive_features']:
        logger.error("❌ CRITICAL: No features have meaningful correlation with target!")
        logger.error("   → Model cannot learn from these features")

    return results


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Check for data quality issues.

    Args:
        df: Training DataFrame

    Returns:
        Dictionary of data quality checks
    """
    logger.info("\n" + "="*70)
    logger.info("DATA QUALITY CHECKS")
    logger.info("="*70)

    results = {}

    # Check for sufficient samples
    results['total_samples'] = int(len(df))
    results['sufficient_samples'] = results['total_samples'] >= 1000

    logger.info(f"Total Samples: {results['total_samples']}")
    if results['sufficient_samples']:
        logger.info("✅ Sufficient training samples")
    else:
        logger.warning(f"⚠️  Only {results['total_samples']} samples (recommend 1000+)")

    # Check for missing values
    results['missing_values'] = {}
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = float(missing / len(df) * 100)
        results['missing_values'][col] = {'count': missing, 'pct': pct}
        if pct > 5:
            logger.warning(f"⚠️  {col}: {pct:.1f}% missing")

    # Check for outliers in target
    td_rate = df['td_rate_pass']
    q1 = td_rate.quantile(0.25)
    q3 = td_rate.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (td_rate < q1 - 3*iqr) | (td_rate > q3 + 3*iqr)
    results['outliers_count'] = int(outlier_mask.sum())
    results['outliers_pct'] = float(outlier_mask.sum() / len(df) * 100)

    logger.info(f"Outliers: {results['outliers_count']} ({results['outliers_pct']:.1f}%)")

    # Check for games with very few attempts (unreliable TD rate)
    min_attempts = 10
    few_attempts = (df['pass_attempt'] < min_attempts).sum()
    results['low_attempt_games'] = int(few_attempts)
    results['low_attempt_pct'] = float(few_attempts / len(df) * 100)

    logger.info(f"Games with <{min_attempts} attempts: {results['low_attempt_games']} "
                f"({results['low_attempt_pct']:.1f}%)")

    if results['low_attempt_pct'] > 20:
        logger.warning(f"⚠️  {results['low_attempt_pct']:.1f}% of games have <{min_attempts} attempts")
        logger.warning("   → TD rate unreliable in these games (small sample noise)")

    return results


def generate_recommendations(
    target_analysis: dict,
    correlation_analysis: dict,
    quality_checks: dict
) -> list:
    """
    Generate recommendations based on diagnostic results.

    Returns:
        List of recommendation strings
    """
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS")
    logger.info("="*70)

    recommendations = []

    # Check variance
    if not target_analysis['has_variance']:
        recommendations.append({
            'severity': 'CRITICAL',
            'issue': 'Target variable has near-zero variance',
            'recommendation': 'This explains why model predicts constants. Options:\n'
                            '  1. Use trailing TD rate directly (current workaround)\n'
                            '  2. Aggregate over multiple games to increase variance\n'
                            '  3. Use Poisson regression instead of XGBoost\n'
                            '  4. Use league-average TD rate with Bayesian shrinkage',
        })

    if not target_analysis['has_sufficient_variance']:
        recommendations.append({
            'severity': 'HIGH',
            'issue': 'Target variable has low variance',
            'recommendation': 'Consider aggregating over 2-3 game rolling windows to smooth noise',
        })

    # Check correlations
    if not correlation_analysis['has_predictive_features']:
        recommendations.append({
            'severity': 'CRITICAL',
            'issue': 'No features correlate with target',
            'recommendation': 'Add more predictive features:\n'
                            '  - Red zone attempts\n'
                            '  - Team scoring environment\n'
                            '  - Opponent pass defense quality\n'
                            '  - Home/away indicator\n'
                            '  - Weather conditions',
        })

    # Check sample size
    if quality_checks['low_attempt_pct'] > 20:
        recommendations.append({
            'severity': 'MEDIUM',
            'issue': f"{quality_checks['low_attempt_pct']:.1f}% of samples have very few attempts",
            'recommendation': 'Filter out games with <15 pass attempts to reduce noise',
        })

    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{i}. [{rec['severity']}] {rec['issue']}")
        logger.info(f"   → {rec['recommendation']}")

    if not recommendations:
        logger.info("✅ No critical issues found!")

    return recommendations


def main():
    """Run QB TD model diagnostics."""
    logger.info("QB TD Rate Model Diagnostic Script")
    logger.info("="*70 + "\n")

    # Paths
    pbp_path = Path('data/processed/pbp_2024.parquet')
    output_dir = Path('data/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if not pbp_path.exists():
        logger.error(f"❌ Play-by-play data not found: {pbp_path}")
        logger.error("   Please run data fetch scripts first")
        return

    df = load_qb_training_data(pbp_path)

    # Run diagnostics
    target_analysis = analyze_target_distribution(df)
    correlation_analysis = analyze_feature_correlations(df)
    quality_checks = check_data_quality(df)
    recommendations = generate_recommendations(
        target_analysis, correlation_analysis, quality_checks
    )

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(pbp_path),
        'sample_count': int(len(df)),
        'target_analysis': target_analysis,
        'correlation_analysis': correlation_analysis,
        'data_quality': quality_checks,
        'recommendations': recommendations,
        'diagnosis_summary': {
            'has_variance_issue': not target_analysis['has_sufficient_variance'],
            'has_correlation_issue': not correlation_analysis['has_predictive_features'],
            'has_quality_issue': quality_checks['low_attempt_pct'] > 20,
            'model_trainable': (
                target_analysis['has_sufficient_variance'] and
                correlation_analysis['has_predictive_features']
            ),
        }
    }

    output_file = output_dir / 'qb_td_model_diagnosis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n" + "="*70)
    logger.info(f"Diagnostic report saved to: {output_file}")
    logger.info("="*70)

    # Final summary
    summary = output['diagnosis_summary']
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"Variance Issue: {'YES ❌' if summary['has_variance_issue'] else 'NO ✅'}")
    logger.info(f"Correlation Issue: {'YES ❌' if summary['has_correlation_issue'] else 'NO ✅'}")
    logger.info(f"Data Quality Issue: {'YES ❌' if summary['has_quality_issue'] else 'NO ✅'}")
    logger.info(f"Model Trainable: {'YES ✅' if summary['model_trainable'] else 'NO ❌'}")

    if not summary['model_trainable']:
        logger.error("\n⚠️  QB TD model is NOT trainable with current data/features")
        logger.error("   Recommend using trailing TD rate directly (current approach)")
    else:
        logger.info("\n✅ QB TD model should be trainable")
        logger.info("   Investigate why model is predicting constants during training")


if __name__ == '__main__':
    main()
