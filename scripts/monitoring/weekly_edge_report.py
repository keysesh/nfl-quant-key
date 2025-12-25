#!/usr/bin/env python3
"""
Weekly Edge Report Generator

Generates a weekly performance report for edge predictions.
Aggregates results by market, source, and overall.

Usage:
    python scripts/monitoring/weekly_edge_report.py --week 15 --season 2024
    python scripts/monitoring/weekly_edge_report.py  # Auto-detect current week
"""
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
import sys
import json

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.config_paths import DATA_DIR, REPORTS_DIR
from nfl_quant.monitoring.edge_logger import EdgePredictionLogger


def get_week_dates(week: int, season: int) -> tuple:
    """
    Get start and end dates for NFL week.

    NFL weeks generally run Tuesday-Monday.
    This is an approximation; adjust as needed.

    Args:
        week: NFL week number
        season: NFL season

    Returns:
        (start_date, end_date)
    """
    # Approximate: Week 1 usually starts around Sept 5-10
    # This is simplified - in production, use official NFL calendar
    season_start = date(season, 9, 5)

    # Find first Tuesday
    days_until_tuesday = (1 - season_start.weekday()) % 7
    first_tuesday = season_start + timedelta(days=days_until_tuesday)

    # Calculate week start
    week_start = first_tuesday + timedelta(weeks=week - 1)
    week_end = week_start + timedelta(days=6)

    return week_start, week_end


def load_week_results(week: int, season: int) -> pd.DataFrame:
    """
    Load matched results for a specific NFL week.

    Args:
        week: NFL week number
        season: NFL season

    Returns:
        DataFrame with matched predictions and results
    """
    logger = EdgePredictionLogger()
    results_dir = logger.log_dir / 'results'

    if not results_dir.exists():
        return pd.DataFrame()

    # Load all result files
    all_files = sorted(results_dir.glob('results_*.csv'))

    if not all_files:
        return pd.DataFrame()

    # Get week date range
    start_date, end_date = get_week_dates(week, season)

    dfs = []
    for f in all_files:
        file_date_str = f.stem.replace('results_', '')
        try:
            file_date = date.fromisoformat(file_date_str)
        except ValueError:
            continue

        if start_date <= file_date <= end_date:
            df = pd.read_csv(f)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate ROI and hit rate metrics."""
    if df.empty or 'hit' not in df.columns:
        return {'n': 0, 'hit_rate': 0, 'roi': 0}

    valid = df[df['hit'].notna()].copy()

    if len(valid) == 0:
        return {'n': 0, 'hit_rate': 0, 'roi': 0}

    hit_rate = valid['hit'].mean()
    # ROI calculation: -110 odds means win 0.909 * stake, lose 1.0 * stake
    roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100

    return {
        'n': len(valid),
        'hits': int(valid['hit'].sum()),
        'hit_rate': float(hit_rate),
        'roi': float(roi),
    }


def generate_report(week: int, season: int) -> dict:
    """
    Generate weekly edge performance report.

    Args:
        week: NFL week
        season: NFL season

    Returns:
        Report dict
    """
    df = load_week_results(week, season)

    report = {
        'week': week,
        'season': season,
        'generated_at': datetime.now().isoformat(),
        'overall': calculate_metrics(df),
        'by_source': {},
        'by_market': {},
        'calibration': {},
    }

    if df.empty:
        return report

    # By source
    for source in df['source'].dropna().unique():
        source_df = df[df['source'] == source]
        report['by_source'][source] = calculate_metrics(source_df)

    # By market
    for market in df['market'].dropna().unique():
        market_df = df[df['market'] == market]
        report['by_market'][market] = calculate_metrics(market_df)

    # Calibration check: predicted confidence vs actual hit rate
    if 'confidence' in df.columns and 'hit' in df.columns:
        valid = df[df['hit'].notna()].copy()
        if len(valid) > 10:
            # Bin by confidence
            valid['conf_bin'] = pd.cut(
                valid['confidence'],
                bins=[0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0],
            )

            calib = valid.groupby('conf_bin').agg({
                'hit': ['count', 'mean'],
                'confidence': 'mean',
            }).reset_index()

            calib.columns = ['bin', 'count', 'actual_rate', 'predicted_rate']
            report['calibration'] = calib.to_dict('records')

    return report


def print_report(report: dict) -> None:
    """Print formatted report to console."""
    print("\n" + "=" * 60)
    print(f"EDGE PERFORMANCE REPORT - Week {report['week']}, {report['season']}")
    print("=" * 60)
    print(f"Generated: {report['generated_at'][:19]}")

    # Overall
    overall = report['overall']
    if overall['n'] > 0:
        print(f"\nOVERALL:")
        print(f"  Bets: {overall['n']}")
        print(f"  Hits: {overall['hits']}")
        print(f"  Hit Rate: {overall['hit_rate']:.1%}")
        print(f"  ROI: {overall['roi']:+.1f}%")
    else:
        print("\nNo results available for this week.")
        return

    # By Source
    if report['by_source']:
        print(f"\nBY SOURCE:")
        print(f"  {'Source':<20} {'Bets':<8} {'Hits':<8} {'Hit %':<10} {'ROI':<10}")
        print("  " + "-" * 56)
        for source, metrics in sorted(report['by_source'].items()):
            print(
                f"  {source:<20} {metrics['n']:<8} {metrics['hits']:<8} "
                f"{metrics['hit_rate']*100:>5.1f}%    {metrics['roi']:>+6.1f}%"
            )

    # By Market
    if report['by_market']:
        print(f"\nBY MARKET:")
        print(f"  {'Market':<25} {'Bets':<8} {'Hits':<8} {'Hit %':<10} {'ROI':<10}")
        print("  " + "-" * 61)
        for market, metrics in sorted(report['by_market'].items()):
            print(
                f"  {market:<25} {metrics['n']:<8} {metrics['hits']:<8} "
                f"{metrics['hit_rate']*100:>5.1f}%    {metrics['roi']:>+6.1f}%"
            )

    # Calibration
    if report.get('calibration'):
        print(f"\nCALIBRATION CHECK:")
        print(f"  {'Conf Bin':<20} {'Count':<8} {'Predicted':<12} {'Actual':<12}")
        print("  " + "-" * 52)
        for row in report['calibration']:
            print(
                f"  {str(row['bin']):<20} {row['count']:<8} "
                f"{row['predicted_rate']*100:>6.1f}%     {row['actual_rate']*100:>6.1f}%"
            )

    print("\n" + "=" * 60)


def save_report(report: dict, filepath: Path = None) -> Path:
    """Save report to JSON file."""
    if filepath is None:
        reports_dir = REPORTS_DIR if REPORTS_DIR.exists() else DATA_DIR / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        filepath = reports_dir / f"edge_report_week{report['week']}_{report['season']}.json"

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Generate Weekly Edge Report")
    parser.add_argument('--week', type=int, help='NFL week number')
    parser.add_argument('--season', type=int, help='NFL season')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    args = parser.parse_args()

    # Auto-detect week/season if not provided
    today = date.today()
    season = args.season or (today.year if today.month >= 9 else today.year - 1)

    if args.week:
        week = args.week
    else:
        # Estimate current week (simplified)
        season_start = date(season, 9, 5)
        days_since_start = (today - season_start).days
        week = max(1, min(18, days_since_start // 7 + 1))
        print(f"Auto-detected: Week {week}, Season {season}")

    # Generate report
    report = generate_report(week, season)

    # Print to console
    print_report(report)

    # Save if requested
    if args.save:
        filepath = save_report(report)
        print(f"\nReport saved to: {filepath}")


if __name__ == '__main__':
    main()
