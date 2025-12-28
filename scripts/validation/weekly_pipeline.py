#!/usr/bin/env python3
"""
Weekly Validation Pipeline - The Scientific Approach to Sports Betting

This script implements the CORRECT methodology for validating and improving
a sports betting model:

For each week N:
1. Pull fresh data (NFLverse, injuries, odds)
2. Train model on weeks < N-1 only (walk-forward, no leakage)
3. Generate predictions for week N
4. Load/capture closing lines
5. After games: record actuals
6. Calculate: Win rate, ROI, CLV
7. Calibration check - is model still calibrated?
8. Output comprehensive report

Usage:
    # Backfill historical week
    python scripts/validation/weekly_pipeline.py --week 10 --season 2025

    # Run for current week (live mode)
    python scripts/validation/weekly_pipeline.py --week 17 --season 2025 --live

    # Backfill multiple weeks
    python scripts/validation/weekly_pipeline.py --weeks 1-16 --season 2025

    # Generate summary report across all weeks
    python scripts/validation/weekly_pipeline.py --summary --season 2025
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Imports
from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.utils.season_utils import get_current_season
from nfl_quant.features.batch_extractor import extract_features_batch
from configs.model_config import (
    FEATURES,
    CLASSIFIER_MARKETS,
    MARKET_DIRECTION_CONSTRAINTS,
    DISABLE_DIRECTION_CONSTRAINTS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

VALIDATION_DIR = PROJECT_ROOT / 'data' / 'validation'
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

# Markets to validate
VALIDATION_MARKETS = [
    'player_receptions',
    'player_reception_yds',
    'player_rush_yds',
    'player_rush_attempts',
    'player_pass_attempts',
    'player_pass_completions',
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_historical_props() -> pd.DataFrame:
    """Load historical props with actuals."""
    path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df['player_norm'] = df['player'].str.lower().str.strip()
    df['global_week'] = (df['season'] - 2023) * 18 + df['week']
    return df


def load_closing_lines(season: int) -> pd.DataFrame:
    """Load historical closing lines for CLV calculation."""
    path = PROJECT_ROOT / 'data' / 'odds' / f'historical_closing_lines_{season}.csv'

    if not path.exists():
        print(f"  Warning: Closing lines not found at {path}")
        print(f"  Run: python scripts/fetch/fetch_historical_closing_lines.py --weeks 1-17")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['player_norm'] = df['player_name'].str.lower().str.strip()
    return df


def load_player_stats() -> pd.DataFrame:
    """Load player stats for trailing calculations."""
    path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Weekly stats not found: {path}")

    df = pd.read_parquet(path)
    df['player_norm'] = df['player_display_name'].str.lower().str.strip()
    df['global_week'] = (df['season'] - 2023) * 18 + df['week']
    return df


def load_production_model() -> Optional[dict]:
    """Load the production XGBoost model."""
    path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'
    if not path.exists():
        print(f"  Warning: Model not found at {path}")
        return None
    return joblib.load(path)


# =============================================================================
# CLV CALCULATION
# =============================================================================

def calculate_clv(
    player_norm: str,
    market: str,
    week: int,
    pick: str,
    model_prob: float,
    closing_lines: pd.DataFrame
) -> Dict:
    """
    Calculate Closing Line Value for a single bet.

    CLV = Model Probability - Closing Line Implied Probability
    Positive CLV = We beat the efficient closing market = REAL EDGE
    """
    if closing_lines.empty:
        return {'clv': None, 'clv_pct': None, 'closing_prob': None}

    match = closing_lines[
        (closing_lines['player_norm'] == player_norm) &
        (closing_lines['market'] == market) &
        (closing_lines['week'] == week)
    ]

    if match.empty:
        return {'clv': None, 'clv_pct': None, 'closing_prob': None}

    closing = match.iloc[0]

    if pick == 'UNDER':
        closing_prob = closing['no_vig_under']
    else:
        closing_prob = closing['no_vig_over']

    clv = model_prob - closing_prob

    return {
        'clv': round(clv, 4),
        'clv_pct': round(clv * 100, 2),
        'closing_prob': round(closing_prob, 4),
    }


# =============================================================================
# CALIBRATION CHECK
# =============================================================================

def check_calibration(results: List[Dict], n_bins: int = 5) -> Dict:
    """
    Check model calibration using reliability diagram approach.

    Returns calibration metrics including ECE (Expected Calibration Error).
    """
    if len(results) < 20:
        return {'ece': None, 'bins': [], 'warning': 'Insufficient data for calibration check'}

    df = pd.DataFrame(results)
    df = df.dropna(subset=['clf_prob_under', 'under_hit'])

    if len(df) < 20:
        return {'ece': None, 'bins': [], 'warning': 'Insufficient data after dropna'}

    # Create probability bins
    df['prob_bin'] = pd.cut(df['clf_prob_under'], bins=n_bins, labels=False)

    bins = []
    weighted_errors = []

    for bin_idx in range(n_bins):
        bin_data = df[df['prob_bin'] == bin_idx]
        if len(bin_data) == 0:
            continue

        avg_predicted = bin_data['clf_prob_under'].mean()
        avg_actual = bin_data['under_hit'].mean()
        n = len(bin_data)

        calibration_error = abs(avg_predicted - avg_actual)
        weighted_errors.append(calibration_error * n)

        bins.append({
            'bin': bin_idx,
            'n': n,
            'avg_predicted': round(avg_predicted, 3),
            'avg_actual': round(avg_actual, 3),
            'calibration_error': round(calibration_error, 3),
        })

    ece = sum(weighted_errors) / len(df) if weighted_errors else None

    return {
        'ece': round(ece, 4) if ece else None,
        'bins': bins,
        'total_bets': len(df),
    }


# =============================================================================
# WEEKLY VALIDATION
# =============================================================================

def validate_week(
    week: int,
    season: int,
    props: pd.DataFrame,
    stats: pd.DataFrame,
    closing_lines: pd.DataFrame,
    model: Optional[dict],
) -> Dict:
    """
    Run complete validation for a single week.

    Returns comprehensive metrics including predictions, actuals, CLV, and calibration.
    """
    print(f"\n{'='*70}")
    print(f"WEEK {week} VALIDATION ({season})")
    print(f"{'='*70}")

    # Get test week data
    test_global_week = (season - 2023) * 18 + week
    week_props = props[
        (props['season'] == season) &
        (props['week'] == week)
    ].copy()

    if len(week_props) == 0:
        print(f"  No props found for week {week}")
        return {'week': week, 'season': season, 'error': 'No props found'}

    print(f"  Props available: {len(week_props)}")

    # Get historical data for training (strictly before test week)
    hist_props = props[props['global_week'] < test_global_week - 1].copy()
    hist_stats = stats[stats['global_week'] < test_global_week].copy()

    print(f"  Training data: {len(hist_props)} props from weeks < {week - 1}")

    results = []

    for market in VALIDATION_MARKETS:
        market_props = week_props[week_props['market'] == market].copy()

        if len(market_props) == 0:
            continue

        # Get predictions from model
        if model and market in model.get('models', {}):
            clf = model['models'][market]

            try:
                # Extract features using batch extractor
                features_df = extract_features_batch(
                    market_props,
                    hist_props,
                    market,
                    test_global_week,
                    for_training=False
                )

                feature_cols = list(clf.feature_names_in_)
                X = features_df[feature_cols].copy()

                # Handle missing values
                for col in feature_cols:
                    if col not in X.columns:
                        X[col] = 0.0

                X = X.fillna(0.0)

                # Get predictions
                probs = clf.predict_proba(X)[:, 1]  # P(UNDER)
                features_df['clf_prob_under'] = probs

            except Exception as e:
                print(f"    {market}: Feature extraction failed - {e}")
                continue
        else:
            print(f"    {market}: No model available")
            continue

        # Process each prediction
        for idx, row in features_df.iterrows():
            clf_prob_under = row.get('clf_prob_under', 0.5)

            # Determine pick (no direction constraints during validation)
            pick = 'UNDER' if clf_prob_under > 0.5 else 'OVER'
            model_prob = clf_prob_under if pick == 'UNDER' else (1 - clf_prob_under)

            # Calculate CLV
            clv_data = calculate_clv(
                row.get('player_norm', ''),
                market,
                week,
                pick,
                model_prob,
                closing_lines
            )

            # Record result
            actual_hit = row['under_hit'] if pick == 'UNDER' else (1 - row['under_hit'])

            results.append({
                'week': week,
                'season': season,
                'player': row.get('player', row.get('player_display_name', 'Unknown')),
                'player_norm': row.get('player_norm', ''),
                'market': market,
                'line': row['line'],
                'pick': pick,
                'clf_prob_under': clf_prob_under,
                'model_prob': model_prob,
                'actual_stat': row.get('actual_stat'),
                'under_hit': row['under_hit'],
                'actual_hit': actual_hit,
                'clv': clv_data.get('clv'),
                'clv_pct': clv_data.get('clv_pct'),
                'closing_prob': clv_data.get('closing_prob'),
            })

        print(f"    {market}: {len([r for r in results if r['market'] == market])} predictions")

    # Calculate metrics
    if len(results) == 0:
        return {'week': week, 'season': season, 'error': 'No predictions generated'}

    results_df = pd.DataFrame(results)

    # Win rate and ROI
    wins = results_df['actual_hit'].sum()
    total = len(results_df)
    win_rate = wins / total if total > 0 else 0
    roi = (win_rate * 0.909 - (1 - win_rate)) * 100  # -110 odds

    # CLV stats
    clv_data = results_df.dropna(subset=['clv'])
    avg_clv = clv_data['clv_pct'].mean() if len(clv_data) > 0 else None
    positive_clv_pct = (clv_data['clv'] > 0).mean() * 100 if len(clv_data) > 0 else None

    # Calibration
    calibration = check_calibration(results)

    # Summary
    print(f"\n  RESULTS:")
    print(f"    Total bets: {total}")
    print(f"    Win rate: {win_rate*100:.1f}%")
    print(f"    ROI: {roi:+.1f}%")
    if avg_clv is not None:
        print(f"    Avg CLV: {avg_clv:+.2f}%")
        print(f"    Positive CLV: {positive_clv_pct:.1f}%")
    if calibration.get('ece'):
        print(f"    ECE: {calibration['ece']:.4f}")

    # By market
    print(f"\n  BY MARKET:")
    for market in VALIDATION_MARKETS:
        market_df = results_df[results_df['market'] == market]
        if len(market_df) == 0:
            continue

        m_wins = market_df['actual_hit'].sum()
        m_total = len(market_df)
        m_wr = m_wins / m_total if m_total > 0 else 0
        m_roi = (m_wr * 0.909 - (1 - m_wr)) * 100

        m_clv = market_df.dropna(subset=['clv'])
        m_avg_clv = m_clv['clv_pct'].mean() if len(m_clv) > 0 else None

        clv_str = f", CLV={m_avg_clv:+.1f}%" if m_avg_clv is not None else ""
        print(f"    {market}: n={m_total}, WR={m_wr*100:.1f}%, ROI={m_roi:+.1f}%{clv_str}")

    # By direction
    print(f"\n  BY DIRECTION:")
    for direction in ['OVER', 'UNDER']:
        dir_df = results_df[results_df['pick'] == direction]
        if len(dir_df) == 0:
            continue

        d_wins = dir_df['actual_hit'].sum()
        d_total = len(dir_df)
        d_wr = d_wins / d_total if d_total > 0 else 0
        d_roi = (d_wr * 0.909 - (1 - d_wr)) * 100

        d_clv = dir_df.dropna(subset=['clv'])
        d_avg_clv = d_clv['clv_pct'].mean() if len(d_clv) > 0 else None

        clv_str = f", CLV={d_avg_clv:+.1f}%" if d_avg_clv is not None else ""
        print(f"    {direction}: n={d_total}, WR={d_wr*100:.1f}%, ROI={d_roi:+.1f}%{clv_str}")

    return {
        'week': week,
        'season': season,
        'total_bets': total,
        'wins': int(wins),
        'win_rate': round(win_rate, 4),
        'roi': round(roi, 2),
        'avg_clv': round(avg_clv, 4) if avg_clv else None,
        'positive_clv_pct': round(positive_clv_pct, 2) if positive_clv_pct else None,
        'clv_bets': len(clv_data),
        'ece': calibration.get('ece'),
        'calibration_bins': calibration.get('bins', []),
        'results': results,
    }


def save_week_results(week_data: Dict, season: int):
    """Save week validation results to file."""
    week = week_data['week']

    # Save detailed results
    results_dir = VALIDATION_DIR / str(season)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f'week_{week:02d}_results.json'

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    week_data_clean = convert_types(week_data)

    with open(results_path, 'w') as f:
        json.dump(week_data_clean, f, indent=2)

    print(f"\n  Saved to: {results_path}")

    # Also save bets as CSV for easy analysis
    if week_data.get('results'):
        csv_path = results_dir / f'week_{week:02d}_bets.csv'
        pd.DataFrame(week_data['results']).to_csv(csv_path, index=False)


def generate_summary(season: int):
    """Generate summary report across all validated weeks."""
    results_dir = VALIDATION_DIR / str(season)

    if not results_dir.exists():
        print(f"No validation results found for {season}")
        return

    # Load all week results
    all_results = []
    all_bets = []

    for week_file in sorted(results_dir.glob('week_*_results.json')):
        with open(week_file) as f:
            data = json.load(f)
            all_results.append(data)
            if data.get('results'):
                all_bets.extend(data['results'])

    if not all_results:
        print("No results to summarize")
        return

    print("\n" + "=" * 70)
    print(f"SEASON {season} SUMMARY")
    print("=" * 70)

    # Aggregate metrics
    total_bets = sum(r.get('total_bets', 0) for r in all_results)
    total_wins = sum(r.get('wins', 0) for r in all_results)

    if total_bets > 0:
        overall_wr = total_wins / total_bets
        overall_roi = (overall_wr * 0.909 - (1 - overall_wr)) * 100
    else:
        overall_wr = 0
        overall_roi = 0

    print(f"\nOVERALL:")
    print(f"  Weeks validated: {len(all_results)}")
    print(f"  Total bets: {total_bets}")
    print(f"  Win rate: {overall_wr*100:.1f}%")
    print(f"  ROI: {overall_roi:+.1f}%")

    # CLV summary
    clv_results = [r for r in all_results if r.get('avg_clv') is not None]
    if clv_results:
        avg_clv = np.mean([r['avg_clv'] for r in clv_results])
        avg_pos_clv = np.mean([r['positive_clv_pct'] for r in clv_results])
        print(f"\nCLV ANALYSIS:")
        print(f"  Weeks with CLV data: {len(clv_results)}")
        print(f"  Average CLV: {avg_clv:+.2f}%")
        print(f"  Average positive CLV rate: {avg_pos_clv:.1f}%")

        if avg_clv > 0:
            print("  -> POSITIVE CLV indicates REAL EDGE")
        else:
            print("  -> Negative CLV suggests edge may be illusory")

    # By market analysis
    if all_bets:
        bets_df = pd.DataFrame(all_bets)

        print(f"\nBY MARKET:")
        for market in VALIDATION_MARKETS:
            m_df = bets_df[bets_df['market'] == market]
            if len(m_df) == 0:
                continue

            m_wr = m_df['actual_hit'].mean()
            m_roi = (m_wr * 0.909 - (1 - m_wr)) * 100

            m_clv = m_df.dropna(subset=['clv'])
            m_avg_clv = m_clv['clv_pct'].mean() if len(m_clv) > 0 else None

            clv_str = f", CLV={m_avg_clv:+.1f}%" if m_avg_clv else ""
            print(f"  {market}: n={len(m_df)}, WR={m_wr*100:.1f}%, ROI={m_roi:+.1f}%{clv_str}")

        print(f"\nBY DIRECTION:")
        for direction in ['OVER', 'UNDER']:
            d_df = bets_df[bets_df['pick'] == direction]
            if len(d_df) == 0:
                continue

            d_wr = d_df['actual_hit'].mean()
            d_roi = (d_wr * 0.909 - (1 - d_wr)) * 100

            d_clv = d_df.dropna(subset=['clv'])
            d_avg_clv = d_clv['clv_pct'].mean() if len(d_clv) > 0 else None
            d_pos_clv = (d_clv['clv'] > 0).mean() * 100 if len(d_clv) > 0 else None

            clv_str = f", CLV={d_avg_clv:+.1f}% ({d_pos_clv:.0f}% positive)" if d_avg_clv else ""
            print(f"  {direction}: n={len(d_df)}, WR={d_wr*100:.1f}%, ROI={d_roi:+.1f}%{clv_str}")

        # Market x Direction analysis (key for setting constraints)
        print(f"\nMARKET x DIRECTION (for constraint decisions):")
        for market in VALIDATION_MARKETS:
            print(f"\n  {market}:")
            for direction in ['OVER', 'UNDER']:
                md_df = bets_df[(bets_df['market'] == market) & (bets_df['pick'] == direction)]
                if len(md_df) < 5:
                    continue

                md_wr = md_df['actual_hit'].mean()
                md_roi = (md_wr * 0.909 - (1 - md_wr)) * 100

                md_clv = md_df.dropna(subset=['clv'])
                md_avg_clv = md_clv['clv_pct'].mean() if len(md_clv) > 0 else None

                recommendation = ""
                if md_avg_clv is not None:
                    if md_avg_clv > 1.0 and md_roi > 0:
                        recommendation = " <- KEEP"
                    elif md_avg_clv < -1.0 or md_roi < -5:
                        recommendation = " <- DISABLE"

                clv_str = f", CLV={md_avg_clv:+.1f}%" if md_avg_clv else ""
                print(f"    {direction}: n={len(md_df)}, WR={md_wr*100:.1f}%, ROI={md_roi:+.1f}%{clv_str}{recommendation}")

    # Calibration summary
    ece_values = [r.get('ece') for r in all_results if r.get('ece') is not None]
    if ece_values:
        avg_ece = np.mean(ece_values)
        print(f"\nCALIBRATION:")
        print(f"  Average ECE: {avg_ece:.4f}")
        if avg_ece < 0.05:
            print("  -> Well calibrated")
        elif avg_ece < 0.10:
            print("  -> Moderately calibrated")
        else:
            print("  -> Poorly calibrated - consider recalibration")

    # Week-by-week summary
    print(f"\nWEEK-BY-WEEK:")
    print(f"  {'Week':<6} {'Bets':<6} {'WR':<8} {'ROI':<10} {'CLV':<10} {'ECE':<8}")
    print(f"  {'-'*50}")

    for r in all_results:
        week = r.get('week', '?')
        bets = r.get('total_bets', 0)
        wr = r.get('win_rate', 0) * 100
        roi = r.get('roi', 0)
        clv = r.get('avg_clv')
        ece = r.get('ece')

        clv_str = f"{clv:+.2f}%" if clv else "N/A"
        ece_str = f"{ece:.4f}" if ece else "N/A"

        print(f"  {week:<6} {bets:<6} {wr:>6.1f}%  {roi:>+7.1f}%  {clv_str:>8}  {ece_str:<8}")

    # Save summary
    summary_path = results_dir / 'season_summary.json'
    summary = {
        'season': season,
        'weeks_validated': len(all_results),
        'total_bets': total_bets,
        'total_wins': total_wins,
        'overall_win_rate': round(overall_wr, 4),
        'overall_roi': round(overall_roi, 2),
        'avg_clv': round(avg_clv, 4) if clv_results else None,
        'avg_ece': round(avg_ece, 4) if ece_values else None,
        'generated_at': datetime.now().isoformat(),
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to: {summary_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Weekly Validation Pipeline - Scientific approach to sports betting'
    )
    parser.add_argument('--week', type=int, help='Single week to validate')
    parser.add_argument('--weeks', type=str, help='Week range (e.g., 1-16)')
    parser.add_argument('--season', type=int, default=2025, help='NFL season')
    parser.add_argument('--live', action='store_true', help='Live mode (fetch fresh data)')
    parser.add_argument('--summary', action='store_true', help='Generate summary report')

    args = parser.parse_args()

    if args.summary:
        generate_summary(args.season)
        return

    if not args.week and not args.weeks:
        parser.error("Either --week, --weeks, or --summary is required")

    # Determine weeks to validate
    if args.weeks:
        start, end = map(int, args.weeks.split('-'))
        weeks = list(range(start, end + 1))
    else:
        weeks = [args.week]

    print("=" * 70)
    print("WEEKLY VALIDATION PIPELINE")
    print("=" * 70)
    print(f"Season: {args.season}")
    print(f"Weeks: {weeks}")
    print(f"Mode: {'LIVE' if args.live else 'HISTORICAL'}")
    print(f"Direction constraints: {'DISABLED' if DISABLE_DIRECTION_CONSTRAINTS else 'ENABLED'}")
    print()

    # Load data
    print("Loading data...")
    props = load_historical_props()
    stats = load_player_stats()
    closing_lines = load_closing_lines(args.season)
    model = load_production_model()

    print(f"  Props: {len(props)}")
    print(f"  Stats: {len(stats)}")
    print(f"  Closing lines: {len(closing_lines)}")
    print(f"  Model: {model.get('version', 'unknown') if model else 'NOT FOUND'}")

    # Validate each week
    for week in weeks:
        try:
            week_data = validate_week(
                week=week,
                season=args.season,
                props=props,
                stats=stats,
                closing_lines=closing_lines,
                model=model,
            )

            if 'error' not in week_data:
                save_week_results(week_data, args.season)

        except Exception as e:
            print(f"\n  ERROR validating week {week}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary if multiple weeks
    if len(weeks) > 1:
        generate_summary(args.season)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
