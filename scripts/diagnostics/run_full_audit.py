#!/usr/bin/env python3
"""
NFL QUANT - Comprehensive Model Diagnostics Framework

Systematic detection of hidden issues across ALL markets:
1. EWMA Sensitivity - How much does a single outlier move projections?
2. Data Coverage - Does each market have historical data for validation?
3. Projection Drift - Are projections systematically biased vs actuals?
4. Edge Calculation - Are edge types being confused?
5. Calibration - When we say 70% confidence, do we win 70%?
6. Feature Importance - Do feature rankings make sense?

Run: python scripts/diagnostics/run_full_audit.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.model_config import (
    MARKET_TO_STAT,
    SUPPORTED_MARKETS,
    FEATURES,
    get_active_model_path,
    MARKET_SNR_CONFIG,
    is_market_enabled,
    get_disabled_markets,
)
from nfl_quant.utils.season_utils import get_current_season


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_weekly_stats() -> pd.DataFrame:
    """Load NFLverse weekly stats."""
    path = project_root / 'data' / 'nflverse' / 'weekly_stats.parquet'
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def load_historical_props() -> Optional[pd.DataFrame]:
    """Load historical props data."""
    path = project_root / 'data' / 'backtest' / 'all_historical_props_2025.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def load_model_predictions(week: int) -> Optional[pd.DataFrame]:
    """Load model predictions for a specific week."""
    path = project_root / 'data' / f'model_predictions_week{week}.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def get_stat_column(market: str) -> Optional[str]:
    """Map market to NFLverse stat column."""
    return MARKET_TO_STAT.get(market)


# =============================================================================
# DIAGNOSTIC 1: EWMA SENSITIVITY AUDIT
# =============================================================================

def audit_ewma_sensitivity(market: str, weekly_stats: pd.DataFrame) -> Dict:
    """
    Test: How much does ONE outlier game move the 4-week EWMA?

    A healthy model should be resistant to single-game variance.
    """
    stat_col = get_stat_column(market)
    if not stat_col or stat_col not in weekly_stats.columns:
        return {
            'market': market,
            'error': f'No stat column mapping for {market}',
            'flag': True,
        }

    # Get current season data
    df = weekly_stats[weekly_stats['season'] == get_current_season()].copy()

    if len(df) == 0:
        return {'market': market, 'error': 'No 2025 data', 'flag': True}

    # Group by player
    results = []

    for player, player_df in df.groupby('player_display_name'):
        player_df = player_df.sort_values('week')

        if len(player_df) < 5:
            continue

        stat_values = player_df[stat_col].dropna().values
        if len(stat_values) < 5:
            continue

        # Calculate baseline EWMA (first 4 games)
        baseline_values = stat_values[:4]
        baseline_ewma = np.average(baseline_values, weights=[0.12, 0.18, 0.27, 0.40])

        if baseline_ewma <= 0:
            continue

        # Simulate: What if next game was 2x the baseline?
        simulated_outlier = baseline_ewma * 2
        values_with_outlier = list(baseline_values[1:]) + [simulated_outlier]
        new_ewma = np.average(values_with_outlier, weights=[0.12, 0.18, 0.27, 0.40])

        # Calculate sensitivity
        pct_change = (new_ewma - baseline_ewma) / baseline_ewma

        results.append({
            'player': player,
            'baseline_ewma': baseline_ewma,
            'ewma_after_2x_outlier': new_ewma,
            'pct_swing': pct_change,
        })

    if len(results) == 0:
        return {'market': market, 'error': 'No valid players', 'flag': True}

    # Calculate average sensitivity
    avg_swing = np.mean([r['pct_swing'] for r in results])
    max_swing = np.max([r['pct_swing'] for r in results])

    # Flag markets where average swing > 25%
    flag = avg_swing > 0.25

    # Get worst cases
    worst = sorted(results, key=lambda x: -x['pct_swing'])[:3]

    return {
        'market': market,
        'avg_sensitivity': avg_swing,
        'max_sensitivity': max_swing,
        'flag': flag,
        'sample_size': len(results),
        'worst_cases': worst,
    }


# =============================================================================
# DIAGNOSTIC 2: DATA COVERAGE AUDIT
# =============================================================================

def audit_data_coverage(weekly_stats: pd.DataFrame, historical_props: Optional[pd.DataFrame]) -> Tuple[Dict, List]:
    """
    Check: Does each market have historical data for sanity checks?
    """
    # Markets to check
    markets = [
        'player_receptions', 'player_reception_yds', 'player_rush_yds',
        'player_pass_yds', 'player_rush_attempts', 'player_pass_completions',
        'player_pass_attempts', 'player_pass_tds',
    ]

    coverage = {}
    missing = []

    for market in markets:
        # Check 1: Historical props
        props_count = 0
        unique_players = 0
        if historical_props is not None and 'market' in historical_props.columns:
            market_props = historical_props[historical_props['market'] == market]
            props_count = len(market_props)
            if 'player_norm' in market_props.columns:
                unique_players = market_props['player_norm'].nunique()
            elif 'player' in market_props.columns:
                unique_players = market_props['player'].nunique()

        # Check 2: NFLverse mapping
        stat_col = get_stat_column(market)
        nflverse_available = stat_col is not None and stat_col in weekly_stats.columns

        # Check 3: Determine if we have sanity check data
        has_sanity_data = props_count > 50 or nflverse_available

        coverage[market] = {
            'hist_props_count': props_count,
            'nflverse_fallback': nflverse_available,
            'nflverse_col': stat_col,
            'unique_players': unique_players,
            'has_sanity_data': has_sanity_data,
        }

        if not has_sanity_data:
            missing.append(market)

    return coverage, missing


# =============================================================================
# DIAGNOSTIC 3: PROJECTION VS ACTUALS DISTRIBUTION
# =============================================================================

def audit_projection_drift(market: str, weekly_stats: pd.DataFrame, weeks: range = range(5, 14)) -> Dict:
    """
    Compare: Projection distribution vs Actual distribution

    A healthy model's projections should roughly match actual outcomes.
    """
    stat_col = get_stat_column(market)
    if not stat_col or stat_col not in weekly_stats.columns:
        return {'market': market, 'error': 'No stat mapping', 'flags': ['No data']}

    # Map market to the actual prediction column name in model_predictions files
    # Model predictions use different naming than NFLverse
    MARKET_TO_PRED_COL = {
        'player_receptions': 'receptions_mean',
        'player_reception_yds': 'receiving_yards_mean',
        'player_rush_yds': 'rushing_yards_mean',
        'player_pass_yds': 'passing_yards_mean',
        'player_rush_attempts': 'rushing_attempts_mean',
        'player_pass_completions': 'passing_completions_mean',
        'player_pass_attempts': 'passing_attempts_mean',
        'player_pass_tds': 'passing_tds_mean',
    }

    results = []

    for week in weeks:
        # Load predictions for this week
        preds = load_model_predictions(week)
        if preds is None:
            continue

        # Determine projection column - use explicit mapping first
        proj_col = MARKET_TO_PRED_COL.get(market)
        if proj_col is None or proj_col not in preds.columns:
            # Fallback to searching
            for col in [f'{stat_col}_mean', 'model_projection', stat_col]:
                if col in preds.columns:
                    proj_col = col
                    break

        if proj_col is None or proj_col not in preds.columns:
            continue

        # Get actuals from weekly_stats
        actuals = weekly_stats[
            (weekly_stats['season'] == get_current_season()) &
            (weekly_stats['week'] == week)
        ].copy()

        if len(actuals) == 0:
            continue

        # Merge on player name
        player_col = 'player_name' if 'player_name' in preds.columns else 'player_display_name'
        if player_col not in preds.columns:
            continue

        merged = preds.merge(
            actuals[['player_display_name', stat_col]],
            left_on=player_col,
            right_on='player_display_name',
            how='inner'
        )

        if len(merged) < 5:
            continue

        proj_values = merged[proj_col].dropna()
        actual_values = merged[stat_col].dropna()

        if len(proj_values) < 5 or len(actual_values) < 5:
            continue

        # Compare distributions
        proj_mean = proj_values.mean()
        actual_mean = actual_values.mean()

        if actual_mean > 0:
            mean_drift = (proj_mean - actual_mean) / actual_mean
        else:
            mean_drift = 0

        # Correlation
        common_idx = proj_values.index.intersection(actual_values.index)
        if len(common_idx) >= 5:
            correlation = merged.loc[common_idx, proj_col].corr(merged.loc[common_idx, stat_col])
        else:
            correlation = np.nan

        results.append({
            'week': week,
            'n': len(merged),
            'proj_mean': proj_mean,
            'actual_mean': actual_mean,
            'mean_drift': mean_drift,
            'correlation': correlation,
        })

    if len(results) == 0:
        return {'market': market, 'error': 'No data for comparison', 'flags': ['No data']}

    # Aggregate
    avg_drift = np.nanmean([r['mean_drift'] for r in results])
    avg_corr = np.nanmean([r['correlation'] for r in results])

    flags = []
    if abs(avg_drift) > 0.15:
        flags.append(f"Systematic drift: {avg_drift:+.1%}")
    if not np.isnan(avg_corr) and avg_corr < 0.25:
        flags.append(f"Low correlation: {avg_corr:.2f}")

    return {
        'market': market,
        'avg_drift': avg_drift,
        'avg_correlation': avg_corr,
        'weeks_analyzed': len(results),
        'flags': flags,
    }


# =============================================================================
# DIAGNOSTIC 4: EDGE CALCULATION AUDIT
# =============================================================================

def audit_edge_calculations(market: str, weekly_stats: pd.DataFrame) -> Dict:
    """
    Check: Are we calculating edges correctly and consistently?
    """
    stat_col = get_stat_column(market)
    if not stat_col:
        return {'market': market, 'error': 'No stat mapping', 'issues_found': 0, 'issues': []}

    # Load latest recommendations
    recs_path = project_root / 'reports' / 'CURRENT_WEEK_RECOMMENDATIONS.csv'
    if not recs_path.exists():
        return {'market': market, 'error': 'No recommendations file', 'issues_found': 0, 'issues': []}

    recs = pd.read_csv(recs_path)
    recs = recs[recs['market'] == market]

    if len(recs) == 0:
        return {'market': market, 'error': 'No recommendations for this market', 'issues_found': 0, 'issues': []}

    issues = []

    for idx, rec in recs.iterrows():
        line = rec.get('line', 0)
        projection = rec.get('model_projection', 0)
        trailing = rec.get('hist_avg')
        player = rec.get('player', 'Unknown')

        # Check for missing trailing
        if pd.isna(trailing) or trailing <= 0:
            issues.append({
                'player': player,
                'issue': 'Missing trailing stat',
                'severity': 'HIGH',
            })
            continue

        if line <= 0 or projection <= 0:
            continue

        # Calculate edge types
        lvt = (line - trailing) / trailing
        model_edge = (projection - line) / line
        proj_vs_trailing = (projection - trailing) / trailing

        # Flag 1: Large model edge but small LVT
        if abs(model_edge) > 0.20 and abs(lvt) < 0.05:
            issues.append({
                'player': player,
                'issue': f'Model edge {model_edge:.1%} but LVT only {lvt:.1%}',
                'severity': 'MEDIUM',
            })

        # Flag 2: Projection far from trailing
        if abs(proj_vs_trailing) > 0.30:
            issues.append({
                'player': player,
                'issue': f'Projection {proj_vs_trailing:.1%} from trailing',
                'severity': 'HIGH',
                'projection': projection,
                'trailing': trailing,
            })

        # Flag 3: Projection and line same direction from trailing but different magnitude
        if (projection > trailing and line > trailing) or (projection < trailing and line < trailing):
            if abs(model_edge) > 0.15:
                issues.append({
                    'player': player,
                    'issue': f'Proj and line both {"above" if projection > trailing else "below"} trailing, but {model_edge:.1%} apart',
                    'severity': 'LOW',
                })

    return {
        'market': market,
        'total_recs': len(recs),
        'issues_found': len(issues),
        'issues': issues,
    }


# =============================================================================
# DIAGNOSTIC 5: BACKTEST CALIBRATION
# =============================================================================

def audit_calibration(market: str, historical_props: Optional[pd.DataFrame]) -> Dict:
    """
    Reality check: When model said 70% confidence, did it win 70%?
    """
    if historical_props is None:
        return {'market': market, 'error': 'No historical props data'}

    market_data = historical_props[historical_props['market'] == market].copy()

    if len(market_data) < 20:
        return {'market': market, 'error': f'Insufficient data ({len(market_data)} samples)'}

    # Check required columns
    required = ['actual_stat', 'line']
    if not all(col in market_data.columns for col in required):
        return {'market': market, 'error': 'Missing required columns'}

    # Calculate hit rates by line bucket
    market_data['over_hit'] = (market_data['actual_stat'] > market_data['line']).astype(int)
    market_data['under_hit'] = (market_data['actual_stat'] < market_data['line']).astype(int)

    overall_over_rate = market_data['over_hit'].mean()
    overall_under_rate = market_data['under_hit'].mean()
    push_rate = 1 - overall_over_rate - overall_under_rate

    # Expected 50/50 - check for systematic bias
    flags = []
    if abs(overall_over_rate - 0.5) > 0.05:
        direction = "OVER" if overall_over_rate > 0.5 else "UNDER"
        flags.append(f"Market bias toward {direction}: {overall_over_rate:.1%} over rate")

    return {
        'market': market,
        'overall_over_rate': overall_over_rate,
        'overall_under_rate': overall_under_rate,
        'push_rate': push_rate,
        'total_samples': len(market_data),
        'flags': flags,
    }


# =============================================================================
# DIAGNOSTIC 6: FEATURE IMPORTANCE SANITY
# =============================================================================

def audit_feature_importance() -> Dict:
    """
    Check: Do feature importances make sense?
    """
    import joblib

    model_path = get_active_model_path()
    if not model_path.exists():
        return {'error': 'Active model not found'}

    try:
        model = joblib.load(model_path)
    except Exception as e:
        return {'error': f'Failed to load model: {e}'}

    if not hasattr(model, 'feature_importances_'):
        return {'error': 'Model has no feature_importances_'}

    importances = dict(zip(FEATURES, model.feature_importances_))

    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])

    # Expected top features
    expected_top = ['line_vs_trailing', 'player_under_rate', 'player_bias', 'line_level']

    flags = []

    # Check if expected features are actually important
    for feat in expected_top:
        if feat in importances:
            rank = [i for i, (f, _) in enumerate(sorted_imp) if f == feat]
            if rank and rank[0] > 7:
                flags.append(f"{feat} ranked #{rank[0]+1} (expected top 7)")

    # Check for suspicious features ranked high
    suspicious_high = []
    for feat, imp in sorted_imp[:5]:
        if imp > 0.15 and feat not in expected_top:
            suspicious_high.append(f"{feat}: {imp:.1%}")

    if suspicious_high:
        flags.append(f"Unexpectedly high: {', '.join(suspicious_high)}")

    return {
        'top_10': sorted_imp[:10],
        'bottom_5': sorted_imp[-5:],
        'flags': flags,
    }


# =============================================================================
# DIAGNOSTIC 7: SNR CONFIG COMPLETENESS
# =============================================================================

def audit_snr_coverage() -> Dict:
    """Check if all markets have SNR configuration."""
    # Markets we recommend on
    active_markets = [
        'player_receptions', 'player_reception_yds', 'player_rush_yds',
        'player_pass_yds', 'player_rush_attempts', 'player_pass_completions',
        'player_pass_attempts', 'player_pass_tds', 'player_anytime_td',
        'player_1st_td', 'player_rush_tds',
    ]

    missing = []
    configured = []

    for market in active_markets:
        if market in MARKET_SNR_CONFIG:
            config = MARKET_SNR_CONFIG[market]
            configured.append({
                'market': market,
                'tier': config.tier,
                'conf_threshold': config.confidence_threshold,
            })
        else:
            missing.append(market)

    return {
        'configured_count': len(configured),
        'missing_count': len(missing),
        'missing': missing,
        'configured': configured,
    }


# =============================================================================
# MASTER AUDIT
# =============================================================================

def run_full_audit() -> List[Dict]:
    """Run all diagnostics and generate report."""

    print("=" * 70)
    print("NFL QUANT MODEL DIAGNOSTICS - COMPREHENSIVE AUDIT")
    print("=" * 70)

    # Load data once
    print("\nLoading data...")
    weekly_stats = load_weekly_stats()
    historical_props = load_historical_props()

    print(f"  Weekly stats: {len(weekly_stats):,} rows")
    print(f"  Historical props: {len(historical_props) if historical_props is not None else 0:,} rows")

    # Show disabled markets first
    disabled_markets = get_disabled_markets()
    if disabled_markets:
        print(f"\n  🚫 DISABLED MARKETS ({len(disabled_markets)}):")
        for market, reason in disabled_markets.items():
            print(f"     {market}: {reason}")

    markets = [
        'player_receptions', 'player_rush_yds', 'player_reception_yds',
        'player_pass_yds', 'player_rush_attempts', 'player_pass_completions',
    ]

    # Separate enabled and disabled markets
    enabled_markets = [m for m in markets if is_market_enabled(m)]
    disabled_in_list = [m for m in markets if not is_market_enabled(m)]

    all_issues = []

    # 1. Data Coverage
    print("\n" + "-" * 70)
    print("[1/7] DATA COVERAGE AUDIT")
    print("-" * 70)
    coverage, missing = audit_data_coverage(weekly_stats, historical_props)

    for market, cov in coverage.items():
        status = "✅" if cov['has_sanity_data'] else "⚠️"
        fallback = "NFLverse" if cov['nflverse_fallback'] else "None"
        print(f"  {status} {market}: {cov['hist_props_count']} props, fallback={fallback}")

    if missing:
        print(f"\n  ⚠️ Markets without sanity data: {missing}")
        all_issues.extend([{'type': 'data_coverage', 'market': m, 'severity': 'HIGH'} for m in missing])

    # 2. EWMA Sensitivity
    print("\n" + "-" * 70)
    print("[2/7] EWMA SENSITIVITY AUDIT")
    print("-" * 70)
    print("  Testing: How much does ONE 2x outlier swing the EWMA?")

    for market in markets:
        result = audit_ewma_sensitivity(market, weekly_stats)
        if result.get('error'):
            print(f"  ⚠️ {market}: {result['error']}")
            continue

        status = "⚠️" if result['flag'] else "✅"
        print(f"  {status} {market}: {result['avg_sensitivity']:.1%} avg swing (n={result['sample_size']})")

        if result['flag']:
            all_issues.append({
                'type': 'ewma_sensitivity',
                'market': market,
                'severity': 'MEDIUM',
                'avg_swing': result['avg_sensitivity'],
            })
            if result.get('worst_cases'):
                for wc in result['worst_cases'][:2]:
                    print(f"      Worst: {wc['player']}: {wc['pct_swing']:.1%} swing")

    # 3. Projection Drift
    print("\n" + "-" * 70)
    print("[3/7] PROJECTION DRIFT AUDIT")
    print("-" * 70)
    print("  Testing: Do projections systematically drift from actuals?")
    print("  (🚫 = market disabled, issues expected)")

    for market in markets:
        market_disabled = not is_market_enabled(market)
        result = audit_projection_drift(market, weekly_stats)
        if result.get('error'):
            disabled_tag = " 🚫" if market_disabled else ""
            print(f"  ⚠️ {market}{disabled_tag}: {result['error']}")
            continue

        disabled_tag = " 🚫" if market_disabled else ""
        status = "⚠️" if result['flags'] else "✅"
        drift_str = f"{result['avg_drift']:+.1%}" if not np.isnan(result['avg_drift']) else "N/A"
        corr_str = f"{result['avg_correlation']:.2f}" if not np.isnan(result['avg_correlation']) else "N/A"
        print(f"  {status} {market}{disabled_tag}: drift={drift_str}, corr={corr_str}, weeks={result['weeks_analyzed']}")

        if result['flags']:
            for flag in result['flags']:
                print(f"      {flag}")
            # Disabled markets get LOW severity (expected issues)
            severity = 'LOW' if market_disabled else ('HIGH' if 'drift' in str(result['flags']) else 'MEDIUM')
            all_issues.append({
                'type': 'projection_drift',
                'market': market,
                'severity': severity,
                'details': result['flags'],
                'disabled': market_disabled,
            })

    # 4. Edge Calculations
    print("\n" + "-" * 70)
    print("[4/7] EDGE CALCULATION AUDIT")
    print("-" * 70)
    print("  Testing: Are edge calculations consistent?")

    for market in markets:
        result = audit_edge_calculations(market, weekly_stats)
        if result.get('error'):
            print(f"  ⚠️ {market}: {result['error']}")
            continue

        status = "⚠️" if result['issues_found'] > 0 else "✅"
        print(f"  {status} {market}: {result['issues_found']} issues in {result['total_recs']} recs")

        if result['issues']:
            for issue in result['issues'][:3]:
                print(f"      {issue['severity']}: {issue['player']} - {issue['issue']}")
            all_issues.append({
                'type': 'edge_calculation',
                'market': market,
                'severity': 'MEDIUM',
                'issues': result['issues'],
            })

    # 5. Calibration
    print("\n" + "-" * 70)
    print("[5/7] CALIBRATION AUDIT")
    print("-" * 70)
    print("  Testing: Historical over/under rates by market")

    for market in markets:
        result = audit_calibration(market, historical_props)
        if result.get('error'):
            print(f"  ⚠️ {market}: {result['error']}")
            continue

        status = "⚠️" if result.get('flags') else "✅"
        print(f"  {status} {market}: {result['overall_over_rate']:.1%} over rate (n={result['total_samples']})")

        if result.get('flags'):
            for flag in result['flags']:
                print(f"      {flag}")
            all_issues.append({
                'type': 'calibration',
                'market': market,
                'severity': 'LOW',
                'details': result['flags'],
            })

    # 6. Feature Importance
    print("\n" + "-" * 70)
    print("[6/7] FEATURE IMPORTANCE AUDIT")
    print("-" * 70)

    result = audit_feature_importance()
    if result.get('error'):
        print(f"  ⚠️ {result['error']}")
    else:
        status = "⚠️" if result.get('flags') else "✅"
        print(f"  {status} Top features:")
        for feat, imp in result['top_10'][:5]:
            print(f"      {feat}: {imp:.1%}")

        if result.get('flags'):
            print("  Flags:")
            for flag in result['flags']:
                print(f"      ⚠️ {flag}")
            all_issues.append({
                'type': 'feature_importance',
                'severity': 'LOW',
                'details': result['flags'],
            })

    # 7. SNR Coverage
    print("\n" + "-" * 70)
    print("[7/7] SNR CONFIGURATION AUDIT")
    print("-" * 70)

    result = audit_snr_coverage()
    if result['missing']:
        print(f"  ⚠️ Missing SNR config for: {result['missing']}")
        all_issues.append({
            'type': 'snr_coverage',
            'severity': 'HIGH',
            'missing': result['missing'],
        })
    else:
        print(f"  ✅ All {result['configured_count']} markets have SNR config")

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    high_severity = [i for i in all_issues if i.get('severity') == 'HIGH']
    medium_severity = [i for i in all_issues if i.get('severity') == 'MEDIUM']
    low_severity = [i for i in all_issues if i.get('severity') == 'LOW']

    print(f"\n  Total Issues: {len(all_issues)}")
    print(f"    🔴 HIGH:   {len(high_severity)}")
    print(f"    🟡 MEDIUM: {len(medium_severity)}")
    print(f"    🟢 LOW:    {len(low_severity)}")

    if high_severity:
        print("\n  🔴 HIGH SEVERITY ISSUES:")
        for issue in high_severity:
            print(f"    - [{issue['type']}] {issue.get('market', 'global')}: {issue.get('details', issue)}")

    print("\n" + "=" * 70)

    return all_issues


if __name__ == "__main__":
    issues = run_full_audit()
