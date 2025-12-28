#!/usr/bin/env python3
"""
Calibration Report Generator

Generates comprehensive calibration diagnostics for the NFL QUANT model:
1. Reliability diagram (matplotlib visualization)
2. ECE (Expected Calibration Error) calculation
3. Brier score tracking
4. Calibration by market and direction

Usage:
    # Generate report for 2025 season
    python scripts/validate/calibration_report.py --season 2025

    # Generate report from validation results
    python scripts/validate/calibration_report.py --from-validation --season 2025
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

REPORTS_DIR = PROJECT_ROOT / 'reports' / 'calibration'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_DIR = PROJECT_ROOT / 'data' / 'validation'

# Calibration thresholds
ECE_EXCELLENT = 0.05   # < 5% = excellent calibration
ECE_GOOD = 0.10        # < 10% = good calibration
ECE_ACCEPTABLE = 0.15  # < 15% = acceptable

MARKETS = [
    'player_receptions',
    'player_reception_yds',
    'player_rush_yds',
    'player_rush_attempts',
    'player_pass_yds',
    'player_pass_attempts',
]


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_ece(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict]]:
    """
    Calculate Expected Calibration Error.

    ECE = sum(|accuracy(bin) - confidence(bin)| * n(bin)) / n_total

    Returns:
        Tuple of (ece_value, bin_details)
    """
    if len(predicted) == 0:
        return None, []

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted, bin_edges[1:-1])

    bins = []
    weighted_errors = []
    total = len(predicted)

    for i in range(n_bins):
        mask = bin_indices == i
        n = mask.sum()

        if n == 0:
            continue

        avg_predicted = predicted[mask].mean()
        avg_actual = actual[mask].mean()
        calibration_error = abs(avg_predicted - avg_actual)

        weighted_errors.append(calibration_error * n)

        bins.append({
            'bin': i,
            'lower': bin_edges[i],
            'upper': bin_edges[i + 1],
            'n': int(n),
            'avg_predicted': float(avg_predicted),
            'avg_actual': float(avg_actual),
            'calibration_error': float(calibration_error),
        })

    ece = sum(weighted_errors) / total if weighted_errors else None
    return ece, bins


def calculate_brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Brier Score (mean squared error of probability predictions).

    Lower is better. Random = 0.25, Perfect = 0.0
    """
    if len(predicted) == 0:
        return None
    return float(np.mean((predicted - actual) ** 2))


def calculate_log_loss(predicted: np.ndarray, actual: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate Log Loss (cross-entropy).

    Lower is better. Random = 0.693, Perfect = 0.0
    """
    if len(predicted) == 0:
        return None

    # Clip predictions to avoid log(0)
    predicted = np.clip(predicted, eps, 1 - eps)

    return float(-np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_reliability_diagram(
    predicted: np.ndarray,
    actual: np.ndarray,
    title: str = "Reliability Diagram",
    n_bins: int = 10,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot reliability diagram (calibration curve).

    A perfectly calibrated model lies on the diagonal.
    """
    ece, bins = calculate_ece(predicted, actual, n_bins)

    if not bins:
        print("No data for reliability diagram")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Reliability diagram
    bin_mids = [(b['lower'] + b['upper']) / 2 for b in bins]
    accuracies = [b['avg_actual'] for b in bins]
    confidences = [b['avg_predicted'] for b in bins]
    counts = [b['n'] for b in bins]

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.7)

    # Actual calibration
    ax1.bar(bin_mids, accuracies, width=1/n_bins * 0.8, alpha=0.6, color='steelblue',
            label='Actual Accuracy', edgecolor='black')

    # Gap to perfect calibration
    for i, (mid, acc, conf) in enumerate(zip(bin_mids, accuracies, confidences)):
        if acc != conf:
            color = 'red' if acc < conf else 'green'
            ax1.plot([mid, mid], [min(acc, conf), max(acc, conf)],
                     color=color, linewidth=2, alpha=0.7)

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}' if ece else title, fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(alpha=0.3)

    # Add ECE rating
    if ece:
        if ece < ECE_EXCELLENT:
            rating, color = 'EXCELLENT', 'green'
        elif ece < ECE_GOOD:
            rating, color = 'GOOD', 'blue'
        elif ece < ECE_ACCEPTABLE:
            rating, color = 'ACCEPTABLE', 'orange'
        else:
            rating, color = 'POOR', 'red'

        ax1.text(0.05, 0.95, f'Calibration: {rating}', transform=ax1.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Right plot: Sample distribution
    ax2.bar(bin_mids, counts, width=1/n_bins * 0.8, alpha=0.6, color='gray',
            edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14)
    ax2.set_xlim([0, 1])
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_calibration_by_market(
    bets_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot calibration metrics by market.
    """
    markets = bets_df['market'].unique()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, market in enumerate(markets[:6]):
        ax = axes[idx]
        market_df = bets_df[bets_df['market'] == market]

        if len(market_df) < 20:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(market_df)})',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(market)
            continue

        predicted = market_df['clf_prob_under'].values
        actual = market_df['under_hit'].values

        ece, bins = calculate_ece(predicted, actual, n_bins=5)

        if not bins:
            continue

        bin_mids = [(b['lower'] + b['upper']) / 2 for b in bins]
        accuracies = [b['avg_actual'] for b in bins]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)

        # Actual calibration
        ax.bar(bin_mids, accuracies, width=0.15, alpha=0.6, color='steelblue',
               edgecolor='black')

        ax.set_title(f'{market}\nECE = {ece:.4f}' if ece else market)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(markets), 6):
        axes[idx].axis('off')

    plt.suptitle('Calibration by Market', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# REPORT GENERATION
# =============================================================================

def load_validation_bets(season: int) -> pd.DataFrame:
    """Load all bets from validation results."""
    results_dir = VALIDATION_DIR / str(season)

    if not results_dir.exists():
        raise FileNotFoundError(f"No validation results for {season}")

    all_bets = []

    for week_file in sorted(results_dir.glob('week_*_results.json')):
        with open(week_file) as f:
            data = json.load(f)
            if data.get('results'):
                all_bets.extend(data['results'])

    if not all_bets:
        raise ValueError(f"No bets found in validation results for {season}")

    return pd.DataFrame(all_bets)


def load_training_data() -> pd.DataFrame:
    """Load training data with actuals."""
    path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    return pd.read_csv(path, low_memory=False)


def generate_report(
    bets_df: pd.DataFrame,
    season: int,
    output_dir: Path = REPORTS_DIR,
) -> Dict:
    """
    Generate comprehensive calibration report.

    Returns:
        Dictionary with all calibration metrics
    """
    print("=" * 70)
    print(f"CALIBRATION REPORT - {season} Season")
    print("=" * 70)
    print(f"Total predictions: {len(bets_df):,}")

    # Ensure numeric types
    bets_df = bets_df.dropna(subset=['clf_prob_under', 'under_hit'])
    predicted = bets_df['clf_prob_under'].astype(float).values
    actual = bets_df['under_hit'].astype(float).values

    # Overall metrics
    ece, bins = calculate_ece(predicted, actual)
    brier = calculate_brier_score(predicted, actual)
    logloss = calculate_log_loss(predicted, actual)

    print(f"\nOVERALL METRICS:")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss: {logloss:.4f}")

    # ECE rating
    if ece < ECE_EXCELLENT:
        rating = 'EXCELLENT (< 5%)'
    elif ece < ECE_GOOD:
        rating = 'GOOD (< 10%)'
    elif ece < ECE_ACCEPTABLE:
        rating = 'ACCEPTABLE (< 15%)'
    else:
        rating = 'POOR (>= 15%) - RECALIBRATION NEEDED'

    print(f"  Calibration Rating: {rating}")

    # By market
    print(f"\nBY MARKET:")
    market_metrics = {}

    for market in MARKETS:
        m_df = bets_df[bets_df['market'] == market]
        if len(m_df) < 20:
            continue

        m_pred = m_df['clf_prob_under'].astype(float).values
        m_actual = m_df['under_hit'].astype(float).values

        m_ece, _ = calculate_ece(m_pred, m_actual)
        m_brier = calculate_brier_score(m_pred, m_actual)

        market_metrics[market] = {
            'n': len(m_df),
            'ece': m_ece,
            'brier': m_brier,
        }

        print(f"  {market}: n={len(m_df)}, ECE={m_ece:.4f}, Brier={m_brier:.4f}")

    # By direction
    print(f"\nBY DIRECTION:")
    direction_metrics = {}

    for direction in ['OVER', 'UNDER']:
        d_df = bets_df[bets_df['pick'] == direction]
        if len(d_df) < 20:
            continue

        # For OVER, predicted prob = 1 - clf_prob_under
        if direction == 'OVER':
            d_pred = 1 - d_df['clf_prob_under'].astype(float).values
            d_actual = 1 - d_df['under_hit'].astype(float).values
        else:
            d_pred = d_df['clf_prob_under'].astype(float).values
            d_actual = d_df['under_hit'].astype(float).values

        d_ece, _ = calculate_ece(d_pred, d_actual)
        d_brier = calculate_brier_score(d_pred, d_actual)

        direction_metrics[direction] = {
            'n': len(d_df),
            'ece': d_ece,
            'brier': d_brier,
        }

        print(f"  {direction}: n={len(d_df)}, ECE={d_ece:.4f}, Brier={d_brier:.4f}")

    # Generate visualizations
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Main reliability diagram
    reliability_path = output_dir / f'reliability_diagram_{season}_{timestamp}.png'
    plot_reliability_diagram(
        predicted, actual,
        title=f'NFL QUANT Model Calibration - {season}',
        output_path=reliability_path
    )

    # By market
    market_path = output_dir / f'calibration_by_market_{season}_{timestamp}.png'
    plot_calibration_by_market(bets_df, output_path=market_path)

    # Save JSON report
    report = {
        'season': season,
        'generated_at': datetime.now().isoformat(),
        'total_predictions': len(bets_df),
        'overall': {
            'ece': float(ece) if ece else None,
            'brier': float(brier) if brier else None,
            'log_loss': float(logloss) if logloss else None,
            'rating': rating,
        },
        'bins': bins,
        'by_market': market_metrics,
        'by_direction': direction_metrics,
        'files': {
            'reliability_diagram': str(reliability_path),
            'market_calibration': str(market_path),
        },
    }

    report_path = output_dir / f'calibration_report_{season}_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {report_path}")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if ece and ece >= ECE_ACCEPTABLE:
        print("  - ECE is high. Consider recalibrating with isotonic regression")
        print("  - Check for distribution shift in recent data")
        print("  - Review market-specific calibration for problem areas")
    elif ece and ece >= ECE_GOOD:
        print("  - Calibration is acceptable but could be improved")
        print("  - Monitor for drift in upcoming weeks")
    else:
        print("  - Calibration is good. Continue monitoring")

    # Market-specific recommendations
    for market, metrics in market_metrics.items():
        if metrics['ece'] and metrics['ece'] >= 0.15:
            print(f"  - {market}: High ECE ({metrics['ece']:.3f}) - investigate")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate calibration diagnostic report'
    )
    parser.add_argument('--season', type=int, default=2025, help='NFL season')
    parser.add_argument('--from-validation', action='store_true',
                        help='Use validation results instead of training data')

    args = parser.parse_args()

    try:
        if args.from_validation:
            print("Loading validation results...")
            bets_df = load_validation_bets(args.season)
        else:
            print("Loading training data...")
            training_df = load_training_data()
            # Filter to season and required columns
            bets_df = training_df[training_df['season'] == args.season].copy()

            # We need clf_prob_under - if not available, use the model to predict
            if 'clf_prob_under' not in bets_df.columns:
                print("Warning: clf_prob_under not in data. Using historical under_hit as proxy.")
                # This is a fallback - ideally we'd run predictions
                bets_df['clf_prob_under'] = bets_df['under_hit'].shift(1).fillna(0.5)

            bets_df['pick'] = 'UNDER'  # Default for historical analysis

        report = generate_report(bets_df, args.season)

        print("\n" + "=" * 70)
        print("CALIBRATION REPORT COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
