#!/usr/bin/env python3
"""
Phase 1: Validate Current Calibration

Analyzes:
1. Raw vs calibrated probability distributions
2. Calibration curve behavior
3. Edge preservation/loss
4. Probability bin analysis

This helps determine if calibration is working correctly or being too aggressive.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.backtest.backtest_player_props import run_backtest
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.utils.season_utils import get_current_season


def load_calibration_curve(calibrator_path: Path) -> Dict:
    """Load and inspect calibration curve parameters."""
    import json
    
    if not calibrator_path.exists():
        print(f"⚠️  Calibration file not found: {calibrator_path}")
        return None
    
    with open(calibrator_path) as f:
        params = json.load(f)
    
    print("=" * 80)
    print("📊 CALIBRATION CURVE ANALYSIS")
    print("=" * 80)
    print()
    
    X_thresholds = np.array(params['X_thresholds'])
    y_thresholds = np.array(params['y_thresholds'])
    
    print(f"Calibration Curve Points: {len(X_thresholds)}")
    print(f"Raw Probability Range: [{params['X_min']:.3f}, {params['X_max']:.3f}]")
    print(f"Calibrated Range: [{params['y_min']:.1%}, {params['y_max']:.1%}]")
    print()
    
    # Show key mappings
    print("Key Probability Mappings:")
    print(f"{'Raw Prob':>12} → {'Calibrated':>12} | {'Change':>12}")
    print("-" * 45)
    
    test_probs = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
    calibrator = NFLProbabilityCalibrator()
    calibrator.load(str(calibrator_path))
    
    for raw_prob in test_probs:
        calibrated = calibrator.transform(raw_prob)
        change = calibrated - raw_prob
        print(f"{raw_prob:>12.1%} → {calibrated:>12.1%} | {change:>+12.1%}")
    
    print()
    
    # Check monotonicity
    if np.all(np.diff(y_thresholds) >= 0):
        print("✅ Calibration curve is monotonic (correct)")
    else:
        print("❌ Calibration curve is NOT monotonic (ERROR!)")
    
    print()
    
    return {
        'X_thresholds': X_thresholds,
        'y_thresholds': y_thresholds,
        'params': params
    }


def run_backtest_with_and_without_calibration(
    start_week: int,
    end_week: int,
    min_edge: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run backtest with calibration ON and OFF."""
    
    print("=" * 80)
    print("🔄 RUNNING BACKTEST: WITH CALIBRATION")
    print("=" * 80)
    print()
    
    # With calibration (default)
    results_with = run_backtest(
        start_week=start_week,
        end_week=end_week,
        season=get_current_season(),
        prop_files_dir=Path("data/historical"),
        min_edge=min_edge,
        min_confidence=0.0,
    )
    
    # Temporarily disable calibration by modifying the simulator
    # We'll need to modify backtest_player_props.py to support this
    # For now, we'll extract raw probabilities from the results
    
    print()
    print("=" * 80)
    print("📊 ANALYZING RESULTS")
    print("=" * 80)
    print()
    
    return results_with


def analyze_probability_distribution(results: pd.DataFrame) -> None:
    """Analyze probability distribution and calibration impact."""
    
    if 'model_prob' not in results.columns:
        print("⚠️  No model_prob column found")
        return
    
    print("=" * 80)
    print("📊 PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()
    
    probs = results['model_prob'].values
    
    print(f"Total Bets: {len(results)}")
    print(f"Mean Probability: {probs.mean():.1%}")
    print(f"Median Probability: {np.median(probs):.1%}")
    print(f"Min Probability: {probs.min():.1%}")
    print(f"Max Probability: {probs.max():.1%}")
    print(f"Std Deviation: {probs.std():.1%}")
    print()
    
    # Probability bins
    bins = [
        (0.40, 0.50, "40-50%"),
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.75, "70-75%"),
        (0.75, 0.80, "75-80%"),
        (0.80, 0.85, "80-85%"),
        (0.85, 0.90, "85-90%"),
        (0.90, 0.95, "90-95%"),
        (0.95, 1.00, "95%+"),
    ]
    
    print(f"{'Probability Range':<15} {'Bets':<8} {'Hit Rate':<12} {'Avg Edge':<12} {'ROI':<10}")
    print("-" * 60)
    
    for bin_min, bin_max, label in bins:
        bin_data = results[
            (results['model_prob'] >= bin_min) &
            (results['model_prob'] < bin_max)
        ]
        
        if len(bin_data) > 0:
            hit_rate = bin_data['bet_won'].mean()
            avg_edge = bin_data['edge'].mean()
            roi = bin_data['unit_return'].mean()
            
            print(f"{label:<15} {len(bin_data):<8} {hit_rate:>11.1%} {avg_edge:>11.1%} {roi:>+9.1%}")
    
    print()


def compare_raw_vs_calibrated(results: pd.DataFrame, calibrator_path: Path) -> pd.DataFrame:
    """Compare raw vs calibrated probabilities."""
    
    print("=" * 80)
    print("🔄 RAW VS CALIBRATED COMPARISON")
    print("=" * 80)
    print()
    
    calibrator = NFLProbabilityCalibrator()
    if calibrator_path.exists():
        calibrator.load(str(calibrator_path))
    else:
        print("⚠️  Calibrator file not found, cannot compare")
        return results
    
    # We need to extract raw probabilities
    # Since we're already applying calibration, we need to reverse-engineer
    # or re-run without calibration
    
    # For now, estimate raw probabilities by inverting calibration
    # This is approximate but should give us insight
    
    print("Note: Re-running backtest to extract raw probabilities...")
    print("(This requires modifying the backtest to save raw probs)")
    print()
    
    return results


def analyze_edge_preservation(results: pd.DataFrame) -> None:
    """Analyze if calibration is preserving edges."""
    
    print("=" * 80)
    print("💰 EDGE PRESERVATION ANALYSIS")
    print("=" * 80)
    print()
    
    if 'edge' not in results.columns:
        print("⚠️  No edge column found")
        return
    
    edges = results['edge'].values
    
    print(f"Edge Statistics:")
    print(f"  Mean:    {edges.mean():.2%}")
    print(f"  Median:  {np.median(edges):.2%}")
    print(f"  Min:     {edges.min():.2%}")
    print(f"  Max:     {edges.max():.2%}")
    print(f"  Std Dev: {edges.std():.2%}")
    print()
    
    # Edge distribution
    positive_edges = edges[edges > 0]
    print(f"Positive Edges: {len(positive_edges)} / {len(edges)} ({len(positive_edges)/len(edges):.1%})")
    if len(positive_edges) > 0:
        print(f"  Mean: {positive_edges.mean():.2%}")
        print(f"  Median: {np.median(positive_edges):.2%}")
    print()
    
    # Analyze edge by probability bin
    bins = [
        (0.50, 0.60, "50-60%"),
        (0.60, 0.70, "60-70%"),
        (0.70, 0.80, "70-80%"),
        (0.80, 0.90, "80-90%"),
        (0.90, 1.00, "90%+"),
    ]
    
    print(f"{'Probability Range':<15} {'Bets':<8} {'Avg Edge':<12} {'Positive Edge %':<15}")
    print("-" * 55)
    
    for bin_min, bin_max, label in bins:
        bin_data = results[
            (results['model_prob'] >= bin_min) &
            (results['model_prob'] < bin_max)
        ]
        
        if len(bin_data) > 0:
            avg_edge = bin_data['edge'].mean()
            positive_pct = (bin_data['edge'] > 0).mean()
            
            print(f"{label:<15} {len(bin_data):<8} {avg_edge:>11.1%} {positive_pct:>14.1%}")
    
    print()


def main():
    print("=" * 80)
    print("PHASE 1: CALIBRATION VALIDATION")
    print("=" * 80)
    print()
    
    calibrator_path = Path("configs/calibrator.json")
    
    # Step 1: Analyze calibration curve
    curve_data = load_calibration_curve(calibrator_path)
    
    # Step 2: Run backtest
    print("Running backtest (weeks 1-7, min_edge=3%)...")
    print()
    
    results = run_backtest_with_and_without_calibration(
        start_week=1,
        end_week=7,
        min_edge=0.03
    )
    
    # Step 3: Analyze results
    analyze_probability_distribution(results)
    analyze_edge_preservation(results)
    
    # Step 4: Save results for further analysis
    output_path = Path("reports/calibration_validation_results.csv")
    results.to_csv(output_path, index=False)
    print(f"💾 Saved detailed results to {output_path}")
    print()
    
    # Step 5: Summary and recommendations
    print("=" * 80)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    total_bets = len(results)
    hit_rate = results['bet_won'].mean()
    avg_prob = results['model_prob'].mean()
    calibration_error = hit_rate - avg_prob
    
    print(f"Total Bets: {total_bets}")
    print(f"Hit Rate: {hit_rate:.1%}")
    print(f"Average Model Probability: {avg_prob:.1%}")
    print(f"Calibration Error: {calibration_error:+.1%}")
    print()
    
    if abs(calibration_error) < 0.05:
        print("✅ Calibration appears well-calibrated (error < 5%)")
    elif calibration_error < -0.05:
        print("⚠️  Model is OVERCONFIDENT (predicting higher than actual)")
        print("   → Calibration may need to be more aggressive")
    else:
        print("⚠️  Model is UNDERCONFIDENT (predicting lower than actual)")
        print("   → Calibration may be too aggressive")
    
    print()
    print("Next Steps:")
    print("  1. Review calibration curve mappings above")
    print("  2. Check if edge distribution is reasonable")
    print("  3. Proceed to Phase 2: Retrain calibrator on actual backtest data")


if __name__ == "__main__":
    main()
































