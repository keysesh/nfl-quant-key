#!/usr/bin/env python3
"""
Analyze the current calibrator's performance and provide recommendations.

This script:
1. Loads the existing calibrator
2. Analyzes its calibration curve
3. Provides insights on how it's adjusting probabilities
4. Identifies potential issues
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator


def analyze_calibrator():
    """Analyze the existing calibrator."""
    
    print("=" * 80)
    print("🔍 CALIBRATOR ANALYSIS")
    print("=" * 80)
    print()
    
    # Load calibrator
    calibrator_path = 'configs/calibrator.json'
    
    if not Path(calibrator_path).exists():
        print(f"❌ Calibrator not found at: {calibrator_path}")
        print("   Run train_calibrator_from_backtest.py first")
        return
    
    calibrator = NFLProbabilityCalibrator()
    calibrator.load(calibrator_path)
    
    print(f"✅ Loaded calibrator from: {calibrator_path}")
    print()
    
    # Analyze calibration curve
    print("📊 CALIBRATION CURVE ANALYSIS")
    print("-" * 80)
    
    # Get calibration parameters
    x_thresh = calibrator.calibrator.X_thresholds_
    y_thresh = calibrator.calibrator.y_thresholds_
    
    print(f"Number of calibration points: {len(x_thresh)}")
    print(f"Raw probability range: {x_thresh.min():.3f} to {x_thresh.max():.3f}")
    print(f"Calibrated probability range: {y_thresh.min():.3f} to {y_thresh.max():.3f}")
    print()
    
    # Test calibration at key probability points
    print("📋 CALIBRATION EXAMPLES")
    print("-" * 80)
    print(f"{'Raw P':>10s} | {'Calibrated P':>15s} | {'Adjustment':>12s} | {'Direction':>10s}")
    print(f"{'-'*10} | {'-'*15} | {'-'*12} | {'-'*10}")
    
    test_probs = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    for raw_p in test_probs:
        cal_p = calibrator.transform(raw_p)
        adjustment = cal_p - raw_p
        direction = "INCREASE" if adjustment > 0.01 else "DECREASE" if adjustment < -0.01 else "NEUTRAL"
        
        print(f"{raw_p:10.2f} | {cal_p:15.3f} | {adjustment:+12.3f} | {direction:>10s}")
    
    print()
    
    # Identify key adjustments
    print("💡 KEY INSIGHTS")
    print("-" * 80)
    
    # Calculate average adjustment in different probability ranges
    low_range = [0.30, 0.35, 0.40]
    mid_range = [0.45, 0.50, 0.55]
    high_range = [0.60, 0.65, 0.70]
    
    low_adj = np.mean([calibrator.transform(p) - p for p in low_range])
    mid_adj = np.mean([calibrator.transform(p) - p for p in mid_range])
    high_adj = np.mean([calibrator.transform(p) - p for p in high_range])
    
    print(f"Average adjustment by range:")
    print(f"  Low (30-40%):  {low_adj:+.3f} ({'increases' if low_adj > 0 else 'decreases'} confidence)")
    print(f"  Mid (45-55%):  {mid_adj:+.3f} ({'increases' if mid_adj > 0 else 'decreases'} confidence)")
    print(f"  High (60-70%): {high_adj:+.3f} ({'increases' if high_adj > 0 else 'decreases'} confidence)")
    print()
    
    # Check if calibrator is conservative or aggressive
    overall_bias = np.mean([calibrator.transform(p) - p for p in np.linspace(0.3, 0.7, 20)])
    
    if abs(overall_bias) < 0.01:
        print("✅ Calibrator is NEUTRAL - minimal systematic bias")
    elif overall_bias > 0.02:
        print("⚠️  Calibrator is AGGRESSIVE - increases probabilities on average")
        print("   This suggests the raw model is underconfident")
    else:
        print("⚠️  Calibrator is CONSERVATIVE - decreases probabilities on average")
        print("   This suggests the raw model is overconfident")
    
    print()
    
    # Create visualization
    print("📊 Creating calibration visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Calibration curve
    raw_range = np.linspace(0.2, 0.8, 100)
    calibrated_range = [calibrator.transform(p) for p in raw_range]
    
    ax1.plot([0, 1], [0, 1], 'k--', label='No Calibration', linewidth=2, alpha=0.5)
    ax1.plot(raw_range, calibrated_range, 'b-', label='Calibration Function', linewidth=2)
    ax1.scatter(x_thresh, y_thresh, c='red', s=30, alpha=0.6, label='Training Points', zorder=10)
    ax1.set_xlabel('Raw Probability', fontsize=12)
    ax1.set_ylabel('Calibrated Probability', fontsize=12)
    ax1.set_title('Calibration Function', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.2, 0.8])
    ax1.set_ylim([0.2, 0.8])
    
    # Plot 2: Adjustment magnitude
    adjustments = [calibrator.transform(p) - p for p in raw_range]
    
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax2.plot(raw_range, adjustments, 'b-', linewidth=2)
    ax2.fill_between(raw_range, 0, adjustments, alpha=0.3)
    ax2.set_xlabel('Raw Probability', fontsize=12)
    ax2.set_ylabel('Calibration Adjustment', fontsize=12)
    ax2.set_title('Adjustment Magnitude', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.2, 0.8])
    
    plt.tight_layout()
    plot_path = 'reports/calibrator_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_path}")
    print()
    
    # Recommendations
    print("=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    print("1. CURRENT STATUS:")
    print("   ✅ Calibrator is trained and functional")
    print(f"   ✅ Covers probability range: {x_thresh.min():.1%} to {x_thresh.max():.1%}")
    print()
    
    print("2. FOR PRODUCTION USE:")
    print("   ✅ Load calibrator in PlayerSimulator initialization")
    print("   ✅ Apply to all probability estimates before betting")
    print("   ✅ Use calibrated probabilities for Kelly Criterion sizing")
    print()
    
    print("3. TO IMPROVE CALIBRATOR:")
    print("   → Collect MORE real predictions vs outcomes")
    print("   → Retrain quarterly or after every 500+ new settled bets")
    print("   → Monitor calibration drift over time")
    print("   → Consider separate calibrators by market type")
    print()
    
    print("4. VALIDATION:")
    print("   → Run backtest with calibrated vs uncalibrated probabilities")
    print("   → Compare ROI, win rate, and Sharpe ratio")
    print("   → Expect better risk-adjusted returns with calibration")
    print()
    
    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_calibrator()

