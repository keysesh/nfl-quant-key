#!/usr/bin/env python3
"""
Train Production Calibrators from Backtest Results

This script:
1. Loads backtest predictions with actual outcomes
2. Trains isotonic regression calibrators for overall, position, and market subsets
3. Saves calibration models for production use
4. Generates a comprehensive calibration report

Run after backtest_v3_system.py to create production-ready calibrators.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest_results"
CALIBRATION_DIR = PROJECT_ROOT / "models" / "calibration"


def load_latest_backtest() -> pd.DataFrame:
    """Load the most recent backtest predictions."""
    csv_files = list(BACKTEST_DIR.glob("backtest_predictions_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No backtest predictions found. Run backtest_v3_system.py first.")

    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading backtest predictions from: {latest_file}")
    return pd.read_csv(latest_file)


def train_calibrator(raw_probs: np.ndarray, actuals: np.ndarray, name: str) -> NFLProbabilityCalibrator:
    """Train an isotonic calibrator on a subset of data."""
    print(f"\nTraining {name} calibrator...")
    print(f"  Samples: {len(raw_probs)}")

    calibrator = NFLProbabilityCalibrator(
        high_prob_threshold=0.70,
        high_prob_shrinkage=0.3
    )
    calibrator.fit(raw_probs, actuals)

    return calibrator


def analyze_calibration_quality(
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
    actuals: np.ndarray
) -> Dict:
    """Analyze how well calibration improves predictions."""
    from sklearn.metrics import brier_score_loss

    # Brier scores
    brier_raw = brier_score_loss(actuals, raw_probs)
    brier_cal = brier_score_loss(actuals, calibrated_probs)

    # Expected Calibration Error
    def calc_ece(probs, actuals, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_pred = probs[mask].mean()
                bin_actual = actuals[mask].mean()
                ece += (mask.sum() / len(probs)) * abs(bin_pred - bin_actual)
        return ece

    ece_raw = calc_ece(raw_probs, actuals)
    ece_cal = calc_ece(calibrated_probs, actuals)

    # ROI simulation
    def calc_roi(probs, actuals):
        # Bet on over if prob > 0.5, else under
        bet_over = probs > 0.5
        hit = np.where(bet_over, actuals, 1 - actuals)
        roi = np.sum(hit * (100 / 110) - (1 - hit)) / len(probs)
        return roi * 100

    roi_raw = calc_roi(raw_probs, actuals)
    roi_cal = calc_roi(calibrated_probs, actuals)

    return {
        'brier_raw': float(brier_raw),
        'brier_calibrated': float(brier_cal),
        'brier_improvement': float(brier_raw - brier_cal),
        'ece_raw': float(ece_raw),
        'ece_calibrated': float(ece_cal),
        'ece_improvement': float(ece_raw - ece_cal),
        'roi_raw': float(roi_raw),
        'roi_calibrated': float(roi_cal),
        'roi_improvement': float(roi_cal - roi_raw),
    }


def main():
    """Train all calibrators from backtest results."""
    print("=" * 70)
    print("TRAINING PRODUCTION CALIBRATORS FROM BACKTEST")
    print("=" * 70)

    # Load backtest data
    df = load_latest_backtest()
    print(f"Loaded {len(df)} predictions")

    # Extract arrays
    raw_probs = df['predicted_prob_over'].values
    actuals = df['hit_over'].astype(float).values

    # Ensure output directory exists
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Train OVERALL calibrator
    overall_cal = train_calibrator(raw_probs, actuals, "Overall")
    overall_cal.save(str(CALIBRATION_DIR / "overall_calibrator.json"))

    overall_calibrated = overall_cal.transform(raw_probs)
    overall_metrics = analyze_calibration_quality(raw_probs, overall_calibrated, actuals)

    # 2. Train POSITION calibrators
    position_calibrators = {}
    position_metrics = {}

    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_mask = df['position'] == position
        if pos_mask.sum() >= 100:  # Need enough data
            pos_raw = raw_probs[pos_mask]
            pos_actuals = actuals[pos_mask]

            pos_cal = train_calibrator(pos_raw, pos_actuals, f"Position {position}")
            pos_cal.save(str(CALIBRATION_DIR / f"position_{position}_calibrator.json"))
            position_calibrators[position] = pos_cal

            pos_calibrated = pos_cal.transform(pos_raw)
            position_metrics[position] = analyze_calibration_quality(pos_raw, pos_calibrated, pos_actuals)

    # 3. Train MARKET calibrators
    market_calibrators = {}
    market_metrics = {}

    for market in df['market'].unique():
        market_mask = df['market'] == market
        if market_mask.sum() >= 100:
            market_raw = raw_probs[market_mask]
            market_actuals = actuals[market_mask]

            market_cal = train_calibrator(market_raw, market_actuals, f"Market {market}")
            market_cal.save(str(CALIBRATION_DIR / f"market_{market}_calibrator.json"))
            market_calibrators[market] = market_cal

            market_calibrated = market_cal.transform(market_raw)
            market_metrics[market] = analyze_calibration_quality(market_raw, market_calibrated, market_actuals)

    # 4. Generate comprehensive report
    report = {
        'overall': overall_metrics,
        'by_position': position_metrics,
        'by_market': market_metrics,
    }

    report_path = CALIBRATION_DIR / "calibration_training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # 5. Print summary
    print("\n" + "=" * 70)
    print("CALIBRATION TRAINING RESULTS")
    print("=" * 70)

    print("\n--- OVERALL CALIBRATION ---")
    print(f"Brier Score: {overall_metrics['brier_raw']:.4f} -> {overall_metrics['brier_calibrated']:.4f} "
          f"(Improvement: {overall_metrics['brier_improvement']:+.4f})")
    print(f"ECE: {overall_metrics['ece_raw']:.4f} -> {overall_metrics['ece_calibrated']:.4f} "
          f"(Improvement: {overall_metrics['ece_improvement']:+.4f})")
    print(f"ROI: {overall_metrics['roi_raw']:.2f}% -> {overall_metrics['roi_calibrated']:.2f}% "
          f"(Change: {overall_metrics['roi_improvement']:+.2f}%)")

    print("\n--- BY POSITION ---")
    for pos, metrics in position_metrics.items():
        print(f"\n{pos}:")
        print(f"  Brier: {metrics['brier_raw']:.4f} -> {metrics['brier_calibrated']:.4f}")
        print(f"  ECE: {metrics['ece_raw']:.4f} -> {metrics['ece_calibrated']:.4f}")
        print(f"  ROI: {metrics['roi_raw']:.2f}% -> {metrics['roi_calibrated']:.2f}%")

    print("\n--- BY MARKET ---")
    for market, metrics in market_metrics.items():
        print(f"\n{market}:")
        print(f"  Brier: {metrics['brier_raw']:.4f} -> {metrics['brier_calibrated']:.4f}")
        print(f"  ROI: {metrics['roi_raw']:.2f}% -> {metrics['roi_calibrated']:.2f}%")

    # 6. Show calibration curve examples
    print("\n--- CALIBRATION CURVE (Overall) ---")
    print("Raw Prob -> Calibrated Prob")
    for raw_p in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        cal_p = overall_cal.transform(np.array([raw_p]))[0]
        print(f"  {raw_p:.2f} -> {cal_p:.3f} (bias correction: {raw_p - cal_p:+.3f})")

    print(f"\nCalibrators saved to: {CALIBRATION_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
