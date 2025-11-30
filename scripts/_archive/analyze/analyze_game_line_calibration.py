#!/usr/bin/env python3
"""
Analyze Game Line Calibration Quality
=====================================

Analyzes historical game line predictions to determine if calibration is needed.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def analyze_calibration(df: pd.DataFrame) -> dict:
    """Analyze calibration quality of game line predictions."""

    results = {
        'moneyline_analysis': {},
        'spread_analysis': {},
        'total_analysis': {},
        'overall_metrics': {}
    }

    # Analyze moneyline calibration
    if 'home_win_prob' in df.columns and 'home_actual_win' in df.columns:
        # Bin probabilities
        bins = np.arange(0, 1.05, 0.05)
        df['prob_bin'] = pd.cut(df['home_win_prob'], bins=bins)

        calibration_data = []
        for bin_group in df.groupby('prob_bin'):
            bin_name, group = bin_group
            if len(group) > 0:
                predicted = group['home_win_prob'].mean()
                actual = group['home_actual_win'].mean()
                count = len(group)
                calibration_data.append({
                    'predicted': predicted,
                    'actual': actual,
                    'error': actual - predicted,
                    'count': count
                })

        calibration_df = pd.DataFrame(calibration_data)

        # Calculate metrics
        mace = np.mean(np.abs(calibration_df['error']))
        ece = np.sum(calibration_df['count'] * np.abs(calibration_df['error'])) / calibration_df['count'].sum()

        results['moneyline_analysis'] = {
            'mace': mace,
            'ece': ece,
            'calibration_data': calibration_df
        }

    return results

def main():
    data_path = Path('reports/game_line_calibration_data.csv')

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("   Run: python scripts/data/collect_game_line_calibration_data.py")
        return

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} game predictions")
    print()

    # Analyze calibration
    results = analyze_calibration(df)

    # Print results
    print("=" * 80)
    print("CALIBRATION ANALYSIS RESULTS")
    print("=" * 80)
    print()

    if results['moneyline_analysis']:
        ml = results['moneyline_analysis']
        print("Moneyline Calibration:")
        print(f"  MACE (Mean Absolute Calibration Error): {ml['mace']:.4f}")
        print(f"  ECE (Expected Calibration Error): {ml['ece']:.4f}")
        print()

        if ml['mace'] > 0.10:
            print("⚠️  CALIBRATION NEEDED: MACE > 0.10 indicates overconfidence")
        elif ml['mace'] > 0.05:
            print("⚠️  CALIBRATION RECOMMENDED: MACE > 0.05 suggests some overconfidence")
        else:
            print("✅ Calibration quality is good (MACE < 0.05)")

    print()
    print("Recommendation:")
    if results['moneyline_analysis'].get('mace', 1.0) > 0.05:
        print("  → Train game line calibrator")
        print("  → Run: python scripts/train/train_game_line_calibrator.py")
    else:
        print("  → Game line probabilities are well-calibrated")
        print("  → No calibration needed")

if __name__ == '__main__':
    main()





























