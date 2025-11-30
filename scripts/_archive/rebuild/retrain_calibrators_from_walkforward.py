#!/usr/bin/env python3
"""
Retrain Calibrators from Walk-Forward Calibration Data

Uses the REAL walk-forward training data generated from actual Monte Carlo simulations
to train isotonic regression calibrators for each market.

This ensures:
1. No data leakage (walk-forward validation)
2. Real model predictions (not synthetic)
3. Actual game outcomes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator


def main():
    print("="*80)
    print("RETRAINING CALIBRATORS FROM WALK-FORWARD DATA")
    print("="*80)

    # Load walk-forward calibration data
    training_file = PROJECT_ROOT / "data" / "training" / "full_walk_forward_calibration_data.parquet"
    if not training_file.exists():
        print(f"ERROR: Training data not found: {training_file}")
        print("Run scripts/rebuild/full_walk_forward_calibration.py first")
        return

    df = pd.read_parquet(training_file)
    print(f"Loaded {len(df):,} training records from walk-forward simulation")
    print(f"Weeks: {sorted(df['week'].unique().tolist())}")
    print()

    configs_dir = PROJECT_ROOT / "configs"
    markets = df['market'].unique()

    for market in markets:
        print(f"\n{'='*60}")
        print(f"TRAINING CALIBRATOR: {market}")
        print(f"{'='*60}")

        market_data = df[df['market'] == market].copy()

        if len(market_data) < 50:
            print(f"  WARNING: Only {len(market_data)} samples for {market}, skipping")
            continue

        # Extract probabilities and outcomes
        probs = market_data['model_prob_raw'].values
        outcomes = market_data['bet_won'].values.astype(int)

        print(f"  Training samples: {len(probs):,}")
        print(f"  Probability range: {probs.min():.3f} - {probs.max():.3f}")
        print(f"  Actual win rate: {outcomes.mean():.1%}")
        print(f"  Raw model avg prob: {probs.mean():.1%}")
        print(f"  Raw calibration error: {abs(outcomes.mean() - probs.mean()):.1%}")

        # Train isotonic calibrator
        calibrator = NFLProbabilityCalibrator()
        calibrator.fit(probs, outcomes)

        # Evaluate calibration improvement
        calibrated_probs = calibrator.transform(probs)

        # Calculate MACE (Mean Absolute Calibration Error)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mace_raw = 0.0
        mace_calibrated = 0.0

        for i in range(n_bins):
            bin_mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
            if bin_mask.sum() > 0:
                bin_actual = outcomes[bin_mask].mean()
                bin_raw_pred = probs[bin_mask].mean()
                bin_cal_pred = calibrated_probs[bin_mask].mean()
                mace_raw += abs(bin_actual - bin_raw_pred) * bin_mask.sum() / len(probs)
                mace_calibrated += abs(bin_actual - bin_cal_pred) * bin_mask.sum() / len(probs)

        print(f"  MACE (raw): {mace_raw:.4f}")
        print(f"  MACE (calibrated): {mace_calibrated:.4f}")
        print(f"  Improvement: {((mace_raw - mace_calibrated) / mace_raw * 100):.1f}%")

        # Save calibrator
        output_file = configs_dir / f"calibrator_{market}.json"
        calibrator.save(str(output_file))

        # Save metadata
        metadata = {
            'market': market,
            'training_samples': int(len(probs)),
            'training_weeks': sorted([int(w) for w in market_data['week'].unique()]),
            'training_win_rate': float(outcomes.mean()),
            'raw_model_avg_prob': float(probs.mean()),
            'raw_calibration_error': float(abs(outcomes.mean() - probs.mean())),
            'mace_raw': float(mace_raw),
            'mace_calibrated': float(mace_calibrated),
            'improvement_percent': float((mace_raw - mace_calibrated) / mace_raw * 100),
            'trained_date': datetime.now().isoformat(),
            'training_source': 'full_walk_forward_calibration_data.parquet'
        }

        metadata_file = configs_dir / f"calibrator_{market}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved: {output_file.name}")
        print(f"  Metadata: {metadata_file.name}")

    print("\n" + "="*80)
    print("CALIBRATOR RETRAINING COMPLETE")
    print("="*80)
    print(f"All calibrators have been retrained from walk-forward simulation data")
    print(f"These calibrators use REAL model predictions, no data leakage")


if __name__ == "__main__":
    main()
