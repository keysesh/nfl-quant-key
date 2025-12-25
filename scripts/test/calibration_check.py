#!/usr/bin/env python3
"""
Calibration Check: Verify Model Calibration with safe_fillna

Measures how well predicted probabilities match actual hit rates
across different probability bins.

Run: python scripts/test/calibration_check.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging
import joblib

from configs.model_config import CLASSIFIER_MARKETS

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_model_and_data(market: str = 'player_receptions'):
    """Load V16 model and historical data with actuals."""
    # Load V16 model (multi-market structure)
    model_path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'
    if not model_path.exists():
        logger.error(f"Active model not found: {model_path}")
        return None, None, None, None

    model_data = joblib.load(model_path)

    # V16 has 'models' dict with per-market models
    if 'models' in model_data and market in model_data['models']:
        market_model = model_data['models'][market]
        model = market_model['model']
        feature_names = market_model.get('features', [])
        scaler = market_model.get('scaler')
        imputer = market_model.get('imputer')
        logger.info(f"Loaded {market} model with {len(feature_names)} features")
    else:
        logger.error(f"Market {market} not found in model. Available: {list(model_data.get('models', {}).keys())}")
        return None, None, None, None

    # Load historical data
    data_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    if not data_path.exists():
        logger.error(f"Historical data not found: {data_path}")
        return None, None, None, None

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} historical props")

    return model, feature_names, df, {'scaler': scaler, 'imputer': imputer}


def prepare_features_old_way(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Prepare features using OLD fillna(0) approach (for comparison only)."""
    from nfl_quant.features.feature_defaults import safe_fillna
    available = [f for f in feature_names if f in df.columns]
    X = df[available].copy()
    # Note: This intentionally uses fillna(0) to compare OLD vs NEW behavior
    X = X.fillna(0).astype(float)
    return X


def prepare_features_new_way(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Prepare features using NEW safe_fillna approach."""
    from nfl_quant.features.feature_defaults import safe_fillna, FEATURE_DEFAULTS

    available = [f for f in feature_names if f in df.columns]
    X = df[available].copy()
    X = safe_fillna(X, FEATURE_DEFAULTS)
    X = X.astype(float)
    return X


def calculate_calibration(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict:
    """
    Calculate calibration metrics.

    Returns:
        Dict with bin-level and overall calibration stats
    """
    # Create probability bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    results = {
        'bins': [],
        'predicted_means': [],
        'actual_means': [],
        'counts': [],
        'calibration_errors': [],
    }

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            pred_mean = y_pred[mask].mean()
            actual_mean = y_true[mask].mean()
            count = mask.sum()
            error = abs(pred_mean - actual_mean)

            results['bins'].append(f"{bins[i]:.1f}-{bins[i+1]:.1f}")
            results['predicted_means'].append(pred_mean)
            results['actual_means'].append(actual_mean)
            results['counts'].append(count)
            results['calibration_errors'].append(error)

    # Overall metrics
    results['ece'] = np.average(
        results['calibration_errors'],
        weights=results['counts']
    ) if results['counts'] else 0.0

    results['max_calibration_error'] = max(results['calibration_errors']) if results['calibration_errors'] else 0.0
    results['brier_score'] = np.mean((y_pred - y_true) ** 2)

    return results


def run_calibration_check(market: str = 'player_receptions'):
    """Run full calibration analysis for a specific market."""
    print("\n" + "="*70)
    print(f"CALIBRATION CHECK: V16 Model ({market}) with safe_fillna")
    print("="*70)

    # Load data
    model, feature_names, df, preprocessing = load_model_and_data(market)
    if model is None:
        return

    # Filter to 2024-2025 data
    df = df[df['season'] >= 2024].copy()
    logger.info(f"Filtered to {len(df)} props from 2024-2025")

    # Filter to specific market
    if 'market' in df.columns:
        df = df[df['market'] == market].copy()
        logger.info(f"Filtered to {len(df)} props for {market}")

    # Filter to rows with actual outcomes
    df = df[df['under_hit'].notna()].copy()
    df['under_hit'] = df['under_hit'].astype(int)
    logger.info(f"Filtered to {len(df)} props with known outcomes")

    if len(df) == 0:
        logger.error("No data with outcomes found")
        return

    # Sample for speed
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
        logger.info(f"Sampled to 3000 props")

    # Prepare features both ways
    X_old = prepare_features_old_way(df, feature_names)
    X_new = prepare_features_new_way(df, feature_names)
    y_true = df['under_hit'].values

    # Handle missing features
    missing_old = set(feature_names) - set(X_old.columns)
    missing_new = set(feature_names) - set(X_new.columns)

    if missing_old:
        logger.warning(f"Missing {len(missing_old)} features from data")
        # Add missing with defaults
        for f in missing_old:
            X_old[f] = 0
        for f in missing_new:
            from nfl_quant.features.feature_defaults import FEATURE_DEFAULTS
            X_new[f] = FEATURE_DEFAULTS.get(f, 0)

    # Reorder to match model
    X_old = X_old[feature_names]
    X_new = X_new[feature_names]

    # Apply preprocessing (imputer, scaler)
    scaler = preprocessing.get('scaler')
    imputer = preprocessing.get('imputer')

    X_old_processed = X_old.copy()
    X_new_processed = X_new.copy()

    if imputer is not None:
        try:
            X_old_processed = pd.DataFrame(imputer.transform(X_old_processed), columns=X_old.columns, index=X_old.index)
            X_new_processed = pd.DataFrame(imputer.transform(X_new_processed), columns=X_new.columns, index=X_new.index)
        except Exception as e:
            logger.warning(f"Imputer failed: {e}")

    if scaler is not None:
        try:
            X_old_processed = pd.DataFrame(scaler.transform(X_old_processed), columns=X_old.columns, index=X_old.index)
            X_new_processed = pd.DataFrame(scaler.transform(X_new_processed), columns=X_new.columns, index=X_new.index)
        except Exception as e:
            logger.warning(f"Scaler failed: {e}")

    # Get predictions
    try:
        # Check if model is sklearn or lightgbm
        if hasattr(model, 'predict_proba'):
            probs_old = model.predict_proba(X_old_processed)[:, 1]
            probs_new = model.predict_proba(X_new_processed)[:, 1]
        else:
            probs_old = model.predict(X_old_processed)
            probs_new = model.predict(X_new_processed)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    # Calculate calibration for both
    cal_old = calculate_calibration(y_true, probs_old)
    cal_new = calculate_calibration(y_true, probs_new)

    # Print results
    print("\n" + "-"*50)
    print("CALIBRATION BY PROBABILITY BIN")
    print("-"*50)

    print("\nOLD fillna(0):")
    print(f"{'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Error':>10} {'N':>8}")
    print("-" * 50)
    for i in range(len(cal_old['bins'])):
        print(f"{cal_old['bins'][i]:<12} "
              f"{cal_old['predicted_means'][i]:>10.3f} "
              f"{cal_old['actual_means'][i]:>10.3f} "
              f"{cal_old['calibration_errors'][i]:>10.3f} "
              f"{cal_old['counts'][i]:>8}")

    print("\nNEW safe_fillna():")
    print(f"{'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Error':>10} {'N':>8}")
    print("-" * 50)
    for i in range(len(cal_new['bins'])):
        print(f"{cal_new['bins'][i]:<12} "
              f"{cal_new['predicted_means'][i]:>10.3f} "
              f"{cal_new['actual_means'][i]:>10.3f} "
              f"{cal_new['calibration_errors'][i]:>10.3f} "
              f"{cal_new['counts'][i]:>8}")

    # Summary metrics
    print("\n" + "-"*50)
    print("CALIBRATION SUMMARY METRICS")
    print("-"*50)

    print(f"\n{'Metric':<30} {'OLD':>12} {'NEW':>12} {'Change':>12}")
    print("-" * 66)
    print(f"{'Expected Calibration Error':<30} {cal_old['ece']:>12.4f} {cal_new['ece']:>12.4f} {cal_new['ece'] - cal_old['ece']:>+12.4f}")
    print(f"{'Max Calibration Error':<30} {cal_old['max_calibration_error']:>12.4f} {cal_new['max_calibration_error']:>12.4f} {cal_new['max_calibration_error'] - cal_old['max_calibration_error']:>+12.4f}")
    print(f"{'Brier Score':<30} {cal_old['brier_score']:>12.4f} {cal_new['brier_score']:>12.4f} {cal_new['brier_score'] - cal_old['brier_score']:>+12.4f}")

    # Calculate accuracy at different thresholds
    print("\n" + "-"*50)
    print("ACCURACY AT DIFFERENT THRESHOLDS")
    print("-"*50)

    thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
    print(f"\n{'Threshold':<12} {'OLD Acc':>10} {'OLD N':>8} {'NEW Acc':>10} {'NEW N':>8} {'Acc Diff':>10}")
    print("-" * 68)

    for thresh in thresholds:
        # OLD
        old_mask = (probs_old > thresh) | (probs_old < (1 - thresh))
        old_preds = (probs_old[old_mask] > 0.5).astype(int)
        old_acc = (old_preds == y_true[old_mask]).mean() if old_mask.sum() > 0 else 0
        old_n = old_mask.sum()

        # NEW
        new_mask = (probs_new > thresh) | (probs_new < (1 - thresh))
        new_preds = (probs_new[new_mask] > 0.5).astype(int)
        new_acc = (new_preds == y_true[new_mask]).mean() if new_mask.sum() > 0 else 0
        new_n = new_mask.sum()

        print(f"{thresh:<12.2f} {old_acc*100:>10.2f}% {old_n:>8} {new_acc*100:>10.2f}% {new_n:>8} {(new_acc-old_acc)*100:>+10.2f}%")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    ece_diff = cal_new['ece'] - cal_old['ece']
    brier_diff = cal_new['brier_score'] - cal_old['brier_score']

    if ece_diff < 0:
        print(f"\n[IMPROVED] ECE reduced by {abs(ece_diff):.4f}")
        print("  -> Predicted probabilities better match actual hit rates")
    else:
        print(f"\n[SIMILAR] ECE increased by {ece_diff:.4f}")
        print("  -> Minimal impact on calibration (expected for small changes)")

    if brier_diff < 0:
        print(f"\n[IMPROVED] Brier Score reduced by {abs(brier_diff):.4f}")
        print("  -> Better overall probability estimates")
    else:
        print(f"\n[SIMILAR] Brier Score increased by {brier_diff:.4f}")
        print("  -> May need recalibration with new defaults")

    print("""
KEY TAKEAWAY:
- If ECE/Brier improved: safe_fillna produces more accurate probabilities
- If ECE/Brier similar: safe_fillna doesn't hurt, still removes OVER bias
- If ECE/Brier worse: Consider recalibrating model with new defaults
""")

    return cal_old, cal_new


if __name__ == '__main__':
    import sys

    # Run for key markets (from central config)
    markets = CLASSIFIER_MARKETS

    if len(sys.argv) > 1:
        # Run for specific market
        run_calibration_check(sys.argv[1])
    else:
        # Run summary for all markets
        print("\n" + "="*70)
        print("CALIBRATION CHECK: ALL MARKETS SUMMARY")
        print("="*70)

        for market in markets:
            try:
                run_calibration_check(market)
            except Exception as e:
                print(f"\n[ERROR] {market}: {e}")
