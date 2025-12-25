#!/usr/bin/env python3
"""
Backtest Comparison: OLD vs NEW fillna behavior

Quantifies the impact of semantic fillna defaults by comparing
predictions made with fillna(0) vs safe_fillna().

Run: python scripts/test/backtest_fillna_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_historical_data():
    """Load historical odds with actuals for comparison."""
    path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'

    if not path.exists():
        logger.error(f"Historical data not found: {path}")
        return None

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} historical props")
    return df


def calculate_features_old_way(row: pd.Series) -> dict:
    """
    Calculate features with OLD fillna(0) behavior.
    This simulates how missing values were handled before the fix.
    """
    features = {
        'line_vs_trailing': row.get('line', 0) - row.get('trailing_stat', 0),
        'line_level': row.get('line', 0),
        'player_under_rate': 0,  # OLD: Missing = 0 (biased toward over)
        'player_bias': 0,
        'market_under_rate': 0,  # OLD: Missing = 0 (biased toward over)
        'snap_share': 0,
        'opp_pass_def_epa': 0,
        'opp_rush_def_epa': 0,
    }
    return features


def calculate_features_new_way(row: pd.Series) -> dict:
    """
    Calculate features with NEW safe_fillna behavior.
    Uses semantic defaults from FEATURE_DEFAULTS.
    """
    from nfl_quant.features.feature_defaults import FEATURE_DEFAULTS

    features = {
        'line_vs_trailing': row.get('line', 0) - row.get('trailing_stat', 0),
        'line_level': row.get('line', 0),
        'player_under_rate': FEATURE_DEFAULTS.get('player_under_rate', 0.5),  # NEW: 0.5
        'player_bias': FEATURE_DEFAULTS.get('player_bias', 0.0),
        'market_under_rate': FEATURE_DEFAULTS.get('market_under_rate', 0.5),  # NEW: 0.5
        'snap_share': FEATURE_DEFAULTS.get('snap_share', 0.0),
        'opp_pass_def_epa': FEATURE_DEFAULTS.get('opp_pass_def_epa', 0.0),
        'opp_rush_def_epa': FEATURE_DEFAULTS.get('opp_rush_def_epa', 0.0),
    }
    return features


def simulate_prediction(features: dict, method: str = 'simple') -> float:
    """
    Simulate model prediction based on features.
    Uses simplified logic to demonstrate impact.
    """
    # Line vs trailing is the primary signal
    lvt = features.get('line_vs_trailing', 0)

    # Base probability from LVT
    # Higher LVT = more likely under
    base_prob = 0.5 + (lvt / 20.0)  # Normalize to ~0.4-0.6 range

    # Adjust for player tendency
    player_under_rate = features.get('player_under_rate', 0.5)
    tendency_adj = (player_under_rate - 0.5) * 0.1

    # Adjust for market regime
    market_under_rate = features.get('market_under_rate', 0.5)
    market_adj = (market_under_rate - 0.5) * 0.1

    prob_under = base_prob + tendency_adj + market_adj
    return max(0.1, min(0.9, prob_under))


def run_comparison():
    """Run the full comparison analysis."""
    print("\n" + "="*70)
    print("FILLNA COMPARISON: OLD (fillna(0)) vs NEW (safe_fillna)")
    print("="*70)

    # Load data
    df = load_historical_data()
    if df is None:
        return

    # Filter to 2024-2025 data for recency
    df = df[(df['season'] >= 2024)]
    logger.info(f"Filtered to {len(df)} props from 2024-2025")

    # Sample for speed (use full dataset for production)
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
        logger.info(f"Sampled to 5000 props for analysis")

    results = []

    for idx, row in df.iterrows():
        # Calculate features both ways
        old_features = calculate_features_old_way(row)
        new_features = calculate_features_new_way(row)

        # Get predictions
        old_prob = simulate_prediction(old_features)
        new_prob = simulate_prediction(new_features)

        # Calculate difference
        prob_diff = abs(new_prob - old_prob)

        # Determine actual outcome
        actual_under = row.get('under_hit', np.nan)

        results.append({
            'player': row.get('player', 'Unknown'),
            'market': row.get('market', 'Unknown'),
            'season': row.get('season', 0),
            'week': row.get('week', 0),
            'line': row.get('line', 0),
            'actual_stat': row.get('actual_stat', np.nan),
            'old_prob_under': old_prob,
            'new_prob_under': new_prob,
            'prob_diff': prob_diff,
            'actual_under': actual_under,
            'old_correct': 1 if (old_prob > 0.5) == actual_under else 0,
            'new_correct': 1 if (new_prob > 0.5) == actual_under else 0,
        })

    results_df = pd.DataFrame(results)

    # Analysis
    print("\n" + "-"*50)
    print("IMPACT ANALYSIS")
    print("-"*50)

    # 1. How many predictions shifted by >5%?
    shifted_5pct = (results_df['prob_diff'] > 0.05).sum()
    shifted_10pct = (results_df['prob_diff'] > 0.10).sum()
    print(f"\nPredictions shifted >5%:  {shifted_5pct} ({shifted_5pct/len(results_df)*100:.1f}%)")
    print(f"Predictions shifted >10%: {shifted_10pct} ({shifted_10pct/len(results_df)*100:.1f}%)")

    # 2. Average probability difference
    avg_diff = results_df['prob_diff'].mean()
    print(f"\nAverage probability difference: {avg_diff:.4f}")

    # 3. Direction of shift
    results_df['prob_change'] = results_df['new_prob_under'] - results_df['old_prob_under']
    increased = (results_df['prob_change'] > 0).sum()
    decreased = (results_df['prob_change'] < 0).sum()
    print(f"\nShifted toward UNDER: {increased} ({increased/len(results_df)*100:.1f}%)")
    print(f"Shifted toward OVER:  {decreased} ({decreased/len(results_df)*100:.1f}%)")

    # 4. Accuracy comparison
    valid_results = results_df[results_df['actual_under'].notna()]
    if len(valid_results) > 0:
        old_accuracy = valid_results['old_correct'].mean()
        new_accuracy = valid_results['new_correct'].mean()
        accuracy_change = new_accuracy - old_accuracy

        print(f"\n" + "-"*50)
        print("ACCURACY COMPARISON")
        print("-"*50)
        print(f"OLD fillna(0) accuracy: {old_accuracy*100:.2f}%")
        print(f"NEW safe_fillna accuracy: {new_accuracy*100:.2f}%")
        print(f"Accuracy change: {accuracy_change*100:+.2f}%")

        # 5. Accuracy on shifted predictions only
        shifted = valid_results[valid_results['prob_diff'] > 0.05]
        if len(shifted) > 0:
            shifted_old_acc = shifted['old_correct'].mean()
            shifted_new_acc = shifted['new_correct'].mean()
            print(f"\nOn shifted predictions (n={len(shifted)}):")
            print(f"  OLD accuracy: {shifted_old_acc*100:.2f}%")
            print(f"  NEW accuracy: {shifted_new_acc*100:.2f}%")
            print(f"  Change: {(shifted_new_acc - shifted_old_acc)*100:+.2f}%")

    # 6. By market analysis
    print(f"\n" + "-"*50)
    print("BY MARKET")
    print("-"*50)

    for market in results_df['market'].unique():
        market_df = results_df[results_df['market'] == market]
        valid_market = market_df[market_df['actual_under'].notna()]

        if len(valid_market) > 10:
            old_acc = valid_market['old_correct'].mean()
            new_acc = valid_market['new_correct'].mean()
            avg_shift = market_df['prob_diff'].mean()

            print(f"\n{market} (n={len(valid_market)}):")
            print(f"  Avg shift: {avg_shift:.4f}")
            print(f"  OLD: {old_acc*100:.1f}% -> NEW: {new_acc*100:.1f}% ({(new_acc-old_acc)*100:+.1f}%)")

    # 7. Save detailed results
    output_path = PROJECT_ROOT / 'data' / 'backtest' / 'fillna_comparison_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n\nDetailed results saved to: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
The fillna fix affects {shifted_5pct} predictions ({shifted_5pct/len(results_df)*100:.1f}%).

Key insight: OLD fillna(0) for player_under_rate and market_under_rate
created a systematic OVER bias (treating missing as "never goes under").

NEW safe_fillna uses 0.5 (neutral) for these features, removing the bias.
""")

    return results_df


if __name__ == '__main__':
    run_comparison()
