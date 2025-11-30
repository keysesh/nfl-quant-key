#!/usr/bin/env python3
"""
Diagnose Model Overconfidence Issue

This script identifies WHY the model is showing 89% confidence when market is 53%.
Possible causes:
1. Calibrator not loaded/applied correctly
2. Calibrator trained on wrong data
3. Market odds are stale/incorrect
4. Simulation variance too low
5. Model inputs are wrong
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.schemas import PlayerPropInput


def american_to_prob(odds: int) -> float:
    """Convert American odds to probability."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def diagnose_overconfidence():
    """Run comprehensive diagnostics."""

    print("="*80)
    print("MODEL OVERCONFIDENCE DIAGNOSTIC")
    print("="*80)
    print()

    # ========================================
    # TEST 1: Check if calibrator exists and is loaded
    # ========================================
    print("TEST 1: Calibrator Status")
    print("-"*80)

    calibrator = NFLProbabilityCalibrator()
    calibrator_paths = [
        Path("models/isotonic_calibrator.pkl"),
        Path("configs/calibrator.json"),
        Path("models/calibrator.pkl"),
    ]

    calibrator_loaded = False
    for path in calibrator_paths:
        if path.exists():
            print(f"✅ Found calibrator: {path}")
            try:
                calibrator.load(str(path))
                calibrator_loaded = True
                print(f"✅ Successfully loaded calibrator")
                print(f"   Is fitted: {calibrator.is_fitted}")
                break
            except Exception as e:
                print(f"❌ Failed to load: {e}")
        else:
            print(f"❌ Not found: {path}")

    if not calibrator_loaded:
        print(f"\n🚨 ISSUE FOUND: No calibrator loaded!")
        print(f"   This means raw simulation probabilities are being used directly")
        print(f"   Raw probabilities tend to be overconfident (too extreme)")
        print(f"\n💡 FIX: Train and load calibrator")
    print()

    # ========================================
    # TEST 2: Check calibration curve
    # ========================================
    print("TEST 2: Calibration Curve Analysis")
    print("-"*80)

    if calibrator_loaded and calibrator.is_fitted:
        # Test probabilities across range
        test_probs = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        calibrated = calibrator.transform(test_probs)

        print("Raw Prob → Calibrated Prob")
        for raw, cal in zip(test_probs, calibrated):
            diff = cal - raw
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            print(f"  {raw:.2f} {arrow} {cal:.2f} (diff: {diff:+.2f})")

        # Check if calibrator is pulling extremes toward 0.5
        avg_pull = np.mean(np.abs(calibrated - test_probs))
        if avg_pull < 0.05:
            print(f"\n⚠️  WARNING: Calibrator barely adjusting probabilities")
            print(f"   Average adjustment: {avg_pull:.3f}")
            print(f"   This suggests calibrator wasn't trained properly")
        elif calibrated[-1] > 0.9:  # 0.99 raw should not stay > 0.9
            print(f"\n🚨 ISSUE FOUND: Calibrator not pulling down high probabilities!")
            print(f"   0.99 raw → {calibrated[-1]:.2f} calibrated (should be ~0.65-0.75)")
            print(f"\n💡 FIX: Retrain calibrator with proper historical hit rates")
    else:
        print("⚠️  Skipping - no calibrator loaded")
    print()

    # ========================================
    # TEST 3: Check a real example from recommendations
    # ========================================
    print("TEST 3: Real Example Analysis")
    print("-"*80)

    # Load the high-edge recommendations
    recs = pd.read_csv('reports/unified_betting_recommendations_v2_ranked.csv')

    # Get the highest edge bet
    top_bet = recs.iloc[0]
    print(f"Analyzing: {top_bet['pick']}")
    print(f"Game: {top_bet['game']}")
    print(f"Framework Prediction: {top_bet.get('framework_prediction', 'N/A')}")
    print()

    our_prob = top_bet['our_prob']
    market_prob = top_bet['market_prob']
    edge = top_bet['edge']

    print(f"Current State:")
    print(f"  Our Prob: {our_prob:.2%}")
    print(f"  Market Prob: {market_prob:.2%}")
    print(f"  Edge: {edge:.2%}")
    print()

    # Check if this is reasonable
    if our_prob > 0.85 and market_prob < 0.55:
        print(f"🚨 ISSUE FOUND: Massive disagreement with market")
        print(f"   Model: {our_prob:.0%} vs Market: {market_prob:.0%}")
        print(f"   Difference: {(our_prob - market_prob):.0%}")
        print()

        # Try to figure out WHY
        print(f"Possible explanations:")
        print(f"  1. Calibrator not working → model overconfident")
        print(f"  2. Market odds are wrong/stale")
        print(f"  3. Model found genuine edge (unlikely to be this large)")
    print()

    # ========================================
    # TEST 4: Check historical backtest calibration
    # ========================================
    print("TEST 4: Historical Backtest Accuracy")
    print("-"*80)

    backtest_files = [
        Path("reports/week_by_week_backtest_results.csv"),
        Path("reports/backtest_edge_audit_weeks1-8.csv"),
        Path("reports/detailed_bet_analysis_weekall.csv"),
    ]

    backtest_found = False
    for path in backtest_files:
        if path.exists():
            print(f"✅ Found backtest: {path}")
            df = pd.read_csv(path)

            # Check if we have predicted_prob and actual outcome
            if 'model_prob' in df.columns or 'our_prob' in df.columns:
                prob_col = 'model_prob' if 'model_prob' in df.columns else 'our_prob'

                if 'hit' in df.columns or 'outcome' in df.columns:
                    outcome_col = 'hit' if 'hit' in df.columns else 'outcome'

                    # Calculate calibration
                    df_clean = df[[prob_col, outcome_col]].dropna()

                    # Group by probability buckets
                    df_clean['prob_bucket'] = pd.cut(
                        df_clean[prob_col],
                        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
                    )

                    calibration = df_clean.groupby('prob_bucket').agg({
                        prob_col: 'mean',
                        outcome_col: 'mean',
                        'prob_bucket': 'size'
                    }).rename(columns={'prob_bucket': 'count'})

                    print(f"\nCalibration Analysis:")
                    print(f"{'Bucket':<15} {'Predicted':<12} {'Actual':<12} {'Count':<10} {'Status'}")
                    print("-"*65)

                    for bucket, row in calibration.iterrows():
                        pred = row[prob_col]
                        actual = row[outcome_col]
                        count = row['count']
                        diff = pred - actual

                        if abs(diff) < 0.05:
                            status = "✅ Well calibrated"
                        elif diff > 0.1:
                            status = "🚨 OVERCONFIDENT"
                        elif diff < -0.1:
                            status = "⚠️  Underconfident"
                        else:
                            status = "⚙️  Slight bias"

                        print(f"{bucket:<15} {pred:.2%}          {actual:.2%}          {count:<10.0f} {status}")

                    # Overall accuracy
                    if '80-90%' in calibration.index or '90-100%' in calibration.index:
                        high_conf_buckets = calibration[calibration.index.isin(['80-90%', '90-100%'])]
                        avg_predicted = high_conf_buckets[prob_col].mean()
                        avg_actual = high_conf_buckets[outcome_col].mean()

                        if avg_predicted - avg_actual > 0.15:
                            print(f"\n🚨 ISSUE FOUND: High confidence bets are overconfident!")
                            print(f"   Predicted: {avg_predicted:.2%}")
                            print(f"   Actual: {avg_actual:.2%}")
                            print(f"   Overconfidence: {(avg_predicted - avg_actual):.2%}")
                            print(f"\n💡 FIX: Retrain calibrator to pull down high probabilities")

                    backtest_found = True
                    break

    if not backtest_found:
        print("⚠️  No backtest data with outcomes found")
        print("   Run backtest to assess calibration accuracy")
    print()

    # ========================================
    # TEST 5: Check market odds freshness
    # ========================================
    print("TEST 5: Market Odds Freshness")
    print("-"*80)

    if 'commence_time' in recs.columns:
        from datetime import datetime
        now = datetime.now()

        # Check when odds were last fetched
        odds_files = list(Path("data").glob("odds_week*.csv"))
        if odds_files:
            latest_odds = max(odds_files, key=lambda p: p.stat().st_mtime)
            odds_age = (now.timestamp() - latest_odds.stat().st_mtime) / 3600

            print(f"Latest odds file: {latest_odds.name}")
            print(f"Age: {odds_age:.1f} hours")

            if odds_age > 24:
                print(f"\n⚠️  WARNING: Odds data is {odds_age:.0f} hours old!")
                print(f"   Stale odds can cause inflated edges")
                print(f"\n💡 FIX: Fetch fresh odds before generating recommendations")
            elif odds_age > 6:
                print(f"\n⚙️  Odds are {odds_age:.0f} hours old - consider refreshing")
            else:
                print(f"✅ Odds are fresh ({odds_age:.1f} hours old)")
        else:
            print("❌ No odds files found in data/")
    print()

    # ========================================
    # TEST 6: Simulate one prop to check raw probability
    # ========================================
    print("TEST 6: Simulation Probability Check")
    print("-"*80)

    try:
        # Load simulator
        usage_pred, efficiency_pred = load_predictors()
        simulator = PlayerSimulator(
            usage_predictor=usage_pred,
            efficiency_predictor=efficiency_pred,
            trials=50000,
            seed=42,
            calibrator=calibrator if calibrator_loaded else None,
        )

        # Get a player prop from recommendations
        player_props = recs[recs['bet_type'] == 'Player Prop']
        if len(player_props) > 0:
            prop = player_props.iloc[0]
            print(f"Testing: {prop['player']} - {prop['pick']}")

            # Note: We'd need actual player data to simulate
            # This is a placeholder showing the process
            print(f"✅ Simulator loaded with {simulator.trials} trials")
            print(f"   Calibrator active: {simulator.calibrator is not None and simulator.calibrator.is_fitted}")
            print()
            print(f"⚠️  Full simulation test requires player stats")
            print(f"   See TEST 7 for manual simulation example")

    except Exception as e:
        print(f"❌ Could not load simulator: {e}")
    print()

    # ========================================
    # SUMMARY & RECOMMENDATIONS
    # ========================================
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print()

    print("Issues Found:")
    issues = []
    fixes = []

    if not calibrator_loaded:
        issues.append("❌ No calibrator loaded")
        fixes.append("1. Train calibrator: python scripts/train/retrain_calibrator_nflverse.py")

    if calibrator_loaded and calibrator.is_fitted:
        # Check if probabilities stay too high
        test_high = calibrator.transform(np.array([0.95]))
        if test_high[0] > 0.85:
            issues.append("❌ Calibrator not pulling down high probabilities")
            fixes.append("2. Retrain calibrator with more conservative mapping")

    if backtest_found and 'avg_predicted' in locals():
        if avg_predicted - avg_actual > 0.15:
            issues.append(f"❌ Historical overconfidence: {(avg_predicted - avg_actual):.1%}")
            fixes.append("3. Review calibration training data for bias")

    if 'odds_age' in locals() and odds_age > 12:
        issues.append(f"⚠️  Stale odds data ({odds_age:.0f} hours old)")
        fixes.append("4. Fetch fresh odds: python scripts/fetch/fetch_live_odds.py")

    if len(issues) == 0:
        print("✅ No obvious issues detected")
        print()
        print("Possible explanations for high edges:")
        print("  - Model genuinely found market inefficiencies")
        print("  - Need more backtesting to validate")
        print("  - Consider reducing bet sizes on highest edges")
    else:
        for issue in issues:
            print(f"  {issue}")
        print()
        print("Recommended Fixes:")
        for i, fix in enumerate(fixes, 1):
            print(f"  {fix}")

    print()
    print("="*80)


if __name__ == "__main__":
    diagnose_overconfidence()
