#!/usr/bin/env python3
"""
Audit high edge calculations to verify they're correct.

Edge of 36% seems very high - this script will:
1. Verify odds-to-probability conversions
2. Check if calibration is working correctly
3. Identify if model is overconfident
"""

import pandas as pd
import numpy as np


def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def audit_high_edges():
    """Audit recommendations with suspiciously high edges."""

    print("="*80)
    print("HIGH EDGE AUDIT")
    print("="*80)
    print()

    # Load recommendations
    df = pd.read_csv('reports/unified_betting_recommendations_v2_ranked.csv')
    print(f"✅ Loaded {len(df)} recommendations")

    # Sort by edge (highest first)
    df_sorted = df.sort_values('edge', ascending=False)

    print(f"\nTop 10 Highest Edge Bets:")
    print("="*80)

    for idx, row in df_sorted.head(10).iterrows():
        pick = row['pick']
        market_odds = row['market_odds']
        our_prob = row['our_prob']
        market_prob = row['market_prob']
        edge = row['edge']
        framework_pred = row.get('framework_prediction', 'N/A')

        print(f"\nPick: {pick}")
        print(f"  Framework Prediction: {framework_pred}")
        print(f"  Market Odds: {market_odds}")
        print(f"  Our Prob: {our_prob:.2%}")
        print(f"  Market Prob: {market_prob:.2%}")
        print(f"  Edge: {edge:.2%}")

        # Verify calculation
        recalc_market_prob = american_to_implied_prob(market_odds)
        recalc_edge = our_prob - recalc_market_prob

        print(f"  ✓ Recalculated Market Prob: {recalc_market_prob:.2%}")
        print(f"  ✓ Recalculated Edge: {recalc_edge:.2%}")

        if abs(recalc_edge - edge) > 0.001:
            print(f"  ⚠️  MISMATCH! Expected {recalc_edge:.2%}, got {edge:.2%}")
        else:
            print(f"  ✅ Edge calculation verified")

        # Check if edge makes sense
        if edge > 0.30:
            print(f"  🚨 WARNING: Edge > 30% is extremely high!")
            print(f"     This suggests either:")
            print(f"     1. Model is overconfident (calibration issue)")
            print(f"     2. Market odds are stale/wrong")
            print(f"     3. Legitimate massive opportunity (rare)")

    # Summary statistics
    print(f"\n{'='*80}")
    print("EDGE DISTRIBUTION SUMMARY")
    print(f"{'='*80}\n")

    print(f"Mean Edge: {df['edge'].mean():.2%}")
    print(f"Median Edge: {df['edge'].median():.2%}")
    print(f"Max Edge: {df['edge'].max():.2%}")
    print(f"Min Edge: {df['edge'].min():.2%}")
    print(f"Std Dev: {df['edge'].std():.2%}")

    print(f"\nEdge Buckets:")
    print(f"  Edge > 30%: {len(df[df['edge'] > 0.30])} bets ({len(df[df['edge'] > 0.30])/len(df)*100:.1f}%)")
    print(f"  Edge 20-30%: {len(df[(df['edge'] >= 0.20) & (df['edge'] <= 0.30)])} bets")
    print(f"  Edge 10-20%: {len(df[(df['edge'] >= 0.10) & (df['edge'] < 0.20)])} bets")
    print(f"  Edge 5-10%: {len(df[(df['edge'] >= 0.05) & (df['edge'] < 0.10)])} bets")
    print(f"  Edge < 5%: {len(df[df['edge'] < 0.05])} bets")

    # Check our_prob distribution
    print(f"\n{'='*80}")
    print("MODEL PROBABILITY DISTRIBUTION")
    print(f"{'='*80}\n")

    print(f"Mean Model Prob: {df['our_prob'].mean():.2%}")
    print(f"Median Model Prob: {df['our_prob'].median():.2%}")

    print(f"\nModel Confidence Buckets:")
    print(f"  Prob > 85%: {len(df[df['our_prob'] > 0.85])} bets ({len(df[df['our_prob'] > 0.85])/len(df)*100:.1f}%)")
    print(f"  Prob 75-85%: {len(df[(df['our_prob'] >= 0.75) & (df['our_prob'] <= 0.85)])} bets")
    print(f"  Prob 65-75%: {len(df[(df['our_prob'] >= 0.65) & (df['our_prob'] < 0.75)])} bets")
    print(f"  Prob 55-65%: {len(df[(df['our_prob'] >= 0.55) & (df['our_prob'] < 0.65)])} bets")
    print(f"  Prob < 55%: {len(df[df['our_prob'] < 0.55])} bets")

    # Check if calibration badge indicates issues
    print(f"\n{'='*80}")
    print("CALIBRATION STATUS")
    print(f"{'='*80}\n")

    if 'calibration_badge' in df.columns:
        print(df['calibration_badge'].value_counts())
    else:
        print("No calibration badge found")

    # Detailed analysis of top edge bet
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS: HIGHEST EDGE BET")
    print(f"{'='*80}\n")

    top_row = df_sorted.iloc[0]
    print(f"Pick: {top_row['pick']}")
    print(f"Game: {top_row['game']}")
    print(f"Player: {top_row.get('player', 'N/A')}")
    print(f"Framework Prediction: {top_row.get('framework_prediction', 'N/A')}")
    print(f"\nOdds Analysis:")
    print(f"  Market Odds: {top_row['market_odds']}")
    print(f"  Implied Prob: {top_row['market_prob']:.2%}")
    print(f"  Our Prob: {top_row['our_prob']:.2%}")
    print(f"  Edge: {top_row['edge']:.2%}")
    print(f"\nValue Analysis:")
    print(f"  EV: ${top_row.get('ev', 0):.2f}")
    print(f"  ROI: {top_row.get('roi', 0):.2%}")
    print(f"  Recommended Bet: ${top_row.get('bet_size', 0):.2f}")
    print(f"\nInterpretation:")

    if top_row['our_prob'] > 0.85 and top_row['market_prob'] < 0.55:
        print(f"  🔍 Model is VERY confident ({top_row['our_prob']:.0%}) but market is not ({top_row['market_prob']:.0%})")
        print(f"  ⚠️  This could indicate:")
        print(f"      - Calibration not working (model overconfident)")
        print(f"      - Model found genuine insight market missed")
        print(f"      - Stale/incorrect market odds")

    # Save audit report
    audit_df = df_sorted[['pick', 'framework_prediction', 'market_odds', 'our_prob', 'market_prob', 'edge', 'ev', 'calibration_badge']].head(20)
    audit_df.to_csv('reports/high_edge_audit.csv', index=False)
    print(f"\n✅ Detailed audit saved to: reports/high_edge_audit.csv")


if __name__ == "__main__":
    audit_high_edges()
