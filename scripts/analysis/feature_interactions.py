"""
V16 Feature Interaction Analysis

Shows how each feature interacts with others in the XGBoost model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def explain_features():
    """Explain what each feature means and how they interact."""

    print("\n" + "=" * 70)
    print("V16 MODEL FEATURE INTERACTION GUIDE")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    CORE DECISION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   LINE_VS_TRAILING (Primary Signal)                                 │
│          │                                                          │
│          │  ┌──────────────────────────────────────────────┐       │
│          ├──┤ × player_under_rate = LVT_x_player_tendency  │       │
│          │  └──────────────────────────────────────────────┘       │
│          │                                                          │
│          │  ┌──────────────────────────────────────────────┐       │
│          ├──┤ × player_bias = LVT_x_player_bias            │       │
│          │  └──────────────────────────────────────────────┘       │
│          │                                                          │
│          │  ┌──────────────────────────────────────────────┐       │
│          ├──┤ × market_under_rate = LVT_x_regime           │       │
│          │  └──────────────────────────────────────────────┘       │
│          │                                                          │
│          │  ┌──────────────────────────────────────────────┐       │
│          └──┤ × line_in_sweet_spot = LVT_in_sweet_spot     │       │
│             └──────────────────────────────────────────────┘       │
│                                                                     │
│   Additional Context:                                               │
│   • line_level: Absolute line value (higher lines = more variance) │
│   • market_bias_strength: How confident is the market?             │
│   • player_market_aligned: Do player & market agree?               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    features = {
        'line_vs_trailing': {
            'meaning': 'How far the line is from player\'s trailing average',
            'formula': 'line - trailing_stat (e.g., line=6.5, trailing=5.0 → LVT=1.5)',
            'interpretation': 'Positive = line above average (favors UNDER), Negative = line below (favors OVER)',
            'monotonic': 'YES (higher LVT → more likely UNDER)',
            'importance': '★★★★★ PRIMARY DRIVER',
        },
        'line_level': {
            'meaning': 'The absolute value of the betting line',
            'formula': 'Just the line itself (e.g., 6.5 receptions)',
            'interpretation': 'Higher lines have more variance; 0.5 lines more volatile than round numbers',
            'importance': '★★★☆☆',
        },
        'line_in_sweet_spot': {
            'meaning': 'Is the line in the optimal betting range?',
            'formula': '1 if line is in [3.5, 8.5] for receptions, similar for other markets',
            'interpretation': 'Lines outside sweet spot may have bookmaker edge',
            'importance': '★★☆☆☆',
        },
        'player_under_rate': {
            'meaning': 'Player\'s historical rate of going UNDER',
            'formula': '(games where actual < line) / total games',
            'interpretation': '0.5 = neutral, >0.5 = tends to go under, <0.5 = tends to go over',
            'importance': '★★★★☆',
        },
        'player_bias': {
            'meaning': 'How much player typically deviates from lines',
            'formula': 'avg(actual - line) over recent games',
            'interpretation': 'Negative = player usually under-performs lines',
            'importance': '★★★☆☆',
        },
        'market_under_rate': {
            'meaning': 'How often UNDERs hit across the entire market',
            'formula': '% of all bets in this market that went under',
            'interpretation': 'If market is 55% under, bookmakers may have set lines too high',
            'importance': '★★★☆☆',
        },
        'LVT_x_player_tendency': {
            'meaning': 'INTERACTION: Line deviation × player tendency',
            'formula': 'line_vs_trailing × (player_under_rate - 0.5)',
            'interpretation': 'Amplifies LVT signal when player has strong under/over tendency',
            'importance': '★★★★☆ KEY INTERACTION',
        },
        'LVT_x_player_bias': {
            'meaning': 'INTERACTION: Line deviation × player bias',
            'formula': 'line_vs_trailing × player_bias',
            'interpretation': 'High line + negative bias = strong under signal',
            'importance': '★★★☆☆',
        },
        'LVT_x_regime': {
            'meaning': 'INTERACTION: Line deviation × market regime',
            'formula': 'line_vs_trailing × (market_under_rate - 0.5)',
            'interpretation': 'Amplifies when market is biased in same direction',
            'importance': '★★★☆☆',
        },
        'LVT_in_sweet_spot': {
            'meaning': 'INTERACTION: Line deviation × sweet spot flag',
            'formula': 'line_vs_trailing × line_in_sweet_spot',
            'interpretation': 'Only count LVT signal when in optimal range',
            'importance': '★★☆☆☆',
        },
        'market_bias_strength': {
            'meaning': 'How extreme is the market bias?',
            'formula': 'abs(market_under_rate - 0.5)',
            'interpretation': 'High = market strongly favors one side',
            'importance': '★★☆☆☆',
        },
        'player_market_aligned': {
            'meaning': 'Do player and market biases agree?',
            'formula': '(player_under_rate - 0.5) × (market_under_rate - 0.5)',
            'interpretation': 'Positive = both favor same direction (stronger signal)',
            'importance': '★★★☆☆',
        },
    }

    print("\n" + "=" * 70)
    print("FEATURE DETAILS")
    print("=" * 70)

    for feat, info in features.items():
        print(f"\n{'─' * 70}")
        print(f"  {feat}")
        print(f"{'─' * 70}")
        print(f"  Meaning:        {info['meaning']}")
        print(f"  Formula:        {info['formula']}")
        print(f"  Interpretation: {info['interpretation']}")
        if 'monotonic' in info:
            print(f"  Monotonic:      {info['monotonic']}")
        print(f"  Importance:     {info['importance']}")


def show_decision_example():
    """Show a concrete example of how the model makes a decision."""

    print("\n" + "=" * 70)
    print("EXAMPLE: How the Model Decides")
    print("=" * 70)

    print("""
SCENARIO: CeeDee Lamb, Receptions Line = 6.5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Calculate Base Signal (line_vs_trailing)
├── Trailing avg receptions: 5.2
├── Line: 6.5
└── LVT = 6.5 - 5.2 = +1.3 ← Line is 1.3 catches ABOVE his average
    └── INITIAL LEAN: UNDER (line seems high)

Step 2: Check Player History (player_under_rate)
├── Lamb has gone UNDER on 58% of his reception props
└── player_under_rate = 0.58 ← Slightly favors UNDER

Step 3: Build Interaction Term (LVT_x_player_tendency)
├── LVT (+1.3) × (0.58 - 0.5) = 1.3 × 0.08 = +0.104
└── REINFORCES UNDER signal (positive LVT + under-tendency)

Step 4: Check Market Regime (market_under_rate)
├── Overall reception unders hitting at 52%
├── LVT_x_regime = 1.3 × (0.52 - 0.5) = 1.3 × 0.02 = +0.026
└── Slight additional UNDER reinforcement

Step 5: Sweet Spot Check
├── Line 6.5 IS in sweet spot [3.5, 8.5]
├── LVT_in_sweet_spot = 1.3 × 1 = +1.3
└── Full LVT signal counts

Step 6: Alignment Check (player_market_aligned)
├── Player bias: 0.08, Market bias: 0.02
├── Aligned = 0.08 × 0.02 = +0.0016
└── Both lean same way (weak but positive)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL DECISION FLOW:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    LVT +1.3 (strong under signal)
         │
         ├─→ × Player tends UNDER (+0.08) → REINFORCED
         │
         ├─→ × Market leans UNDER (+0.02) → SLIGHTLY REINFORCED
         │
         ├─→ × In sweet spot (1.0) → FULL SIGNAL COUNTED
         │
         └─→ XGBoost combines all → P(under) = 0.67

    ✓ RECOMMENDATION: UNDER 6.5 receptions (67% confidence)
""")


def load_and_show_importance():
    """Load the actual model and show feature importance."""

    print("\n" + "=" * 70)
    print("ACTUAL MODEL FEATURE IMPORTANCE")
    print("=" * 70)

    model_path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'

    if not model_path.exists():
        print("  Model not found. Run training first.")
        return

    try:
        bundle = joblib.load(model_path)

        # Check if it's a multi-market model
        if isinstance(bundle, dict) and 'markets' in bundle:
            # Get first market's model
            first_market = list(bundle['markets'].keys())[0]
            model = bundle['markets'][first_market]['model']
            feature_cols = bundle['markets'][first_market].get('feature_cols', [])
        elif isinstance(bundle, dict) and 'model' in bundle:
            model = bundle['model']
            feature_cols = bundle.get('feature_cols', [])
        else:
            print("  Unknown model format")
            return

        # Get importance
        importance = model.get_score(importance_type='gain')
        total = sum(importance.values())

        print(f"\n  Market: {first_market if 'markets' in bundle else 'unified'}")
        print(f"  Features: {len(feature_cols)}")
        print("\n  Feature Importance (by information gain):")
        print("  " + "─" * 50)

        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for feat, score in sorted_imp:
            pct = score / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {feat:30s} {pct:5.1f}% {bar}")

    except Exception as e:
        print(f"  Error loading model: {e}")


def show_xgboost_constraints():
    """Explain the XGBoost interaction constraints."""

    print("\n" + "=" * 70)
    print("XGBOOST INTERACTION CONSTRAINTS")
    print("=" * 70)

    print("""
The V16 model uses XGBoost interaction constraints to ensure meaningful
feature combinations:

┌─────────────────────────────────────────────────────────────────────┐
│ CONSTRAINT 1: LVT Hub Constraint                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   line_vs_trailing ─┬─── can interact with ALL other features       │
│                     │                                               │
│                     ├─── player_under_rate                          │
│                     ├─── player_bias                                │
│                     ├─── market_under_rate                          │
│                     ├─── line_level                                 │
│                     └─── etc.                                       │
│                                                                     │
│   This makes LVT the "hub" of all predictions.                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ CONSTRAINT 2: Monotonic Constraint on LVT                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   line_vs_trailing has MONOTONIC = +1                               │
│                                                                     │
│   This means: As LVT increases, P(under) can ONLY increase          │
│               (or stay the same, never decrease)                    │
│                                                                     │
│   Why? Logically, if the line is WAY above a player's average,      │
│         they should be MORE likely to go under, not less.           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ WHY THESE CONSTRAINTS MATTER                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Without constraints, XGBoost might learn spurious correlations:   │
│   • "Week 14 + Thursday = OVER" (coincidental pattern)              │
│   • Overfitting to noise in small samples                           │
│                                                                     │
│   With constraints, the model MUST use LVT as the primary signal,   │
│   modified by context features. This is more interpretable and      │
│   generalizes better to new data.                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


def main():
    """Run all explanations."""
    explain_features()
    show_decision_example()
    load_and_show_importance()
    show_xgboost_constraints()

    print("\n" + "=" * 70)
    print("END OF FEATURE INTERACTION GUIDE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
