#!/usr/bin/env python3
"""
Walk-Forward Validation with FULL Historical Data
==================================================

Uses bet_outcomes_consolidated.csv which has:
- 13,773 props across weeks 1-8
- ~1,700 props per week (realistic volume)
- Actual outcomes (bet_won)
- Model probabilities and projections

This provides a more robust test than the sparse real_props_backtest.csv.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.isotonic import IsotonicRegression

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
REPORTS_DIR = Path(__file__).parent.parent.parent / 'reports'


def run_full_walk_forward():
    """Run walk-forward validation on complete historical data."""
    print("=" * 70)
    print("WALK-FORWARD VALIDATION - FULL HISTORICAL DATA")
    print("13,773 Props Across Weeks 1-8")
    print("=" * 70)

    # Load consolidated data
    df = pd.read_csv(DATA_DIR / 'betting_history' / 'bet_outcomes_consolidated.csv')
    print(f"\nLoaded {len(df)} historical prop outcomes")

    weeks = sorted(df['week'].unique())
    print(f"Weeks available: {weeks}")

    # Week distribution
    print("\nProp distribution per week:")
    for w in weeks:
        count = len(df[df['week'] == w])
        print(f"  Week {w}: {count:,} props")

    # Derive went_over from actual_outcome
    df['went_over'] = (df['actual_outcome'] == 'OVER').astype(int)

    # Derive prob_over_raw from model_prob and pick direction
    # If pick is OVER, model_prob is prob of over
    # If pick is UNDER, model_prob is prob of under, so prob_over = 1 - model_prob
    df['prob_over_raw'] = df.apply(
        lambda r: r['model_prob'] if 'Over' in r['pick'] else 1 - r['model_prob'],
        axis=1
    )

    # Market prob adjustment (currently shows implied prob of bet side)
    # Convert to prob_over format
    df['market_prob_over'] = df.apply(
        lambda r: r['market_prob'] if 'Over' in r['pick'] else 1 - r['market_prob'],
        axis=1
    )

    print("\nDerived fields:")
    print(f"  went_over: {df['went_over'].mean():.1%} OVER rate")
    print(f"  prob_over_raw range: {df['prob_over_raw'].min():.2f} - {df['prob_over_raw'].max():.2f}")

    # Initialize tracking
    cumulative = {
        'bankroll': 1000.0,
        'total_bets': 0,
        'wins': 0,
        'losses': 0,
        'profit': 0.0,
        'peak': 1000.0,
        'max_drawdown': 0.0
    }
    weekly_results = []

    # Sequential validation
    print("\n" + "=" * 70)
    print("SEQUENTIAL VALIDATION (OUT-OF-SAMPLE)")
    print("=" * 70)

    for test_week in weeks:
        train_weeks = [w for w in weeks if w < test_week]

        if len(train_weeks) < 2:
            print(f"\nWeek {test_week}: Skipping (need 2+ weeks training)")
            continue

        train_df = df[df['week'].isin(train_weeks)]
        test_df = df[df['week'] == test_week].copy()

        print(f"\n--- WEEK {test_week} ---")
        print(f"Train on weeks {train_weeks} ({len(train_df):,} props)")
        print(f"Test on week {test_week} ({len(test_df):,} props)")

        # Train isotonic calibrator
        X_train = train_df['prob_over_raw'].values
        y_train = train_df['went_over'].values

        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(X_train, y_train)

        # Calibrate test week
        test_df['prob_cal'] = calibrator.predict(test_df['prob_over_raw'].values)

        # Apply shrinkage for extreme probabilities
        high_mask = test_df['prob_cal'] > 0.70
        test_df.loc[high_mask, 'prob_cal'] = 0.70 + (test_df.loc[high_mask, 'prob_cal'] - 0.70) * 0.5

        low_mask = test_df['prob_cal'] < 0.30
        test_df.loc[low_mask, 'prob_cal'] = 0.30 - (0.30 - test_df.loc[low_mask, 'prob_cal']) * 0.5

        # Calculate edges
        test_df['edge_over'] = test_df['prob_cal'] - test_df['market_prob_over']
        test_df['edge_under'] = (1 - test_df['prob_cal']) - (1 - test_df['market_prob_over'])

        # Recommendation
        test_df['rec'] = np.where(test_df['edge_over'] > test_df['edge_under'], 'OVER', 'UNDER')
        test_df['best_edge'] = np.maximum(test_df['edge_over'], test_df['edge_under'])

        # Simulate betting (min 5% edge, 1/4 Kelly)
        min_edge = 0.05
        kelly_fraction = 0.25

        betting_df = test_df[test_df['best_edge'] >= min_edge].copy()

        week_profit = 0.0
        week_wins = 0
        week_losses = 0

        for _, row in betting_df.iterrows():
            rec = row['rec']
            edge = row['best_edge']

            # Kelly sizing
            full_kelly = edge / 0.909
            bet_pct = min(full_kelly * kelly_fraction, 0.05)
            bet_size = cumulative['bankroll'] * bet_pct

            # Check if bet won
            if rec == 'OVER':
                bet_won = row['went_over'] == 1
            else:
                bet_won = row['went_over'] == 0

            # Calculate profit (-110 odds)
            if bet_won:
                profit = bet_size * 0.909
                week_wins += 1
            else:
                profit = -bet_size
                week_losses += 1

            week_profit += profit

        # Update cumulative
        cumulative['bankroll'] += week_profit
        cumulative['total_bets'] += len(betting_df)
        cumulative['wins'] += week_wins
        cumulative['losses'] += week_losses
        cumulative['profit'] += week_profit

        if cumulative['bankroll'] > cumulative['peak']:
            cumulative['peak'] = cumulative['bankroll']

        drawdown = (cumulative['peak'] - cumulative['bankroll']) / cumulative['peak']
        if drawdown > cumulative['max_drawdown']:
            cumulative['max_drawdown'] = drawdown

        # Metrics
        win_rate = week_wins / len(betting_df) if len(betting_df) > 0 else 0
        brier = np.mean((test_df['went_over'] - test_df['prob_cal']) ** 2)

        week_result = {
            'week': int(test_week),
            'train_weeks': [int(w) for w in train_weeks],
            'props_tested': int(len(test_df)),
            'bets_placed': int(len(betting_df)),
            'wins': int(week_wins),
            'losses': int(week_losses),
            'win_rate': float(win_rate),
            'profit': float(week_profit),
            'brier_score': float(brier),
            'bankroll': float(cumulative['bankroll'])
        }
        weekly_results.append(week_result)

        print(f"  Bets: {len(betting_df):,} | Win Rate: {win_rate:.1%} | Profit: ${week_profit:+.2f}")
        print(f"  Brier: {brier:.4f} | Bankroll: ${cumulative['bankroll']:.2f}")

    # Final report
    print("\n" + "=" * 70)
    print("CUMULATIVE RESULTS")
    print("=" * 70)

    overall_wr = cumulative['wins'] / cumulative['total_bets'] if cumulative['total_bets'] > 0 else 0

    print(f"Total Bets: {cumulative['total_bets']:,}")
    print(f"Wins: {cumulative['wins']:,} | Losses: {cumulative['losses']:,}")
    print(f"Overall Win Rate: {overall_wr:.1%}")
    print(f"Total Profit: ${cumulative['profit']:+.2f}")
    print(f"Final Bankroll: ${cumulative['bankroll']:.2f} (started $1000)")
    print(f"Total ROI: {(cumulative['bankroll'] - 1000) / 1000 * 100:+.1f}%")
    print(f"Max Drawdown: {cumulative['max_drawdown']:.1%}")

    # Profitability check
    breakeven = 0.524
    print(f"\nBreak-even at -110: {breakeven:.1%}")

    if overall_wr > breakeven:
        edge_pp = (overall_wr - breakeven) * 100
        print(f"✅ PROFITABLE: {edge_pp:.1f}pp above break-even")
    else:
        deficit_pp = (breakeven - overall_wr) * 100
        print(f"❌ NOT PROFITABLE: {deficit_pp:.1f}pp below break-even")

    # Brier score
    avg_brier = np.mean([w['brier_score'] for w in weekly_results])
    print(f"\nAverage Brier Score: {avg_brier:.4f}")
    if avg_brier < 0.25:
        print("✅ Better than random (0.25)")
    else:
        print("❌ Worse than random (0.25)")

    # Week-by-week table
    print("\n" + "=" * 70)
    print("WEEK-BY-WEEK BREAKDOWN")
    print("=" * 70)
    print(f"{'Week':<6} {'Props':<8} {'Bets':<8} {'Wins':<6} {'Win%':<8} {'Profit':<12} {'Bankroll'}")
    print("-" * 70)

    for w in weekly_results:
        print(f"{w['week']:<6} {w['props_tested']:<8} {w['bets_placed']:<8} {w['wins']:<6} "
              f"{w['win_rate']:.1%}   ${w['profit']:+9.2f}  ${w['bankroll']:.2f}")

    # Save results
    REPORTS_DIR.mkdir(exist_ok=True)
    results_file = REPORTS_DIR / 'walk_forward_full_data_results.json'

    with open(results_file, 'w') as f:
        json.dump({
            'weekly_results': weekly_results,
            'cumulative': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in cumulative.items()},
            'generated': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")

    return weekly_results, cumulative


if __name__ == "__main__":
    weekly_results, cumulative = run_full_walk_forward()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("\nThis validation uses 13,773 historical props (weeks 1-8).")
    print("Each week tested with calibrator trained ONLY on prior weeks.")
    print("No look-ahead bias - true out-of-sample performance.")

    if cumulative['wins'] / cumulative['total_bets'] > 0.524:
        print("\n✅ Model shows genuine predictive edge!")
        print("   Sequential calibration improves accuracy over time.")
    else:
        print("\n❌ Model needs improvement.")
        print("   Review feature engineering and data quality.")
