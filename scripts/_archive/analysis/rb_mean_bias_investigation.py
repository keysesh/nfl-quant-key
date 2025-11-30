"""
RB Mean Bias Investigation - Quantify systematic underestimation

Analyzes:
1. Model predicted mean vs actual mean for RB rushing yards
2. Distribution of prediction errors (model - actual)
3. Betting line comparison (model vs market)
4. Week-by-week bias trends
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List


def load_predictions():
    """Load backtest predictions."""
    pred_file = Path("reports/v3_backtest_predictions.csv")
    if not pred_file.exists():
        print("❌ Predictions file not found")
        return None

    df = pd.read_csv(pred_file)

    # Parse samples
    df['samples_array'] = df['samples'].apply(
        lambda x: np.array([float(v) for v in x.split(',')]) if pd.notna(x) else np.array([])
    )

    return df


def load_actuals():
    """Load actual stats for weeks 1-8."""
    all_actuals = []

    for week in range(1, 9):
        file_path = Path(f"data/sleeper_stats/stats_week{week}_2025.csv")
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['week'] = week

            # Standardize column names
            df = df.rename(columns={
                'rec_yd': 'receiving_yards',
                'rush_yd': 'rushing_yards',
                'pass_yd': 'passing_yards',
            })

            all_actuals.append(df)

    if all_actuals:
        return pd.concat(all_actuals, ignore_index=True)

    return pd.DataFrame()


def analyze_mean_bias_by_position():
    """Analyze model mean vs actual mean by position and stat type."""
    print("="*80)
    print("MEAN BIAS ANALYSIS BY POSITION")
    print("="*80)
    print()

    predictions = load_predictions()
    actuals = load_actuals()

    if predictions is None or actuals.empty:
        print("❌ Could not load data")
        return

    # Merge predictions with actuals
    # Note: position comes from predictions, not actuals
    merged = predictions.merge(
        actuals[['player_name', 'team', 'week', 'rushing_yards', 'receiving_yards', 'passing_yards']],
        on=['player_name', 'team', 'week'],
        how='inner'
    )

    print(f"Matched {len(merged)} predictions with actuals\n")

    # Analyze by position and stat type
    for position in ['QB', 'RB', 'WR', 'TE']:
        print(f"{'='*80}")
        print(f"{position} ANALYSIS")
        print(f"{'='*80}")
        print()

        pos_data = merged[merged['position'] == position]

        if len(pos_data) == 0:
            print(f"No data for {position}\n")
            continue

        # Analyze each relevant stat type
        stat_types = []
        if position == 'QB':
            stat_types = ['passing_yards', 'rushing_yards']
        elif position == 'RB':
            stat_types = ['rushing_yards', 'receiving_yards']
        else:  # WR, TE
            stat_types = ['receiving_yards']

        for stat_type in stat_types:
            stat_data = pos_data[pos_data['stat_type'] == stat_type].copy()

            if len(stat_data) == 0:
                continue

            # Calculate means
            model_mean = stat_data['mean'].mean()
            actual_mean = stat_data[stat_type].mean()
            bias = model_mean - actual_mean
            bias_pct = (bias / actual_mean * 100) if actual_mean > 0 else 0

            # Calculate median
            model_median = stat_data['median'].mean()
            actual_median = stat_data[stat_type].median()

            # Calculate error distribution
            stat_data['error'] = stat_data['mean'] - stat_data[stat_type]
            mean_error = stat_data['error'].mean()
            std_error = stat_data['error'].std()
            mae = stat_data['error'].abs().mean()

            # Underestimate vs overestimate
            underestimate_pct = (stat_data['error'] < 0).sum() / len(stat_data) * 100
            overestimate_pct = (stat_data['error'] > 0).sum() / len(stat_data) * 100

            print(f"📊 {stat_type.upper()}:")
            print(f"   Samples: {len(stat_data)}")
            print(f"   Model Mean: {model_mean:.1f} yards")
            print(f"   Actual Mean: {actual_mean:.1f} yards")
            print(f"   Bias: {bias:+.1f} yards ({bias_pct:+.1f}%)")
            print(f"   Model Median: {model_median:.1f} yards")
            print(f"   Actual Median: {actual_median:.1f} yards")
            print(f"   Mean Absolute Error: {mae:.1f} yards")
            print(f"   Error Std Dev: {std_error:.1f} yards")
            print(f"   Underestimates: {underestimate_pct:.1f}%")
            print(f"   Overestimates: {overestimate_pct:.1f}%")

            if abs(bias_pct) > 5:
                if bias_pct < -5:
                    print(f"   ⚠️  SYSTEMATIC UNDERESTIMATION: Model {abs(bias_pct):.1f}% too low")
                else:
                    print(f"   ⚠️  SYSTEMATIC OVERESTIMATION: Model {bias_pct:.1f}% too high")
            else:
                print(f"   ✅ Well-calibrated (bias < 5%)")

            print()

    return merged


def analyze_betting_line_comparison():
    """Compare model predictions to betting market lines."""
    print("="*80)
    print("MODEL VS MARKET LINES COMPARISON")
    print("="*80)
    print()

    # Load betting results
    bets_file = Path("reports/betting_roi_detailed.csv")
    if not bets_file.exists():
        print("❌ Betting results not found")
        return

    bets = pd.read_csv(bets_file)

    # For each position, compare model prob to line crossings
    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        type_bets = bets[bets['stat_type'] == stat_type]

        if len(type_bets) == 0:
            continue

        print(f"📊 {stat_type.upper()}:")

        # Calculate how often model vs market disagrees
        # Model thinks OVER but market prices UNDER as favorite (and vice versa)

        # Separate by side
        over_bets = type_bets[type_bets['side'] == 'over']
        under_bets = type_bets[type_bets['side'] == 'under']

        print(f"   Total bets: {len(type_bets)}")
        print(f"   OVER bets: {len(over_bets)} ({len(over_bets)/len(type_bets)*100:.1f}%)")
        print(f"   UNDER bets: {len(under_bets)} ({len(under_bets)/len(type_bets)*100:.1f}%)")

        # Average line vs actual
        if len(type_bets) > 0:
            avg_line = type_bets['line'].mean()
            avg_actual = type_bets['actual'].mean()
            line_bias = avg_line - avg_actual

            print(f"   Avg Line: {avg_line:.1f} yards")
            print(f"   Avg Actual: {avg_actual:.1f} yards")
            print(f"   Market Bias: {line_bias:+.1f} yards")

            # If model heavily bets UNDER but actuals > lines, model underestimates
            if len(under_bets) > len(type_bets) * 0.7:  # 70%+ UNDER
                if line_bias < 0:  # Lines lower than actuals
                    print(f"   ⚠️  Model bets UNDER heavily, but market already underestimates")
                    print(f"   → Model is EVEN MORE pessimistic than market")

        print()


def analyze_week_by_week_trends():
    """Analyze if bias changes over weeks (model drift)."""
    print("="*80)
    print("WEEK-BY-WEEK BIAS TRENDS")
    print("="*80)
    print()

    predictions = load_predictions()
    actuals = load_actuals()

    if predictions is None or actuals.empty:
        return

    # Note: position comes from predictions DataFrame
    merged = predictions.merge(
        actuals[['player_name', 'team', 'week', 'rushing_yards', 'receiving_yards', 'passing_yards']],
        on=['player_name', 'team', 'week'],
        how='inner'
    )

    # Focus on RB rushing yards
    rb_rushing = merged[
        (merged['position'] == 'RB') &
        (merged['stat_type'] == 'rushing_yards')
    ].copy()

    if len(rb_rushing) == 0:
        print("No RB rushing data")
        return

    print("RB RUSHING YARDS BIAS BY WEEK:")
    print(f"{'Week':<6} {'N':<5} {'Model':<8} {'Actual':<8} {'Bias':<8} {'Bias%':<8}")
    print("-" * 60)

    for week in range(1, 9):
        week_data = rb_rushing[rb_rushing['week'] == week]

        if len(week_data) == 0:
            continue

        model_mean = week_data['mean'].mean()
        actual_mean = week_data['rushing_yards'].mean()
        bias = model_mean - actual_mean
        bias_pct = (bias / actual_mean * 100) if actual_mean > 0 else 0

        indicator = "⚠️" if abs(bias_pct) > 10 else "✅"

        print(f"{week:<6} {len(week_data):<5} {model_mean:<8.1f} {actual_mean:<8.1f} {bias:+8.1f} {bias_pct:+7.1f}% {indicator}")

    # Overall
    model_mean = rb_rushing['mean'].mean()
    actual_mean = rb_rushing['rushing_yards'].mean()
    bias = model_mean - actual_mean
    bias_pct = (bias / actual_mean * 100) if actual_mean > 0 else 0

    print("-" * 60)
    print(f"{'Overall':<6} {len(rb_rushing):<5} {model_mean:<8.1f} {actual_mean:<8.1f} {bias:+8.1f} {bias_pct:+7.1f}%")
    print()


def recommend_mean_adjustment():
    """Calculate recommended mean adjustment multiplier."""
    print("="*80)
    print("MEAN ADJUSTMENT RECOMMENDATIONS")
    print("="*80)
    print()

    predictions = load_predictions()
    actuals = load_actuals()

    if predictions is None or actuals.empty:
        return

    # Note: position comes from predictions DataFrame
    merged = predictions.merge(
        actuals[['player_name', 'team', 'week', 'rushing_yards', 'receiving_yards', 'passing_yards']],
        on=['player_name', 'team', 'week'],
        how='inner'
    )

    # Calculate adjustments for each position/stat combo
    adjustments = []

    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = merged[merged['position'] == position]

        stat_types = []
        if position == 'QB':
            stat_types = ['passing_yards', 'rushing_yards']
        elif position == 'RB':
            stat_types = ['rushing_yards', 'receiving_yards']
        else:
            stat_types = ['receiving_yards']

        for stat_type in stat_types:
            stat_data = pos_data[pos_data['stat_type'] == stat_type]

            if len(stat_data) < 10:  # Need minimum sample
                continue

            model_mean = stat_data['mean'].mean()
            actual_mean = stat_data[stat_type].mean()

            if model_mean > 0:
                adjustment_multiplier = actual_mean / model_mean
                bias_pct = (1 - adjustment_multiplier) * 100

                config_key = f"{position.lower()}_{stat_type.replace('_yards', '')}_mean_adjustment"

                adjustments.append({
                    'position': position,
                    'stat_type': stat_type,
                    'model_mean': model_mean,
                    'actual_mean': actual_mean,
                    'adjustment_multiplier': adjustment_multiplier,
                    'bias_pct': bias_pct,
                    'config_key': config_key,
                    'samples': len(stat_data)
                })

    # Print recommendations
    print("Recommended config additions for simulation_config.json:")
    print()
    print('"mean_adjustments": {')

    for adj in sorted(adjustments, key=lambda x: abs(x['bias_pct']), reverse=True):
        if abs(adj['bias_pct']) > 3:  # Only adjust if >3% bias
            print(f'  "{adj["config_key"]}": {adj["adjustment_multiplier"]:.3f},  '
                  f'// {adj["bias_pct"]:+.1f}% bias ({adj["samples"]} samples)')

    print('}')
    print()

    # Show most critical fixes
    print("\n🔧 MOST CRITICAL FIXES (>10% bias):")
    for adj in sorted(adjustments, key=lambda x: abs(x['bias_pct']), reverse=True):
        if abs(adj['bias_pct']) > 10:
            direction = "UNDERESTIMATE" if adj['bias_pct'] < 0 else "OVERESTIMATE"
            print(f"   {adj['position']} {adj['stat_type']}: {direction} by {abs(adj['bias_pct']):.1f}%")
            print(f"      Model: {adj['model_mean']:.1f}, Actual: {adj['actual_mean']:.1f}")
            print(f"      → Multiply predictions by {adj['adjustment_multiplier']:.3f}")
    print()


def main():
    print("\n")

    # Run all analyses
    merged = analyze_mean_bias_by_position()
    print("\n")

    analyze_betting_line_comparison()
    print("\n")

    analyze_week_by_week_trends()
    print("\n")

    recommend_mean_adjustment()

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Add mean_adjustment parameters to simulation_config.json
2. Update player_simulator_v3_correlated.py to apply adjustments
3. Re-run backtest to validate fixes
4. Re-run ROI analysis to measure improvement

Expected result: RB rushing yards ROI should improve from -1.3% to positive
    """)


if __name__ == '__main__':
    main()
