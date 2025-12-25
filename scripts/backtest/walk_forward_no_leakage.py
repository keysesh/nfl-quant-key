#!/usr/bin/env python3
"""
Walk-Forward Backtest with NO Data Leakage

For each week W:
1. Load player stats from weeks 1 to W-1 only
2. Generate predictions for week W (using only historical data)
3. Compare to actual week W results
4. Store results
5. Move to week W+1

Key Anti-Leakage Principles:
- Temporal separation: Predictions for week W use only weeks 1 to W-1
- No future data: Model never sees outcomes it's predicting
- Correlation check: Model should NOT beat line correlation (if it does, leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_historical_stats(max_week: int, season: int = 2025) -> pd.DataFrame:
    """
    Load player stats ONLY up to max_week (exclusive).
    This prevents data leakage.
    """
    stats_path = Path('data/nflverse/weekly_stats.parquet')

    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    df = pd.read_parquet(stats_path)

    # Filter to current season and ONLY weeks BEFORE prediction week
    df = df[(df['season'] == season) & (df['week'] < max_week)]

    return df


def generate_predictions_for_week(
    historical_stats: pd.DataFrame,
    week: int,
    rolling_window: int = 4
) -> pd.DataFrame:
    """
    Generate predictions for a specific week using ONLY historical data.

    Method: Rolling average of last N weeks for each player.
    """
    if len(historical_stats) == 0:
        return pd.DataFrame()

    # Calculate rolling averages per player
    # Sort by week to ensure proper rolling
    historical_stats = historical_stats.sort_values(['player_display_name', 'week'])

    predictions = []

    for player, group in historical_stats.groupby('player_display_name'):
        # Use last N weeks (or all available if less)
        recent = group.tail(rolling_window)

        pred = {
            'player': player,
            'pred_receptions': recent['receptions'].mean() if 'receptions' in recent else np.nan,
            'pred_rec_yards': recent['receiving_yards'].mean() if 'receiving_yards' in recent else np.nan,
            'pred_rush_yards': recent['rushing_yards'].mean() if 'rushing_yards' in recent else np.nan,
            'pred_pass_yards': recent['passing_yards'].mean() if 'passing_yards' in recent else np.nan,
            'games_used': len(recent),
            'prediction_week': week,
        }
        predictions.append(pred)

    return pd.DataFrame(predictions)


def load_props_for_week(week: int, season: int = 2025) -> pd.DataFrame:
    """Load betting lines and actuals for a specific week."""
    odds_path = Path('data/backtest/combined_odds_actuals_2023_2024_2025.csv')

    df = pd.read_csv(odds_path)
    df = df[(df['week'] == week) & (df['season'] == season)]

    return df


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ''
    # Remove suffixes, lowercase, strip
    name = str(name).lower().strip()
    for suffix in [' jr.', ' sr.', ' ii', ' iii', ' iv']:
        name = name.replace(suffix, '')
    return name


def run_walk_forward_backtest(
    start_week: int = 5,
    end_week: int = 13,
    season: int = 2025,
    rolling_window: int = 4
) -> pd.DataFrame:
    """
    Run walk-forward backtest from start_week to end_week.

    No data leakage: each week's predictions only use prior weeks' data.
    """
    all_results = []

    print("=" * 70)
    print("WALK-FORWARD BACKTEST (NO DATA LEAKAGE)")
    print("=" * 70)
    print(f"\nSeason: {season}")
    print(f"Weeks: {start_week} to {end_week}")
    print(f"Rolling window: {rolling_window} weeks")

    for week in range(start_week, end_week + 1):
        print(f"\n--- Week {week} ---")

        # Step 1: Load ONLY historical data (weeks 1 to week-1)
        try:
            historical = load_historical_stats(max_week=week, season=season)
            print(f"  Historical data: weeks 1-{week-1}, {len(historical)} player-games")
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        if len(historical) == 0:
            print(f"  No historical data available for week {week}")
            continue

        # Step 2: Generate predictions using only historical data
        predictions = generate_predictions_for_week(historical, week, rolling_window)
        print(f"  Generated predictions for {len(predictions)} players")

        # Step 3: Load actual props/lines for this week
        props = load_props_for_week(week, season)
        print(f"  Loaded {len(props)} props for week {week}")

        if len(props) == 0:
            print(f"  No props data for week {week}")
            continue

        # Step 4: Match predictions to props
        predictions['player_norm'] = predictions['player'].apply(normalize_player_name)
        props['player_norm'] = props['player'].apply(normalize_player_name)

        # Map market to prediction column
        market_to_pred = {
            'player_receptions': 'pred_receptions',
            'player_reception_yds': 'pred_rec_yards',
            'player_rush_yds': 'pred_rush_yards',
            'player_pass_yds': 'pred_pass_yards',
        }

        # Merge predictions
        merged = props.merge(
            predictions[['player_norm', 'pred_receptions', 'pred_rec_yards',
                        'pred_rush_yards', 'pred_pass_yards', 'games_used']],
            on='player_norm',
            how='left'
        )

        # Set pred_mean based on market
        merged['pred_mean'] = np.nan
        for market, pred_col in market_to_pred.items():
            mask = merged['market'] == market
            if pred_col in merged.columns:
                merged.loc[mask, 'pred_mean'] = merged.loc[mask, pred_col]

        # Calculate divergence
        merged['divergence_pct'] = np.where(
            merged['line'] > 0,
            (merged['pred_mean'] - merged['line']) / merged['line'] * 100,
            np.nan
        )

        # Calculate under hit
        merged['under_hit'] = merged['actual_stat'] < merged['line']

        # Step 5: Store results
        merged['backtest_week'] = week
        merged['prediction_timestamp'] = datetime.now().isoformat()

        all_results.append(merged)

        matched = merged['pred_mean'].notna().sum()
        print(f"  Matched {matched}/{len(merged)} props with predictions")

    if not all_results:
        print("\nNo results generated!")
        return pd.DataFrame()

    # Combine all weeks
    results = pd.concat(all_results, ignore_index=True)

    # Save results
    output_path = Path(f'data/backtest/walk_forward_no_leakage_{season}.csv')
    results.to_csv(output_path, index=False)
    print(f"\nSaved {len(results)} results to: {output_path}")

    return results


def validate_no_leakage(results: pd.DataFrame) -> bool:
    """
    Validate that there's no data leakage.

    Key check: Model predictions should NOT be better correlated with actuals
    than the line is. If model beats line, something is wrong.
    """
    print("\n" + "=" * 70)
    print("VALIDATION: Check for Data Leakage")
    print("=" * 70)

    valid = results.dropna(subset=['pred_mean', 'actual_stat', 'line'])

    if len(valid) < 50:
        print(f"\nInsufficient data for validation ({len(valid)} rows)")
        return False

    pred_actual_corr = np.corrcoef(valid['pred_mean'], valid['actual_stat'])[0, 1]
    line_actual_corr = np.corrcoef(valid['line'], valid['actual_stat'])[0, 1]

    print(f"\nCorrelation checks:")
    print(f"  Model vs Actual: {pred_actual_corr:.3f}")
    print(f"  Line vs Actual:  {line_actual_corr:.3f}")

    if pred_actual_corr > line_actual_corr + 0.05:
        print(f"\n⚠️ WARNING: Model suspiciously better than line!")
        print(f"   Difference: {pred_actual_corr - line_actual_corr:.3f}")
        print(f"   This may indicate data leakage.")
        return False
    else:
        print(f"\n✅ No data leakage detected.")
        print(f"   Line is better predictor than model (expected)")
        return True


def analyze_results(results: pd.DataFrame):
    """Analyze walk-forward backtest results."""
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS ANALYSIS")
    print("=" * 70)

    # Filter to valid predictions
    valid = results.dropna(subset=['pred_mean', 'line', 'actual_stat']).copy()
    print(f"\nValid predictions: {len(valid)}")

    # Focus on receptions (our validated market)
    rec = valid[valid['market'] == 'player_receptions'].copy()
    print(f"Receptions bets: {len(rec)}")

    if len(rec) < 30:
        print("Insufficient receptions data for analysis")
        return

    print("\n--- LINE-LEVEL PERFORMANCE ---")
    for thresh in [3.5, 4.5, 5.5]:
        subset = rec[rec['line'] >= thresh]
        if len(subset) > 20:
            wr = subset['under_hit'].mean()
            roi = (wr * 0.91 - (1 - wr)) * 100
            print(f"Line >= {thresh}: n={len(subset)}, WR={wr:.1%}, ROI={roi:+.1f}%")

    print("\n--- MODEL DIVERGENCE FILTER ---")
    baseline = rec[rec['line'] >= 4.5]
    if len(baseline) > 20:
        base_wr = baseline['under_hit'].mean()
        base_roi = (base_wr * 0.91 - (1 - base_wr)) * 100
        print(f"Baseline (line >= 4.5): n={len(baseline)}, WR={base_wr:.1%}, ROI={base_roi:+.1f}%")

        for thresh in [0, 5, 10, 15, 20]:
            filtered = rec[(rec['line'] >= 4.5) & (rec['divergence_pct'] > thresh)]
            if len(filtered) >= 10:
                wr = filtered['under_hit'].mean()
                roi = (wr * 0.91 - (1 - wr)) * 100
                lift = roi - base_roi
                status = "✅" if lift > 3 else ("⚠️" if lift > 0 else "❌")
                print(f"+ Divergence > {thresh}%: n={len(filtered)}, WR={wr:.1%}, ROI={roi:+.1f}%, lift={lift:+.1f}% {status}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Walk-forward backtest with no data leakage')
    parser.add_argument('--start-week', type=int, default=5, help='Start week')
    parser.add_argument('--end-week', type=int, default=13, help='End week')
    parser.add_argument('--season', type=int, default=2025, help='Season')
    parser.add_argument('--window', type=int, default=4, help='Rolling window size')
    args = parser.parse_args()

    # Run backtest
    results = run_walk_forward_backtest(
        start_week=args.start_week,
        end_week=args.end_week,
        season=args.season,
        rolling_window=args.window
    )

    if len(results) > 0:
        # Validate no leakage
        is_valid = validate_no_leakage(results)

        # Analyze results
        analyze_results(results)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total bets: {len(results)}")
        print(f"Data leakage check: {'PASSED' if is_valid else 'FAILED'}")
