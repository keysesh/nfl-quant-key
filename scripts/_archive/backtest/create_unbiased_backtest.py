#!/usr/bin/env python3
"""
Create Unbiased Backtesting Dataset
====================================

Problem: Consolidated bets only contain HIGH EDGE bets that were placed.
This creates selection bias - we're only testing bets that already passed our filter.

Solution: Generate synthetic prop lines for ALL players using NFLverse actual stats.
- Get all player-week stats from NFLverse (passing_yards, rushing_yards, etc.)
- Generate realistic prop lines based on historical averages
- Test model predictions on ENTIRE universe, not just high-edge picks

This provides true unbiased out-of-sample testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
REPORTS_DIR = Path(__file__).parent.parent.parent / 'reports'


def load_nflverse_weekly_stats():
    """Load NFLverse weekly player stats."""
    stats_file = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'

    if not stats_file.exists():
        raise FileNotFoundError(f"NFLverse stats not found: {stats_file}")

    df = pd.read_parquet(stats_file)

    # Filter to 2025 season, regular season weeks
    df = df[(df['season'] == 2025) & (df['week'] <= 11)].copy()

    print(f"Loaded {len(df)} player-week records")
    print(f"Weeks: {sorted(df['week'].unique())}")

    return df


def generate_synthetic_prop_lines(stats_df):
    """
    Generate realistic prop lines for all players based on their performance.

    For each player-week-stat, we:
    1. Calculate their season average up to that point
    2. Add market adjustment (slightly under actual to favor books)
    3. Create over/under at standard -110 odds
    """
    print("\nGenerating synthetic prop lines...")

    all_props = []

    # Stats we care about
    stat_mappings = {
        'passing_yards': 'passing_yards',
        'rushing_yards': 'rushing_yards',
        'receiving_yards': 'receiving_yards',
        'receptions': 'receptions'
    }

    # Process each week
    for week in sorted(stats_df['week'].unique()):
        if week < 3:  # Need at least 2 weeks of history
            continue

        week_stats = stats_df[stats_df['week'] == week]
        prior_stats = stats_df[stats_df['week'] < week]

        print(f"  Week {week}: {len(week_stats)} players")

        for _, player_row in week_stats.iterrows():
            player_name = player_row['player_display_name']
            team = player_row['team']
            opponent = player_row['opponent_team']

            # Get player's historical average
            player_history = prior_stats[
                prior_stats['player_display_name'] == player_name
            ]

            if len(player_history) < 2:
                continue  # Need history to set line

            # Generate props for each stat type
            for stat_name, col_name in stat_mappings.items():
                if col_name not in player_row.index:
                    continue

                actual_value = player_row[col_name]

                # Skip if no meaningful stat (0 or NaN)
                if pd.isna(actual_value) or actual_value == 0:
                    # Only skip if not a relevant position
                    if stat_name == 'passing_yards' and player_row.get('position') != 'QB':
                        continue
                    if stat_name in ['rushing_yards'] and player_row.get('position') not in ['RB', 'QB']:
                        continue
                    if stat_name in ['receiving_yards', 'receptions'] and player_row.get('position') not in ['WR', 'TE', 'RB']:
                        continue

                # Calculate historical average
                hist_avg = player_history[col_name].mean()
                hist_std = player_history[col_name].std()

                if pd.isna(hist_avg) or hist_avg < 1:
                    continue

                # Set line slightly below average (books are smart)
                # Round to .5 for realism
                line = round(hist_avg * 0.95 * 2) / 2

                if line < 0.5:
                    continue

                # Standard -110 odds
                over_odds = -110
                under_odds = -110

                # Calculate went_over
                went_over = 1 if actual_value > line else 0

                # Generate raw probability prediction using our model logic
                # P(over) = P(actual > line) using normal distribution
                if hist_std > 0:
                    z_score = (line - hist_avg) / hist_std
                    prob_over_raw = 1 - norm.cdf(z_score)
                else:
                    prob_over_raw = 0.5

                # Cap extreme probabilities
                prob_over_raw = np.clip(prob_over_raw, 0.05, 0.95)

                prop = {
                    'player': player_name,
                    'team': team,
                    'opponent': opponent,
                    'week': week,
                    'stat_type': stat_name,
                    'line': line,
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'actual_value': actual_value,
                    'went_over': went_over,
                    'prob_over_raw': prob_over_raw,
                    'hist_avg': hist_avg,
                    'hist_std': hist_std,
                    'position': player_row.get('position', 'UNK')
                }
                all_props.append(prop)

    props_df = pd.DataFrame(all_props)
    print(f"\nGenerated {len(props_df)} synthetic props")

    return props_df


def add_market_probabilities(props_df):
    """Add implied market probabilities from odds."""
    # At -110 odds, implied prob = 110/210 = 0.524
    # But we need to account for vig
    # True prob closer to 0.50
    props_df['market_prob_over'] = 0.5238  # Standard -110 implied
    props_df['market_prob_under'] = 0.5238

    return props_df


def validate_unbiased_data(props_df):
    """Validate the synthetic data is unbiased."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # Overall over rate should be ~50%
    overall_over_rate = props_df['went_over'].mean()
    print(f"Overall OVER rate: {overall_over_rate:.1%}")
    if 0.45 <= overall_over_rate <= 0.55:
        print("  ✅ Close to 50% - unbiased")
    else:
        print("  ⚠️ Biased - adjust line generation")

    # Check by stat type
    print("\nBy stat type:")
    for stat in props_df['stat_type'].unique():
        stat_data = props_df[props_df['stat_type'] == stat]
        over_rate = stat_data['went_over'].mean()
        print(f"  {stat}: {over_rate:.1%} OVER rate ({len(stat_data)} props)")

    # Check by week
    print("\nBy week:")
    for week in sorted(props_df['week'].unique()):
        week_data = props_df[props_df['week'] == week]
        over_rate = week_data['went_over'].mean()
        print(f"  Week {week}: {over_rate:.1%} OVER rate ({len(week_data)} props)")

    # Check probability distribution
    print(f"\nRaw probability distribution:")
    print(f"  Mean: {props_df['prob_over_raw'].mean():.3f}")
    print(f"  Std: {props_df['prob_over_raw'].std():.3f}")
    print(f"  Min: {props_df['prob_over_raw'].min():.3f}")
    print(f"  Max: {props_df['prob_over_raw'].max():.3f}")

    return props_df


def main():
    print("=" * 60)
    print("CREATING UNBIASED BACKTESTING DATASET")
    print("=" * 60)

    # Load NFLverse stats
    stats_df = load_nflverse_weekly_stats()

    # Generate synthetic props for ALL players
    props_df = generate_synthetic_prop_lines(stats_df)

    # Add market probabilities
    props_df = add_market_probabilities(props_df)

    # Validate unbiasedness
    props_df = validate_unbiased_data(props_df)

    # Save
    output_file = DATA_DIR / 'backtest' / 'unbiased_props_backtest.csv'
    DATA_DIR.joinpath('backtest').mkdir(exist_ok=True)
    props_df.to_csv(output_file, index=False)

    print(f"\n✅ Saved unbiased backtest data to: {output_file}")
    print(f"   Total props: {len(props_df)}")
    print(f"   Weeks: {sorted(props_df['week'].unique())}")
    print(f"   Props per week: ~{len(props_df) // len(props_df['week'].unique())}")

    return props_df


if __name__ == "__main__":
    props_df = main()
