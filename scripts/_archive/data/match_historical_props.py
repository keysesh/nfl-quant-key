#!/usr/bin/env python3
"""
Step 2: Match Historical Props to Outcomes

This script:
1. Loads historical weekly stats from NFLverse
2. Creates synthetic prop lines based on actual outcomes
3. Matches props to actual outcomes (bet_won)
4. Saves matched dataset for simulation

This prepares data for Step 3 (running simulations with PlayerSimulator).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_historical_weekly_stats() -> pd.DataFrame:
    """Load historical weekly stats from NFLverse."""
    data_dir = Path("data/nflverse")
    weekly_file = data_dir / "weekly_historical.parquet"

    if not weekly_file.exists():
        raise FileNotFoundError(
            f"Weekly stats not found at {weekly_file}. "
            "Run fetch_historical_nflverse.py first."
        )

    df = pd.read_parquet(weekly_file)
    logger.info(f"✅ Loaded {len(df):,} player-week records")
    return df


def generate_realistic_prop_lines(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Generate realistic prop lines based on actual player performances.

    Strategy:
    - For each player-week with sufficient volume, create 2-3 prop lines
    - Place lines at strategic points relative to actual outcome
    - This simulates what sportsbooks would have offered
    """

    # Standard line placements for each stat type
    placements = {
        'passing_yards': [199.5, 224.5, 249.5, 274.5, 299.5, 324.5],
        'rushing_yards': [19.5, 34.5, 49.5, 64.5, 79.5, 99.5],
        'receiving_yards': [19.5, 34.5, 49.5, 64.5, 79.5, 99.5],
        'receptions': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    }

    props = []

    for _, row in weekly_stats.iterrows():
        player_name = row['player_display_name']
        player_id = row.get('player_id', '')
        position = row['position']
        season = row['season']
        week = row['week']
        team = row.get('recent_team', '')
        opponent = row.get('opponent_team', '')

        # Map stats to markets
        stat_market_map = {
            'passing_yards': ('player_pass_yds', 'QB'),
            'rushing_yards': ('player_rush_yds', ['QB', 'RB', 'FB', 'WR']),
            'receiving_yards': ('player_reception_yds', ['WR', 'TE', 'RB']),
            'receptions': ('player_receptions', ['WR', 'TE', 'RB'])
        }

        for stat_col, (market, valid_positions) in stat_market_map.items():
            # Check if position is valid for this market
            if isinstance(valid_positions, list):
                if position not in valid_positions:
                    continue
            else:
                if position != valid_positions:
                    continue

            actual_value = row.get(stat_col, np.nan)

            # Skip if no data or zero
            if pd.isna(actual_value) or actual_value == 0:
                continue

            # Generate 2-3 lines per player-week-stat
            # Strategy: Place lines around actual outcome to ensure variety
            relevant_lines = []
            for line_value in placements[stat_col]:
                # Only include lines that are reasonable for this player's outcome
                # Include lines within 2x of actual value
                if abs(line_value - actual_value) <= max(actual_value * 1.5, 50):
                    relevant_lines.append(line_value)

            # Take up to 3 lines around the actual value
            if len(relevant_lines) > 0:
                # Sort by distance from actual
                relevant_lines = sorted(
                    relevant_lines,
                    key=lambda x: abs(x - actual_value)
                )[:3]

                for line_value in relevant_lines:
                    # Determine outcome
                    if actual_value > line_value:
                        over_won = True
                        under_won = False
                    elif actual_value < line_value:
                        over_won = False
                        under_won = True
                    else:
                        # Push - skip these for now
                        continue

                    # Add both Over and Under as separate training examples
                    props.append({
                        'player_name': player_name,
                        'player_id': player_id,
                        'position': position,
                        'season': season,
                        'week': week,
                        'team': team,
                        'opponent': opponent,
                        'market': market,
                        'line': line_value,
                        'pick_type': 'Over',
                        'actual_value': actual_value,
                        'bet_won': over_won,
                        # These will be filled in by simulation step
                        'model_prob_raw': np.nan,
                        'model_projection': np.nan,
                        'model_std': np.nan
                    })

                    props.append({
                        'player_name': player_name,
                        'player_id': player_id,
                        'position': position,
                        'season': season,
                        'week': week,
                        'team': team,
                        'opponent': opponent,
                        'market': market,
                        'line': line_value,
                        'pick_type': 'Under',
                        'actual_value': actual_value,
                        'bet_won': under_won,
                        'model_prob_raw': np.nan,
                        'model_projection': np.nan,
                        'model_std': np.nan
                    })

    df = pd.DataFrame(props)
    logger.info(f"✅ Generated {len(df):,} historical prop betting opportunities")

    return df


def main():
    print("=" * 80)
    print("STEP 2: MATCHING HISTORICAL PROPS TO OUTCOMES")
    print("=" * 80)
    print()

    # Load historical stats
    print("Loading historical weekly stats...")
    weekly_stats = load_historical_weekly_stats()

    print(f"  Seasons: {sorted(weekly_stats['season'].unique())}")
    print(f"  Weeks: {weekly_stats['week'].min()} to {weekly_stats['week'].max()}")
    print(f"  Players: {weekly_stats['player_display_name'].nunique():,}")
    print()

    # Generate prop lines
    print("Generating historical prop lines...")
    historical_props = generate_realistic_prop_lines(weekly_stats)

    print()
    print("📊 Dataset Summary:")
    print(f"  Total props: {len(historical_props):,}")
    print(f"  Over bets: {(historical_props['pick_type'] == 'Over').sum():,}")
    print(f"  Under bets: {(historical_props['pick_type'] == 'Under').sum():,}")
    print(f"  Winning bets: {historical_props['bet_won'].sum():,}")
    print(f"  Win rate: {historical_props['bet_won'].mean():.2%}")
    print()

    print("Markets breakdown:")
    for market in historical_props['market'].unique():
        market_props = historical_props[historical_props['market'] == market]
        print(f"  {market}: {len(market_props):,} props, "
              f"{market_props['bet_won'].mean():.2%} win rate")
    print()

    print("Seasons breakdown:")
    for season in sorted(historical_props['season'].unique()):
        season_props = historical_props[historical_props['season'] == season]
        print(f"  {season}: {len(season_props):,} props, "
              f"{season_props['bet_won'].mean():.2%} win rate")
    print()

    # Save matched props
    output_dir = Path("data/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "historical_props_matched.parquet"

    historical_props.to_parquet(output_file, index=False)
    print(f"✅ Saved matched props to: {output_file}")
    print()

    print("Next steps:")
    print("  1. Run simulations on these historical props (Step 3)")
    print("  2. This will populate model_prob_raw using PlayerSimulator")
    print("  3. Then retrain calibrator with expanded dataset (Step 4)")
    print()


if __name__ == "__main__":
    main()
