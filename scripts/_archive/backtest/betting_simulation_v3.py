"""
Betting Simulation for V3 Model - Historical ROI Analysis

Loads historical odds data and V3 predictions to simulate betting performance
and calculate ROI for weeks 1-8.

Usage:
    python scripts/backtest/betting_simulation_v3.py --weeks 1-8 --min-edge 0.05
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
from datetime import datetime

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.utils.odds import OddsEngine
from nfl_quant.utils.season_utils import get_current_season


# Map dates to weeks
WEEK_DATES = {
    1: "20250909",
    2: "20250916",
    3: "20250923",
    4: "20250930",
    5: "20251007",
    6: "20251014",
    7: "20251021",
    8: "20251028",
}

# Map prop types to stat types
PROP_TYPE_MAP = {
    'player_pass_yds': 'passing_yards',
    'player_rush_yds': 'rushing_yards',
    'player_rec_yds': 'receiving_yards',
}


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def load_historical_props(week: int) -> pd.DataFrame:
    """Load historical player props for a given week."""
    date_str = WEEK_DATES.get(week)
    if not date_str:
        print(f"⚠️  No historical props for Week {week}")
        return pd.DataFrame()

    file_path = Path(f"data/historical/backfill/player_props_history_{date_str}T000000Z.csv")

    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Filter to relevant markets
    df = df[df['market'].isin(PROP_TYPE_MAP.keys())].copy()

    # Add week column
    df['week'] = week

    return df


def load_sleeper_actuals(week: int) -> pd.DataFrame:
    """Load actual player stats from NFLverse for a given week."""
    file_path = Path("data/nflverse_cache/stats_player_week_2025.csv")

    if not file_path.exists():
        print(f"⚠️  No NFLverse stats found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    df = df[df['week'] == week].copy()  # Filter to specific week

    # Standardize column names (NFLverse already uses standard names)
    df = df.rename(columns={
        'rec_yd': 'receiving_yards',
        'rush_yd': 'rushing_yards',
        'pass_yd': 'passing_yards',
    })

    return df


def generate_v3_predictions(week: int, props: pd.DataFrame, simulator: PlayerSimulator) -> pd.DataFrame:
    """Generate V3 predictions for all players with props."""
    predictions = []

    # Get unique players
    unique_players = props[['player', 'market']].drop_duplicates()

    print(f"   Generating predictions for {len(unique_players)} player-market combinations...")

    for _, row in unique_players.iterrows():
        player_name = row['player']
        market = row['market']
        stat_type = PROP_TYPE_MAP[market]

        # Create PlayerPropInput (simplified - in reality you'd need more context)
        # For backtest, we'll use defaults since we just need the distribution shape
        player_input = PlayerPropInput(
            player_id=player_name,
            player_name=player_name,
            position='QB' if 'pass' in market else 'RB' if 'rush' in market else 'WR',
            team='UNK',
            opponent='UNK',
            game_id=f"week_{week}",
            week=week,
            season=get_current_season(),
            is_home=True,
            trailing_stats={
                'attempts_per_game': 35.0 if 'pass' in market else 15.0,
                'completions_per_game': 23.0,
                'pass_yds_per_game': 250.0 if 'pass' in market else 0,
                'completion_pct': 0.65,
                'yards_per_completion': 11.0,
                'pass_td_per_game': 1.5,
            }
        )

        try:
            result = simulator.simulate_player(player_input)

            if stat_type in result:
                samples = result[stat_type]
                predictions.append({
                    'player': player_name,
                    'market': market,
                    'stat_type': stat_type,
                    'week': week,
                    'samples': samples,
                    'mean': np.mean(samples),
                    'std': np.std(samples),
                    'p5': np.percentile(samples, 5),
                    'p50': np.percentile(samples, 50),
                    'p95': np.percentile(samples, 95),
                })
        except Exception as e:
            print(f"   ⚠️  Failed to predict {player_name} {market}: {e}")
            continue

    return pd.DataFrame(predictions)


def calculate_edge_and_bet(row: pd.Series) -> Dict:
    """
    Calculate betting edge and make bet decision.

    Args:
        row: Contains prediction samples, line, odds, and actual outcome

    Returns:
        Dictionary with bet info and results
    """
    samples = row['samples']
    line = row['line']
    over_odds = row['over_odds']
    under_odds = row['under_odds']
    actual = row['actual']

    # Calculate model probabilities
    over_prob_model = np.mean(samples > line)
    under_prob_model = np.mean(samples < line)

    # Calculate implied probabilities from odds (with vig)
    over_prob_implied = OddsEngine.american_to_implied_prob(over_odds)
    under_prob_implied = OddsEngine.american_to_implied_prob(under_odds)

    # Calculate edge (model prob - implied prob)
    over_edge = over_prob_model - over_prob_implied
    under_edge = under_prob_model - under_prob_implied

    # Determine best bet
    if over_edge > under_edge and over_edge > 0:
        bet_side = 'over'
        edge = over_edge
        bet_prob = over_prob_model
        odds = over_odds
        won = actual > line if pd.notna(actual) else None
    elif under_edge > 0:
        bet_side = 'under'
        edge = under_edge
        bet_prob = under_prob_model
        odds = under_odds
        won = actual < line if pd.notna(actual) else None
    else:
        bet_side = None
        edge = 0
        bet_prob = 0
        odds = 0
        won = None

    return {
        'bet_side': bet_side,
        'edge': edge,
        'model_prob': bet_prob,
        'implied_prob': over_prob_implied if bet_side == 'over' else under_prob_implied,
        'odds': odds,
        'won': won,
    }


def simulate_betting(predictions: pd.DataFrame, props: pd.DataFrame, actuals: pd.DataFrame,
                     min_edge: float = 0.05, unit_size: float = 100.0) -> pd.DataFrame:
    """
    Simulate betting based on predictions vs odds.

    Args:
        predictions: V3 model predictions with samples
        props: Historical odds data
        actuals: Actual player performance
        min_edge: Minimum edge required to place bet (default 5%)
        unit_size: Bet size in dollars (default $100)

    Returns:
        DataFrame with all bets and results
    """
    bets = []

    # For each prop, find matching prediction and actual
    for _, prop in props.iterrows():
        player = prop['player']
        market = prop['market']
        stat_type = PROP_TYPE_MAP[market]
        week = prop['week']

        # Skip if not over/under market
        if prop['prop_type'] not in ['over', 'under']:
            continue

        # Get line
        line = prop['line']
        if pd.isna(line):
            continue

        # Find matching prediction
        pred_match = predictions[
            (predictions['player'] == player) &
            (predictions['market'] == market) &
            (predictions['week'] == week)
        ]

        if len(pred_match) == 0:
            continue

        pred = pred_match.iloc[0]

        # Find actual outcome
        actual_match = actuals[
            (actuals['player_name'] == player) &
            (actuals['week'] == week)
        ]

        actual_value = None
        if len(actual_match) > 0:
            actual_value = actual_match.iloc[0].get(stat_type)

        # Get over/under odds
        over_prop = props[
            (props['player'] == player) &
            (props['market'] == market) &
            (props['week'] == week) &
            (props['prop_type'] == 'over')
        ]
        under_prop = props[
            (props['player'] == player) &
            (props['market'] == market) &
            (props['week'] == week) &
            (props['prop_type'] == 'under')
        ]

        if len(over_prop) == 0 or len(under_prop) == 0:
            continue

        over_odds = over_prop.iloc[0]['price']
        under_odds = under_prop.iloc[0]['price']

        if pd.isna(over_odds) or pd.isna(under_odds):
            continue

        # Calculate edge and make bet decision
        bet_info = calculate_edge_and_bet(pd.Series({
            'samples': pred['samples'],
            'line': line,
            'over_odds': over_odds,
            'under_odds': under_odds,
            'actual': actual_value,
        }))

        # Only bet if edge > min_edge
        if bet_info['edge'] < min_edge or bet_info['bet_side'] is None:
            continue

        # Calculate payout
        decimal_odds = american_to_decimal(bet_info['odds'])
        potential_payout = unit_size * decimal_odds
        profit = potential_payout - unit_size if bet_info['won'] else -unit_size

        bets.append({
            'week': week,
            'player': player,
            'market': market,
            'line': line,
            'bet_side': bet_info['bet_side'],
            'edge': bet_info['edge'],
            'model_prob': bet_info['model_prob'],
            'implied_prob': bet_info['implied_prob'],
            'odds': bet_info['odds'],
            'decimal_odds': decimal_odds,
            'unit_size': unit_size,
            'actual': actual_value,
            'won': bet_info['won'],
            'profit': profit if bet_info['won'] is not None else None,
        })

    return pd.DataFrame(bets)


def calculate_roi_metrics(bets: pd.DataFrame) -> Dict:
    """Calculate comprehensive ROI and betting metrics."""
    # Filter to bets with known outcomes
    completed_bets = bets[bets['won'].notna()].copy()

    if len(completed_bets) == 0:
        return {'error': 'No completed bets'}

    total_wagered = completed_bets['unit_size'].sum()
    total_profit = completed_bets['profit'].sum()
    roi = (total_profit / total_wagered) * 100

    wins = completed_bets[completed_bets['won'] == True]
    losses = completed_bets[completed_bets['won'] == False]

    win_rate = len(wins) / len(completed_bets) * 100
    avg_win = wins['profit'].mean() if len(wins) > 0 else 0
    avg_loss = losses['profit'].mean() if len(losses) > 0 else 0

    # By market
    by_market = completed_bets.groupby('market').agg({
        'won': ['count', 'sum'],
        'profit': 'sum',
        'unit_size': 'sum',
    }).round(2)

    # By edge bucket
    completed_bets['edge_bucket'] = pd.cut(
        completed_bets['edge'],
        bins=[0, 0.05, 0.10, 0.15, 0.20, 1.0],
        labels=['5-10%', '10-15%', '15-20%', '20%+', 'Error']
    )

    by_edge = completed_bets.groupby('edge_bucket').agg({
        'won': ['count', 'sum'],
        'profit': 'sum',
    }).round(2)

    return {
        'total_bets': len(completed_bets),
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate,
        'wins': len(wins),
        'losses': len(losses),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'by_market': by_market,
        'by_edge': by_edge,
    }


def main():
    parser = argparse.ArgumentParser(description='V3 Betting Simulation')
    parser.add_argument('--weeks', default='1-8', help='Weeks to backtest (e.g., 1-8)')
    parser.add_argument('--min-edge', type=float, default=0.05, help='Minimum edge to bet (default 5%)')
    parser.add_argument('--unit-size', type=float, default=100.0, help='Bet unit size (default $100)')

    args = parser.parse_args()

    # Parse weeks
    if '-' in args.weeks:
        start, end = map(int, args.weeks.split('-'))
        weeks = list(range(start, end + 1))
    else:
        weeks = [int(args.weeks)]

    print("="*80)
    print("V3 BETTING SIMULATION: WEEKS", args.weeks)
    print("="*80)
    print(f"\nSettings:")
    print(f"  Min Edge: {args.min_edge * 100:.1f}%")
    print(f"  Unit Size: ${args.unit_size:.0f}")
    print()

    # Load predictors
    print("📥 Loading V3 predictors...")
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(usage_predictor, efficiency_predictor, trials=50000, seed=42)
    print("✅ V3 simulator ready\n")

    all_bets = []

    for week in weeks:
        print(f"{'='*80}")
        print(f"WEEK {week}")
        print(f"{'='*80}\n")

        # Load historical props
        print(f"📥 Loading historical props for Week {week}...")
        props = load_historical_props(week)
        if props.empty:
            continue
        print(f"   ✅ Loaded {len(props)} props\n")

        # Load actuals
        print(f"📥 Loading actuals for Week {week}...")
        actuals = load_sleeper_actuals(week)
        if actuals.empty:
            print(f"   ⚠️  No actuals available\n")
        else:
            print(f"   ✅ Loaded {len(actuals)} player performances\n")

        # Generate predictions
        print(f"📊 Generating V3 predictions for Week {week}...")
        predictions = generate_v3_predictions(week, props, simulator)
        print(f"   ✅ Generated {len(predictions)} predictions\n")

        # Simulate betting
        print(f"💰 Simulating bets for Week {week}...")
        week_bets = simulate_betting(predictions, props, actuals,
                                     min_edge=args.min_edge, unit_size=args.unit_size)
        print(f"   ✅ Placed {len(week_bets)} bets with edge >= {args.min_edge * 100:.0f}%\n")

        if len(week_bets) > 0:
            all_bets.append(week_bets)

    # Aggregate results
    if len(all_bets) == 0:
        print("\n❌ No bets placed")
        return

    all_bets_df = pd.concat(all_bets, ignore_index=True)

    print("\n" + "="*80)
    print("AGGREGATE RESULTS")
    print("="*80 + "\n")

    metrics = calculate_roi_metrics(all_bets_df)

    if 'error' in metrics:
        print(f"❌ {metrics['error']}")
        return

    print(f"📊 Overall Performance:")
    print(f"   Total Bets: {metrics['total_bets']}")
    print(f"   Total Wagered: ${metrics['total_wagered']:,.0f}")
    print(f"   Total Profit: ${metrics['total_profit']:,.2f}")
    print(f"   ROI: {metrics['roi']:.2f}%")
    print(f"   Win Rate: {metrics['win_rate']:.1f}%")
    print(f"   Wins: {metrics['wins']}")
    print(f"   Losses: {metrics['losses']}")
    print(f"   Avg Win: ${metrics['avg_win']:.2f}")
    print(f"   Avg Loss: ${metrics['avg_loss']:.2f}")

    print(f"\n📊 By Market:")
    print(metrics['by_market'])

    print(f"\n📊 By Edge Bucket:")
    print(metrics['by_edge'])

    # Save results
    output_path = Path("reports/betting_simulation_v3_results.csv")
    all_bets_df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
