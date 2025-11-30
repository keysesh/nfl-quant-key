"""
Betting ROI Analysis - Calculate historical ROI from backtest predictions + odds

Loads V3 backtest predictions and historical odds to calculate betting ROI.
Much simpler than betting_simulation_v3.py - just analyzes existing predictions.

Usage:
    python scripts/backtest/betting_roi_analysis.py --min-edge 0.05
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict
import argparse

from nfl_quant.utils.odds import OddsEngine
from nfl_quant.calibration.calibrator_loader import load_calibrator_for_market


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
    'player_reception_yds': 'receiving_yards',  # Fixed: was player_rec_yds
}

STAT_TO_PROP_MAP = {v: k for k, v in PROP_TYPE_MAP.items()}


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
        return pd.DataFrame()

    file_path = Path(f"data/historical/backfill/player_props_history_{date_str}T000000Z.csv")

    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    # Filter to relevant markets
    df = df[df['market'].isin(PROP_TYPE_MAP.keys())].copy()
    df['week'] = week

    # Parse into over/under pairs
    lines = []
    for (player, market), group in df.groupby(['player', 'market']):
        over_row = group[group['prop_type'] == 'over']
        under_row = group[group['prop_type'] == 'under']

        if len(over_row) == 0 or len(under_row) == 0:
            continue

        over_line = over_row.iloc[0]['line']
        under_line = under_row.iloc[0]['line']

        if pd.isna(over_line) or pd.isna(under_line) or over_line != under_line:
            continue

        lines.append({
            'week': week,
            'player': player,
            'market': market,
            'stat_type': PROP_TYPE_MAP[market],
            'line': over_line,
            'over_odds': over_row.iloc[0]['price'],
            'under_odds': under_row.iloc[0]['price'],
        })

    return pd.DataFrame(lines)


def load_sleeper_actuals(week: int) -> pd.DataFrame:
    """Load actual player stats from NFLverse for a given week."""
    file_path = Path("data/nflverse_cache/stats_player_week_2025.csv")

    if not file_path.exists():
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


def analyze_betting_roi(predictions_file: str, min_edge: float = 0.05, unit_size: float = 100.0):
    """
    Analyze betting ROI from saved V3 backtest predictions.

    Args:
        predictions_file: Path to V3 backtest predictions CSV
        min_edge: Minimum edge to bet (default 5%)
        unit_size: Bet size (default $100)
    """
    print("="*80)
    print("V3 BETTING ROI ANALYSIS")
    print("="*80)
    print(f"\nSettings:")
    print(f"  Min Edge: {min_edge * 100:.1f}%")
    print(f"  Unit Size: ${unit_size:.0f}")
    print(f"  Predictions File: {predictions_file}\n")

    # Load all V3 predictions from backtest
    print("📥 Loading V3 backtest predictions...")
    if not Path(predictions_file).exists():
        print(f"❌ File not found: {predictions_file}")
        print("Run the backtest first to generate predictions!")
        return

    predictions_df = pd.read_csv(predictions_file)
    print(f"✅ Loaded {len(predictions_df)} predictions\n")

    # Parse samples column back to arrays
    print("📊 Parsing prediction samples...")
    predictions_df['samples_array'] = predictions_df['samples'].apply(
        lambda x: np.array([float(v) for v in x.split(',')]) if pd.notna(x) else np.array([])
    )
    print(f"✅ Parsed samples for {len(predictions_df)} predictions\n")

    # Load market-specific calibrators
    print("📊 Loading calibrators...")
    calibrators = {}
    for market in PROP_TYPE_MAP.keys():
        try:
            calibrators[market] = load_calibrator_for_market(market)
            print(f"   ✅ Loaded calibrator for {market}")
        except Exception as e:
            print(f"   ⚠️  No calibrator for {market}: {e}")
            calibrators[market] = None
    print()

    all_bets = []
    total_props_found = 0
    total_matched = 0

    for week in range(1, 9):
        print(f"Week {week}:")

        # Load props
        props = load_historical_props(week)
        if props.empty:
            print(f"  No props found")
            continue

        total_props_found += len(props)

        # Load actuals
        actuals = load_sleeper_actuals(week)
        if actuals.empty:
            print(f"  No actuals found")
            continue

        # Filter predictions for this week
        week_preds = predictions_df[predictions_df['week'] == week].copy()

        # Match props to predictions and actuals
        for _, prop in props.iterrows():
            # Find matching prediction
            pred_match = week_preds[
                (week_preds['player_name'] == prop['player']) &
                (week_preds['stat_type'] == prop['stat_type'])
            ]

            if len(pred_match) == 0:
                continue

            # Find matching actual
            actual_match = actuals[actuals['player_name'] == prop['player']]
            if len(actual_match) == 0:
                continue

            actual_value = actual_match.iloc[0].get(prop['stat_type'])
            if pd.isna(actual_value):
                continue

            # Get prediction samples
            pred = pred_match.iloc[0]
            samples = pred['samples_array']

            if len(samples) == 0:
                continue

            # Calculate RAW model probabilities from samples
            line = prop['line']
            over_prob_raw = np.mean(samples > line)
            under_prob_raw = np.mean(samples < line)

            # Apply calibration if available
            calibrator = calibrators.get(prop['market'])
            if calibrator:
                over_prob_model = calibrator.transform(over_prob_raw)
                under_prob_model = calibrator.transform(under_prob_raw)
            else:
                over_prob_model = over_prob_raw
                under_prob_model = under_prob_raw

            # Calculate implied probabilities from odds
            over_prob_implied = 1 / american_to_decimal(prop['over_odds'])
            under_prob_implied = 1 / american_to_decimal(prop['under_odds'])

            # Calculate edges (using calibrated probabilities)
            over_edge = over_prob_model - over_prob_implied
            under_edge = under_prob_model - under_prob_implied

            # Determine best bet
            best_side = None
            best_edge = 0
            best_prob_model = 0
            best_odds = 0

            if over_edge > min_edge:
                best_side = 'over'
                best_edge = over_edge
                best_prob_model = over_prob_model
                best_odds = prop['over_odds']
            elif under_edge > min_edge:
                best_side = 'under'
                best_edge = under_edge
                best_prob_model = under_prob_model
                best_odds = prop['under_odds']

            if best_side:
                # Calculate result
                if best_side == 'over':
                    won = actual_value > line
                else:
                    won = actual_value < line

                # Calculate profit/loss
                decimal_odds = american_to_decimal(best_odds)
                if won:
                    profit = unit_size * (decimal_odds - 1)
                else:
                    profit = -unit_size

                total_matched += 1

                all_bets.append({
                    'week': week,
                    'player': prop['player'],
                    'market': prop['market'],
                    'stat_type': prop['stat_type'],
                    'line': line,
                    'side': best_side,
                    'odds': best_odds,
                    'prob_model': best_prob_model,
                    'prob_implied': over_prob_implied if best_side == 'over' else under_prob_implied,
                    'edge': best_edge,
                    'actual': actual_value,
                    'won': won,
                    'profit': profit,
                })

        print(f"  Props: {len(props)}, Matched: {len([b for b in all_bets if b['week'] == week])}")

    print(f"\n✅ Matched {total_matched} of {total_props_found} props with predictions\n")

    if not all_bets:
        print("❌ No bets met the minimum edge criteria")
        return

    # Calculate ROI metrics
    bets_df = pd.DataFrame(all_bets)

    print("="*80)
    print("BETTING ROI RESULTS")
    print("="*80)
    print()

    total_bets = len(bets_df)
    wins = bets_df['won'].sum()
    losses = total_bets - wins
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0

    total_wagered = total_bets * unit_size
    total_profit = bets_df['profit'].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    print(f"📊 Overall Results:")
    print(f"   Total Bets: {total_bets}")
    print(f"   Wins: {wins} ({win_rate:.1f}%)")
    print(f"   Losses: {losses}")
    print(f"   Total Wagered: ${total_wagered:.2f}")
    print(f"   Total Profit: ${total_profit:+.2f}")
    print(f"   ROI: {roi:+.1f}%\n")

    # Breakdown by stat type
    print("📊 By Market:")
    for stat_type in ['passing_yards', 'rushing_yards', 'receiving_yards']:
        type_bets = bets_df[bets_df['stat_type'] == stat_type]
        if len(type_bets) == 0:
            continue

        type_wins = type_bets['won'].sum()
        type_total = len(type_bets)
        type_win_rate = (type_wins / type_total * 100) if type_total > 0 else 0
        type_profit = type_bets['profit'].sum()
        type_wagered = type_total * unit_size
        type_roi = (type_profit / type_wagered * 100) if type_wagered > 0 else 0

        print(f"   {stat_type}:")
        print(f"      Bets: {type_total}, Wins: {type_wins} ({type_win_rate:.1f}%)")
        print(f"      Profit: ${type_profit:+.2f}, ROI: {type_roi:+.1f}%")

    # Breakdown by week
    print(f"\n📊 By Week:")
    for week in sorted(bets_df['week'].unique()):
        week_bets = bets_df[bets_df['week'] == week]
        week_wins = week_bets['won'].sum()
        week_total = len(week_bets)
        week_win_rate = (week_wins / week_total * 100) if week_total > 0 else 0
        week_profit = week_bets['profit'].sum()

        print(f"   Week {week}: {week_total} bets, {week_wins} wins ({week_win_rate:.1f}%), ${week_profit:+.2f}")

    # Save detailed results
    output_csv = Path("reports/betting_roi_detailed.csv")
    bets_df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed results saved to: {output_csv}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='V3 Betting ROI Analysis')
    parser.add_argument('--predictions', default='reports/v3_backtest_predictions.csv',
                       help='Path to V3 backtest predictions CSV')
    parser.add_argument('--min-edge', type=float, default=0.05, help='Minimum edge (default 5%)')
    parser.add_argument('--unit-size', type=float, default=100.0, help='Bet size (default $100)')

    args = parser.parse_args()

    analyze_betting_roi(args.predictions, args.min_edge, args.unit_size)


if __name__ == '__main__':
    main()
