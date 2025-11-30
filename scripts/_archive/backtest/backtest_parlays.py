#!/usr/bin/env python3
"""
Backtest Parlay Recommendations
Tests parlay logic on historical data to validate theoretical edge matches actual ROI.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "nfl_quant" / "parlay"))

from nfl_quant.parlay.odds_calculator import ParlayOddsCalculator
from nfl_quant.parlay.correlation import CorrelationChecker, ParlayLeg
from nfl_quant.parlay.recommendation import ParlayRecommender, SingleBet


def load_backtest_results(backtest_csv: str = "reports/BACKTEST_WEEKS_1_8_VALIDATION.csv") -> pd.DataFrame:
    """Load single bet backtest results."""
    print(f"\n📂 Loading backtest results from {backtest_csv}...")

    if not Path(backtest_csv).exists():
        print(f"❌ File not found: {backtest_csv}")
        return pd.DataFrame()

    df = pd.read_csv(backtest_csv)
    print(f"✅ Loaded {len(df)} backtest records")

    # Add 'hit' column from 'bet_won' for consistency
    if 'bet_won' in df.columns and 'hit' not in df.columns:
        df['hit'] = df['bet_won']

    return df


def convert_to_single_bets(df: pd.DataFrame) -> List[SingleBet]:
    """Convert backtest dataframe to SingleBet objects."""
    bets = []

    for _, row in df.iterrows():
        # Only include positive edge bets
        if row.get('filter_status') != 'positive_edge':
            continue

        # Extract game string
        game = f"{row['team']} Game"  # Simplified for backtest

        bet = SingleBet(
            name=f"{row['player']} {row['market']} {row['prop_type']} {row['line']}",
            bet_type='Player Prop',
            game=game,
            team=row['team'],
            player=row['player'],
            market=row['market'],
            odds=int(row['american_price']),
            our_prob=float(row['model_prob']),
            market_prob=float(row['implied_prob']),
            edge=float(row['edge']),
            bet_size=None,  # Will be calculated
        )

        bets.append(bet)

    print(f"✅ Converted {len(bets)} positive-edge bets to SingleBet objects")
    return bets


def load_actual_results(backtest_csv: str) -> Dict[str, bool]:
    """Load actual hit/miss results for each bet."""
    df = pd.read_csv(backtest_csv)

    # Create a lookup key for each bet
    results = {}

    for _, row in df.iterrows():
        if 'hit' in row:
            key = f"{row['player']}_{row['market']}_{row['prop_type']}_{row['line']}_{row['week']}"
            results[key] = bool(row['hit'])

    return results


def generate_parlay_combinations_by_week(
    df: pd.DataFrame,
    max_legs: int = 4
) -> List[Dict]:
    """Generate parlay combinations grouped by week."""
    print(f"\n🎲 Generating parlay combinations (max {max_legs} legs)...")

    all_parlays = []

    # Filter to only positive edge bets (edge > 0)
    df = df[df['edge'] > 0].copy()

    # Add filter_status for compatibility
    df['filter_status'] = 'positive_edge'

    # Group by week
    weeks = df['week'].unique()

    for week in sorted(weeks):
        week_df = df[df['week'] == week].copy()

        if len(week_df) < 2:
            print(f"  Week {week}: Insufficient bets ({len(week_df)}), skipping")
            continue

        # LIMIT: Take only top 15 bets by edge to avoid combinatorial explosion
        week_df = week_df.nlargest(15, 'edge')

        print(f"\n  Week {week}: Using top 15 of {len(df[df['week'] == week])} positive-edge bets")

        # Convert to SingleBet objects
        bets = []
        for _, row in week_df.iterrows():
            # Convert odds from probability (need to derive from market_prob)
            # Using -110 as default (standard odds)
            odds = -110  # Simplified for backtest

            bet = SingleBet(
                name=f"{row['player']} {row['pick']}",
                bet_type='Player Prop',
                game=f"{row['player']} Week {week}",  # Group by player+week to avoid same-game issues
                team=None,  # Not in this dataset
                player=row['player'],
                market=row['market'],
                odds=odds,
                our_prob=float(row['model_prob']),
                market_prob=float(row['market_prob']),
                edge=float(row['edge']),
                bet_size=None,
            )
            bets.append(bet)

        # Generate parlays using the recommender
        recommender = ParlayRecommender(
            correlation_threshold=0.70,
            max_legs=max_legs,
            min_confidence=0.52,
            min_edge=0.02
        )

        # Get top parlays
        parlays = recommender.generate_parlays(bets, num_parlays=20)

        if not parlays:
            print(f"    ⚠️  No valid parlays generated (correlation blocks)")
            continue

        print(f"    ✅ Generated {len(parlays)} parlay combinations")

        # Store parlays with week info
        for parlay in parlays:
            # Add actual results for each leg
            leg_results = []
            all_hit = True

            for leg in parlay.legs:
                # Try to find actual result from dataframe
                # Match by player and pick (which contains market + direction + line)
                leg_row = week_df[
                    (week_df['player'] == leg.player) &
                    (week_df['market'] == leg.market) &
                    (week_df['pick'] == leg.name.split(maxsplit=1)[1] if ' ' in leg.name else leg.name)
                ]

                if not leg_row.empty and 'hit' in leg_row.columns:
                    hit = bool(leg_row.iloc[0]['hit'])
                    leg_results.append(hit)
                    if not hit:
                        all_hit = False
                else:
                    # Try simpler match - just by player and market
                    leg_row = week_df[
                        (week_df['player'] == leg.player) &
                        (week_df['market'] == leg.market)
                    ]

                    if not leg_row.empty and 'hit' in leg_row.columns:
                        # Take first match
                        hit = bool(leg_row.iloc[0]['hit'])
                        leg_results.append(hit)
                        if not hit:
                            all_hit = False
                    else:
                        # Missing result, skip this parlay
                        all_hit = None
                        break

            if all_hit is None:
                continue  # Skip parlays with missing results

            parlay_result = {
                'week': week,
                'num_legs': len(parlay.legs),
                'legs': [leg.name for leg in parlay.legs],
                'true_odds': parlay.true_odds,
                'model_odds': parlay.model_odds,
                'true_prob': parlay.true_prob,
                'model_prob': parlay.model_prob,
                'edge': parlay.edge,
                'recommended_stake': parlay.recommended_stake,
                'potential_win': parlay.potential_win,
                'expected_value': parlay.expected_value,
                'leg_results': leg_results,
                'parlay_hit': all_hit,
                'actual_profit': parlay.potential_win if all_hit else -parlay.recommended_stake,
            }

            all_parlays.append(parlay_result)

    print(f"\n✅ Total parlays with results: {len(all_parlays)}")
    return all_parlays


def calculate_parlay_metrics(parlays: List[Dict]) -> Dict:
    """Calculate comprehensive metrics for parlay performance."""
    if not parlays:
        return {}

    df = pd.DataFrame(parlays)

    # Overall metrics
    total_parlays = len(df)
    total_staked = df['recommended_stake'].sum()
    total_profit = df['actual_profit'].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    # Hit rate
    hit_rate = df['parlay_hit'].sum() / total_parlays

    # Expected metrics
    expected_profit = df['expected_value'].sum()
    expected_roi = (expected_profit / total_staked * 100) if total_staked > 0 else 0

    # Metrics by number of legs
    by_legs = {}
    for num_legs in sorted(df['num_legs'].unique()):
        leg_df = df[df['num_legs'] == num_legs]
        by_legs[num_legs] = {
            'count': len(leg_df),
            'hit_rate': leg_df['parlay_hit'].sum() / len(leg_df),
            'avg_edge': leg_df['edge'].mean(),
            'avg_model_prob': leg_df['model_prob'].mean(),
            'avg_true_prob': leg_df['true_prob'].mean(),
            'total_staked': leg_df['recommended_stake'].sum(),
            'total_profit': leg_df['actual_profit'].sum(),
            'roi': (leg_df['actual_profit'].sum() / leg_df['recommended_stake'].sum() * 100) if leg_df['recommended_stake'].sum() > 0 else 0,
            'expected_roi': (leg_df['expected_value'].sum() / leg_df['recommended_stake'].sum() * 100) if leg_df['recommended_stake'].sum() > 0 else 0,
        }

    # Metrics by week
    by_week = {}
    for week in sorted(df['week'].unique()):
        week_df = df[df['week'] == week]
        by_week[week] = {
            'count': len(week_df),
            'hit_rate': week_df['parlay_hit'].sum() / len(week_df),
            'total_profit': week_df['actual_profit'].sum(),
            'roi': (week_df['actual_profit'].sum() / week_df['recommended_stake'].sum() * 100) if week_df['recommended_stake'].sum() > 0 else 0,
        }

    return {
        'total_parlays': total_parlays,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': roi,
        'hit_rate': hit_rate,
        'expected_profit': expected_profit,
        'expected_roi': expected_roi,
        'roi_vs_expected': roi - expected_roi,
        'by_legs': by_legs,
        'by_week': by_week,
    }


def print_results(metrics: Dict, parlays: List[Dict]):
    """Print backtest results in formatted output."""
    print("\n" + "=" * 80)
    print("PARLAY BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n📊 OVERALL PERFORMANCE")
    print(f"   Total Parlays: {metrics['total_parlays']}")
    print(f"   Hit Rate: {metrics['hit_rate']:.1%}")
    print(f"   Total Staked: ${metrics['total_staked']:.2f}")
    print(f"   Total Profit: ${metrics['total_profit']:.2f}")
    print(f"   ROI: {metrics['roi']:.2f}%")

    print(f"\n🎯 THEORETICAL vs ACTUAL")
    print(f"   Expected ROI: {metrics['expected_roi']:.2f}%")
    print(f"   Actual ROI: {metrics['roi']:.2f}%")
    print(f"   Variance: {metrics['roi_vs_expected']:+.2f}%")

    if abs(metrics['roi_vs_expected']) < 5:
        print(f"   ✅ Actual ROI within 5% of expected (VALIDATED)")
    elif abs(metrics['roi_vs_expected']) < 10:
        print(f"   ⚠️  Actual ROI within 10% of expected (acceptable variance)")
    else:
        print(f"   ❌ Large variance - may need calibration adjustment")

    print(f"\n📈 PERFORMANCE BY NUMBER OF LEGS")
    print(f"   {'Legs':<6} {'Count':<8} {'Hit%':<8} {'Avg Edge':<10} {'ROI':<10} {'Expected ROI':<12}")
    print(f"   {'-'*60}")

    for num_legs in sorted(metrics['by_legs'].keys()):
        leg_metrics = metrics['by_legs'][num_legs]
        print(f"   {num_legs:<6} {leg_metrics['count']:<8} "
              f"{leg_metrics['hit_rate']*100:<7.1f}% "
              f"{leg_metrics['avg_edge']*100:<9.1f}% "
              f"{leg_metrics['roi']:<9.2f}% "
              f"{leg_metrics['expected_roi']:<11.2f}%")

    print(f"\n📅 PERFORMANCE BY WEEK")
    print(f"   {'Week':<6} {'Count':<8} {'Hit%':<8} {'ROI':<10}")
    print(f"   {'-'*40}")

    for week in sorted(metrics['by_week'].keys()):
        week_metrics = metrics['by_week'][week]
        print(f"   {week:<6} {week_metrics['count']:<8} "
              f"{week_metrics['hit_rate']*100:<7.1f}% "
              f"{week_metrics['roi']:<9.2f}%")

    # Show top 5 winning parlays
    df = pd.DataFrame(parlays)
    winning_df = df[df['parlay_hit'] == True].sort_values('potential_win', ascending=False)

    if not winning_df.empty:
        print(f"\n🏆 TOP 5 WINNING PARLAYS")
        for i, (_, row) in enumerate(winning_df.head(5).iterrows(), 1):
            print(f"\n   {i}. Week {row['week']} - {row['num_legs']}-Leg Parlay")
            print(f"      Odds: {row['true_odds']:+d} ({row['model_prob']:.1%} model prob)")
            print(f"      Edge: {row['edge']:.1%}")
            print(f"      Stake: ${row['recommended_stake']:.2f}")
            print(f"      Win: ${row['potential_win']:.2f}")
            print(f"      Legs:")
            for j, leg in enumerate(row['legs'], 1):
                hit_emoji = "✅" if row['leg_results'][j-1] else "❌"
                print(f"         {j}. {hit_emoji} {leg}")

    # Show worst losses
    losing_df = df[df['parlay_hit'] == False].sort_values('recommended_stake', ascending=False)

    if not losing_df.empty:
        print(f"\n❌ TOP 5 LOSING PARLAYS (by stake)")
        for i, (_, row) in enumerate(losing_df.head(5).iterrows(), 1):
            print(f"\n   {i}. Week {row['week']} - {row['num_legs']}-Leg Parlay")
            print(f"      Edge: {row['edge']:.1%} (Expected to win)")
            print(f"      Stake Lost: ${row['recommended_stake']:.2f}")
            print(f"      Legs:")
            for j, leg in enumerate(row['legs'], 1):
                hit_emoji = "✅" if row['leg_results'][j-1] else "❌"
                print(f"         {j}. {hit_emoji} {leg}")


def save_results(parlays: List[Dict], metrics: Dict):
    """Save backtest results to files."""
    # Save parlays to CSV
    df = pd.DataFrame(parlays)

    # Flatten legs for CSV
    df['legs_str'] = df['legs'].apply(lambda x: ' | '.join(x))
    df['leg_results_str'] = df['leg_results'].apply(lambda x: ','.join(['✅' if h else '❌' for h in x]))

    csv_path = Path("reports/parlay_backtest_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved parlay results to: {csv_path}")

    # Save metrics to JSON
    json_path = Path("reports/parlay_backtest_metrics.json")

    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    metrics_clean = convert_types(metrics)

    with open(json_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    print(f"💾 Saved metrics to: {json_path}")


def main():
    """Run parlay backtest."""
    print("=" * 80)
    print("🎰 NFL PARLAY BACKTEST")
    print("=" * 80)
    print(f"Testing parlay logic on historical data (Weeks 1-8)")
    print(f"Validates: Theoretical edge vs actual ROI")

    # Load backtest results
    backtest_csv = "reports/BACKTEST_WEEKS_1_8_VALIDATION.csv"
    df = load_backtest_results(backtest_csv)

    if df.empty:
        print("\n❌ No backtest data found. Run single bet backtest first:")
        print("   python scripts/backtest/backtest_player_props.py --min-edge 0.0 --end-week 8")
        return

    # Check if we have 'hit' column (actual results)
    if 'hit' not in df.columns:
        print("\n❌ Backtest data missing 'hit' column with actual results")
        print("   Re-run backtest to include actual outcomes")
        return

    # Generate parlay combinations with actual results
    parlays = generate_parlay_combinations_by_week(df, max_legs=4)

    if not parlays:
        print("\n❌ No parlays generated with complete results")
        return

    # Calculate metrics
    print("\n📊 Calculating performance metrics...")
    metrics = calculate_parlay_metrics(parlays)

    # Print results
    print_results(metrics, parlays)

    # Save results
    save_results(parlays, metrics)

    print("\n" + "=" * 80)
    print("✅ BACKTEST COMPLETE")
    print("=" * 80)

    # Summary
    print(f"\nSummary:")
    print(f"  • Parlays tested: {metrics['total_parlays']}")
    print(f"  • Hit rate: {metrics['hit_rate']:.1%}")
    print(f"  • Actual ROI: {metrics['roi']:.2f}%")
    print(f"  • Expected ROI: {metrics['expected_roi']:.2f}%")
    print(f"  • Variance: {metrics['roi_vs_expected']:+.2f}%")

    if abs(metrics['roi_vs_expected']) < 5:
        print(f"\n✅ VALIDATION PASSED: Theoretical edge matches actual ROI")
    elif abs(metrics['roi_vs_expected']) < 10:
        print(f"\n⚠️  ACCEPTABLE: Within 10% variance (sample size may be small)")
    else:
        print(f"\n❌ CALIBRATION NEEDED: Large variance between expected and actual ROI")


if __name__ == "__main__":
    main()
