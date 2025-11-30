#!/usr/bin/env python3
"""
Backtest the main unified framework for weeks 1-7
Generates predictions and evaluates against actual outcomes to calibrate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Try to import framework components
try:
    from nfl_quant.simulation.player_simulator import (
        PlayerSimulator, load_predictors)
    from nfl_quant.schemas import PlayerPropInput
    from nfl_quant.calibration.isotonic_calibrator import (
        NFLProbabilityCalibrator)
    from nfl_quant.constants import (
        FALLBACK_VALUES, RECEIVING_POSITIONS, RUSHING_POSITIONS)
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    print("⚠️  PlayerSimulator not available")

print("=" * 80)
print("FRAMEWORK BACKTEST - Weeks 1-7")
print("=" * 80)
print()

# Load training dataset with actual outcomes
print("📂 Loading training dataset...")
df = pd.read_csv('data/historical/player_prop_training_dataset.csv')
print(f"✅ Loaded {len(df):,} props")

# Load trailing stats if available
trailing_stats_path = Path('data/week_specific_trailing_stats.json')
if trailing_stats_path.exists():
    with open(trailing_stats_path) as f:
        trailing_stats = json.load(f)
    print(f"✅ Loaded {len(trailing_stats)} week-specific profiles")
else:
    print("⚠️  No week-specific stats - using fallbacks")
    trailing_stats = {}

# Keep only weeks 1-7
df = df[df['week'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
print(f"📅 Filtering to weeks 1-7: {len(df):,} props")
print()

# Filter to props with actual values and over/under only
df = df[df['actual_value'].notna()].copy()
df = df[df['prop_type'].isin(['over', 'under'])].copy()
print(f"📊 Props with outcomes: {len(df):,}")
print()

if FRAMEWORK_AVAILABLE:
    print("🔧 Building framework...")

    # Load predictors
    usage_predictor, efficiency_predictor = load_predictors()
    print("✅ Loaded predictors")

    # Load calibrator if exists
    calibrator = NFLProbabilityCalibrator()
    calibrator_path = Path('configs/calibrator.json')
    if calibrator_path.exists():
        calibrator.load(str(calibrator_path))
        print("✅ Loaded calibrator")
    else:
        print("⚠️  No calibrator found")

    # Build simulator
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=5000,  # Faster for backtesting
        seed=42,
        calibrator=calibrator,
    )
    print("✅ Created PlayerSimulator")
    print()
else:
    simulator = None

# Run predictions
print("🧮 Running framework predictions...")
print()

results = []
cache = {}

for idx, row in df.iterrows():
    if idx % 100 == 0 and idx > 0:
        print(f"   Processed {idx}/{len(df)} props...")

    player_name = row['player']
    team = row['team']
    position = row['position']
    week = int(row['week'])
    market = row['market']
    prop_type = row['prop_type']
    line = row['line']
    actual_value = row['actual_value']
    american_price = row.get('american_price', -110)

    # Skip anytime TD (no line to compare)
    if pd.isna(line):
        continue

    # Create cache key

    cache_key = (week, player_name.lower(), team)

    # Try to get prediction from framework
    if FRAMEWORK_AVAILABLE and simulator is not None:
        if cache_key not in cache:
            # Create player input
            key = f"{player_name}_week{week}"
            hist = trailing_stats.get(key, {})

            # Fallback to most recent week if this week not found
            if not hist and trailing_stats:
                available_weeks = [int(k.split('_week')[1]) for k in trailing_stats.keys()
                                  if k.startswith(f"{player_name}_week") and k.split('_week')[1].isdigit()]
                if available_weeks:
                    previous_weeks = [w for w in available_weeks if w < week]
                    if previous_weeks:
                        most_recent_week = max(previous_weeks)
                        key_fallback = f"{player_name}_week{most_recent_week}"
                        hist = trailing_stats.get(key_fallback, {})

            player_input = PlayerPropInput(
                player_id=player_name,
                player_name=player_name,
                team=team,
                position=position,
                week=week,
                opponent=hist.get("opponent", "UNK"),
                projected_team_total=float(hist.get("projected_team_total", 25.0)),
                projected_opponent_total=float(hist.get("projected_opponent_total", 22.0)),
                projected_game_script=float(hist.get("projected_game_script", 0.0)),
                projected_pace=float(hist.get("projected_pace", 28.0)),
                trailing_snap_share=float(hist.get("trailing_snap_share", FALLBACK_VALUES["snap_share"])),
                trailing_target_share=hist.get("trailing_target_share") if position in RECEIVING_POSITIONS else None,
                trailing_carry_share=hist.get("trailing_carry_share") if position in RUSHING_POSITIONS else None,
                trailing_yards_per_opportunity=float(
                    hist.get("trailing_yards_per_opportunity",
                             FALLBACK_VALUES["yards_per_opportunity"])),
                trailing_td_rate=float(
                    hist.get("trailing_td_rate",
                             FALLBACK_VALUES["td_rate"])),
                opponent_def_epa_vs_position=hist.get(
                    "opponent_def_epa_vs_position", 0.0),
            )

            try:
                cache[cache_key] = simulator.simulate_player(player_input)
            except Exception:
                cache[cache_key] = {}

        sim_result = cache.get(cache_key, {})

        # Map market to simulator key
        market_map = {
            'player_pass_yds': 'passing_yards',
            'player_rush_yds': 'rushing_yards',
            'player_reception_yds': 'receiving_yards',
            'player_receptions': 'receptions',
        }

        sim_key = market_map.get(market)
        if sim_key and sim_key in sim_result:
            distribution = sim_result[sim_key]
            if prop_type == 'over':
                model_prob = float(np.mean(distribution > line))
            else:
                model_prob = float(np.mean(distribution < line))
        else:
            model_prob = None
    else:
        model_prob = None

    # Determine bet outcome
    if prop_type == 'over':
        bet_won = actual_value > line
        bet_outcome = 1.0 if bet_won else 0.0
    else:
        bet_won = actual_value < line
        bet_outcome = 1.0 if bet_won else 0.0

    # Calculate implied probability from American odds
    if pd.notna(american_price):
        if american_price >= 0:
            implied_prob = 100.0 / (american_price + 100.0)
        else:
            implied_prob = abs(american_price) / (abs(american_price) + 100.0)
    else:
        implied_prob = 0.524  # -110 default

    # Calculate edge if we have model prediction
    edge = (model_prob - implied_prob) if model_prob is not None else None

    # Calculate profit (assuming $100 bet)
    if american_price >= 0:
        profit = american_price / 100.0
    else:
        profit = 100.0 / abs(american_price)

    unit_return = profit if bet_won else -1.0

    results.append({
        'week': week,
        'player': player_name,
        'team': team,
        'position': position,
        'market': market,
        'prop_type': prop_type,
        'line': line,
        'actual_value': actual_value,
        'american_price': american_price,
        'implied_prob': implied_prob,
        'model_prob': model_prob,
        'edge': edge,
        'bet_won': bet_won,
        'bet_outcome': bet_outcome,
        'unit_return': unit_return,
        'profit': profit if bet_won else -1.0,
    })

results_df = pd.DataFrame(results)
print(f"\n✅ Generated predictions for {len(results_df):,} props")
print()

# Save detailed results
output_file = 'reports/framework_backtest_weeks_1_7.csv'
results_df.to_csv(output_file, index=False)
print(f"💾 Saved detailed results to: {output_file}")
print()

# Analysis
print("=" * 80)
print("FRAMEWORK BACKTEST RESULTS")
print("=" * 80)
print()

# Overall stats
total_bets = len(results_df)
wins = results_df['bet_won'].sum()
losses = total_bets - wins
win_rate = wins / total_bets if total_bets > 0 else 0
total_roi = results_df['unit_return'].sum()
avg_roi = total_roi / total_bets if total_bets > 0 else 0

print(f"📊 OVERALL RESULTS:")
print(f"   Total Props: {total_bets:,}")
print(f"   Wins: {wins:,}")
print(f"   Losses: {losses:,}")
print(f"   Win Rate: {win_rate:.1%}")
print(f"   Total ROI: {total_roi:,.2f} units")
print(f"   Average ROI: {avg_roi:.2%}")
print()

# Only props with framework predictions
if FRAMEWORK_AVAILABLE:
    framework_results = results_df[results_df['model_prob'].notna()].copy()

    if len(framework_results) > 0:
        fw_bets = len(framework_results)
        fw_wins = framework_results['bet_won'].sum()
        fw_losses = fw_bets - fw_wins
        fw_wr = fw_wins / fw_bets if fw_bets > 0 else 0
        fw_roi = framework_results['unit_return'].sum()
        fw_avg_roi = fw_roi / fw_bets if fw_bets > 0 else 0

        print(f"🤖 FRAMEWORK PREDICTIONS:")
        print(f"   Props with predictions: {fw_bets:,}")
        print(f"   Wins: {fw_wins:,}")
        print(f"   Losses: {fw_losses:,}")
        print(f"   Win Rate: {fw_wr:.1%}")
        print(f"   Total ROI: {fw_roi:,.2f} units")
        print(f"   Average ROI: {fw_avg_roi:.2%}")
        print()

        # Filter by edge
        if len(framework_results[framework_results['edge'].notna()]) > 0:
            positive_edge = framework_results[framework_results['edge'] > 0.03].copy()

            if len(positive_edge) > 0:
                pe_bets = len(positive_edge)
                pe_wins = positive_edge['bet_won'].sum()
                pe_losses = pe_bets - pe_wins
                pe_wr = pe_wins / pe_bets if pe_bets > 0 else 0
                pe_roi = positive_edge['unit_return'].sum()
                pe_avg_roi = pe_roi / pe_bets if pe_bets > 0 else 0

                print(f"🎯 FILTERED (Edge > 3%):")
                print(f"   Props: {pe_bets:,}")
                print(f"   Wins: {pe_wins:,}")
                print(f"   Losses: {pe_losses:,}")
                print(f"   Win Rate: {pe_wr:.1%}")
                print(f"   Total ROI: {pe_roi:,.2f} units")
                print(f"   Average ROI: {pe_avg_roi:.2%}")
                print()

# By week
print(f"📅 WEEK-BY-WEEK (Framework Predictions):")
for week in sorted(results_df['week'].unique()):
    week_data = framework_results[framework_results['week'] == week] if FRAMEWORK_AVAILABLE else results_df[results_df['week'] == week]
    if len(week_data) > 0:
        w_bets = len(week_data)
        w_wins = week_data['bet_won'].sum()
        w_wr = w_wins / w_bets if w_bets > 0 else 0
        w_roi = week_data['unit_return'].sum()
        print(f"   Week {int(week)}: {w_bets:3} bets | WR: {w_wr:5.1%} | ROI: {w_roi:+7.2f} units")
print()

# By market
print(f"🏈 BY MARKET (Framework Predictions):")
for market in sorted(results_df['market'].unique()):
    market_data = framework_results[framework_results['market'] == market] if FRAMEWORK_AVAILABLE else results_df[results_df['market'] == market]
    if len(market_data) > 0:
        m_bets = len(market_data)
        m_wins = market_data['bet_won'].sum()
        m_wr = m_wins / m_bets if m_bets > 0 else 0
        m_roi = market_data['unit_return'].sum()
        print(f"   {market:25s}: {m_bets:4} bets | WR: {m_wr:5.1%} | ROI: {m_roi:+7.2f} units")
print()

print("=" * 80)
print("✅ BACKTEST COMPLETE")
print("=" * 80)
