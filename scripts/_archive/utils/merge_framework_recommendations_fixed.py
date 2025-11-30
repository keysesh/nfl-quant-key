#!/usr/bin/env python3
"""Merge full framework picks into unified recommendations - FIXED V2"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def american_to_decimal(odds):
    """Convert American odds to decimal odds"""
    if pd.isna(odds):
        return 1.0
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def kelly_bet_size(american_odds, win_prob, bankroll, kelly_fraction=0.25, max_bet_pct=3.0, min_bet=1.0):
    """Calculate Kelly criterion bet size."""
    if american_odds < 0:
        b = 100 / abs(american_odds)
    else:
        b = american_odds / 100

    p = win_prob
    q = 1 - p

    # Full Kelly
    full_kelly = (b * p - q) / b
    full_kelly = max(0, full_kelly)  # No negative bets

    # Apply Kelly fraction
    kelly_scaled = full_kelly * kelly_fraction

    # Cap at max bet percentage
    max_bet_amount = bankroll * (max_bet_pct / 100)
    kelly_bet_amount = bankroll * kelly_scaled
    bet_amount = min(kelly_bet_amount, max_bet_amount)

    # Apply minimum bet
    bet_amount = max(bet_amount, min_bet)

    return bet_amount

print("="*80)
print("MERGING GAME LINES + PLAYER PROPS INTO UNIFIED RECOMMENDATIONS")
print("="*80)

# Load bankroll configuration
config_path = Path('configs/bankroll_config.json')
if config_path.exists():
    with open(config_path) as f:
        bankroll_config = json.load(f)
    BANKROLL = bankroll_config['total_bankroll']
    KELLY_FRACTION = bankroll_config.get('kelly_fraction', 0.25)
    MAX_BET_PCT = bankroll_config.get('max_bet_pct', 3.0)
    MIN_BET = bankroll_config.get('min_bet_amount', 1.0)
    print(f"✅ Loaded bankroll config: ${BANKROLL:.2f} (Kelly: {KELLY_FRACTION*100:.0f}%, Max: {MAX_BET_PCT}%)")
else:
    BANKROLL = 50.0
    KELLY_FRACTION = 0.25
    MAX_BET_PCT = 3.0
    MIN_BET = 1.0
    print(f"⚠️  Using default bankroll config: ${BANKROLL:.2f}")

# Load game line recommendations from CURRENT_WEEK_RECOMMENDATIONS.csv (includes game lines)
# Fallback to FINAL_RECOMMENDATIONS.csv for backwards compatibility
game_lines_path = Path('reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
if game_lines_path.exists():
    all_recommendations = pd.read_csv(game_lines_path)
    # Filter to game lines only (rows where player is empty or market starts with 'game_')
    game_line = all_recommendations[
        (all_recommendations['player'].isna() | (all_recommendations['player'] == '')) |
        (all_recommendations['market'].str.startswith('game_', na=False))
    ].copy()
    print(f"\n✅ Loaded {len(game_line)} game line recommendations from CURRENT_WEEK_RECOMMENDATIONS.csv")

    # Also try FINAL_RECOMMENDATIONS.csv if it exists and has more game lines
    final_path = Path('reports/FINAL_RECOMMENDATIONS.csv')
    if final_path.exists():
        final_df = pd.read_csv(final_path)
        if len(final_df) > len(game_line):
            print(f"   ⚠️  Also found FINAL_RECOMMENDATIONS.csv with {len(final_df)} lines, using that instead")
            game_line = final_df
else:
    # Fallback to FINAL_RECOMMENDATIONS.csv
    final_path = Path('reports/FINAL_RECOMMENDATIONS.csv')
    if final_path.exists():
        game_line = pd.read_csv(final_path)
        print(f"\n✅ Loaded {len(game_line)} game line recommendations from FINAL_RECOMMENDATIONS.csv")
    else:
        print(f"\n⚠️  No game line recommendations found (checked CURRENT_WEEK_RECOMMENDATIONS.csv and FINAL_RECOMMENDATIONS.csv)")
        game_line = pd.DataFrame()

# Load the player props from current week recommendations
# Try new naming first, fall back to old for compatibility
player_props_path = Path('reports/CURRENT_WEEK_PLAYER_PROPS.csv')
if not player_props_path.exists():
    player_props_path = Path('reports/full_framework_tonight.csv')  # Fallback to old naming
if not player_props_path.exists():
    player_props_path = Path('reports/archive_stale_20251029/full_framework_tonight.csv')  # Check archive

if player_props_path.exists():
    player_props = pd.read_csv(player_props_path)
    print(f"✅ Loaded {len(player_props)} player props from {player_props_path.name}")
    print(f"   Columns: {player_props.columns.tolist()}")
else:
    print(f"\n❌ No player props file found!")
    print(f"   Expected: reports/CURRENT_WEEK_PLAYER_PROPS.csv")
    print(f"   Run: python scripts/predict/generate_current_week_recommendations.py")
    player_props = pd.DataFrame()

# Convert player props to unified format
player_unified = []
for idx, row in player_props.iterrows():
    # Extract player name - should be a string column
    player_name = str(row['player']).strip()

    # Skip if player name is invalid
    if not player_name or player_name == 'nan':
        print(f"Skipping row {idx}: invalid player name")
        continue

    decimal_odds = american_to_decimal(row['odds'])
    ev = (row['model_prob'] * decimal_odds - 1) if not pd.isna(row['odds']) else 0
    bet_size = max(1.0, min(10.0, 5.0 * (row['edge_pct'] / 100)))
    potential_profit = bet_size * (decimal_odds - 1)

    # Get line and hist_over_rate
    line = float(row['line'])
    hist_over_rate = float(row['hist_over_rate'])

    # Estimate predicted value from historical over rate
    # If 100% over rate, predict ~50% above line
    # If 0% over rate, predict ~50% below line
    # If 50% over rate, predict at line
    adjustment_factor = (hist_over_rate - 0.5) * 1.0  # More aggressive adjustment
    predicted_value = line * (1 + adjustment_factor)

    # Make sure prediction makes sense
    if 'Over' in row['pick'] and predicted_value < line:
        predicted_value = line * 1.25  # At least 25% above for Over
    elif 'Under' in row['pick'] and predicted_value > line:
        predicted_value = line * 0.75  # At least 25% below for Under

    player_unified.append({
        'bet_type': 'Player Prop',
        'game': 'WAS @ KC',
        'player': player_name,
        'pick': row['pick'],
        'market': row['market'],
        'market_odds': int(row['odds']) if pd.notna(row['odds']) else None,
        'model_value': float(round(predicted_value, 1)),
        'edge': float(row['edge_pct'] / 100),
        'our_prob': float(row['model_prob']),
        'market_prob': float(row['market_prob']),
        'ev': float(ev),
        'roi': float(row['roi_pct'] / 100),
        'bet_size': float(bet_size),
        'potential_profit': float(potential_profit),
        'rank': len(player_unified) + 2
    })

# Convert game lines to unified format if they exist
game_unified = []
if len(game_line) > 0:
    print(f"\n📊 Converting {len(game_line)} game lines to unified format...")

    # Load simulation results to get actual scores for display
    import json
    import glob
    sim_data_dict = {}
    sim_files = glob.glob('reports/sim_*.json')
    for sim_file in sim_files:
        try:
            with open(sim_file) as f:
                sim_data = json.load(f)
                game_id = sim_data.get('game_id', '')
                if game_id:
                    sim_data_dict[game_id] = sim_data
        except:
            pass

    for idx, row in game_line.iterrows():
        # Parse model_prob from string like "58.0%" to float 0.58, or use as-is if already float
        model_prob = row.get('model_prob', 0.5)
        if isinstance(model_prob, str):
            model_prob_str = str(model_prob).replace('%', '')
            model_prob = float(model_prob_str) / 100.0
        else:
            model_prob = float(model_prob)

        market_prob = row.get('market_prob', 0.5)
        if isinstance(market_prob, str):
            market_prob_str = str(market_prob).replace('%', '')
            market_prob = float(market_prob_str) / 100.0
        else:
            market_prob = float(market_prob)

        # Get edge - can be edge_pct (percentage) or edge (decimal)
        edge = row.get('edge', 0.0)
        if isinstance(edge, str):
            edge_str = str(edge).replace('%', '')
            edge = float(edge_str) / 100.0
        elif 'edge_pct' in row and pd.notna(row.get('edge_pct')):
            edge = float(row['edge_pct']) / 100.0
        else:
            edge = float(edge)

        # Get odds
        market_odds = int(row.get('odds', -110)) if pd.notna(row.get('odds')) else -110

        # Recalculate bet size using proper Kelly criterion
        bet_size = kelly_bet_size(market_odds, model_prob, BANKROLL, KELLY_FRACTION, MAX_BET_PCT, MIN_BET)

        # Calculate potential profit
        if market_odds < 0:
            potential_profit = bet_size * (100 / abs(market_odds))
        else:
            potential_profit = bet_size * (market_odds / 100)

        # Get model projection (fair_spread, fair_total, or model_prob for moneylines)
        model_projection = row.get('model_projection', None)
        if pd.isna(model_projection):
            model_projection = None

        # Format model_value based on market type
        market = str(row.get('market', ''))
        game_display = str(row.get('game', ''))

        # Try to get actual scores from simulation data
        model_value_display = None
        if model_projection is not None:
            if 'spread' in market.lower():
                model_value_display = f"Model Spread: {model_projection:.1f}"
            elif 'total' in market.lower():
                model_value_display = f"Model Total: {model_projection:.1f}"
            elif 'moneyline' in market.lower():
                model_value_display = f"Model Win Prob: {model_prob:.1%}"

        # Try to get actual scores from simulation JSON files
        # Game format: "AWAY @ HOME" or from game_id
        game_id = None
        for gid, sim_data in sim_data_dict.items():
            # Parse game_id: 2025_09_AWAY_HOME
            parts = gid.split('_')
            if len(parts) >= 4:
                away_team = parts[2]
                home_team = parts[3]
                sim_game_display = f"{away_team} @ {home_team}"
                if sim_game_display == game_display:
                    game_id = gid
                    # Add score predictions
                    home_score = sim_data.get('home_score_median', None)
                    away_score = sim_data.get('away_score_median', None)
                    if home_score is not None and away_score is not None:
                        if 'spread' in market.lower():
                            model_value_display = f"{away_score:.0f}-{home_score:.0f} (Model Spread: {model_projection:.1f})"
                        elif 'total' in market.lower():
                            model_value_display = f"{away_score:.0f}-{home_score:.0f} (Model Total: {model_projection:.1f})"
                        elif 'moneyline' in market.lower():
                            model_value_display = f"{away_score:.0f}-{home_score:.0f} (Model Win Prob: {model_prob:.1%})"
                    break

        if model_value_display is None:
            model_value_display = 'Game Simulation'  # Fallback if no projection available

        game_unified.append({
            'bet_type': str(row.get('bet_type', 'Game Line')),
            'game': game_display,
            'player': '',  # No player for game lines
            'pick': str(row.get('pick', row.get('recommendation', row.get('bet_on', '')))),
            'market': market,
            'market_odds': market_odds,
            'model_value': model_value_display,
            'edge': edge,
            'our_prob': model_prob,
            'market_prob': market_prob,
            'ev': float(row.get('ev', row.get('roi_pct', 0.0) / 100.0 if pd.notna(row.get('roi_pct')) else 0.0)),
            'roi': float(row.get('roi', row.get('roi_pct', 0.0) / 100.0 if pd.notna(row.get('roi_pct')) else 0.0)),
            'bet_size': float(bet_size),
            'potential_profit': float(potential_profit),
            'rank': 0  # Will be reassigned after sorting
        })
    print(f"   ✅ Converted {len(game_unified)} game lines")

# Combine game lines and player props
print(f"\n🔄 Merging {len(game_unified)} game lines + {len(player_unified)} player props...")
unified_df = pd.concat([
    pd.DataFrame(game_unified),
    pd.DataFrame(player_unified)
], ignore_index=True)

# Sort by our_prob (confidence) descending
unified_df = unified_df.sort_values('our_prob', ascending=False).reset_index(drop=True)
unified_df['rank'] = range(1, len(unified_df) + 1)

# Validation
expected_min = len(game_line) + len(player_props)
if len(unified_df) < expected_min:
    print(f"\n⚠️  WARNING: Expected at least {expected_min} picks, got {len(unified_df)}")
    print(f"   Game lines input: {len(game_line)}")
    print(f"   Player props input: {len(player_props)}")
    print(f"   Unified output: {len(unified_df)}")
else:
    print(f"\n✅ Validation passed: {len(unified_df)} total picks")

# Save
output_path = Path('reports/unified_betting_recommendations.csv')
unified_df.to_csv(output_path, index=False)

print(f'\n{"="*80}')
print(f'✅ MERGE COMPLETE: unified_betting_recommendations.csv')
print(f'{"="*80}')
print(f'📊 Summary:')
print(f'   - Game lines: {len(game_unified)}')
print(f'   - Player props: {len(player_unified)}')
print(f'   - Total picks: {len(unified_df)}')
print(f'   - Output: {output_path}')

if len(unified_df) > 0:
    print(f'\n📋 Top 5 picks by confidence:')
    display_cols = ['bet_type', 'pick', 'our_prob', 'edge', 'ev']
    available_cols = [col for col in display_cols if col in unified_df.columns]
    print(unified_df[available_cols].head(5).to_string(index=False))
else:
    print(f'\n⚠️  No picks to display')
