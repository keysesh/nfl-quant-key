#!/usr/bin/env python3
"""Diagnose why backtest finds no +EV bets"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

def normalize_name(name):
    if pd.isna(name):
        return ""
    return name.strip().lower().replace("'", "").replace(".", "").replace(" jr", "").replace(" sr", "")

def american_to_prob(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

# Load Week 5 data
print("Loading Week 5 data...")
preds = pd.read_csv(project_root / "data/model_predictions_week5.csv")
props = pd.read_csv(project_root / "data/backtest/historical_by_week/week_5_props.csv")

print(f"Predictions: {len(preds)} players")
print(f"Props: {len(props)} lines")

# Normalize names
preds['player_normalized'] = preds['player_name'].apply(normalize_name)
props['player_normalized'] = props['player'].apply(normalize_name)

# Pair over/under
props_paired = []
grouped = props.groupby(['player_normalized', 'market', 'line'])

for (player, market, line), group in grouped:
    over_row = group[group['prop_type'] == 'over']
    under_row = group[group['prop_type'] == 'under']

    if len(over_row) > 0 and len(under_row) > 0:
        props_paired.append({
            'player_normalized': player,
            'player': over_row.iloc[0]['player'],
            'market': market,
            'line': line,
            'over_odds': over_row.iloc[0]['american_odds'],
            'under_odds': under_row.iloc[0]['american_odds']
        })

props_df = pd.DataFrame(props_paired)
print(f"Paired: {len(props_df)} markets")

# Check rushing yards specifically
rush_props = props_df[props_df['market'] == 'player_rush_yds'].head(10)
print(f"\nFirst 10 rushing yards props:")
for _, prop in rush_props.iterrows():
    player_norm = prop['player_normalized']

    # Find prediction
    pred_match = preds[preds['player_normalized'] == player_norm]

    if len(pred_match) == 0:
        print(f"  {prop['player']:20} - NO PREDICTION MATCH")
        continue

    if 'rushing_yards_mean' not in pred_match.columns or 'rushing_yards_std' not in pred_match.columns:
        print(f"  {prop['player']:20} - MISSING COLUMNS")
        continue

    mean = pred_match.iloc[0]['rushing_yards_mean']
    std = pred_match.iloc[0]['rushing_yards_std']

    if pd.isna(mean) or pd.isna(std):
        print(f"  {prop['player']:20} - NA values (mean={mean}, std={std})")
        continue

    line = prop['line']
    over_odds = prop['over_odds']
    under_odds = prop['under_odds']

    # Calculate probabilities
    over_prob_implied = american_to_prob(over_odds)
    under_prob_implied = american_to_prob(under_odds)
    total_implied = over_prob_implied + under_prob_implied
    over_prob_fair = over_prob_implied / total_implied
    under_prob_fair = under_prob_implied / total_implied

    # Model probability
    if std <= 0:
        print(f"  {prop['player']:20} - STD <= 0 ({std})")
        continue

    z_score = (line - mean) / std
    model_over_prob = 1 - norm.cdf(z_score)

    # Edge
    over_edge = model_over_prob - over_prob_fair
    under_edge = (1 - model_over_prob) - under_prob_fair

    best_side = 'OVER' if over_edge > under_edge else 'UNDER'
    best_edge = max(over_edge, under_edge)

    print(f"  {prop['player']:20} Line:{line:5.1f} Mean:{mean:5.1f} Std:{std:5.1f} Edge:{best_edge:+6.1%} ({best_side})")
