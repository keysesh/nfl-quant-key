#!/usr/bin/env python3
"""Find TD props with positive edge"""
import pandas as pd
import numpy as np

preds = pd.read_csv('data/model_predictions_week9.csv')
odds = pd.read_csv('data/nfl_player_props_draftkings.csv')
td_odds = odds[odds['market'] == 'player_anytime_td'].copy()

print("="*80)
print("FINDING TD PROPS WITH POSITIVE EDGE")
print("="*80)

edges = []

for _, odds_row in td_odds.iterrows():
    player_name = odds_row['player_name']
    market_odds = odds_row['odds']

    # Find prediction
    pred_row = preds[preds['player_name'] == player_name]
    if pred_row.empty:
        continue

    pred = pred_row.iloc[0]
    position = pred.get('position', '')

    # Calculate P(any TD)
    prob_any_td = 0.0

    if position == 'QB':
        pass_td = pred.get('passing_tds_mean', 0.0)
        rush_td = pred.get('rushing_tds_mean', 0.0)
        prob_any_td = 1 - (np.exp(-pass_td) * np.exp(-rush_td))
    elif position == 'RB':
        rush_td = pred.get('rushing_tds_mean', 0.0)
        rec_td = pred.get('receiving_tds_mean', 0.0)
        prob_any_td = 1 - (np.exp(-rush_td) * np.exp(-rec_td))
    elif position in ['WR', 'TE']:
        rec_td = pred.get('receiving_tds_mean', 0.0)
        prob_any_td = 1 - np.exp(-rec_td)

    if prob_any_td == 0:
        continue

    # Market prob
    if market_odds > 0:
        market_prob = 100 / (market_odds + 100)
    else:
        market_prob = abs(market_odds) / (abs(market_odds) + 100)

    edge = prob_any_td - market_prob

    edges.append({
        'player': player_name,
        'position': position,
        'model_prob': prob_any_td,
        'market_prob': market_prob,
        'edge': edge,
        'edge_pct': edge * 100,
        'odds': market_odds
    })

# Convert to DataFrame and sort
df = pd.DataFrame(edges)
df = df.sort_values('edge', ascending=False)

print(f"\nProcessed {len(df)} players with TD odds and predictions\n")

print("TOP 20 POSITIVE EDGES:")
print("="*80)
print(f"{'Player':<20s} {'Pos':<4s} {'Model':<8s} {'Market':<8s} {'Edge':<10s} {'Odds':<8s}")
print("-"*80)

for _, row in df.head(20).iterrows():
    print(f"{row['player']:<20s} {row['position']:<4s} {row['model_prob']:<8.1%} {row['market_prob']:<8.1%} {row['edge']:<+10.1%} {row['odds']:<+8.0f}")

print("\n\nBOTTOM 10 (Most Overpriced):")
print("="*80)
print(f"{'Player':<20s} {'Pos':<4s} {'Model':<8s} {'Market':<8s} {'Edge':<10s} {'Odds':<8s}")
print("-"*80)

for _, row in df.tail(10).iterrows():
    print(f"{row['player']:<20s} {row['position']:<4s} {row['model_prob']:<8.1%} {row['market_prob']:<8.1%} {row['edge']:<+10.1%} {row['odds']:<+8.0f}")

print("\n\nPlayers with edge > 2%:")
good_edges = df[df['edge'] > 0.02]
print(f"Found: {len(good_edges)} players")

if len(good_edges) > 0:
    print("\n" + "="*80)
    for _, row in good_edges.iterrows():
        print(f"✅ {row['player']:<20s} {row['position']:<4s} | Model: {row['model_prob']:.1%} vs Market: {row['market_prob']:.1%} | Edge: {row['edge']:+.1%} ({row['odds']:+.0f})")
else:
    print("\n❌ No TD props meet the 2% edge threshold")

print("\n" + "="*80)
