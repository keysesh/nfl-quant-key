"""Test TD recommendations generation"""
import pandas as pd
import numpy as np

# Load predictions
predictions = pd.read_csv('data/model_predictions_week9.csv')

# Load odds
odds = pd.read_csv('data/draftkings_week9_props_live.csv')

# Filter to anytime TD props
td_props = odds[odds['market'] == 'player_anytime_td'].copy()
print(f"Total TD props: {len(td_props)}")

# Merge with predictions
merged = td_props.merge(
    predictions[['player_dk', 'position', 'rushing_tds_mean', 'receiving_tds_mean', 'passing_tds_mean']],
    left_on='player_name',
    right_on='player_dk',
    how='left'
)

print(f"Merged TD props with predictions: {len(merged)}")

# Calculate model probability for each player
def calc_td_prob(row):
    """Calculate P(any TD) from lambda values"""
    lambda_total = 0

    rush_tds = row.get('rushing_tds_mean', 0)
    if pd.notna(rush_tds):
        lambda_total += rush_tds

    rec_tds = row.get('receiving_tds_mean', 0)
    if pd.notna(rec_tds):
        lambda_total += rec_tds

    pass_tds = row.get('passing_tds_mean', 0)
    if pd.notna(pass_tds):
        lambda_total += pass_tds

    # P(any TD) = 1 - P(no TD) = 1 - exp(-lambda)
    prob_any_td = 1 - np.exp(-lambda_total)

    # Cap at reasonable bounds
    return max(0.01, min(0.95, prob_any_td))

merged['model_prob'] = merged.apply(calc_td_prob, axis=1)

# Calculate market probability from odds
merged['market_prob'] = 1 / (1 + merged['price'])

# Calculate edge
merged['edge'] = merged['model_prob'] - merged['market_prob']

# Filter positive edges
positive_edges = merged[merged['edge'] > 0.05].sort_values('edge', ascending=False)

print(f"\nTD props with >5% edge: {len(positive_edges)}")
print("\nTop 10 TD recommendations:")
print("=" * 100)
for _, row in positive_edges.head(10).iterrows():
    name = row['player_name']
    pos = row['position']
    model = row['model_prob']
    market = row['market_prob']
    edge = row['edge']
    odds = row['price']
    print(f"{name:25s} {pos:3s}  Model: {model:5.1%}  Market: {market:5.1%}  Edge: {edge:+6.1%}  Odds: {odds:+.0f}")
