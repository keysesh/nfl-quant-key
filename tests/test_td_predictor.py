"""Quick test of TD predictor"""
from pathlib import Path
from nfl_quant.models.td_predictor import TouchdownPredictor, estimate_usage_factors
import pandas as pd

# Initialize TD predictor
hist_stats_path = Path('data/nflverse_cache/stats_player_week_2025.csv')
pbp_path = Path('data/processed/pbp_2025.parquet')
td_predictor = TouchdownPredictor(
    historical_stats_path=hist_stats_path if hist_stats_path.exists() else None,
    pbp_path=pbp_path if pbp_path.exists() else None
)

print("\n✅ TD Predictor initialized")
if td_predictor.defensive_metrics:
    print("✅ Defensive metrics loaded")
print()

# Test with a workhorse RB
print("=" * 80)
print("Test 1: Derrick Henry (Workhorse RB)")
print("=" * 80)

# Create mock player data
player_data = pd.Series({
    'player_name': 'Derrick Henry',
    'position': 'RB',
    'rushing_attempts_mean': 20.0,
    'receptions_mean': 1.5,
    'team': 'BAL'
})

usage_factors = estimate_usage_factors(player_data, 'RB')
print(f"Usage factors: {usage_factors}")

td_pred = td_predictor.predict_touchdown_probability(
    player_name='Derrick Henry',
    position='RB',
    projected_carries=20.0,
    projected_targets=2.0,
    red_zone_share=usage_factors['red_zone_share'],
    goal_line_role=usage_factors['goal_line_role'],
    team_projected_total=26.0,
    opponent_team='DEN',  # Test with opponent
    current_week=9,
)

print(f"\nTD Prediction:")
for key, value in td_pred.items():
    print(f"  {key}: {value:.4f}")

# Test with a backup RB
print("\n" + "=" * 80)
print("Test 2: Nathan Carter (Backup RB)")
print("=" * 80)

player_data2 = pd.Series({
    'player_name': 'Nathan Carter',
    'position': 'RB',
    'rushing_attempts_mean': 4.0,
    'receptions_mean': 0.5,
    'team': 'IND'
})

usage_factors2 = estimate_usage_factors(player_data2, 'RB')
print(f"Usage factors: {usage_factors2}")

td_pred2 = td_predictor.predict_touchdown_probability(
    player_name='Nathan Carter',
    position='RB',
    projected_carries=4.0,
    projected_targets=0.7,
    red_zone_share=usage_factors2['red_zone_share'],
    goal_line_role=usage_factors2['goal_line_role'],
    team_projected_total=22.0,
)

print(f"\nTD Prediction:")
for key, value in td_pred2.items():
    print(f"  {key}: {value:.4f}")

# Test with a WR1
print("\n" + "=" * 80)
print("Test 3: Amon-Ra St. Brown (WR1)")
print("=" * 80)

player_data3 = pd.Series({
    'player_name': 'Amon-Ra St. Brown',
    'position': 'WR',
    'receptions_mean': 8.0,
    'team': 'DET'
})

usage_factors3 = estimate_usage_factors(player_data3, 'WR')
print(f"Usage factors: {usage_factors3}")

td_pred3 = td_predictor.predict_touchdown_probability(
    player_name='Amon-Ra St. Brown',
    position='WR',
    projected_targets=12.0,
    red_zone_share=usage_factors3['red_zone_share'],
    goal_line_role=usage_factors3['goal_line_role'],
    team_projected_total=28.0,
)

print(f"\nTD Prediction:")
for key, value in td_pred3.items():
    print(f"  {key}: {value:.4f}")

print("\n" + "=" * 80)
print("✅ All tests completed!")
print("=" * 80)
