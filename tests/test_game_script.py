"""Test game script impact on TD predictions"""
from pathlib import Path
from nfl_quant.models.td_predictor import TouchdownPredictor, estimate_usage_factors
import pandas as pd


def get_pbp_path(season: int = 2025) -> Path:
    """Get PBP path using cascading lookup (fresh → season-specific → processed)."""
    paths = [
        Path('data/nflverse/pbp.parquet'),
        Path(f'data/nflverse/pbp_{season}.parquet'),
        Path(f'data/processed/pbp_{season}.parquet'),
    ]
    for p in paths:
        if p.exists():
            return p
    return paths[-1]  # Return last path even if doesn't exist


# Initialize TD predictor
hist_stats_path = Path('data/nflverse_cache/stats_player_week_2025.csv')
pbp_path = get_pbp_path(2025)
td_predictor = TouchdownPredictor(
    historical_stats_path=hist_stats_path if hist_stats_path.exists() else None,
    pbp_path=pbp_path if pbp_path.exists() else None
)

print("\n✅ TD Predictor initialized")
print("="*80)
print("Testing Game Script Impact")
print("="*80)

# Test RB in different game scripts
player_data = pd.Series({
    'player_name': 'Derrick Henry',
    'position': 'RB',
    'rushing_attempts_mean': 20.0,
    'receptions_mean': 1.5,
    'team': 'BAL'
})

usage_factors = estimate_usage_factors(player_data, 'RB')

# Test 1: Neutral game script (0 point differential)
print("\n1. Neutral Game Script (0 point differential)")
print("-" * 80)
td_neutral = td_predictor.predict_touchdown_probability(
    player_name='Derrick Henry',
    position='RB',
    projected_carries=20.0,
    projected_targets=2.0,
    red_zone_share=usage_factors['red_zone_share'],
    goal_line_role=usage_factors['goal_line_role'],
    team_projected_total=24.0,
    projected_point_differential=0.0,
)
print(f"Rushing TDs: {td_neutral.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_neutral.get('receiving_tds_mean', 0):.4f}")
print(f"Prob Any TD: {td_neutral.get('prob_any_td', 0):.4f}")

# Test 2: Leading by 7 (positive game script - more rushing)
print("\n2. Leading by 7 points (+7 differential)")
print("-" * 80)
td_leading = td_predictor.predict_touchdown_probability(
    player_name='Derrick Henry',
    position='RB',
    projected_carries=20.0,
    projected_targets=2.0,
    red_zone_share=usage_factors['red_zone_share'],
    goal_line_role=usage_factors['goal_line_role'],
    team_projected_total=27.0,
    projected_point_differential=7.0,
)
print(f"Rushing TDs: {td_leading.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_leading.get('receiving_tds_mean', 0):.4f}")
print(f"Prob Any TD: {td_leading.get('prob_any_td', 0):.4f}")
rush_increase = (td_leading.get('rushing_tds_mean', 0) / td_neutral.get('rushing_tds_mean', 1) - 1) * 100
print(f"Rushing TD increase: +{rush_increase:.1f}%")

# Test 3: Trailing by 7 (negative game script - less rushing)
print("\n3. Trailing by 7 points (-7 differential)")
print("-" * 80)
td_trailing = td_predictor.predict_touchdown_probability(
    player_name='Derrick Henry',
    position='RB',
    projected_carries=20.0,
    projected_targets=2.0,
    red_zone_share=usage_factors['red_zone_share'],
    goal_line_role=usage_factors['goal_line_role'],
    team_projected_total=21.0,
    projected_point_differential=-7.0,
)
print(f"Rushing TDs: {td_trailing.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_trailing.get('receiving_tds_mean', 0):.4f}")
print(f"Prob Any TD: {td_trailing.get('prob_any_td', 0):.4f}")
rush_decrease = (1 - td_trailing.get('rushing_tds_mean', 0) / td_neutral.get('rushing_tds_mean', 1)) * 100
rec_increase = (td_trailing.get('receiving_tds_mean', 0) / td_neutral.get('receiving_tds_mean', 1) - 1) * 100
print(f"Rushing TD decrease: -{rush_decrease:.1f}%")
print(f"Receiving TD increase: +{rec_increase:.1f}%")

# Test WR in trailing game script
print("\n" + "="*80)
print("WR in Trailing Game Script")
print("="*80)

player_data_wr = pd.Series({
    'player_name': 'CeeDee Lamb',
    'position': 'WR',
    'receptions_mean': 8.0,
    'team': 'DAL'
})

usage_factors_wr = estimate_usage_factors(player_data_wr, 'WR')

# Neutral
td_wr_neutral = td_predictor.predict_touchdown_probability(
    player_name='CeeDee Lamb',
    position='WR',
    projected_targets=11.0,
    red_zone_share=usage_factors_wr['red_zone_share'],
    goal_line_role=usage_factors_wr['goal_line_role'],
    team_projected_total=24.0,
    projected_point_differential=0.0,
)

# Trailing
td_wr_trailing = td_predictor.predict_touchdown_probability(
    player_name='CeeDee Lamb',
    position='WR',
    projected_targets=11.0,
    red_zone_share=usage_factors_wr['red_zone_share'],
    goal_line_role=usage_factors_wr['goal_line_role'],
    team_projected_total=21.0,
    projected_point_differential=-7.0,
)

print(f"\nNeutral: {td_wr_neutral.get('receiving_tds_mean', 0):.4f} receiving TDs")
print(f"Trailing by 7: {td_wr_trailing.get('receiving_tds_mean', 0):.4f} receiving TDs")
wr_increase = (td_wr_trailing.get('receiving_tds_mean', 0) / td_wr_neutral.get('receiving_tds_mean', 1) - 1) * 100
print(f"Increase when trailing: +{wr_increase:.1f}%")

print("\n" + "="*80)
print("✅ Game script tests completed!")
print("="*80)
