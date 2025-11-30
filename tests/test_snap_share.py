"""Test snap share impact on TD predictions (starter vs backup)"""
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
print("="*80)
print("Testing Snap Share Impact - Starter vs Backup Discrimination")
print("="*80)

# Base RB parameters
player_data = pd.Series({
    'player_name': 'RB Test',
    'position': 'RB',
    'rushing_attempts_mean': 15.0,
    'receptions_mean': 2.0,
    'team': 'TEST'
})

usage_factors = estimate_usage_factors(player_data, 'RB')

# Test 1: Full starter (85% snap share)
print("\n1. Starter RB (85% snap share)")
print("-" * 80)
td_starter = td_predictor.predict_touchdown_probability(
    player_name='Starter RB',
    position='RB',
    projected_carries=18.0,
    projected_targets=3.0,
    red_zone_share=usage_factors['red_zone_share'],
    goal_line_role=usage_factors['goal_line_role'],
    team_projected_total=24.0,
    projected_snap_share=0.85,
)
print(f"Snap Share: 85%")
print(f"Rushing TDs: {td_starter.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_starter.get('receiving_tds_mean', 0):.4f}")
print(f"Total TDs: {td_starter.get('lambda_total', 0):.4f}")
print(f"Prob Any TD: {td_starter.get('prob_any_td', 0):.4f}")

# Test 2: Committee back (60% snap share)
print("\n2. Committee RB (60% snap share)")
print("-" * 80)
td_committee = td_predictor.predict_touchdown_probability(
    player_name='Committee RB',
    position='RB',
    projected_carries=12.0,
    projected_targets=2.5,
    red_zone_share=usage_factors['red_zone_share'] * 0.7,  # Slightly less RZ usage
    goal_line_role=usage_factors['goal_line_role'] * 0.7,
    team_projected_total=24.0,
    projected_snap_share=0.60,
)
print(f"Snap Share: 60%")
print(f"Rushing TDs: {td_committee.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_committee.get('receiving_tds_mean', 0):.4f}")
print(f"Total TDs: {td_committee.get('lambda_total', 0):.4f}")
print(f"Prob Any TD: {td_committee.get('prob_any_td', 0):.4f}")

# Test 3: Backup (30% snap share)
print("\n3. Backup RB (30% snap share)")
print("-" * 80)
td_backup = td_predictor.predict_touchdown_probability(
    player_name='Backup RB',
    position='RB',
    projected_carries=6.0,
    projected_targets=1.0,
    red_zone_share=usage_factors['red_zone_share'] * 0.3,
    goal_line_role=usage_factors['goal_line_role'] * 0.2,
    team_projected_total=24.0,
    projected_snap_share=0.30,
)
print(f"Snap Share: 30%")
print(f"Rushing TDs: {td_backup.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_backup.get('receiving_tds_mean', 0):.4f}")
print(f"Total TDs: {td_backup.get('lambda_total', 0):.4f}")
print(f"Prob Any TD: {td_backup.get('prob_any_td', 0):.4f}")

# Test 4: Deep backup (15% snap share)
print("\n4. Deep Backup RB (15% snap share)")
print("-" * 80)
td_deep_backup = td_predictor.predict_touchdown_probability(
    player_name='Deep Backup RB',
    position='RB',
    projected_carries=2.0,
    projected_targets=0.5,
    red_zone_share=usage_factors['red_zone_share'] * 0.1,
    goal_line_role=usage_factors['goal_line_role'] * 0.05,
    team_projected_total=24.0,
    projected_snap_share=0.15,
)
print(f"Snap Share: 15%")
print(f"Rushing TDs: {td_deep_backup.get('rushing_tds_mean', 0):.4f}")
print(f"Receiving TDs: {td_deep_backup.get('receiving_tds_mean', 0):.4f}")
print(f"Total TDs: {td_deep_backup.get('lambda_total', 0):.4f}")
print(f"Prob Any TD: {td_deep_backup.get('prob_any_td', 0):.4f}")

# Summary
print("\n" + "="*80)
print("Summary - TD Discrimination by Role")
print("="*80)
print(f"Starter (85% snaps):       {td_starter.get('prob_any_td', 0):.1%} TD prob")
print(f"Committee (60% snaps):     {td_committee.get('prob_any_td', 0):.1%} TD prob ({td_committee.get('prob_any_td', 0)/td_starter.get('prob_any_td', 1)*100:.0f}% of starter)")
print(f"Backup (30% snaps):        {td_backup.get('prob_any_td', 0):.1%} TD prob ({td_backup.get('prob_any_td', 0)/td_starter.get('prob_any_td', 1)*100:.0f}% of starter)")
print(f"Deep Backup (15% snaps):   {td_deep_backup.get('prob_any_td', 0):.1%} TD prob ({td_deep_backup.get('prob_any_td', 0)/td_starter.get('prob_any_td', 1)*100:.0f}% of starter)")

print("\n" + "="*80)
print("✅ Snap share discrimination tests completed!")
print("="*80)
