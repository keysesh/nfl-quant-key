#!/usr/bin/env python3
"""
Re-simulate QB Pass TD Props with Fixed TD Rates

After fixing the QB TD rate simulation to use actual trailing TD rates
instead of the broken efficiency predictor model, we need to re-simulate
all QB pass TD props in the historical dataset.

This will update the probability distributions used to train the calibrator.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput

def load_historical_props():
    """Load historical simulated props"""
    props_file = Path("data/calibration/historical_props_simulated.parquet")

    if not props_file.exists():
        print(f"❌ Historical props file not found: {props_file}")
        return None

    df = pd.read_parquet(props_file)
    print(f"✅ Loaded {len(df):,} historical props")

    return df

def resimulate_qb_pass_tds(df, simulator):
    """Re-simulate QB pass TD props with fixed TD rate logic"""

    # Filter to QB pass TD props
    qb_td_props = df[
        (df['position'] == 'QB') &
        (df['market'] == 'player_pass_tds')
    ].copy()

    print(f"\n📊 Found {len(qb_td_props):,} QB pass TD props to re-simulate")
    print(f"   Seasons: {sorted(qb_td_props['season'].unique())}")
    print(f"   Weeks: {qb_td_props['week'].min()}-{qb_td_props['week'].max()}")
    print()

    if len(qb_td_props) == 0:
        print("⚠️  No QB pass TD props found")
        return df

    # Re-simulate each prop
    updated_count = 0

    for idx in tqdm(qb_td_props.index, desc="Re-simulating QB pass TDs"):
        row = df.loc[idx]

        # Create PlayerPropInput
        player_input = PlayerPropInput(
            player_id=row['player_id'],
            player_name=row['player_name'],
            position=row['position'],
            team=row['team'],
            opponent=row['opponent'],
            week=row['week'],
            projected_team_total=row.get('projected_team_total', 24.0),
            projected_opponent_total=row.get('projected_opponent_total', 22.0),
            projected_game_script=row.get('projected_game_script', 0.0),
            projected_pace=row.get('projected_pace', 30.0),
            trailing_snap_share=row.get('trailing_snap_share', 1.0),
            trailing_target_share=row.get('trailing_target_share'),
            trailing_carry_share=row.get('trailing_carry_share'),
            trailing_yards_per_opportunity=row.get('trailing_yards_per_opportunity', 7.0),
            trailing_td_rate=row.get('trailing_td_rate', 0.04),  # Key: actual trailing TD rate
            opponent_def_epa_vs_position=row.get('opponent_def_epa_vs_position', 0.0),
        )

        # Run simulation (now uses fixed TD rate logic)
        result = simulator.simulate_player(player_input)

        # Extract pass TDs
        pass_tds = result['passing_tds']

        # Calculate probabilities for different lines
        for line in [0.5, 1.5, 2.5]:
            df.at[idx, f'prob_over_{line}'] = np.mean(pass_tds > line)
            df.at[idx, f'prob_under_{line}'] = np.mean(pass_tds < line)

        # Store distribution stats
        df.at[idx, 'sim_mean'] = np.mean(pass_tds)
        df.at[idx, 'sim_std'] = np.std(pass_tds)
        df.at[idx, 'sim_median'] = np.median(pass_tds)

        updated_count += 1

    print()
    print(f"✅ Re-simulated {updated_count:,} QB pass TD props")
    print()

    return df

def compare_before_after(df_before, df_after):
    """Compare TD probabilities before and after fix"""
    print("=" * 80)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 80)
    print()

    qb_td_mask = (df_before['position'] == 'QB') & (df_before['market'] == 'player_pass_tds')

    if 'prob_over_1.5' not in df_before.columns or 'prob_over_1.5' not in df_after.columns:
        print("⚠️  Probability columns not found for comparison")
        return

    # Compare average probabilities
    before_avg = df_before.loc[qb_td_mask, 'prob_over_1.5'].mean()
    after_avg = df_after.loc[qb_td_mask, 'prob_over_1.5'].mean()

    print(f"Average P(>1.5 TDs):")
    print(f"  Before fix: {before_avg:.1%}")
    print(f"  After fix:  {after_avg:.1%}")
    print(f"  Change:     {(after_avg - before_avg)*100:+.1f} percentage points")
    print()

    # Show distribution shifts
    before_dist = df_before.loc[qb_td_mask, 'prob_over_1.5'].describe()
    after_dist = df_after.loc[qb_td_mask, 'prob_over_1.5'].describe()

    print("Distribution of P(>1.5 TDs):")
    print()
    print(f"{'Stat':10s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print("-" * 45)
    for stat in ['min', '25%', '50%', '75%', 'max']:
        before_val = before_dist[stat]
        after_val = after_dist[stat]
        change = after_val - before_val
        print(f"{stat:10s} {before_val:>9.1%} {after_val:>9.1%} {change:>+9.1%}")

    print()

def verify_fix_quality(df_after):
    """Verify the fix improved TD rate accuracy"""
    print("=" * 80)
    print("FIX QUALITY VERIFICATION")
    print("=" * 80)
    print()

    qb_td_props = df_after[
        (df_after['position'] == 'QB') &
        (df_after['market'] == 'player_pass_tds')
    ].copy()

    # Check if trailing TD rates are being reflected
    if 'trailing_td_rate' in qb_td_props.columns and 'sim_mean' in qb_td_props.columns:
        print("Checking if simulated TDs reflect actual trailing TD rates:")
        print()

        # Bin by trailing TD rate
        qb_td_props['td_rate_bin'] = pd.cut(
            qb_td_props['trailing_td_rate'],
            bins=[0, 0.02, 0.04, 0.06, 1.0],
            labels=['Low (<2%)', 'Medium (2-4%)', 'High (4-6%)', 'Elite (>6%)']
        )

        for bin_label, bin_data in qb_td_props.groupby('td_rate_bin'):
            if len(bin_data) == 0:
                continue

            avg_trailing_rate = bin_data['trailing_td_rate'].mean()
            avg_sim_mean = bin_data['sim_mean'].mean()
            avg_prob_over_1_5 = bin_data['prob_over_1.5'].mean()

            print(f"{bin_label}:")
            print(f"  Avg trailing TD rate: {avg_trailing_rate:.1%}")
            print(f"  Avg simulated TDs: {avg_sim_mean:.2f}")
            print(f"  Avg P(>1.5 TDs): {avg_prob_over_1_5:.1%}")
            print(f"  Count: {len(bin_data)}")
            print()

        # Check correlation
        correlation = qb_td_props['trailing_td_rate'].corr(qb_td_props['sim_mean'])
        print(f"Correlation between trailing TD rate and simulated mean: {correlation:.3f}")

        if correlation > 0.7:
            print("✅ STRONG correlation - Fix is working!")
        elif correlation > 0.4:
            print("⚠️  MODERATE correlation - Some improvement but could be better")
        else:
            print("❌ WEAK correlation - Fix may not be working properly")

        print()

def save_updated_props(df):
    """Save updated props file"""
    output_file = Path("data/calibration/historical_props_simulated.parquet")

    # Backup original
    backup_file = Path("data/calibration/historical_props_simulated_backup_before_td_fix.parquet")
    if output_file.exists() and not backup_file.exists():
        print(f"📦 Creating backup: {backup_file}")
        df_original = pd.read_parquet(output_file)
        df_original.to_parquet(backup_file, index=False)

    # Save updated
    df.to_parquet(output_file, index=False)
    print(f"✅ Saved updated props: {output_file}")
    print()

def main():
    print()
    print("=" * 80)
    print("RE-SIMULATE QB PASS TD PROPS WITH FIXED TD RATES")
    print("=" * 80)
    print()

    # Load historical props
    df_before = load_historical_props()
    if df_before is None:
        return 1

    # Load simulator with fix
    print("Loading simulator (now with fixed TD rate logic)...")
    usage_predictor, efficiency_predictor = load_predictors()
    simulator = PlayerSimulator(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,
        seed=42
    )
    print("✅ Simulator loaded")
    print()

    # Re-simulate
    df_after = resimulate_qb_pass_tds(df_before.copy(), simulator)

    # Compare
    compare_before_after(df_before, df_after)

    # Verify fix quality
    verify_fix_quality(df_after)

    # Save
    save_updated_props(df_after)

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. ✅ QB pass TD props re-simulated with fixed TD rates")
    print("2. 🎯 Next: Retrain calibrator with corrected data")
    print("3. 🎯 Then: Run backtest to validate improvement")
    print()
    print("Run:")
    print("  python scripts/train/retrain_calibrator_nflverse.py")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
