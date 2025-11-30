#!/usr/bin/env python3
"""
Add historical DraftKings odds to training data to improve probability estimates.

Option 1 Enhancement: Merge odds from odds_archive.csv with player_prop_evaluation.csv
This gives the calibrator both model predictions AND market odds for better training.
"""

import pandas as pd
from pathlib import Path

def american_to_prob(odds):
    """Convert American odds to implied probability"""
    if pd.isna(odds) or odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def main():
    print("=" * 80)
    print("ADDING HISTORICAL ODDS TO TRAINING DATA")
    print("=" * 80)
    print()

    # Load evaluation data
    eval_path = Path('data/historical/player_prop_evaluation.csv')
    if not eval_path.exists():
        print(f"❌ {eval_path} not found")
        return

    df_eval = pd.read_csv(eval_path)
    print(f"📂 Loaded {len(df_eval):,} props from evaluation data")
    print(f"   Markets: {sorted(df_eval['market'].unique())}")
    print()

    # Check if american_price column exists
    if 'american_price' in df_eval.columns:
        has_odds = df_eval['american_price'].notna().sum()
        print(f"📊 Current odds coverage: {has_odds:,} rows ({has_odds/len(df_eval):.1%})")

        if has_odds > len(df_eval) * 0.9:
            print("✅ Already has odds for >90% of rows - no update needed")
            return
    else:
        print("📊 Adding odds column...")
        df_eval['american_price'] = None

    # Load historical odds from backfill
    odds_data = []
    backfill_dir = Path('data/historical/backfill')

    if backfill_dir.exists():
        print(f"📂 Loading odds from backfill directory...")
        for csv_file in backfill_dir.glob('player_props_history_*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if 'price' in df.columns:  # American odds column
                    # Convert to match evaluation format
                    df_odds = df[['event_id', 'market', 'player', 'prop_type', 'line', 'price']].copy()
                    df_odds['american_price'] = df_odds['price']
                    odds_data.append(df_odds)
                    print(f"   ✅ {csv_file.name}: {len(df_odds):,} props with odds")
            except Exception as e:
                print(f"   ⚠️  {csv_file.name}: {e}")

    if not odds_data:
        print("❌ No odds data found in backfill directory")
        return

    # Combine odds data
    df_odds_combined = pd.concat(odds_data, ignore_index=True)
    print(f"\n📊 Combined odds data: {len(df_odds_combined):,} props")

    # Merge odds into evaluation data by matching event_id, market, player
    print("\n🔄 Merging odds data...")

    # Create merge keys
    df_eval['merge_key'] = (
        df_eval['event_id'].astype(str) + '|' +
        df_eval['market'].astype(str) + '|' +
        df_eval['player'].astype(str) + '|' +
        df_eval['prop_type'].astype(str)
    )

    df_odds_combined['merge_key'] = (
        df_odds_combined['event_id'].astype(str) + '|' +
        df_odds_combined['market'].astype(str) + '|' +
        df_odds_combined['player'].astype(str) + '|' +
        df_odds_combined['prop_type'].astype(str)
    )

    # Merge
    df_eval = df_eval.merge(
        df_odds_combined[['merge_key', 'american_price']].drop_duplicates('merge_key'),
        on='merge_key',
        how='left',
        suffixes=('_old', '')
    )

    # Fill in missing odds from the new column
    if 'american_price_old' in df_eval.columns:
        df_eval['american_price'] = df_eval['american_price'].fillna(df_eval['american_price_old'])
        df_eval = df_eval.drop(columns=['american_price_old'])

    # Clean up merge key
    df_eval = df_eval.drop(columns=['merge_key'])

    # Report results
    final_has_odds = df_eval['american_price'].notna().sum()
    print(f"✅ Updated odds coverage: {final_has_odds:,} rows ({final_has_odds/len(df_eval):.1%})")
    print(f"   Added odds to: {final_has_odds - has_odds if 'has_odds' in locals() else final_has_odds:,} rows")

    # Add implied probability column for calibration
    df_eval['implied_prob'] = df_eval['american_price'].apply(american_to_prob)

    # Save updated data
    output_path = eval_path  # Overwrite original
    df_eval.to_csv(output_path, index=False)
    print(f"\n💾 Saved updated training data to: {output_path}")
    print()

    # Show market breakdown with odds
    print("📊 Market Odds Coverage:")
    print(f"{'Market':<30} {'With Odds':>12} {'Total':>10} {'%':>8}")
    print("-" * 65)

    for market in sorted(df_eval['market'].unique()):
        market_df = df_eval[df_eval['market'] == market]
        with_odds = market_df['american_price'].notna().sum()
        total = len(market_df)
        pct = (with_odds / total * 100) if total > 0 else 0
        print(f"{market:<30} {with_odds:>12,} {total:>10,} {pct:>7.1f}%")

    print()
    print("✅ Historical odds added successfully!")
    print("   Next step: Retrain the calibrator with this enhanced data")

if __name__ == "__main__":
    main()
