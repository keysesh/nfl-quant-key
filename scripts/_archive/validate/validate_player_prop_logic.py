#!/usr/bin/env python3
"""
Validate Player Prop Logic Against Recent Performance
Check if model projections align with actual recent performance
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def load_actual_stats():
    """Load actual player stats using unified interface - FAIL EXPLICITLY if unavailable."""
    from nfl_quant.data.stats_loader import load_weekly_stats, is_data_available

    all_stats = []

    for week in range(1, 9):
        if not is_data_available(week, 2025, source='auto'):
            raise FileNotFoundError(
                f"Stats data not available for week {week}, season 2025. "
                f"Run data fetching scripts to populate data."
            )

        df = load_weekly_stats(week, 2025, source='auto')
        df['week'] = week
        all_stats.append(df)

    if not all_stats:
        raise FileNotFoundError(
            "No stats data available for weeks 1-8, season 2025. "
            "Run data fetching scripts to populate data."
        )

    return pd.concat(all_stats, ignore_index=True)

def get_player_recent_performance(player_name, stats_df, weeks_back=4):
    """Get player's recent performance"""
    player_stats = stats_df[stats_df['player_name'].str.contains(player_name, case=False, na=False)].copy()

    if player_stats.empty:
        return None

    # Sort by week
    player_stats = player_stats.sort_values('week')

    # Get recent weeks
    recent = player_stats.tail(weeks_back)

    return {
        'weeks': recent['week'].tolist(),
        'pass_yds': recent['pass_yd'].mean() if 'pass_yd' in recent.columns else 0,
        'rush_yds': recent['rush_yd'].mean() if 'rush_yd' in recent.columns else 0,
        'pass_attempts': recent['pass_att'].mean() if 'pass_att' in recent.columns else 0,
        'rush_attempts': recent['rush_att'].mean() if 'rush_att' in recent.columns else 0,
        'completions': recent['pass_cmp'].mean() if 'pass_cmp' in recent.columns else 0,
        'recent_data': recent
    }

def load_model_projections():
    """Load current model projections from recommendations"""
    recs_path = Path('reports/unified_betting_recommendations.csv')
    if recs_path.exists():
        df = pd.read_csv(recs_path)
        return df
    return pd.DataFrame()

def analyze_player_logic(player_name, stats_df, projections_df):
    """Analyze if model logic makes sense for a player"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {player_name}")
    print(f"{'='*80}\n")

    # Get recent performance
    recent = get_player_recent_performance(player_name, stats_df)

    if recent is None:
        print(f"❌ No recent performance data found for {player_name}")
        return

    print(f"📊 RECENT PERFORMANCE (Last 4 Weeks):")
    print(f"   Weeks: {recent['weeks']}")
    print(f"   Avg Pass Yds: {recent['pass_yds']:.1f}")
    print(f"   Avg Rush Yds: {recent['rush_yds']:.1f}")
    print(f"   Avg Pass Attempts: {recent['pass_attempts']:.1f}")
    print(f"   Avg Rush Attempts: {recent['rush_attempts']:.1f}")
    print()

    # Get model projections
    player_props = projections_df[projections_df['player'].str.contains(player_name, case=False, na=False)]

    if player_props.empty:
        print(f"⚠️  No current projections found for {player_name}")
        return

    print(f"📈 MODEL PROJECTIONS:")
    for _, prop in player_props.iterrows():
        market = prop['market']
        model_val = prop['model_value']
        pick = prop['pick']
        line = pick.split()[-1] if ' ' in pick else 'N/A'
        our_prob = prop['our_prob']

        print(f"\n   {market}:")
        print(f"     Model Value: {model_val:.1f}")
        print(f"     Line: {line}")
        print(f"     Our Probability: {our_prob:.1%}")
        print(f"     Pick: {pick}")

        # Compare to recent performance
        if 'pass_yds' in market:
            recent_val = recent['pass_yds']
            diff = model_val - recent_val
            pct_diff = (diff / recent_val * 100) if recent_val > 0 else 0
            print(f"     Recent Avg: {recent_val:.1f}")
            print(f"     Difference: {diff:.1f} ({pct_diff:+.1f}%)")

            if abs(pct_diff) > 50:
                print(f"     ⚠️  WARNING: Model projection differs by >50% from recent performance")
                if pct_diff < -50:
                    print(f"     ℹ️  Model is projecting MUCH LOWER than recent performance")
                    print(f"     ℹ️  Possible reasons: Injury, matchup, or model needs update")

        elif 'rush_yds' in market:
            recent_val = recent['rush_yds']
            diff = model_val - recent_val
            pct_diff = (diff / recent_val * 100) if recent_val > 0 else 0
            print(f"     Recent Avg: {recent_val:.1f}")
            print(f"     Difference: {diff:.1f} ({pct_diff:+.1f}%)")

            if abs(pct_diff) > 50:
                print(f"     ⚠️  WARNING: Model projection differs by >50% from recent performance")
                if pct_diff < -50:
                    print(f"     ℹ️  Model is projecting MUCH LOWER than recent performance")
                    print(f"     ℹ️  Possible reasons: Injury, matchup, or model needs update")

    print()

def check_injury_status(player_name):
    """Check if player has injury status"""
    print(f"🏥 INJURY STATUS CHECK:")

    # Check week-specific trailing stats for injury indicators
    try:
        with open('data/week_specific_trailing_stats.json') as f:
            stats = json.load(f)

        player_keys = [k for k in stats.keys() if player_name.lower() in k.lower()]

        if player_keys:
            latest_key = sorted(player_keys, key=lambda x: int(x.split('_week')[1]) if '_week' in x else 0)[-1]
            latest_data = stats[latest_key]

            print(f"   Latest week data: {latest_key}")
            print(f"   Trailing weeks: {latest_data.get('trailing_weeks', 'N/A')}")
            print(f"   Yards per opportunity: {latest_data.get('trailing_yards_per_opportunity', 'N/A')}")

            # Check if yards per opportunity is very low (injury indicator)
            ypo = latest_data.get('trailing_yards_per_opportunity', 0)
            if ypo < 5 and latest_data.get('trailing_weeks', 0) > 0:
                print(f"   ⚠️  Low yards per opportunity ({ypo:.1f}) may indicate injury/limited usage")
        else:
            print(f"   ⚠️  No recent trailing stats found")

    except Exception as e:
        print(f"   ❌ Error checking injury status: {e}")

    print()

def main():
    print("="*80)
    print("PLAYER PROP LOGIC VALIDATION")
    print("="*80)
    print()

    # Load data
    print("Loading data...")
    stats_df = load_actual_stats()
    projections_df = load_model_projections()

    if stats_df.empty:
        print("❌ No stats data found")
        return

    if projections_df.empty:
        print("❌ No projections found")
        return

    print(f"✅ Loaded {len(stats_df)} stat records")
    print(f"✅ Loaded {len(projections_df)} projections")
    print()

    # Analyze specific players
    players_to_check = ['Lamar Jackson']

    # Also check top projections with large edges
    top_players = projections_df.nlargest(10, 'edge')['player'].unique()
    players_to_check.extend([p for p in top_players if p not in players_to_check])

    for player_name in players_to_check[:5]:  # Limit to 5 for now
        analyze_player_logic(player_name, stats_df, projections_df)
        check_injury_status(player_name)

    # Summary recommendations
    print("\n" + "="*80)
    print("VALIDATION SUMMARY & RECOMMENDATIONS")
    print("="*80)
    print("\n✅ Check completed. Review warnings above.")
    print("\n📋 Key Things to Validate:")
    print("   1. Model projections vs recent performance")
    print("   2. Large differences (>50%) may indicate:")
    print("      - Injury/limited usage")
    print("      - Matchup difficulty")
    print("      - Model needs update")
    print("   3. Confidence should be reduced if:")
    print("      - Projection differs significantly from recent form")
    print("      - Recent data is limited (< 3 games)")
    print("      - Injury status is unclear")

if __name__ == '__main__':
    main()
