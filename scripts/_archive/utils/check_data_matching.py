#!/usr/bin/env python3
"""Check for data matching issues between props and outcomes"""

import pandas as pd
from pathlib import Path

print('🔍 DATA MATCHING INVESTIGATION')
print('=' * 80)
print()

# Load datasets
print('📂 Loading datasets...')
archive = pd.read_csv('data/historical/live_archive/player_props_archive.csv')
training = pd.read_csv('data/historical/player_prop_training_dataset.csv')

# Load sample stats files to check format
stats_files = list(Path('data/sleeper_stats').glob('stats_week*_2025.csv'))
stats_sample = pd.read_csv(stats_files[0]) if stats_files else None

print(f'✅ Archive props: {len(archive):,}')
print(f'✅ Training (matched): {len(training):,}')
print(f'📊 Match rate: {len(training)/len(archive)*100:.1f}%')
print()

# ============================================================================
# SECTION 1: NAME FORMAT COMPARISON
# ============================================================================
print('📋 SECTION 1: PLAYER NAME FORMAT COMPARISON')
print('-' * 80)

print('Archive (Odds API) - Sample names:')
for name in sorted(archive['player'].dropna().unique())[:15]:
    print(f'  "{name}"')
print()

if stats_sample is not None:
    print('Sleeper Stats - Sample names:')
    for name in sorted(stats_sample['player_name'].dropna().unique())[:15]:
        print(f'  "{name}"')
    print()

# ============================================================================
# SECTION 2: CASE SENSITIVITY CHECK
# ============================================================================
print('📋 SECTION 2: CASE SENSITIVITY & FORMATTING')
print('-' * 80)

# Check if names are lowercase in training dataset
training_sample_names = training['player'].head(10).tolist()
print('Training dataset names (should be lowercase):')
for name in training_sample_names:
    print(f'  "{name}"')
print()

# ============================================================================
# SECTION 3: UNMATCHED PLAYERS
# ============================================================================
print('📋 SECTION 3: UNMATCHED PLAYERS ANALYSIS')
print('-' * 80)

archive_players = set(archive['player'].dropna().str.strip().unique())
training_players = set(training['player'].dropna().str.strip().unique())
unmatched = archive_players - training_players

print(f'Players in archive: {len(archive_players):,}')
print(f'Players in training: {len(training_players):,}')
print(f'Unmatched players: {len(unmatched):,} ({len(unmatched)/len(archive_players)*100:.1f}%)')
print()

if len(unmatched) > 0:
    # Get prop counts for unmatched players
    unmatched_stats = []
    for player in unmatched:
        count = len(archive[archive['player'] == player])
        markets = archive[archive['player'] == player]['market'].unique()
        unmatched_stats.append({
            'player': player,
            'props': count,
            'markets': ', '.join(markets[:3])
        })
    
    unmatched_df = pd.DataFrame(unmatched_stats).sort_values('props', ascending=False)
    
    print('Top 20 unmatched players (most props):')
    for _, row in unmatched_df.head(20).iterrows():
        print(f'  {row["player"]:35s} - {row["props"]:3} props ({row["markets"]})')
    print()

# ============================================================================
# SECTION 4: ZERO WIN RATE PLAYERS
# ============================================================================
print('📋 SECTION 4: PLAYERS WITH 0% WIN RATE')
print('-' * 80)

# Check for anytime_td market (yes/no, not over/under)
td_props = training[training['market'] == 'player_anytime_td']
print(f'Anytime TD props: {len(td_props):,} ({len(td_props)/len(training)*100:.1f}% of total)')
print(f'Note: TD props are Yes/No bets, not Over/Under')
print()

# Check actual_value to see if it's populated
null_actual = training['actual_value'].isna().sum()
print(f'Props with NULL actual_value: {null_actual:,} ({null_actual/len(training)*100:.1f}%)')
print()

# Check for players with all losses
over_losses = training.groupby('player')['bet_outcome_over'].agg(['sum', 'count'])
zero_wins = over_losses[over_losses['sum'] == 0]

print(f'Players with 0 over wins: {len(zero_wins)}')
if len(zero_wins) > 0:
    print('Sample (min 5 props):')
    for player, row in zero_wins[zero_wins['count'] >= 5].head(10).iterrows():
        props_count = int(row['count'])
        # Get actual values for this player
        player_data = training[training['player'] == player]
        avg_line = player_data['line'].mean()
        avg_actual = player_data['actual_value'].mean()
        print(f'  {player:35s} - {props_count:2} props | Avg Line: {avg_line:5.1f} | Avg Actual: {avg_actual:5.1f}')
print()

# ============================================================================
# SECTION 5: MATCHING LOGIC VERIFICATION
# ============================================================================
print('📋 SECTION 5: MATCHING LOGIC VERIFICATION')
print('-' * 80)

# Check how build_prop_training_dataset.py matches names
print('Checking name normalization (from build_prop_training_dataset.py):')
print('  - Archive names are used as-is')
print('  - Sleeper names are converted to lowercase via player_key')
print('  - Matching uses (player_key, team) tuple')
print()

# Verify team abbreviations
archive_teams = set(archive['home_team'].dropna().unique()) | set(archive['away_team'].dropna().unique())
training_teams = set(training['team'].dropna().unique())

print(f'Unique teams in archive: {len(archive_teams)}')
print(f'Unique teams in training: {len(training_teams)}')
print()

if archive_teams != training_teams:
    missing_in_training = archive_teams - training_teams
    extra_in_training = training_teams - archive_teams
    if missing_in_training:
        print(f'Teams in archive but not training: {missing_in_training}')
    if extra_in_training:
        print(f'Teams in training but not archive: {extra_in_training}')
    print()

# ============================================================================
# SECTION 6: RECOMMENDATIONS
# ============================================================================
print('=' * 80)
print('💡 RECOMMENDATIONS')
print('=' * 80)
print()

match_rate = len(training) / len(archive) * 100

if match_rate < 50:
    print('❌ CRITICAL: Match rate below 50%')
    print('   Issue: Severe name matching problems')
    print('   Action: Review player name normalization logic')
elif match_rate < 70:
    print('⚠️  WARNING: Match rate below 70%')
    print('   Issue: Significant unmatched players')
    print('   Action: Improve name matching algorithm')
else:
    print('✅ GOOD: Match rate above 70%')
    print('   Status: Acceptable matching performance')

print()

if len(zero_wins) > 50:
    print('⚠️  Many players with 0% win rate')
    print('   Possible causes:')
    print('   1. Player name variations (Jr., Sr., II, III)')
    print('   2. Special characters in names')
    print('   3. Team mismatches')
    print('   4. Position-specific issues')
    print()

if len(unmatched) > len(archive_players) * 0.3:
    print('⚠️  High unmatched player count')
    print('   Action Items:')
    print('   1. Add fuzzy name matching')
    print('   2. Create player name alias dictionary')
    print('   3. Use player IDs instead of names')
    print()

# Save unmatched players report
if len(unmatched) > 0:
    unmatched_df.to_csv('reports/unmatched_players.csv', index=False)
    print(f'📄 Saved unmatched players report: reports/unmatched_players.csv')

print()
print('✅ Analysis complete!')

