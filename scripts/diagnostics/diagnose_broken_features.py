#!/usr/bin/env python3
"""
Deep diagnosis of all 13 broken features.
Traces the full data path from source -> enriched_odds -> model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 90)
print("DEEP DIAGNOSIS: TRACING DATA PATH FOR EACH BROKEN FEATURE")
print("=" * 90)

# Load all relevant data sources
print("\n[LOADING DATA SOURCES]")

enriched_path = PROJECT_ROOT / 'data' / 'processed' / 'enriched_odds.parquet'
if enriched_path.exists():
    enriched = pd.read_parquet(enriched_path)
    print(f"  enriched_odds: {len(enriched)} rows, {len(enriched.columns)} columns")
else:
    print(f"  enriched_odds: NOT FOUND at {enriched_path}")
    enriched = None

weekly_stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
if weekly_stats_path.exists():
    weekly_stats = pd.read_parquet(weekly_stats_path)
    print(f"  weekly_stats: {len(weekly_stats)} rows, {len(weekly_stats.columns)} columns")
else:
    print(f"  weekly_stats: NOT FOUND")
    weekly_stats = None

participation_path = PROJECT_ROOT / 'data' / 'nflverse' / 'participation.parquet'
if participation_path.exists():
    participation = pd.read_parquet(participation_path)
    print(f"  participation: {len(participation)} rows, {len(participation.columns)} columns")
else:
    print(f"  participation: NOT FOUND")
    participation = None

ngs_receiving_path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_receiving.parquet'
if ngs_receiving_path.exists():
    ngs_receiving = pd.read_parquet(ngs_receiving_path)
    print(f"  ngs_receiving: {len(ngs_receiving)} rows, {len(ngs_receiving.columns)} columns")
else:
    print(f"  ngs_receiving: NOT FOUND")
    ngs_receiving = None

ngs_passing_path = PROJECT_ROOT / 'data' / 'nflverse' / 'ngs_passing.parquet'
if ngs_passing_path.exists():
    ngs_passing = pd.read_parquet(ngs_passing_path)
    print(f"  ngs_passing: {len(ngs_passing)} rows, {len(ngs_passing.columns)} columns")
else:
    print(f"  ngs_passing: NOT FOUND")
    ngs_passing = None

injuries_path = PROJECT_ROOT / 'data' / 'nflverse' / 'injuries.parquet'
if injuries_path.exists():
    injuries = pd.read_parquet(injuries_path)
    print(f"  injuries: {len(injuries)} rows, {len(injuries.columns)} columns")
else:
    print(f"  injuries: NOT FOUND")
    injuries = None

# Load model
model_path = PROJECT_ROOT / 'data' / 'models' / 'active_model.joblib'
if model_path.exists():
    model = joblib.load(model_path)
    print(f"  active_model: loaded (version {model.get('version', 'unknown')})")
else:
    print(f"  active_model: NOT FOUND")
    model = None

# Zero importance features to diagnose
zero_importance_features = [
    'adot', 'game_pace', 'lvt_x_defense', 'lvt_x_rest',
    'man_coverage_adjustment', 'oline_health_score',
    'opp_man_coverage_rate_trailing', 'opp_pressure_rate',
    'opp_wr1_receptions_allowed', 'pressure_rate',
    'slot_funnel_score', 'slot_snap_pct', 'trailing_catch_rate'
]

# Source mapping for each feature
source_mapping = {
    'adot': {
        'description': 'Average Depth of Target - yards downfield per target',
        'source_file': 'weekly_stats',
        'source_cols': ['receiving_air_yards', 'targets', 'target_share'],
        'calculation': 'receiving_air_yards / targets (trailing EWMA)',
        'alt_sources': ['ngs_receiving'],
    },
    'game_pace': {
        'description': 'Expected plays per game based on team tendencies',
        'source_file': 'weekly_stats',
        'source_cols': ['plays', 'passing_plays', 'rushing_plays'],
        'calculation': 'Sum of team plays per game (trailing EWMA)',
        'alt_sources': ['pbp aggregated'],
    },
    'pressure_rate': {
        'description': 'How often this player\'s QB is pressured',
        'source_file': 'ngs_passing',
        'source_cols': ['times_pressured', 'dropbacks', 'avg_time_to_throw'],
        'calculation': 'times_pressured / dropbacks (trailing)',
        'alt_sources': [],
    },
    'opp_pressure_rate': {
        'description': 'How often opponent defense pressures QBs',
        'source_file': 'ngs_passing',
        'source_cols': ['times_pressured', 'dropbacks'],
        'calculation': 'Opponent team aggregate: times_pressured / dropbacks',
        'alt_sources': [],
    },
    'slot_snap_pct': {
        'description': 'Percentage of snaps lined up in slot',
        'source_file': 'participation',
        'source_cols': ['snap_counts_slot', 'snap_counts_wide', 'snap_counts_total', 'route_participation'],
        'calculation': 'slot_snaps / total_route_snaps',
        'alt_sources': ['ngs_receiving'],
    },
    'trailing_catch_rate': {
        'description': 'Trailing catch rate (receptions / targets)',
        'source_file': 'weekly_stats',
        'source_cols': ['receptions', 'targets'],
        'calculation': 'receptions / targets (trailing EWMA)',
        'alt_sources': [],
    },
    'oline_health_score': {
        'description': 'O-line health based on injury status',
        'source_file': 'injuries',
        'source_cols': ['position', 'injury_status', 'gsis_id'],
        'calculation': 'Count healthy OL starters / 5',
        'alt_sources': [],
    },
    'man_coverage_adjustment': {
        'description': 'Adjustment for opponent man coverage rate',
        'source_file': 'participation',
        'source_cols': ['man_coverage', 'zone_coverage'],
        'calculation': 'man_coverage_rate vs league average',
        'alt_sources': [],
    },
    'opp_man_coverage_rate_trailing': {
        'description': 'Trailing opponent man coverage rate',
        'source_file': 'participation',
        'source_cols': ['man_coverage', 'zone_coverage'],
        'calculation': 'opponent man plays / total coverage plays (trailing)',
        'alt_sources': [],
    },
    'opp_wr1_receptions_allowed': {
        'description': 'Receptions allowed to opposing WR1s',
        'source_file': 'weekly_stats',
        'source_cols': ['receptions', 'position', 'target_share'],
        'calculation': 'Sum of WR1 receptions vs this opponent defense',
        'alt_sources': [],
    },
    'lvt_x_defense': {
        'description': 'Interaction: line_vs_trailing * opponent_def_epa',
        'source_file': 'enriched',
        'source_cols': ['line_vs_trailing', 'opponent_def_epa', 'opp_def_epa_pass'],
        'calculation': 'line_vs_trailing * opponent_def_epa',
        'alt_sources': [],
    },
    'lvt_x_rest': {
        'description': 'Interaction: line_vs_trailing * rest_days',
        'source_file': 'enriched',
        'source_cols': ['line_vs_trailing', 'rest_days'],
        'calculation': 'line_vs_trailing * normalized_rest_days',
        'alt_sources': [],
    },
    'slot_funnel_score': {
        'description': 'Slot receiver target funnel opportunity',
        'source_file': 'participation',
        'source_cols': ['slot_snaps', 'slot_targets', 'route_participation'],
        'calculation': 'slot_target_rate * route_participation',
        'alt_sources': [],
    },
}

# Diagnose each feature
results = {}

for feat in zero_importance_features:
    print(f"\n{'='*90}")
    print(f"FEATURE: {feat}")
    print(f"{'='*90}")

    info = source_mapping.get(feat, {})
    print(f"Description: {info.get('description', 'Unknown')}")
    print(f"Expected source: {info.get('source_file', 'Unknown')}")
    print(f"Calculation: {info.get('calculation', 'Unknown')}")

    result = {
        'feature': feat,
        'in_enriched': False,
        'null_pct': 100,
        'zero_pct': 100,
        'unique_values': 0,
        'variance': 0,
        'source_available': False,
        'source_cols_found': [],
        'source_cols_missing': [],
        'root_cause': 'Unknown',
        'fix_category': 'Unknown',
    }

    # Step 1: Check if in enriched_odds (final training data)
    print("\n[1] IN TRAINING DATA (enriched_odds)?")
    if enriched is not None and feat in enriched.columns:
        col = enriched[feat]
        null_pct = col.isna().mean() * 100
        zero_pct = (col == 0).mean() * 100 if pd.api.types.is_numeric_dtype(col) else 0
        unique = col.nunique()
        variance = col.var() if pd.api.types.is_numeric_dtype(col) else 0

        result['in_enriched'] = True
        result['null_pct'] = null_pct
        result['zero_pct'] = zero_pct
        result['unique_values'] = unique
        result['variance'] = variance

        print(f"    YES - Null: {null_pct:.1f}% | Zeros: {zero_pct:.1f}% | Unique: {unique} | Var: {variance:.6f}")
        if pd.api.types.is_numeric_dtype(col):
            non_null = col.dropna()
            if len(non_null) > 0:
                print(f"    Stats: mean={non_null.mean():.4f}, std={non_null.std():.4f}, min={non_null.min():.4f}, max={non_null.max():.4f}")
                print(f"    Sample non-null values: {non_null.head(5).tolist()}")
    else:
        print(f"    NO - Feature not found in enriched_odds!")
        result['root_cause'] = 'Not extracted to training data'
        result['fix_category'] = 'A - Add to extraction'

    # Step 2: Check source data availability
    print(f"\n[2] SOURCE DATA CHECK:")
    source_file = info.get('source_file', '')
    source_cols = info.get('source_cols', [])

    source_df = None
    if source_file == 'weekly_stats':
        source_df = weekly_stats
    elif source_file == 'participation':
        source_df = participation
    elif source_file == 'ngs_passing':
        source_df = ngs_passing
    elif source_file == 'ngs_receiving':
        source_df = ngs_receiving
    elif source_file == 'injuries':
        source_df = injuries
    elif source_file == 'enriched':
        source_df = enriched

    if source_df is not None:
        result['source_available'] = True
        print(f"    Source file '{source_file}' loaded: {len(source_df)} rows")

        available_cols = [c for c in source_cols if c in source_df.columns]
        missing_cols = [c for c in source_cols if c not in source_df.columns]
        result['source_cols_found'] = available_cols
        result['source_cols_missing'] = missing_cols

        print(f"    Available columns: {available_cols}")
        print(f"    Missing columns: {missing_cols}")

        # Check coverage of available columns
        for col in available_cols[:3]:  # Limit to first 3
            coverage = source_df[col].notna().mean() * 100
            print(f"      {col}: {coverage:.1f}% coverage")
    else:
        print(f"    Source file '{source_file}' NOT LOADED")
        if source_file not in ['enriched']:
            result['root_cause'] = f'Source file {source_file} not found'
            result['fix_category'] = 'B - Load source data'

    # Step 3: Determine root cause
    print(f"\n[3] ROOT CAUSE ANALYSIS:")

    if not result['in_enriched']:
        result['root_cause'] = 'Feature not extracted at all'
        result['fix_category'] = 'A - Add extraction'
        print(f"    Category A: Feature never added to extraction pipeline")
    elif result['null_pct'] > 95:
        if not result['source_available']:
            result['root_cause'] = 'Source data not loaded'
            result['fix_category'] = 'B - Load source'
        elif len(result['source_cols_missing']) > 0:
            result['root_cause'] = f"Missing source columns: {result['source_cols_missing']}"
            result['fix_category'] = 'B - Find column names'
        else:
            result['root_cause'] = 'Join/merge failure in extraction'
            result['fix_category'] = 'A - Fix join logic'
        print(f"    {result['fix_category']}: {result['root_cause']}")
    elif result['zero_pct'] > 95:
        if 'lvt_x' in feat:
            result['root_cause'] = 'Base features for interaction are 0'
            result['fix_category'] = 'D - Fix base features first'
        else:
            result['root_cause'] = 'Calculation returns 0 (division issue or missing data)'
            result['fix_category'] = 'C - Fix calculation'
        print(f"    {result['fix_category']}: {result['root_cause']}")
    elif result['unique_values'] <= 3:
        result['root_cause'] = 'Constant/near-constant values'
        result['fix_category'] = 'C - Fix calculation variance'
        print(f"    {result['fix_category']}: {result['root_cause']}")
    else:
        result['root_cause'] = 'Unknown - feature has variance but 0 importance'
        result['fix_category'] = 'E - Investigate model'
        print(f"    {result['fix_category']}: {result['root_cause']}")

    results[feat] = result

# Summary
print("\n" + "=" * 90)
print("SUMMARY: ROOT CAUSE CATEGORIES")
print("=" * 90)

categories = {}
for feat, res in results.items():
    cat = res['fix_category']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(feat)

for cat, feats in sorted(categories.items()):
    print(f"\n{cat}:")
    for f in feats:
        print(f"    - {f}: {results[f]['root_cause']}")

print("\n" + "=" * 90)
print("DETAILED SOURCE COLUMN INVESTIGATION")
print("=" * 90)

# Check what columns are actually available in each source
if weekly_stats is not None:
    print("\n[weekly_stats columns relevant to broken features]:")
    relevant = ['receiving_air_yards', 'targets', 'receptions', 'target_share',
                'air_yards_share', 'wopr', 'racr', 'receiving_epa']
    for col in relevant:
        if col in weekly_stats.columns:
            cov = weekly_stats[col].notna().mean() * 100
            print(f"    {col}: {cov:.1f}% coverage")
        else:
            print(f"    {col}: NOT FOUND")

if participation is not None:
    print("\n[participation columns relevant to broken features]:")
    print(f"    Available columns: {list(participation.columns)[:20]}...")
    slot_cols = [c for c in participation.columns if 'slot' in c.lower()]
    route_cols = [c for c in participation.columns if 'route' in c.lower()]
    snap_cols = [c for c in participation.columns if 'snap' in c.lower()]
    print(f"    Slot-related: {slot_cols}")
    print(f"    Route-related: {route_cols}")
    print(f"    Snap-related: {snap_cols}")

if ngs_passing is not None:
    print("\n[ngs_passing columns relevant to broken features]:")
    print(f"    Available columns: {list(ngs_passing.columns)}")
    pressure_cols = [c for c in ngs_passing.columns if 'pressure' in c.lower() or 'sack' in c.lower()]
    print(f"    Pressure-related: {pressure_cols}")

if ngs_receiving is not None:
    print("\n[ngs_receiving columns relevant to broken features]:")
    print(f"    Available columns: {list(ngs_receiving.columns)}")

if injuries is not None:
    print("\n[injuries columns]:")
    print(f"    Available columns: {list(injuries.columns)}")
    if 'position' in injuries.columns:
        print(f"    Positions: {injuries['position'].unique()[:20]}")

print("\n" + "=" * 90)
print("EXTRACTION CODE LOCATIONS TO CHECK")
print("=" * 90)

print("""
Key files to investigate:
1. nfl_quant/features/core.py - Main feature extraction
2. nfl_quant/features/batch_extractor.py - Batch extraction for training
3. scripts/train/train_model.py - Training data preparation
4. scripts/data/enrich_odds.py - Enrichment pipeline

Run these searches:
    grep -rn 'adot' nfl_quant/features/
    grep -rn 'slot_snap' nfl_quant/features/
    grep -rn 'pressure_rate' nfl_quant/features/
    grep -rn 'game_pace' nfl_quant/features/
""")
