#!/usr/bin/env python3
"""
Generate Summary Tables - Quick Reference

Create formatted tables summarizing all key findings for easy reference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from nfl_quant.utils.player_names import normalize_player_name

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_data():
    """Load and prepare data."""
    odds_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    odds = pd.read_csv(odds_path)
    odds['player_norm'] = odds['player'].apply(normalize_player_name)

    stats_2024 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)
    stats_2023 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2023.csv', low_memory=False)
    stats = pd.concat([stats_2024, stats_2023], ignore_index=True)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    stats['trailing_receptions'] = (
        stats.groupby('player_norm')['receptions']
        .transform(lambda x: x.rolling(4, min_periods=1).mean().shift(1))
    )

    merged = odds.merge(
        stats[['player_norm', 'season', 'week', 'trailing_receptions']].drop_duplicates(),
        on=['player_norm', 'season', 'week'],
        how='left'
    )

    return merged


def table_1_year_market_distribution():
    """Table 1: UNDER Rate by Year and Market"""
    print("\n" + "="*80)
    print("TABLE 1: UNDER HIT RATE BY YEAR AND MARKET")
    print("="*80)

    df = load_data()

    year_market = df.groupby(['season', 'market']).agg({
        'under_hit': ['count', 'mean']
    }).round(3)
    year_market.columns = ['Count', 'UNDER Rate']

    print("\n", year_market)

    # Summary by year
    print("\n" + "-"*80)
    print("SUMMARY BY YEAR:")
    year_summary = df.groupby('season').agg({
        'under_hit': ['count', 'mean']
    }).round(3)
    year_summary.columns = ['Total Bets', 'UNDER Rate']
    print(year_summary)

    print("\n" + "-"*80)
    print("KEY INSIGHT: UNDER rate increasing from 50.0% (2023) to 54.4% (2025)")


def table_2_high_confidence_players():
    """Table 2: High-Confidence UNDER/OVER Players"""
    print("\n" + "="*80)
    print("TABLE 2: HIGH-CONFIDENCE PLAYERS (Min 15 bets)")
    print("="*80)

    df = load_data()
    rec_df = df[df['market'] == 'player_receptions'].copy()

    player_stats = rec_df.groupby('player_norm').agg({
        'under_hit': ['count', 'mean']
    }).round(3)
    player_stats.columns = ['Bets', 'UNDER Rate']
    player_stats = player_stats[player_stats['Bets'] >= 15]

    print("\nTOP 15 UNDER PLAYERS:")
    print(player_stats.sort_values('UNDER Rate', ascending=False).head(15))

    print("\nTOP 10 OVER PLAYERS:")
    print(player_stats.sort_values('UNDER Rate').head(10))


def table_3_lvt_performance():
    """Table 3: LVT Threshold Performance"""
    print("\n" + "="*80)
    print("TABLE 3: LVT THRESHOLD PERFORMANCE (Receptions, 2025)")
    print("="*80)

    df = load_data()
    rec_df = df[df['market'] == 'player_receptions'].dropna(subset=['trailing_receptions']).copy()
    rec_df['lvt'] = rec_df['line'] - rec_df['trailing_receptions']
    test_df = rec_df[rec_df['season'] == 2025].copy()

    results = []
    for thresh in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        mask = test_df['lvt'] > thresh
        if mask.sum() > 0:
            hit_rate = test_df[mask]['under_hit'].mean()
            roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
            results.append({
                'LVT Threshold': f'> {thresh}',
                'N Bets': mask.sum(),
                'Hit Rate': f'{hit_rate:.1%}',
                'ROI': f'{roi:+.1f}%'
            })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    print("\n" + "-"*80)
    print("KEY INSIGHT: LVT > 1.5 has 76.0% hit rate, +45.1% ROI")


def table_4_ensemble_comparison():
    """Table 4: V8 vs V10 vs V11 Ensemble"""
    print("\n" + "="*80)
    print("TABLE 4: STRATEGY COMPARISON (Receptions, 2025)")
    print("="*80)

    df = load_data()
    rec_df = df[df['market'] == 'player_receptions'].dropna(subset=['trailing_receptions']).copy()
    rec_df['lvt'] = rec_df['line'] - rec_df['trailing_receptions']

    rec_df = rec_df.sort_values(['player_norm', 'season', 'week'])
    rec_df['player_under_rate'] = (
        rec_df.groupby('player_norm')['under_hit']
        .transform(lambda x: x.rolling(10, min_periods=5).mean().shift(1))
    )

    test_df = rec_df[rec_df['season'] == 2025].dropna(subset=['player_under_rate']).copy()

    strategies = {
        'Baseline (bet all)': test_df.index,
        'V8 (LVT > 1.5)': test_df[test_df['lvt'] > 1.5].index,
        'V10 (Player > 0.65)': test_df[test_df['player_under_rate'] > 0.65].index,
        'V11 Ensemble (LVT>1.5 OR Player>0.65)': test_df[(test_df['lvt'] > 1.5) | (test_df['player_under_rate'] > 0.65)].index,
        'V11 Conservative (LVT>1.5 OR Player>0.70)': test_df[(test_df['lvt'] > 1.5) | (test_df['player_under_rate'] > 0.70)].index,
    }

    results = []
    for name, idx in strategies.items():
        segment = test_df.loc[idx]
        if len(segment) > 0:
            hit_rate = segment['under_hit'].mean()
            roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
            results.append({
                'Strategy': name,
                'N Bets': len(segment),
                'Hit Rate': f'{hit_rate:.1%}',
                'ROI': f'{roi:+.1f}%',
                'Expected Weekly': f'{len(segment) * 18 / 52:.0f}'  # Estimate weekly volume
            })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    print("\n" + "-"*80)
    print("RECOMMENDATION: V11 Ensemble for balanced volume and ROI")


def table_5_line_level_analysis():
    """Table 5: Performance by Line Level"""
    print("\n" + "="*80)
    print("TABLE 5: PERFORMANCE BY LINE LEVEL (Receptions, 2025)")
    print("="*80)

    df = load_data()
    rec_df = df[df['market'] == 'player_receptions'].copy()
    test_df = rec_df[rec_df['season'] == 2025].copy()

    test_df['line_bin'] = pd.cut(test_df['line'], bins=[0, 1.5, 2.5, 3.5, 5.5, 7.5, 100],
                                   labels=['0-1.5', '1.5-2.5', '2.5-3.5', '3.5-5.5', '5.5-7.5', '7.5+'])

    results = []
    for line_bin in test_df['line_bin'].unique():
        if pd.isna(line_bin):
            continue
        segment = test_df[test_df['line_bin'] == line_bin]
        if len(segment) >= 10:
            hit_rate = segment['under_hit'].mean()
            roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
            results.append({
                'Line Range': str(line_bin),
                'N Bets': len(segment),
                'UNDER Rate': f'{hit_rate:.1%}',
                'ROI': f'{roi:+.1f}%'
            })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    print("\n" + "-"*80)
    print("KEY INSIGHT: Lines 3.5-7.5 have strongest edge")


def table_6_position_analysis():
    """Table 6: Performance by Position"""
    print("\n" + "="*80)
    print("TABLE 6: PERFORMANCE BY POSITION (Receptions, 2024-2025)")
    print("="*80)

    df = load_data()
    rec_df = df[df['market'] == 'player_receptions'].copy()

    stats_2024 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2024_2025.csv', low_memory=False)
    stats_2023 = pd.read_csv(PROJECT_ROOT / 'data' / 'nflverse' / 'player_stats_2023.csv', low_memory=False)
    stats = pd.concat([stats_2024, stats_2023], ignore_index=True)
    stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)

    position_map = stats[['player_norm', 'position']].drop_duplicates()
    rec_df = rec_df.merge(position_map, on='player_norm', how='left')

    test_df = rec_df[(rec_df['season'] >= 2024) & (rec_df['season'] <= 2025)].copy()

    results = []
    for pos in ['WR', 'RB', 'TE']:
        segment = test_df[test_df['position'] == pos]
        if len(segment) >= 50:
            hit_rate = segment['under_hit'].mean()
            roi = (hit_rate * 0.909 + (1 - hit_rate) * -1.0) * 100
            results.append({
                'Position': pos,
                'N Bets': len(segment),
                'UNDER Rate': f'{hit_rate:.1%}',
                'ROI': f'{roi:+.1f}%'
            })

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))

    print("\n" + "-"*80)
    print("KEY INSIGHT: Skip TEs (no edge), focus on WR and RB")


def generate_quick_reference():
    """Generate quick reference card"""
    print("\n" + "="*80)
    print("QUICK REFERENCE CARD")
    print("="*80)

    print("""
RECOMMENDED STRATEGY: V11 Ensemble
----------------------------------

Strategy A (LVT Edge):
  Rule: Bet UNDER when (line - trailing_4game_avg) > 1.5
  Expected: 76% hit rate, +45% ROI, ~25 bets/week
  Markets: All

Strategy B (Player Bias Edge):
  Rule: Bet UNDER when player_historical_under_rate > 0.65
  Expected: 56% hit rate, +7% ROI, ~190 bets/week
  Markets: Receptions only

Combined Performance:
  N bets: ~200/week
  Hit rate: 57-60%
  ROI: +9-12%

FILTERS (apply after strategy):
  ✓ Skip market: player_reception_yds (no edge)
  ✓ Skip lines: < 2.5 (coin flip)
  ✓ Skip position: TE for receptions (no edge)
  ✓ Focus lines: 3.5-7.5 (strongest edge)

HIGH-CONFIDENCE UNDER PLAYERS:
  Travis Kelce, Marvin Harrison Jr, Marquise Brown,
  DeMarcus Robinson, Zach Charbonnet, Jahan Dotson

AVOID:
  ✗ Reception yards market
  ✗ Lines below 2.5
  ✗ TEs for receptions
  ✗ Betting 2023 data (regime shift)

MONITORING:
  - Track weekly baseline UNDER rate
  - If < 50% for 3+ weeks: pause LVT strategy
  - Track Strategy A and B separately
  - Adjust thresholds if ROI < +5% for 2 weeks
""")


def main():
    """Generate all summary tables."""
    print("="*80)
    print("NFL QUANT - SUMMARY TABLES AND QUICK REFERENCE")
    print("="*80)

    table_1_year_market_distribution()
    table_2_high_confidence_players()
    table_3_lvt_performance()
    table_4_ensemble_comparison()
    table_5_line_level_analysis()
    table_6_position_analysis()
    generate_quick_reference()

    print("\n" + "="*80)
    print("TABLES GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nFor detailed analysis, see:")
    print("  - /reports/ROOT_CAUSE_ANALYSIS_FINAL.md")
    print("  - /reports/EXECUTIVE_SUMMARY_ROOT_CAUSE.md")


if __name__ == '__main__':
    main()
