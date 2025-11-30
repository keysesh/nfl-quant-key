#!/usr/bin/env python3
"""
NFL QUANT - Comprehensive Market Discovery

Systematic feature discovery for ALL prop markets:
1. pass_yds, rush_yds, rec_yds
2. receptions, pass_tds, rush_tds, rec_tds
3. completions, interceptions, pass_attempts

For each market:
- Phase 1: Data Audit
- Phase 2: Feature Engineering
- Phase 3: Correlation Analysis
- Phase 4: Betting Edge Validation
- Phase 5: Walk-Forward Validation
- Phase 6: Feature Importance

Usage:
    python scripts/analysis/comprehensive_market_discovery.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Market to stat column mapping
MARKET_MAPPING = {
    'player_pass_yds': {'stat_col': 'passing_yards', 'position': 'QB', 'play_type': 'pass'},
    'player_rush_yds': {'stat_col': 'rushing_yards', 'position': 'RB', 'play_type': 'run'},
    'player_reception_yds': {'stat_col': 'receiving_yards', 'position': ['WR', 'TE', 'RB'], 'play_type': 'pass'},
    'player_receptions': {'stat_col': 'receptions', 'position': ['WR', 'TE', 'RB'], 'play_type': 'pass'},
    'player_pass_tds': {'stat_col': 'passing_tds', 'position': 'QB', 'play_type': 'pass'},
    'player_rush_tds': {'stat_col': 'rushing_tds', 'position': 'RB', 'play_type': 'run'},
    'player_pass_completions': {'stat_col': 'completions', 'position': 'QB', 'play_type': 'pass'},
    'player_interceptions': {'stat_col': 'passing_interceptions', 'position': 'QB', 'play_type': 'pass'},
    'player_pass_attempts': {'stat_col': 'attempts', 'position': 'QB', 'play_type': 'pass'},
}


def load_all_data():
    """Load all required data sources."""
    logger.info("="*80)
    logger.info("PHASE 1: DATA AUDIT - Loading all data sources")
    logger.info("="*80)

    # Load weekly stats
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'
    stats = pd.read_parquet(stats_path)
    logger.info(f"Weekly stats: {len(stats):,} records")
    logger.info(f"  Columns: {len(stats.columns)}")
    logger.info(f"  Seasons: {sorted(stats['season'].unique())}")

    # Load PBP for all seasons
    pbp_dfs = []
    for year in [2023, 2024, 2025]:
        path = PROJECT_ROOT / 'data' / 'nflverse' / f'pbp_{year}.parquet'
        if path.exists():
            df = pd.read_parquet(path)
            df['season'] = year
            pbp_dfs.append(df)
            logger.info(f"  PBP {year}: {len(df):,} plays")
    pbp = pd.concat(pbp_dfs, ignore_index=True) if pbp_dfs else None

    # Load backtest data
    backtest_path = PROJECT_ROOT / 'data' / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
    backtest = pd.read_csv(backtest_path)
    logger.info(f"Backtest data: {len(backtest):,} records")

    # Show available markets
    logger.info(f"\nAvailable markets in backtest:")
    for market in sorted(backtest['market'].unique()):
        count = len(backtest[backtest['market'] == market])
        logger.info(f"  {market}: {count:,} records")

    return stats, pbp, backtest


def document_available_features(stats, pbp):
    """Document all available features from data sources."""
    logger.info("\n" + "="*80)
    logger.info("AVAILABLE FEATURES BY CATEGORY")
    logger.info("="*80)

    # Categorize columns
    categories = {
        'passing': [c for c in stats.columns if 'pass' in c.lower() or 'complet' in c.lower() or 'attempt' in c.lower()],
        'rushing': [c for c in stats.columns if 'rush' in c.lower() or 'carr' in c.lower()],
        'receiving': [c for c in stats.columns if 'recei' in c.lower() or 'target' in c.lower()],
        'epa': [c for c in stats.columns if 'epa' in c.lower()],
        'advanced': [c for c in stats.columns if any(x in c.lower() for x in ['cpoe', 'racr', 'wopr', 'share'])],
    }

    for cat, cols in categories.items():
        logger.info(f"\n{cat.upper()}:")
        for col in cols:
            non_null = stats[col].notna().sum()
            logger.info(f"  {col}: {non_null:,} non-null values")

    # PBP columns
    if pbp is not None:
        pbp_useful = [c for c in pbp.columns if any(x in c.lower() for x in
                     ['epa', 'wpa', 'success', 'yards', 'score', 'spread', 'total', 'pace'])]
        logger.info(f"\nPBP USEFUL COLUMNS ({len(pbp_useful)}):")
        for col in pbp_useful[:20]:
            logger.info(f"  {col}")

    return categories


def calculate_defense_metrics(pbp):
    """Calculate defensive metrics from PBP data."""
    logger.info("\n" + "="*80)
    logger.info("CALCULATING DEFENSIVE METRICS")
    logger.info("="*80)

    defense_metrics = {}

    # Pass defense
    pass_plays = pbp[pbp['play_type'] == 'pass']
    pass_def = pass_plays.groupby(['defteam', 'week', 'season']).agg(
        pass_def_epa=('epa', 'mean'),
        pass_yards_allowed=('passing_yards', 'sum'),
        pass_tds_allowed=('pass_touchdown', 'sum'),
        completions_allowed=('complete_pass', 'sum'),
        pass_plays=('play_id', 'count')
    ).reset_index()

    # Calculate trailing metrics
    pass_def = pass_def.sort_values(['defteam', 'season', 'week'])
    pass_def['trailing_pass_def_epa'] = pass_def.groupby('defteam')['pass_def_epa'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
    pass_def['trailing_pass_yards_allowed'] = pass_def.groupby('defteam')['pass_yards_allowed'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    defense_metrics['pass'] = pass_def
    logger.info(f"Pass defense: {len(pass_def):,} team-week records")

    # Rush defense
    rush_plays = pbp[pbp['play_type'] == 'run']
    rush_def = rush_plays.groupby(['defteam', 'week', 'season']).agg(
        rush_def_epa=('epa', 'mean'),
        rush_yards_allowed=('rushing_yards', 'sum'),
        rush_tds_allowed=('rush_touchdown', 'sum'),
        rush_plays=('play_id', 'count')
    ).reset_index()

    rush_def = rush_def.sort_values(['defteam', 'season', 'week'])
    rush_def['trailing_rush_def_epa'] = rush_def.groupby('defteam')['rush_def_epa'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
    rush_def['trailing_rush_yards_allowed'] = rush_def.groupby('defteam')['rush_yards_allowed'].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    defense_metrics['rush'] = rush_def
    logger.info(f"Rush defense: {len(rush_def):,} team-week records")

    return defense_metrics


def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace('.', '').replace("'", "")


def analyze_market(market, stats, pbp, backtest, defense_metrics):
    """Run complete discovery process for a single market."""
    logger.info("\n" + "="*80)
    logger.info(f"MARKET ANALYSIS: {market}")
    logger.info("="*80)

    config = MARKET_MAPPING.get(market)
    if not config:
        logger.warning(f"No config for {market}")
        return None

    stat_col = config['stat_col']
    position = config['position']
    play_type = config['play_type']

    # Filter backtest to this market
    bt = backtest[backtest['market'] == market].copy()
    logger.info(f"Backtest records: {len(bt):,}")

    if len(bt) == 0:
        logger.warning(f"No backtest data for {market}")
        return None

    # Filter stats by position
    if isinstance(position, list):
        player_stats = stats[stats['position'].isin(position)].copy()
    else:
        player_stats = stats[stats['position'] == position].copy()

    # Check if stat column exists
    if stat_col not in player_stats.columns:
        logger.warning(f"Stat column {stat_col} not in data")
        return None

    # Filter to players with data
    player_stats = player_stats[player_stats[stat_col] > 0].copy()
    logger.info(f"Player records with {stat_col} > 0: {len(player_stats):,}")

    # Calculate trailing stats for player
    player_stats = player_stats.sort_values(['player_id', 'season', 'week'])
    player_stats[f'trailing_{stat_col}'] = player_stats.groupby('player_id')[stat_col].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )

    # Calculate EPA trailing if available
    epa_col = f'{play_type}ing_epa' if play_type == 'pass' else f'{play_type}ing_epa'
    if epa_col in player_stats.columns:
        player_stats[f'trailing_{epa_col}'] = player_stats.groupby('player_id')[epa_col].transform(
            lambda x: x.shift(1).rolling(4, min_periods=1).mean()
        )

    # Add opponent defense metrics
    def_df = defense_metrics.get(play_type)
    if def_df is not None:
        def_col = f'trailing_{play_type}_def_epa'
        if def_col in def_df.columns:
            player_stats = player_stats.merge(
                def_df[['defteam', 'week', 'season', def_col]],
                left_on=['opponent_team', 'week', 'season'],
                right_on=['defteam', 'week', 'season'],
                how='left'
            )
            logger.info(f"Added {def_col} from defense metrics")

    # Normalize names for merge
    player_stats['player_norm'] = player_stats['player_display_name'].apply(normalize_name)
    bt['player_norm'] = bt['player'].apply(normalize_name)

    # Merge backtest with player stats
    merged = player_stats.merge(
        bt[['player_norm', 'season', 'week', 'line', 'under_hit', 'over_hit']],
        on=['player_norm', 'season', 'week'],
        how='inner'
    )

    # Drop rows without trailing data
    merged = merged[merged[f'trailing_{stat_col}'].notna()].copy()
    logger.info(f"Merged records with trailing data: {len(merged):,}")

    if len(merged) < 50:
        logger.warning(f"Insufficient data for analysis ({len(merged)} records)")
        return None

    # Phase 2: Feature Engineering
    logger.info(f"\n--- PHASE 2: Feature Engineering ---")

    # LVT (Line vs Trailing)
    merged['line_vs_trailing'] = merged['line'] / merged[f'trailing_{stat_col}'].replace(0, np.nan)

    # Line vs Actual (for residual analysis)
    merged['residual'] = merged[stat_col] - merged['line']
    merged['prediction_error'] = merged[stat_col] - merged[f'trailing_{stat_col}']

    # Defense-adjusted prediction
    def_col = f'trailing_{play_type}_def_epa'
    if def_col in merged.columns:
        merged['def_adjusted_pred'] = merged[f'trailing_{stat_col}'] * (1 + merged[def_col].fillna(0) * 5)

    # Phase 3: Correlation Analysis
    logger.info(f"\n--- PHASE 3: Correlation Analysis ---")

    # Features to analyze
    features_to_test = ['line_vs_trailing']
    if def_col in merged.columns:
        features_to_test.append(def_col)

    epa_trailing = f'trailing_{play_type}ing_epa'
    if epa_trailing in merged.columns:
        features_to_test.append(epa_trailing)

    correlations = {}
    logger.info(f"\nCorrelations with UNDER hit:")
    for feat in features_to_test:
        if feat in merged.columns:
            corr = merged[feat].corr(merged['under_hit'])
            correlations[feat] = corr
            logger.info(f"  {feat}: {corr:.4f}")

    logger.info(f"\nCorrelations with prediction residual:")
    for feat in features_to_test:
        if feat in merged.columns:
            corr = merged[feat].corr(merged['residual'])
            logger.info(f"  {feat}: {corr:.4f}")

    # Phase 4: Betting Edge Validation
    logger.info(f"\n--- PHASE 4: Betting Edge Validation ---")

    # Build model features
    model_features = []
    for feat in features_to_test:
        if feat in merged.columns:
            model_features.append(feat)

    if len(model_features) == 0:
        logger.warning("No valid features for model")
        return None

    X = merged[model_features].fillna(0)
    y = merged['under_hit']

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    merged['p_under'] = model.predict_proba(X)[:, 1]

    # Calculate ROI at thresholds
    logger.info(f"\nROI at probability thresholds (UNDER bets):")
    results = []
    for threshold in [0.50, 0.55, 0.575, 0.60, 0.625, 0.65]:
        mask = merged['p_under'] > threshold
        if mask.sum() == 0:
            continue

        hits = merged.loc[mask, 'under_hit'].sum()
        total = mask.sum()
        hit_rate = hits / total
        profit = hits * 0.909 - (total - hits) * 1.0
        roi = profit / total * 100

        results.append({
            'threshold': threshold,
            'bets': total,
            'wins': hits,
            'hit_rate': hit_rate,
            'roi': roi
        })
        logger.info(f"  P > {threshold:.0%}: {int(hits)}W-{int(total-hits)}L ({hit_rate:.1%}), ROI: {roi:+.1f}%")

    # LVT-only baseline
    logger.info(f"\nLVT-only baseline:")
    for lvt_thresh in [1.1, 1.2, 1.3, 1.4, 1.5]:
        mask = merged['line_vs_trailing'] > lvt_thresh
        if mask.sum() == 0:
            continue

        hits = merged.loc[mask, 'under_hit'].sum()
        total = mask.sum()
        hit_rate = hits / total
        profit = hits * 0.909 - (total - hits) * 1.0
        roi = profit / total * 100
        logger.info(f"  LVT > {lvt_thresh}: {int(hits)}W-{int(total-hits)}L ({hit_rate:.1%}), ROI: {roi:+.1f}%")

    # Phase 5: Walk-Forward Validation
    logger.info(f"\n--- PHASE 5: Walk-Forward Validation ---")

    merged['global_week'] = (merged['season'] - 2023) * 17 + merged['week']
    merged = merged.sort_values('global_week')

    wf_results = []
    for test_gw in range(15, merged['global_week'].max() + 1):
        train = merged[merged['global_week'] < test_gw]
        test = merged[merged['global_week'] == test_gw]

        if len(test) == 0 or len(train) < 100:
            continue

        X_train = train[model_features].fillna(0)
        y_train = train['under_hit']
        X_test = test[model_features].fillna(0)
        y_test = test['under_hit']

        wf_model = LogisticRegression(max_iter=1000)
        wf_model.fit(X_train, y_train)

        test = test.copy()
        test['p_under'] = wf_model.predict_proba(X_test)[:, 1]

        for _, row in test.iterrows():
            wf_results.append({
                'global_week': row['global_week'],
                'p_under': row['p_under'],
                'under_hit': row['under_hit']
            })

    wf_df = pd.DataFrame(wf_results)

    if len(wf_df) > 0:
        logger.info(f"\nWalk-forward results ({len(wf_df)} predictions):")
        for threshold in [0.55, 0.60, 0.65]:
            mask = wf_df['p_under'] > threshold
            if mask.sum() == 0:
                continue

            hits = wf_df.loc[mask, 'under_hit'].sum()
            total = mask.sum()
            hit_rate = hits / total
            profit = hits * 0.909 - (total - hits) * 1.0
            roi = profit / total * 100
            logger.info(f"  P > {threshold:.0%}: {int(hits)}W-{int(total-hits)}L ({hit_rate:.1%}), ROI: {roi:+.1f}%")

    # Phase 6: Feature Importance
    logger.info(f"\n--- PHASE 6: Feature Importance ---")
    logger.info(f"\nModel coefficients:")
    for feat, coef in zip(model_features, model.coef_[0]):
        logger.info(f"  {feat}: {coef:.4f}")
    logger.info(f"  intercept: {model.intercept_[0]:.4f}")

    # Interpretation
    logger.info(f"\nInterpretation:")
    for feat, coef in zip(model_features, model.coef_[0]):
        if 'lvt' in feat.lower() or 'trailing' in feat.lower():
            direction = "MORE" if coef > 0 else "LESS"
            logger.info(f"  - Higher {feat} → {direction} likely UNDER")
        elif 'def' in feat.lower() or 'epa' in feat.lower():
            direction = "MORE" if coef < 0 else "LESS"
            logger.info(f"  - Good defense (negative EPA) → {direction} likely UNDER")

    return {
        'market': market,
        'n_records': len(merged),
        'correlations': correlations,
        'results': results,
        'features': model_features,
        'coefficients': dict(zip(model_features, model.coef_[0]))
    }


def main():
    """Run comprehensive market discovery."""
    print("="*80)
    print("NFL QUANT - COMPREHENSIVE MARKET DISCOVERY")
    print("="*80)

    # Load data
    stats, pbp, backtest = load_all_data()

    # Document features
    document_available_features(stats, pbp)

    # Calculate defense metrics
    defense_metrics = calculate_defense_metrics(pbp)

    # Analyze each market
    all_results = {}

    for market in MARKET_MAPPING.keys():
        try:
            result = analyze_market(market, stats, pbp, backtest, defense_metrics)
            if result:
                all_results[market] = result
        except Exception as e:
            logger.error(f"Error analyzing {market}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: MARKETS WITH EDGE")
    print("="*80)

    for market, result in all_results.items():
        if result.get('results'):
            best = max(result['results'], key=lambda x: x['roi'])
            if best['roi'] > 5 and best['bets'] >= 30:
                print(f"\n{market}:")
                print(f"  Best threshold: {best['threshold']:.0%}")
                print(f"  Bets: {best['bets']}, Hit rate: {best['hit_rate']:.1%}")
                print(f"  ROI: {best['roi']:+.1f}%")
                print(f"  Features: {result['features']}")


if __name__ == '__main__':
    main()
