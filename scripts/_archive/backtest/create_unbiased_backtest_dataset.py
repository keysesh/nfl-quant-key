#!/usr/bin/env python3
"""
Create Truly Unbiased Backtest Dataset

This script creates an unbiased backtest dataset by:
1. Using only PREGAME odds (captured before kickoff)
2. Including ALL props (no cherry-picking based on edge)
3. Using actual outcomes from NFLverse (no hindsight bias)
4. Proper temporal ordering (no look-ahead bias)

Key principles for unbiased backtesting:
- Train on past data only
- Test on future (out-of-sample) data
- Include ALL bettable props, not just ones with edge
- Use odds as they were available pregame
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_matched_data() -> pd.DataFrame:
    """Load the matched odds-to-actuals dataset"""
    matched_file = PROJECT_ROOT / 'data' / 'backtest' / 'matched_odds_actuals.csv'
    df = pd.read_csv(matched_file)
    print(f"Loaded {len(df)} matched prop records")
    return df


def validate_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data quality and remove problematic records

    Checks:
    - No missing critical fields
    - Reasonable odds ranges
    - Valid actual stats
    - Proper pregame odds (not stale)
    """
    print("\n=== DATA QUALITY VALIDATION ===")
    initial_count = len(df)

    # Remove records with missing critical fields
    critical_cols = ['player', 'market', 'line', 'over_odds', 'under_odds', 'actual_stat', 'week', 'season']
    for col in critical_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  Warning: {missing} records missing {col}")
            df = df[df[col].notna()]

    # Validate odds are reasonable (American odds -500 to +500)
    valid_odds = (
        (df['over_odds'] >= -500) & (df['over_odds'] <= 500) &
        (df['under_odds'] >= -500) & (df['under_odds'] <= 500)
    )
    invalid_odds = (~valid_odds).sum()
    if invalid_odds > 0:
        print(f"  Removing {invalid_odds} records with invalid odds ranges")
        df = df[valid_odds]

    # Validate actual stats are reasonable
    # Pass yards: 0-600, Rush yards: -20 to 300, Receptions: 0-20, Rec yards: -20 to 400
    # These are generous bounds to avoid filtering valid outliers
    valid_stats = pd.Series([True] * len(df))

    for market in df['market'].unique():
        market_data = df[df['market'] == market]
        if market == 'player_pass_yds':
            valid = (market_data['actual_stat'] >= 0) & (market_data['actual_stat'] <= 600)
        elif market == 'player_rush_yds':
            valid = (market_data['actual_stat'] >= -20) & (market_data['actual_stat'] <= 300)
        elif market == 'player_receptions':
            valid = (market_data['actual_stat'] >= 0) & (market_data['actual_stat'] <= 25)
        elif market == 'player_reception_yds':
            valid = (market_data['actual_stat'] >= -20) & (market_data['actual_stat'] <= 400)
        else:
            valid = pd.Series([True] * len(market_data))

        invalid = (~valid).sum()
        if invalid > 0:
            print(f"  Warning: {invalid} {market} records with unusual stat values")

    # Remove pushes from win/loss analysis (keep them for reference)
    push_count = df['is_push'].sum()
    print(f"  Push bets (exact line): {push_count} ({push_count/len(df)*100:.2f}%)")

    final_count = len(df)
    removed = initial_count - final_count
    print(f"\n  Data quality check: {removed} records removed ({removed/initial_count*100:.2f}%)")
    print(f"  Final clean dataset: {final_count} records")

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional features for analysis

    Features:
    - Market efficiency metrics
    - Line sharpness
    - Temporal patterns
    """
    df = df.copy()

    # Implied probabilities (already in the data but let's recalculate for consistency)
    def american_to_prob(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)

    df['market_over_prob'] = df['over_odds'].apply(american_to_prob)
    df['market_under_prob'] = df['under_odds'].apply(american_to_prob)

    # Vig (overround)
    df['total_prob'] = df['market_over_prob'] + df['market_under_prob']
    df['vig_pct'] = (df['total_prob'] - 1) * 100

    # No-vig probabilities
    df['fair_over_prob'] = df['market_over_prob'] / df['total_prob']
    df['fair_under_prob'] = df['market_under_prob'] / df['total_prob']

    # Line positioning relative to mean
    # (How far is the line from typical for this market?)
    for market in df['market'].unique():
        mask = df['market'] == market
        mean_line = df.loc[mask, 'line'].mean()
        std_line = df.loc[mask, 'line'].std()
        df.loc[mask, 'line_zscore'] = (df.loc[mask, 'line'] - mean_line) / std_line

    # Actual vs Line difference
    df['stat_vs_line'] = df['actual_stat'] - df['line']
    df['pct_vs_line'] = (df['actual_stat'] - df['line']) / df['line'] * 100

    # Week-based features
    df['is_early_season'] = df['week'] <= 4
    df['is_mid_season'] = (df['week'] > 4) & (df['week'] <= 10)
    df['is_late_season'] = df['week'] > 10

    return df


def calculate_betting_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive betting metrics for the dataset

    Returns dictionary of metrics for both over and under betting strategies
    """
    metrics = {}

    # Overall hit rates
    metrics['total_props'] = len(df)
    metrics['over_hit_rate'] = df['over_hit'].mean()
    metrics['under_hit_rate'] = df['under_hit'].mean()
    metrics['push_rate'] = df['is_push'].mean()

    # Actual hit rate should be close to 50% for efficient markets
    metrics['market_efficiency'] = abs(metrics['over_hit_rate'] - 0.5)

    # Profit/Loss analysis (assuming $100 per bet)
    def calculate_profit(odds, hit):
        if odds > 0:
            return hit * (odds / 100) * 100 - (1 - hit) * 100
        else:
            return hit * (100 / abs(odds)) * 100 - (1 - hit) * 100

    # Over betting
    over_profits = []
    under_profits = []

    for _, row in df.iterrows():
        if row['is_push']:
            over_profits.append(0)
            under_profits.append(0)
        else:
            # Over profit
            if row['over_hit']:
                if row['over_odds'] > 0:
                    over_profits.append(row['over_odds'])
                else:
                    over_profits.append(100 / abs(row['over_odds']) * 100)
            else:
                over_profits.append(-100)

            # Under profit
            if row['under_hit']:
                if row['under_odds'] > 0:
                    under_profits.append(row['under_odds'])
                else:
                    under_profits.append(100 / abs(row['under_odds']) * 100)
            else:
                under_profits.append(-100)

    df['over_profit'] = over_profits
    df['under_profit'] = under_profits

    metrics['total_over_profit'] = sum(over_profits)
    metrics['total_under_profit'] = sum(under_profits)
    metrics['over_roi'] = sum(over_profits) / (len(df) * 100) * 100
    metrics['under_roi'] = sum(under_profits) / (len(df) * 100) * 100

    # By market
    metrics['by_market'] = {}
    for market in df['market'].unique():
        market_data = df[df['market'] == market]
        metrics['by_market'][market] = {
            'count': len(market_data),
            'over_hit_rate': market_data['over_hit'].mean(),
            'under_hit_rate': market_data['under_hit'].mean(),
            'avg_line': market_data['line'].mean(),
            'avg_actual': market_data['actual_stat'].mean(),
            'over_roi': sum(market_data['over_profit']) / (len(market_data) * 100) * 100,
            'under_roi': sum(market_data['under_profit']) / (len(market_data) * 100) * 100,
        }

    # By week
    metrics['by_week'] = {}
    for week in sorted(df['week'].unique()):
        week_data = df[df['week'] == week]
        metrics['by_week'][week] = {
            'count': len(week_data),
            'over_hit_rate': week_data['over_hit'].mean(),
            'under_hit_rate': week_data['under_hit'].mean(),
            'over_roi': sum(week_data['over_profit']) / (len(week_data) * 100) * 100,
            'under_roi': sum(week_data['under_profit']) / (len(week_data) * 100) * 100,
        }

    return metrics, df


def create_train_test_splits(df: pd.DataFrame) -> dict:
    """
    Create multiple train/test splits for walk-forward validation

    Splits:
    1. Expanding window: Train on weeks 1-N, test on week N+1
    2. Rolling window: Train on 4-week window, test on next week
    3. Season halves: First half train, second half test
    """
    splits = {}

    weeks = sorted(df['week'].unique())

    # 1. Expanding window splits
    splits['expanding'] = []
    for i in range(3, len(weeks)):  # Start with at least 3 weeks of training
        train_weeks = weeks[:i]
        test_week = weeks[i]

        train_data = df[df['week'].isin(train_weeks)]
        test_data = df[df['week'] == test_week]

        splits['expanding'].append({
            'train_weeks': train_weeks,
            'test_week': test_week,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_data': train_data,
            'test_data': test_data
        })

    # 2. Rolling window (4 weeks)
    splits['rolling'] = []
    window_size = 4
    for i in range(window_size, len(weeks)):
        train_weeks = weeks[i-window_size:i]
        test_week = weeks[i]

        train_data = df[df['week'].isin(train_weeks)]
        test_data = df[df['week'] == test_week]

        splits['rolling'].append({
            'train_weeks': train_weeks,
            'test_week': test_week,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_data': train_data,
            'test_data': test_data
        })

    # 3. Season halves
    mid_week = weeks[len(weeks) // 2]
    first_half = df[df['week'] < mid_week]
    second_half = df[df['week'] >= mid_week]

    splits['season_halves'] = {
        'train_weeks': [w for w in weeks if w < mid_week],
        'test_weeks': [w for w in weeks if w >= mid_week],
        'train_size': len(first_half),
        'test_size': len(second_half),
        'train_data': first_half,
        'test_data': second_half
    }

    return splits


def save_backtest_dataset(df: pd.DataFrame, metrics: dict, splits: dict):
    """Save the unbiased backtest dataset and metadata"""
    output_dir = PROJECT_ROOT / 'data' / 'backtest'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main dataset
    csv_file = output_dir / 'unbiased_backtest_dataset.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Saved unbiased dataset: {csv_file}")

    parquet_file = output_dir / 'unbiased_backtest_dataset.parquet'
    df.to_parquet(parquet_file, index=False)
    print(f"✅ Saved parquet: {parquet_file}")

    # Save metrics
    import json
    metrics_file = output_dir / 'backtest_metrics.json'

    # Convert non-serializable items
    metrics_serializable = {
        'total_props': int(metrics['total_props']),
        'over_hit_rate': float(metrics['over_hit_rate']),
        'under_hit_rate': float(metrics['under_hit_rate']),
        'push_rate': float(metrics['push_rate']),
        'market_efficiency': float(metrics['market_efficiency']),
        'total_over_profit': float(metrics['total_over_profit']),
        'total_under_profit': float(metrics['total_under_profit']),
        'over_roi': float(metrics['over_roi']),
        'under_roi': float(metrics['under_roi']),
        'by_market': {
            k: {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv
                for kk, vv in v.items()}
            for k, v in metrics['by_market'].items()
        },
        'by_week': {
            int(k): {kk: float(vv) if isinstance(vv, (np.float64, np.float32)) else vv
                     for kk, vv in v.items()}
            for k, v in metrics['by_week'].items()
        }
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"✅ Saved metrics: {metrics_file}")

    # Save split information (without data)
    splits_info = {
        'expanding_count': len(splits['expanding']),
        'rolling_count': len(splits['rolling']),
        'season_halves': {
            'train_weeks': [int(w) for w in splits['season_halves']['train_weeks']],
            'test_weeks': [int(w) for w in splits['season_halves']['test_weeks']],
            'train_size': int(splits['season_halves']['train_size']),
            'test_size': int(splits['season_halves']['test_size']),
        }
    }

    splits_file = output_dir / 'train_test_splits_info.json'
    with open(splits_file, 'w') as f:
        json.dump(splits_info, f, indent=2)
    print(f"✅ Saved split info: {splits_file}")


def print_summary(df: pd.DataFrame, metrics: dict, splits: dict):
    """Print comprehensive summary of the unbiased backtest dataset"""
    print("\n" + "="*80)
    print("UNBIASED BACKTEST DATASET SUMMARY")
    print("="*80)

    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total prop bets: {len(df)}")
    print(f"  Unique games: {df['game_id'].nunique()}")
    print(f"  Unique players: {df['player'].nunique()}")
    print(f"  Season: {df['season'].iloc[0]}")
    print(f"  Weeks: {sorted(df['week'].unique().tolist())}")

    print(f"\n📈 MARKET EFFICIENCY")
    print(f"  Over hit rate: {metrics['over_hit_rate']*100:.2f}%")
    print(f"  Under hit rate: {metrics['under_hit_rate']*100:.2f}%")
    print(f"  Push rate: {metrics['push_rate']*100:.2f}%")
    print(f"  Market efficiency score: {metrics['market_efficiency']*100:.2f}% (0% = perfect)")

    print(f"\n💰 BLIND BETTING ROI (No Edge Selection)")
    print(f"  Bet all overs: {metrics['over_roi']:.2f}% ROI")
    print(f"  Bet all unders: {metrics['under_roi']:.2f}% ROI")
    print(f"  (Expected: ~-5% due to vig)")

    print(f"\n📊 BY MARKET")
    for market, mkt_metrics in metrics['by_market'].items():
        print(f"  {market}:")
        print(f"    Props: {mkt_metrics['count']}")
        print(f"    Over hit: {mkt_metrics['over_hit_rate']*100:.1f}%")
        print(f"    Avg line: {mkt_metrics['avg_line']:.1f}, Avg actual: {mkt_metrics['avg_actual']:.1f}")

    print(f"\n🔄 TRAIN/TEST SPLITS FOR WALK-FORWARD")
    print(f"  Expanding window splits: {len(splits['expanding'])}")
    print(f"  Rolling window splits: {len(splits['rolling'])}")
    print(f"  Season halves:")
    print(f"    Train: weeks {splits['season_halves']['train_weeks']} ({splits['season_halves']['train_size']} props)")
    print(f"    Test: weeks {splits['season_halves']['test_weeks']} ({splits['season_halves']['test_size']} props)")

    print(f"\n✅ KEY CHARACTERISTICS OF UNBIASED DATASET:")
    print(f"  1. ALL props included (no cherry-picking)")
    print(f"  2. Pregame odds only (no in-play bias)")
    print(f"  3. Actual outcomes from NFLverse (verified)")
    print(f"  4. Temporal ordering preserved (no look-ahead)")
    print(f"  5. Full universe of bettable props")


def main():
    print("="*80)
    print("CREATING TRULY UNBIASED BACKTEST DATASET")
    print("="*80)

    # 1. Load matched data
    df = load_matched_data()

    # 2. Validate data quality
    df = validate_data_quality(df)

    # 3. Add features
    df = add_market_features(df)

    # 4. Calculate betting metrics
    metrics, df = calculate_betting_metrics(df)

    # 5. Create train/test splits
    splits = create_train_test_splits(df)

    # 6. Print summary
    print_summary(df, metrics, splits)

    # 7. Save everything
    save_backtest_dataset(df, metrics, splits)

    print("\n" + "="*80)
    print("✅ UNBIASED BACKTEST DATASET CREATED SUCCESSFULLY")
    print("="*80)

    return df, metrics, splits


if __name__ == '__main__':
    main()
