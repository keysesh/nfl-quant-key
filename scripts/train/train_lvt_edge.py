#!/usr/bin/env python3
"""
Train LVT Edge Model

Trains the LVT (Line vs Trailing) edge model for statistical reversion.
This edge captures when Vegas lines diverge significantly from trailing performance.

Target: 65-70% hit rate at low volume

Usage:
    python scripts/train/train_lvt_edge.py
    python scripts/train/train_lvt_edge.py --market player_receptions
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.edges.lvt_edge import LVTEdge
from nfl_quant.config_paths import MODELS_DIR, DATA_DIR
from configs.edge_config import EDGE_MARKETS, get_lvt_threshold


def load_training_data() -> pd.DataFrame:
    """Load historical odds/actuals data for training."""
    # Use enriched training data
    enriched_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_ENRICHED.csv'

    if enriched_path.exists():
        print(f"Loading enriched data: {enriched_path}")
        df = pd.read_csv(enriched_path, low_memory=False)
    else:
        # Fallback to regular combined data
        combined_path = DATA_DIR / 'backtest' / 'combined_odds_actuals_2023_2024_2025.csv'
        print(f"Loading combined data: {combined_path}")
        df = pd.read_csv(combined_path, low_memory=False)

    print(f"Loaded {len(df)} rows")
    return df


def prepare_trailing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing stats for LVT calculation."""
    from nfl_quant.utils.player_names import normalize_player_name

    df = df.copy()

    # Normalize player names
    if 'player_norm' not in df.columns:
        df['player_norm'] = df['player'].apply(normalize_player_name)

    # Sort by player and time
    df = df.sort_values(['player_norm', 'season', 'week'])

    # Compute trailing stats per market
    market_stat_map = {
        'player_receptions': 'receptions',
        'player_rush_yds': 'rushing_yards',
        'player_reception_yds': 'receiving_yards',
        'player_rush_attempts': 'carries',
        'player_pass_attempts': 'attempts',
        'player_pass_completions': 'completions',
    }

    # Load player stats for trailing calculations
    stats_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'
    if stats_path.exists():
        stats = pd.read_csv(stats_path, low_memory=False)
        stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
        stats = stats.sort_values(['player_norm', 'season', 'week'])

        # Calculate trailing stats
        for market, stat_col in market_stat_map.items():
            if stat_col in stats.columns:
                trailing_col = f'trailing_{stat_col}'
                stats[trailing_col] = (
                    stats.groupby('player_norm')[stat_col]
                    .transform(lambda x: x.ewm(span=6, min_periods=1).mean().shift(1))
                )

                # Merge into main df
                df = df.merge(
                    stats[['player_norm', 'season', 'week', trailing_col]].drop_duplicates(),
                    on=['player_norm', 'season', 'week'],
                    how='left',
                    suffixes=('', '_new')
                )

                # Handle duplicate columns
                if f'{trailing_col}_new' in df.columns:
                    df[trailing_col] = df[trailing_col].fillna(df[f'{trailing_col}_new'])
                    df.drop(columns=[f'{trailing_col}_new'], inplace=True)

    return df


def add_v23_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V23 opponent defense context features for LVT edge."""
    df = df.copy()

    # Try to load team defensive EPA
    def_epa_path = DATA_DIR / 'nflverse' / 'team_defensive_epa.parquet'
    if def_epa_path.exists():
        try:
            def_epa = pd.read_parquet(def_epa_path)

            # Merge by opponent
            if 'opponent' in df.columns:
                # Build lookup keyed by team/season/week
                if 'team' in def_epa.columns and 'def_epa' in def_epa.columns:
                    lookup = def_epa[['team', 'season', 'week', 'def_epa']].copy()
                    lookup = lookup.rename(columns={'team': 'opponent', 'def_epa': 'opp_def_epa'})
                    lookup = lookup.drop_duplicates(['opponent', 'season', 'week'])

                    df = df.merge(lookup, on=['opponent', 'season', 'week'], how='left')
                    df['opp_def_epa'] = df['opp_def_epa'].fillna(0.0)
                else:
                    df['opp_def_epa'] = 0.0
            else:
                df['opp_def_epa'] = 0.0

            print(f"  Opponent EPA coverage: {(df['opp_def_epa'] != 0).mean():.1%}")
        except Exception as e:
            print(f"  Warning: Could not load defensive EPA: {e}")
            df['opp_def_epa'] = 0.0
    else:
        print(f"  Warning: {def_epa_path} not found, using fallback")
        df['opp_def_epa'] = 0.0

    # Set has_opponent_context flag
    df['has_opponent_context'] = (df['opp_def_epa'] != 0).astype(int)

    return df


def add_v28_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V28 rest/elo features for LVT edge."""
    df = df.copy()

    # Add rest_days from schedules
    schedule_path = DATA_DIR / 'nflverse' / 'schedules.parquet'
    if schedule_path.exists():
        try:
            schedule = pd.read_parquet(schedule_path)

            # Build rest lookup for both home and away teams
            rest_data = []
            if 'away_team' in schedule.columns and 'away_rest' in schedule.columns:
                away = schedule[['season', 'week', 'away_team', 'away_rest']].copy()
                away.columns = ['season', 'week', 'team', 'rest_days']
                rest_data.append(away)
            if 'home_team' in schedule.columns and 'home_rest' in schedule.columns:
                home = schedule[['season', 'week', 'home_team', 'home_rest']].copy()
                home.columns = ['season', 'week', 'team', 'rest_days']
                rest_data.append(home)

            if rest_data:
                rest_lookup = pd.concat(rest_data).drop_duplicates(['season', 'week', 'team'])

                # Merge by player's team
                team_col = 'team' if 'team' in df.columns else 'player_team'
                if team_col in df.columns:
                    rest_lookup = rest_lookup.rename(columns={'team': team_col})
                    df = df.merge(rest_lookup, on=['season', 'week', team_col], how='left')
                    df['rest_days'] = df['rest_days'].fillna(7.0)
                else:
                    df['rest_days'] = 7.0

                print(f"  Rest days coverage: {(df['rest_days'] != 7.0).mean():.1%}")
            else:
                df['rest_days'] = 7.0
        except Exception as e:
            print(f"  Warning: Could not load schedules: {e}")
            df['rest_days'] = 7.0
    else:
        df['rest_days'] = 7.0

    # Add elo_diff (simple placeholder - would need elo_ratings.json)
    # For training, use 0.0 as default (neutral matchup)
    elo_path = DATA_DIR / 'models' / 'elo_ratings.json'
    if elo_path.exists():
        try:
            import json
            with open(elo_path) as f:
                elo_data = json.load(f)

            # Get team elos
            team_elos = elo_data.get('team_elos', {})

            def get_elo_diff(row):
                team = row.get('team') or row.get('player_team')
                opp = row.get('opponent')
                if team and opp and team in team_elos and opp in team_elos:
                    return team_elos[team] - team_elos[opp]
                return 0.0

            df['elo_diff'] = df.apply(get_elo_diff, axis=1)
            print(f"  Elo diff coverage: {(df['elo_diff'] != 0).mean():.1%}")
        except Exception as e:
            print(f"  Warning: Could not load elo ratings: {e}")
            df['elo_diff'] = 0.0
    else:
        df['elo_diff'] = 0.0

    # Add lvt_x_defense interaction feature
    # Captures how LVT signal interacts with opponent defense quality
    if 'line_vs_trailing' in df.columns and 'opp_def_epa' in df.columns:
        df['lvt_x_defense'] = df['line_vs_trailing'] * df['opp_def_epa']
        print(f"  lvt_x_defense coverage: {df['lvt_x_defense'].notna().mean():.1%}")
    else:
        df['lvt_x_defense'] = 0.0

    return df


def train_market(edge: LVTEdge, df: pd.DataFrame, market: str) -> dict:
    """Train LVT edge for a single market."""
    print(f"\n{'='*60}")
    print(f"Training LVT edge for: {market}")
    print(f"{'='*60}")

    # Filter to market
    market_df = df[df['market'] == market].copy()

    if len(market_df) < 100:
        print(f"  Insufficient data: {len(market_df)} samples")
        return None

    # Add trailing stats column name
    trailing_col_map = {
        'player_receptions': 'trailing_receptions',
        'player_rush_yds': 'trailing_rushing_yards',
        'player_reception_yds': 'trailing_receiving_yards',
        'player_rush_attempts': 'trailing_carries',
        'player_pass_attempts': 'trailing_attempts',
        'player_pass_completions': 'trailing_completions',
    }
    trailing_col = trailing_col_map.get(market)

    if trailing_col and trailing_col not in market_df.columns:
        print(f"  Missing {trailing_col} column")
        return None

    # Ensure we have under_hit column
    if 'under_hit' not in market_df.columns:
        if 'actual' in market_df.columns and 'line' in market_df.columns:
            market_df['under_hit'] = (market_df['actual'] < market_df['line']).astype(int)
        else:
            print("  Missing under_hit, actual, or line columns")
            return None

    # Add market context features
    # FIXED: shift BEFORE expanding to prevent data leakage
    if 'market_under_rate' not in market_df.columns:
        market_df['market_under_rate'] = market_df['under_hit'].shift(1).expanding().mean().fillna(0.5)

    if 'vegas_spread' not in market_df.columns:
        market_df['vegas_spread'] = 0

    if 'implied_team_total' not in market_df.columns:
        market_df['implied_team_total'] = 24.0

    # Train
    try:
        metrics = edge.train(market_df, market)
        print(f"\n  Results for {market}:")
        print(f"    Samples (high-LVT): {metrics['n_samples']}")
        print(f"    Filter rate: {metrics['filter_rate']:.1%}")
        print(f"    Train accuracy: {metrics['train_accuracy']:.1%}")
        print(f"\n    Top features:")
        sorted_imp = sorted(metrics['feature_importance'].items(), key=lambda x: -x[1])
        for feat, imp in sorted_imp[:5]:
            print(f"      {feat}: {imp:.1%}")
        return metrics
    except Exception as e:
        print(f"  Error training: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train LVT Edge Model")
    parser.add_argument('--market', type=str, help='Train single market only')
    args = parser.parse_args()

    print("="*60)
    print("LVT EDGE TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Load data
    df = load_training_data()

    # Prepare trailing stats
    print("\nPreparing trailing stats...")
    df = prepare_trailing_stats(df)

    # Add V23/V28 features
    print("\nAdding V23 opponent context features...")
    df = add_v23_opponent_features(df)

    print("\nAdding V28 situational features...")
    df = add_v28_situational_features(df)

    # Initialize edge
    edge = LVTEdge()

    # Determine markets to train
    if args.market:
        markets = [args.market]
    else:
        markets = EDGE_MARKETS

    # Train each market
    all_metrics = {}
    for market in markets:
        metrics = train_market(edge, df, market)
        if metrics:
            all_metrics[market] = metrics

    # Save model
    if all_metrics:
        save_path = MODELS_DIR / 'lvt_edge_model.joblib'
        edge.save(save_path)

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Trained markets: {list(all_metrics.keys())}")
        print(f"Model saved to: {save_path}")

        # Summary table
        print("\n" + "-"*60)
        print(f"{'Market':<25} {'Samples':<10} {'Filter %':<10} {'Accuracy':<10}")
        print("-"*60)
        for market, metrics in all_metrics.items():
            print(
                f"{market:<25} "
                f"{metrics['n_samples']:<10} "
                f"{metrics['filter_rate']*100:<9.1f}% "
                f"{metrics['train_accuracy']*100:<9.1f}%"
            )
    else:
        print("\nNo markets trained successfully")


if __name__ == '__main__':
    main()
