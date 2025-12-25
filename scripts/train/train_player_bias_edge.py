#!/usr/bin/env python3
"""
Train Player Bias Edge Model

Trains the Player Bias edge model for player-specific tendencies.
This edge captures players who consistently go over or under their lines.

Target: 55-60% hit rate at higher volume

Usage:
    python scripts/train/train_player_bias_edge.py
    python scripts/train/train_player_bias_edge.py --market player_receptions
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

from nfl_quant.edges.player_bias_edge import PlayerBiasEdge
from nfl_quant.config_paths import MODELS_DIR, DATA_DIR
from configs.edge_config import EDGE_MARKETS, get_player_bias_threshold


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


def prepare_player_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute player-specific features."""
    from nfl_quant.utils.player_names import normalize_player_name

    df = df.copy()

    # Normalize player names
    if 'player_norm' not in df.columns:
        df['player_norm'] = df['player'].apply(normalize_player_name)

    # Sort by player and time
    df = df.sort_values(['player_norm', 'season', 'week'])

    # Ensure we have under_hit column
    if 'under_hit' not in df.columns:
        if 'actual' in df.columns and 'line' in df.columns:
            df['under_hit'] = (df['actual'] < df['line']).astype(int)

    # Compute player_under_rate (rolling, shifted to prevent leakage)
    if 'player_under_rate' not in df.columns or df['player_under_rate'].isna().all():
        df['player_under_rate'] = (
            df.groupby('player_norm')['under_hit']
            .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
        )

    # Compute current_season_under_rate (current season only)
    current_season = df['season'].max()
    season_mask = df['season'] == current_season
    df['current_season_under_rate'] = 0.5  # Default
    if season_mask.any():
        current_season_df = df[season_mask].copy()
        current_season_df['_season_rate'] = (
            current_season_df.groupby('player_norm')['under_hit']
            .transform(lambda x: x.rolling(8, min_periods=2).mean().shift(1))
        )
        df.loc[season_mask, 'current_season_under_rate'] = current_season_df['_season_rate'].fillna(0.5)

    # Season games played (for current season only)
    df['season_games_played'] = 0
    if season_mask.any():
        df.loc[season_mask, 'season_games_played'] = df[season_mask].groupby('player_norm').cumcount()

    # Compute player_bias (average actual - line)
    if 'player_bias' not in df.columns or df['player_bias'].isna().all():
        if 'actual' in df.columns and 'line' in df.columns:
            df['_diff'] = df['actual'] - df['line']
            df['player_bias'] = (
                df.groupby('player_norm')['_diff']
                .transform(lambda x: x.rolling(10, min_periods=3).mean().shift(1))
            )
            df.drop(columns=['_diff'], inplace=True)

    # Compute player bet count
    df['player_bet_count'] = df.groupby('player_norm').cumcount()

    # Add LVT features if not present (needed for alignment features)
    if 'line_vs_trailing' not in df.columns:
        df['line_vs_trailing'] = 0.0

    # Compute alignment features
    player_tendency = df['player_under_rate'].fillna(0.5) - 0.5
    df['LVT_x_player_tendency'] = df['line_vs_trailing'] * player_tendency
    df['LVT_x_player_bias'] = df['line_vs_trailing'] * df['player_bias'].fillna(0)

    # Market alignment
    # FIXED: shift BEFORE expanding to prevent data leakage
    if 'market_under_rate' not in df.columns:
        df['market_under_rate'] = df['under_hit'].shift(1).expanding().mean().fillna(0.5)

    df['player_market_aligned'] = np.where(
        (df['player_under_rate'].fillna(0.5) > 0.5) == (df['market_under_rate'] > 0.5),
        1.0,
        -1.0
    )

    # Market bias strength
    df['market_bias_strength'] = abs(df['market_under_rate'] - 0.5) * 2

    # Load player stats for usage features
    stats_path = DATA_DIR / 'nflverse' / 'player_stats_2024_2025.csv'
    if stats_path.exists():
        stats = pd.read_csv(stats_path, low_memory=False)
        stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
        stats = stats.sort_values(['player_norm', 'season', 'week'])

        # Snap share and target share
        if 'offense_pct' in stats.columns:
            stats['snap_share'] = stats['offense_pct'] / 100
        else:
            stats['snap_share'] = 0.70

        if 'target_share' in stats.columns:
            pass
        elif 'targets' in stats.columns and 'team_targets' in stats.columns:
            stats['target_share'] = stats['targets'] / stats['team_targets'].replace(0, 1)
        else:
            stats['target_share'] = 0.15

        # Catch rate
        if 'receptions' in stats.columns and 'targets' in stats.columns:
            stats['trailing_catch_rate'] = (
                stats.groupby('player_norm')
                .apply(lambda x: x['receptions'].rolling(6, min_periods=1).sum() /
                       x['targets'].rolling(6, min_periods=1).sum().replace(0, 1))
                .reset_index(level=0, drop=True)
            ).shift(1)
        else:
            stats['trailing_catch_rate'] = np.nan  # Leave NaN for training - XGBoost handles it

        # Merge into main df
        merge_cols = ['player_norm', 'season', 'week', 'snap_share', 'target_share', 'trailing_catch_rate']
        available_cols = [c for c in merge_cols if c in stats.columns]
        df = df.merge(
            stats[available_cols].drop_duplicates(),
            on=['player_norm', 'season', 'week'],
            how='left',
            suffixes=('', '_stats')
        )

        # Handle duplicate columns
        for col in ['snap_share', 'target_share', 'trailing_catch_rate']:
            if f'{col}_stats' in df.columns:
                df[col] = df[col].fillna(df[f'{col}_stats'])
                df.drop(columns=[f'{col}_stats'], inplace=True)

    # For training: leave NaN for XGBoost to handle natively (learns missing patterns)
    # Only ensure columns exist - don't fill with fake defaults
    if 'target_share' not in df.columns:
        df['target_share'] = np.nan
    if 'snap_share' not in df.columns:
        df['snap_share'] = np.nan
    if 'trailing_catch_rate' not in df.columns:
        df['trailing_catch_rate'] = np.nan
    df['pos_rank'] = df.get('pos_rank', pd.Series(2, index=df.index)).fillna(2)
    df['is_starter'] = (df['pos_rank'] == 1).astype(int)

    return df


def add_v23_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V23 opponent defense context features for Player Bias edge."""
    from nfl_quant.utils.player_names import normalize_player_name

    df = df.copy()

    # Try to calculate opponent defense z-scores from weekly stats
    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'
    if stats_path.exists():
        try:
            stats = pd.read_parquet(stats_path)

            # Calculate team defense stats
            team_defense = stats.groupby(['season', 'week', 'recent_team']).agg({
                'passing_yards': 'sum',
                'rushing_yards': 'sum',
            }).reset_index()
            team_defense.columns = ['season', 'week', 'team', 'pass_yds_allowed', 'rush_yds_allowed']

            # Calculate league averages and z-scores
            for yds_col, z_col in [('pass_yds_allowed', 'opp_pass_yds_def_vs_avg'),
                                    ('rush_yds_allowed', 'opp_rush_yds_def_vs_avg')]:
                mean_val = team_defense[yds_col].mean()
                std_val = team_defense[yds_col].std()
                if std_val > 0:
                    team_defense[z_col] = (team_defense[yds_col] - mean_val) / std_val
                else:
                    team_defense[z_col] = 0.0

            # Merge by opponent
            if 'opponent' in df.columns:
                lookup = team_defense[['season', 'week', 'team',
                                       'opp_pass_yds_def_vs_avg', 'opp_rush_yds_def_vs_avg']].copy()
                lookup = lookup.rename(columns={'team': 'opponent'})
                lookup = lookup.drop_duplicates(['season', 'week', 'opponent'])

                df = df.merge(lookup, on=['season', 'week', 'opponent'], how='left')
                df['opp_pass_yds_def_vs_avg'] = df['opp_pass_yds_def_vs_avg'].fillna(0.0)
                df['opp_rush_yds_def_vs_avg'] = df['opp_rush_yds_def_vs_avg'].fillna(0.0)
            else:
                df['opp_pass_yds_def_vs_avg'] = 0.0
                df['opp_rush_yds_def_vs_avg'] = 0.0

            print(f"  Opponent defense z-score coverage: {(df['opp_pass_yds_def_vs_avg'] != 0).mean():.1%}")
        except Exception as e:
            print(f"  Warning: Could not compute opponent defense z-scores: {e}")
            df['opp_pass_yds_def_vs_avg'] = 0.0
            df['opp_rush_yds_def_vs_avg'] = 0.0
    else:
        df['opp_pass_yds_def_vs_avg'] = 0.0
        df['opp_rush_yds_def_vs_avg'] = 0.0

    # Add opp_def_epa from team_defensive_epa.parquet
    def_epa_path = DATA_DIR / 'nflverse' / 'team_defensive_epa.parquet'
    if def_epa_path.exists():
        try:
            def_epa = pd.read_parquet(def_epa_path)
            if 'opponent' in df.columns and 'team' in def_epa.columns and 'def_epa' in def_epa.columns:
                lookup = def_epa[['team', 'season', 'week', 'def_epa']].copy()
                lookup = lookup.rename(columns={'team': 'opponent', 'def_epa': 'opp_def_epa'})
                lookup = lookup.drop_duplicates(['opponent', 'season', 'week'])

                df = df.merge(lookup, on=['opponent', 'season', 'week'], how='left')
                df['opp_def_epa'] = df['opp_def_epa'].fillna(0.0)
            else:
                df['opp_def_epa'] = 0.0
            print(f"  Opponent EPA coverage: {(df['opp_def_epa'] != 0).mean():.1%}")
        except Exception as e:
            print(f"  Warning: Could not load defensive EPA: {e}")
            df['opp_def_epa'] = 0.0
    else:
        df['opp_def_epa'] = 0.0

    return df


def add_v28_injury_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add V28.1 injury features for Player Bias edge."""
    from nfl_quant.utils.player_names import normalize_player_name

    df = df.copy()

    injuries_path = DATA_DIR / 'nflverse' / 'injuries.parquet'
    if injuries_path.exists():
        try:
            injuries = pd.read_parquet(injuries_path)

            # Encode injury status: 0=None, 1=Questionable, 2=Doubtful, 3=Out
            status_map = {'Out': 3, 'Doubtful': 2, 'Questionable': 1, 'Probable': 0}
            if 'report_status' in injuries.columns:
                injuries['injury_status_encoded'] = injuries['report_status'].map(status_map).fillna(0).astype(int)
            else:
                injuries['injury_status_encoded'] = 0

            # Get player identifier
            if 'full_name' in injuries.columns:
                injuries['player_norm'] = injuries['full_name'].apply(normalize_player_name)
            elif 'player_name' in injuries.columns:
                injuries['player_norm'] = injuries['player_name'].apply(normalize_player_name)
            else:
                df['injury_status_encoded'] = 0
                df['has_injury_designation'] = 0
                return df

            # Create lookup
            injury_lookup = injuries[['player_norm', 'season', 'week', 'injury_status_encoded']].drop_duplicates(
                ['player_norm', 'season', 'week'], keep='last'
            )

            df = df.merge(injury_lookup, on=['player_norm', 'season', 'week'], how='left')
            df['injury_status_encoded'] = df['injury_status_encoded'].fillna(0).astype(int)
            df['has_injury_designation'] = (df['injury_status_encoded'] > 0).astype(int)

            print(f"  Injury designation coverage: {df['has_injury_designation'].mean():.1%}")
        except Exception as e:
            print(f"  Warning: Could not load injuries: {e}")
            df['injury_status_encoded'] = 0
            df['has_injury_designation'] = 0
    else:
        df['injury_status_encoded'] = 0
        df['has_injury_designation'] = 0

    return df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest_days feature for Player Bias edge."""
    df = df.copy()

    schedule_path = DATA_DIR / 'nflverse' / 'schedules.parquet'
    if schedule_path.exists():
        try:
            schedule = pd.read_parquet(schedule_path)

            # Build rest lookup
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

    return df


def train_market(edge: PlayerBiasEdge, df: pd.DataFrame, market: str) -> dict:
    """Train Player Bias edge for a single market."""
    print(f"\n{'='*60}")
    print(f"Training Player Bias edge for: {market}")
    print(f"{'='*60}")

    # Filter to market
    market_df = df[df['market'] == market].copy()

    if len(market_df) < 100:
        print(f"  Insufficient data: {len(market_df)} samples")
        return None

    threshold = get_player_bias_threshold(market)
    print(f"  Threshold: min_bets={threshold.min_bets}, min_rate={threshold.min_rate}")

    # Check available players
    player_counts = market_df.groupby('player_norm').size()
    qualified_players = (player_counts >= threshold.min_bets).sum()
    print(f"  Players with {threshold.min_bets}+ bets: {qualified_players}")

    # Train
    try:
        metrics = edge.train(market_df, market)
        print(f"\n  Results for {market}:")
        print(f"    Samples (strong bias): {metrics['n_samples']}")
        print(f"    Filter rate: {metrics['filter_rate']:.1%}")
        print(f"    Train accuracy: {metrics['train_accuracy']:.1%}")
        print(f"\n    Top features:")
        sorted_imp = sorted(metrics['feature_importance'].items(), key=lambda x: -x[1])
        for feat, imp in sorted_imp[:5]:
            print(f"      {feat}: {imp:.1%}")
        return metrics
    except Exception as e:
        print(f"  Error training: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Train Player Bias Edge Model")
    parser.add_argument('--market', type=str, help='Train single market only')
    args = parser.parse_args()

    print("="*60)
    print("PLAYER BIAS EDGE TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Load data
    df = load_training_data()

    # Prepare player features
    print("\nPreparing player features...")
    df = prepare_player_features(df)

    # Add V23/V28 features
    print("\nAdding V23 opponent context features...")
    df = add_v23_opponent_features(df)

    print("\nAdding V28.1 injury features...")
    df = add_v28_injury_features(df)

    print("\nAdding rest days...")
    df = add_rest_days(df)

    # Show sample stats
    print(f"\nData prepared:")
    print(f"  Total rows: {len(df)}")
    print(f"  With player_under_rate: {df['player_under_rate'].notna().sum()}")
    print(f"  With player_bias: {df['player_bias'].notna().sum()}")

    # Initialize edge
    edge = PlayerBiasEdge()

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
        save_path = MODELS_DIR / 'player_bias_edge_model.joblib'
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
