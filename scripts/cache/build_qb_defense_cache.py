"""
Build QB-specific defense ranking caches.

Creates rankings for:
1. Defense vs QB Pass Attempts
2. Defense vs QB Completions
3. Defense vs QB Pass TDs
4. Defense Pressure Rate (sacks per game)
5. Defense vs Mobile QBs
6. Defense vs Pocket QBs

Usage:
    python scripts/cache/build_qb_defense_cache.py --season 2025
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / 'data' / 'cache'
NFLVERSE_DIR = PROJECT_ROOT / 'data' / 'nflverse'

# Mobile QB threshold (rushing yards per game)
MOBILE_QB_THRESHOLD = 25


def load_qb_stats(season: int) -> pd.DataFrame:
    """Load QB game stats from weekly_stats."""
    stats_path = NFLVERSE_DIR / 'weekly_stats.parquet'
    if not stats_path.exists():
        raise FileNotFoundError(f"weekly_stats.parquet not found at {stats_path}")

    stats = pd.read_parquet(stats_path)
    qb_stats = stats[
        (stats['position'] == 'QB') &
        (stats['season'] == season) &
        (stats['attempts'] >= 10)  # Only starters
    ].copy()

    logger.info(f"Loaded {len(qb_stats)} QB games for season {season}")
    return qb_stats


def classify_qb_mobility(qb_stats: pd.DataFrame) -> pd.DataFrame:
    """Classify QBs as Mobile or Pocket based on rushing yards."""
    # Calculate per-QB average rushing yards
    qb_mobility = qb_stats.groupby('player_id').agg({
        'rushing_yards': 'mean',
        'player_display_name': 'first',
        'team': 'last'
    })

    qb_mobility['is_mobile'] = qb_mobility['rushing_yards'] >= MOBILE_QB_THRESHOLD
    qb_mobility['qb_style'] = np.where(qb_mobility['is_mobile'], 'Mobile', 'Pocket')

    mobile_count = qb_mobility['is_mobile'].sum()
    pocket_count = (~qb_mobility['is_mobile']).sum()
    logger.info(f"Classified QBs: {mobile_count} Mobile, {pocket_count} Pocket")

    return qb_mobility


def build_defense_vs_qb_cache(qb_stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build defense vs QB statistics by week."""
    # Classify QB mobility
    qb_mobility = classify_qb_mobility(qb_stats)
    qb_stats = qb_stats.merge(
        qb_mobility[['qb_style']],
        left_on='player_id',
        right_index=True,
        how='left'
    )

    # Calculate trailing (prior weeks) stats for each defense
    results = []

    for week in sorted(qb_stats['week'].unique()):
        if week < 3:  # Need at least 2 weeks of data
            continue

        # Use only prior weeks
        prior = qb_stats[qb_stats['week'] < week]

        if len(prior) == 0:
            continue

        # League averages for normalization
        league_avgs = {
            'attempts': prior['attempts'].mean(),
            'completions': prior['completions'].mean(),
            'passing_tds': prior['passing_tds'].mean(),
            'passing_yards': prior['passing_yards'].mean(),
            'sacks_suffered': prior['sacks_suffered'].mean(),
        }

        # Defense vs ALL QBs
        def_all = prior.groupby('opponent_team').agg({
            'attempts': 'mean',
            'completions': 'mean',
            'passing_tds': 'mean',
            'passing_yards': 'mean',
            'sacks_suffered': 'mean',
            'player_id': 'count'
        }).rename(columns={'player_id': 'qb_games'})

        # Calculate vs average
        for col in ['attempts', 'completions', 'passing_tds', 'passing_yards']:
            if league_avgs[col] > 0:
                def_all[f'{col}_vs_avg'] = (def_all[col] - league_avgs[col]) / league_avgs[col]
            else:
                def_all[f'{col}_vs_avg'] = 0

        # Defense vs Mobile QBs
        mobile_prior = prior[prior['qb_style'] == 'Mobile']
        if len(mobile_prior) > 0:
            def_mobile = mobile_prior.groupby('opponent_team').agg({
                'attempts': 'mean',
                'completions': 'mean',
                'passing_yards': 'mean',
                'rushing_yards': 'mean',  # QB rushing allowed
            })
            def_mobile.columns = [f'vs_mobile_{c}' for c in def_mobile.columns]
            def_all = def_all.join(def_mobile, how='left')

        # Defense vs Pocket QBs
        pocket_prior = prior[prior['qb_style'] == 'Pocket']
        if len(pocket_prior) > 0:
            def_pocket = pocket_prior.groupby('opponent_team').agg({
                'attempts': 'mean',
                'completions': 'mean',
                'passing_yards': 'mean',
            })
            def_pocket.columns = [f'vs_pocket_{c}' for c in def_pocket.columns]
            def_all = def_all.join(def_pocket, how='left')

        def_all['week'] = week
        def_all['season'] = season
        def_all = def_all.reset_index()

        results.append(def_all)

    if not results:
        return pd.DataFrame()

    cache_df = pd.concat(results, ignore_index=True)
    logger.info(f"Built defense vs QB cache: {len(cache_df)} rows for {cache_df['week'].nunique()} weeks")

    return cache_df


def build_qb_style_cache(qb_stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build QB mobility style cache."""
    qb_mobility = classify_qb_mobility(qb_stats)
    qb_mobility['season'] = season
    qb_mobility = qb_mobility.reset_index()

    # Add game counts
    games = qb_stats.groupby('player_id').size().rename('games_played')
    qb_mobility = qb_mobility.merge(games, left_on='player_id', right_index=True, how='left')

    logger.info(f"Built QB style cache: {len(qb_mobility)} QBs")
    return qb_mobility


def main():
    parser = argparse.ArgumentParser(description='Build QB defense caches')
    parser.add_argument('--season', type=int, default=2025, help='Season to process')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("QB DEFENSE CACHE BUILDER")
    logger.info("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load QB stats
    qb_stats = load_qb_stats(args.season)

    # Build defense vs QB cache
    def_vs_qb = build_defense_vs_qb_cache(qb_stats, args.season)
    if len(def_vs_qb) > 0:
        # Save with year suffix
        path = CACHE_DIR / f'defense_vs_qb_{args.season}.parquet'
        def_vs_qb.to_parquet(path)
        logger.info(f"Saved defense_vs_qb to {path}")

        # Also save without suffix for dashboard
        path_main = CACHE_DIR / 'defense_vs_qb.parquet'
        def_vs_qb.to_parquet(path_main)
        logger.info(f"Saved defense_vs_qb to {path_main}")

    # Build QB style cache
    qb_style = build_qb_style_cache(qb_stats, args.season)
    if len(qb_style) > 0:
        path = CACHE_DIR / f'qb_style_{args.season}.parquet'
        qb_style.to_parquet(path)
        logger.info(f"Saved qb_style to {path}")

        path_main = CACHE_DIR / 'qb_style.parquet'
        qb_style.to_parquet(path_main)
        logger.info(f"Saved qb_style to {path_main}")

    logger.info("=" * 60)
    logger.info("QB DEFENSE CACHE BUILD COMPLETE")
    logger.info("=" * 60)

    # Print summary
    if len(def_vs_qb) > 0:
        print("\nDefense vs QB Rankings (Week 15):")
        latest = def_vs_qb[def_vs_qb['week'] == def_vs_qb['week'].max()]
        latest = latest.sort_values('attempts')
        print(latest[['opponent_team', 'attempts', 'completions', 'sacks_suffered', 'qb_games']].head(10).to_string())


if __name__ == '__main__':
    main()
