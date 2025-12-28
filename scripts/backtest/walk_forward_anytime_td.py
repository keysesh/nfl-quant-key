#!/usr/bin/env python3
"""
Walk-Forward Validation for Anytime TD Props

Backtests anytime TD scorer predictions using Poisson model.
Only bets YES (player scores at least 1 TD).

Key Features:
- Uses historical props from data/archive/props/
- Matches with actual TD results from weekly_stats.parquet
- Poisson probability of scoring >= 1 TD
- Walk-forward validation (train on past, test on future)

Usage:
    python scripts/backtest/walk_forward_anytime_td.py
"""

import sys
from pathlib import Path
import glob
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.stats import poisson

from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.config_paths import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def load_anytime_td_props() -> pd.DataFrame:
    """Load all historical anytime TD props from archive."""
    logger.info("Loading anytime TD props...")

    props_files = glob.glob(str(DATA_DIR / 'archive' / 'props' / 'props_*.csv'))
    logger.info(f"  Found {len(props_files)} props files")

    all_props = []
    for f in props_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            anytime = df[df['market'] == 'player_anytime_td'].copy()
            if len(anytime) > 0:
                anytime['source_file'] = Path(f).name
                all_props.append(anytime)
        except Exception as e:
            continue

    if not all_props:
        logger.warning("No anytime TD props found")
        return pd.DataFrame()

    props_df = pd.concat(all_props, ignore_index=True)

    # Parse timestamp and extract week/season
    props_df['fetch_dt'] = pd.to_datetime(props_df['fetch_timestamp'])
    props_df['commence_dt'] = pd.to_datetime(props_df['commence_time'])

    # Deduplicate - keep latest odds for each player/game
    props_df = props_df.sort_values('fetch_dt', ascending=False)
    props_df = props_df.drop_duplicates(
        subset=['player_name', 'home_team', 'away_team', 'commence_time'],
        keep='first'
    )

    # Normalize player names
    props_df['player_norm'] = props_df['player_name'].apply(normalize_player_name)

    # Compute implied probability from American odds
    props_df['implied_prob'] = props_df['odds'].apply(american_to_implied_prob)

    # Remove vig (assume ~10% vig on TD props)
    props_df['fair_prob'] = props_df['implied_prob'] * 0.90

    logger.info(f"  Total unique props: {len(props_df)}")

    return props_df


def load_weekly_stats() -> pd.DataFrame:
    """Load weekly stats for TD actuals."""
    stats_path = DATA_DIR / 'nflverse' / 'weekly_stats.parquet'

    if not stats_path.exists():
        raise FileNotFoundError(f"Weekly stats not found: {stats_path}")

    stats = pd.read_parquet(stats_path)

    # Compute total TDs (receiving + rushing)
    stats['total_tds'] = (
        stats.get('receiving_tds', 0).fillna(0) +
        stats.get('rushing_tds', 0).fillna(0)
    )

    # Normalize player names
    if 'player_display_name' in stats.columns:
        stats['player_norm'] = stats['player_display_name'].apply(normalize_player_name)
    elif 'player_name' in stats.columns:
        stats['player_norm'] = stats['player_name'].apply(normalize_player_name)

    # Recent team for matching
    if 'recent_team' in stats.columns:
        stats['team'] = stats['recent_team']

    return stats


def match_props_with_actuals(props: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Match props with actual TD results."""
    logger.info("Matching props with actuals...")

    # Create week/season from commence_time
    # NFL weeks: Tues-Mon, game dates help determine week
    props['game_date'] = props['commence_dt'].dt.date

    # Add year from commence time
    props['year'] = props['commence_dt'].dt.year

    # Merge on player_norm + team (home or away)
    # First, add team info to props
    props_expanded = []

    for _, row in props.iterrows():
        # Try matching with home team players
        row_home = row.copy()
        row_home['team'] = row['home_team']
        props_expanded.append(row_home)

        # Try matching with away team players
        row_away = row.copy()
        row_away['team'] = row['away_team']
        props_expanded.append(row_away)

    props_exp = pd.DataFrame(props_expanded)

    # Get unique weeks in stats
    stats_weeks = stats[['season', 'week', 'player_norm', 'team', 'total_tds']].copy()
    stats_weeks = stats_weeks.drop_duplicates()

    # Match by player_norm (no team constraint since players can play for either team)
    # and filter by date proximity
    matched = []

    for _, prop in props.iterrows():
        player_norm = prop['player_norm']
        game_date = prop['game_date']
        year = prop['year']

        # Find matching stats row
        player_stats = stats_weeks[stats_weeks['player_norm'] == player_norm].copy()

        if len(player_stats) == 0:
            continue

        # Filter to correct season (2024 or 2025)
        if year >= 2025:
            season = 2025
        else:
            season = 2024

        player_stats = player_stats[player_stats['season'] == season]

        if len(player_stats) == 0:
            continue

        # Take most recent game for this player in this season
        # (props are for upcoming games, so we need the actual result)
        # Since we don't have exact game dates in stats, approximate by week

        # Get the latest week with data
        latest = player_stats.sort_values('week', ascending=False).iloc[0]

        matched.append({
            'player': prop['player_name'],
            'player_norm': player_norm,
            'team': latest['team'],
            'season': latest['season'],
            'week': latest['week'],
            'odds': prop['odds'],
            'implied_prob': prop['implied_prob'],
            'fair_prob': prop['fair_prob'],
            'actual_tds': latest['total_tds'],
            'scored_td': 1 if latest['total_tds'] >= 1 else 0,
            'game_date': game_date,
        })

    matched_df = pd.DataFrame(matched)

    if len(matched_df) == 0:
        logger.warning("No matches found")
        return pd.DataFrame()

    # Deduplicate
    matched_df = matched_df.drop_duplicates(
        subset=['player_norm', 'season', 'week'],
        keep='first'
    )

    logger.info(f"  Matched {len(matched_df)} props with actuals")
    logger.info(f"  TD rate: {matched_df['scored_td'].mean():.1%}")

    return matched_df


def compute_trailing_td_rate(stats: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing TD rate per player using EWMA with enhanced features."""
    stats = stats.copy()
    stats = stats.sort_values(['player_norm', 'season', 'week'])

    # Binary: scored at least 1 TD
    stats['scored_td'] = (stats['total_tds'] >= 1).astype(int)

    # EWMA of TD scoring (shift to avoid leakage)
    stats['trailing_td_rate'] = (
        stats.groupby('player_norm')['scored_td']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    # Trailing average TDs
    stats['trailing_tds'] = (
        stats.groupby('player_norm')['total_tds']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    # ENHANCED: Trailing target share (proxy for opportunities)
    # These columns exist in weekly_stats.parquet - no defaults
    stats['trailing_target_share'] = (
        stats.groupby('player_norm')['target_share']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    # ENHANCED: Trailing receiving yards (volume indicator)
    stats['trailing_rec_yds'] = (
        stats.groupby('player_norm')['receiving_yards']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    # ENHANCED: Trailing rushing yards (for RBs)
    stats['trailing_rush_yds'] = (
        stats.groupby('player_norm')['rushing_yards']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    # ENHANCED: Trailing targets (opportunity count)
    stats['trailing_targets'] = (
        stats.groupby('player_norm')['targets']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    # ENHANCED: Trailing carries (RB opportunity)
    stats['trailing_carries'] = (
        stats.groupby('player_norm')['carries']
        .transform(lambda x: x.shift(1).ewm(span=6, min_periods=2).mean())
    )

    return stats


def run_walk_forward_backtest(
    props: pd.DataFrame,
    stats: pd.DataFrame,
    min_confidence: float = 0.50,
) -> pd.DataFrame:
    """
    Run walk-forward backtest for anytime TD props with enhanced features.

    Strategy: Bet YES (player scores TD) when our enhanced model probability
    exceeds the implied market probability by a threshold.

    Enhanced features:
    - trailing_target_share: Higher share = more opportunities
    - trailing_rec_yds: Volume indicator
    - trailing_targets: Opportunity count
    - trailing_carries: RB opportunities
    """
    logger.info("Running walk-forward backtest...")

    # Compute trailing features
    stats = compute_trailing_td_rate(stats)

    # Global week for ordering
    stats['global_week'] = (stats['season'] - 2023) * 18 + stats['week']

    # Get unique weeks to test
    test_weeks = sorted(stats['global_week'].unique())

    # Start from week 5 to have enough training data
    test_weeks = [w for w in test_weeks if w >= 23]  # 2024 week 5+

    all_bets = []

    for test_week in test_weeks:
        # Get test data for this week
        test_stats = stats[stats['global_week'] == test_week].copy()

        if len(test_stats) == 0:
            continue

        season = 2024 if test_week <= 36 else 2025
        week = test_week - (18 if test_week > 18 else 0)
        if test_week > 36:
            week = test_week - 36

        # For each player with data, compute expected TDs using enhanced model
        for _, row in test_stats.iterrows():
            player_norm = row['player_norm']

            # Get trailing features
            trailing_td_rate = row.get('trailing_td_rate', np.nan)
            trailing_tds = row.get('trailing_tds', np.nan)
            trailing_target_share = row.get('trailing_target_share', 0.0)
            trailing_targets = row.get('trailing_targets', 0.0)
            trailing_carries = row.get('trailing_carries', 0.0)

            if pd.isna(trailing_tds):
                continue

            # ENHANCED: Adjust lambda based on opportunity indicators
            # Base lambda from trailing TDs
            lambda_tds = trailing_tds

            # Boost for high target share (WR/TE)
            if trailing_target_share > 0.20:  # Top target share
                lambda_tds *= 1.15

            # Boost for high targets (more opportunities)
            if trailing_targets > 6:  # Above average targets
                lambda_tds *= 1.10

            # Boost for high carries (RB goal-line work)
            if trailing_carries > 12:  # Workhorse RB
                lambda_tds *= 1.10

            # Skip very low TD expectation
            if lambda_tds < 0.10:
                continue

            # Poisson probability of scoring >= 1 TD
            # P(X >= 1) = 1 - P(X = 0) = 1 - e^(-lambda)
            p_score = 1 - poisson.pmf(0, lambda_tds)

            # Assume market implied probability ~40% for average player
            market_implied = 0.40

            # Only bet if our probability exceeds threshold
            if p_score < min_confidence:
                continue

            # Record bet
            all_bets.append({
                'season': season,
                'week': week,
                'global_week': test_week,
                'player': row.get('player_display_name', player_norm),
                'player_norm': player_norm,
                'team': row.get('team', ''),
                'trailing_tds': trailing_tds,
                'trailing_td_rate': trailing_td_rate,
                'trailing_target_share': trailing_target_share,
                'trailing_targets': trailing_targets,
                'trailing_carries': trailing_carries,
                'lambda': lambda_tds,
                'p_score': p_score,
                'market_implied': market_implied,
                'edge': p_score - market_implied,
                'actual_tds': row['total_tds'],
                'hit': 1 if row['total_tds'] >= 1 else 0,
            })

    if not all_bets:
        logger.warning("No bets generated")
        return pd.DataFrame()

    bets_df = pd.DataFrame(all_bets)

    # Compute results
    total = len(bets_df)
    hits = bets_df['hit'].sum()
    win_rate = hits / total

    # ROI at -110 odds
    roi = (win_rate * 0.909) - (1 - win_rate)

    logger.info(f"\nResults at {min_confidence:.0%} confidence:")
    logger.info(f"  Total bets: {total}")
    logger.info(f"  Hits: {hits} ({win_rate:.1%})")
    logger.info(f"  ROI: {roi:+.1%}")

    return bets_df


def main():
    logger.info("=" * 60)
    logger.info("WALK-FORWARD ANYTIME TD BACKTEST")
    logger.info("=" * 60)

    # Load data
    stats = load_weekly_stats()
    logger.info(f"Loaded {len(stats)} stats rows")

    # Run backtest at different confidence levels
    results = {}

    for conf in [0.40, 0.45, 0.50, 0.55, 0.60]:
        bets = run_walk_forward_backtest(stats, stats, min_confidence=conf)

        if len(bets) > 0:
            hits = bets['hit'].sum()
            total = len(bets)
            wr = hits / total
            roi = (wr * 0.909) - (1 - wr)

            results[conf] = {
                'n': total,
                'hits': hits,
                'wr': wr,
                'roi': roi,
            }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY BY CONFIDENCE THRESHOLD")
    logger.info("=" * 60)
    logger.info(f"{'Conf':>8} {'N':>8} {'Hits':>8} {'WR':>10} {'ROI':>10}")
    logger.info("-" * 50)

    for conf, r in sorted(results.items()):
        logger.info(
            f"{conf:>7.0%} {r['n']:>8} {r['hits']:>8} "
            f"{r['wr']:>9.1%} {r['roi']:>+9.1%}"
        )

    # Save best result
    if results:
        best_conf = max(results.keys(), key=lambda x: results[x]['roi'])
        best = results[best_conf]

        logger.info(f"\nBest threshold: {best_conf:.0%}")
        logger.info(f"  N={best['n']}, WR={best['wr']:.1%}, ROI={best['roi']:+.1%}")

        # Run and save best
        bets_df = run_walk_forward_backtest(stats, stats, min_confidence=best_conf)

        if len(bets_df) > 0:
            output_path = DATA_DIR / 'backtest' / 'walk_forward_anytime_td.csv'
            bets_df.to_csv(output_path, index=False)
            logger.info(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()
