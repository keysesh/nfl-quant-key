#!/usr/bin/env python3
"""
NFL QUANT - Bet Results Tracker

Tracks V5 Edge Classifier bet results week-over-week to validate:
1. Predicted P(UNDER) vs actual hit rate
2. ROI by market and threshold
3. Feature importance (what's driving winning bets?)

Usage:
    python scripts/tracking/track_bet_results.py --week 12  # Track Week 12 results
    python scripts/tracking/track_bet_results.py --summary  # Show all-time summary
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRACKING_DIR = PROJECT_ROOT / 'data' / 'tracking'
TRACKING_FILE = TRACKING_DIR / 'bet_results_history.json'


def load_tracking_history() -> dict:
    """Load historical bet tracking data."""
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {'bets': [], 'summary': {}}


def save_tracking_history(data: dict):
    """Save bet tracking data."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRACKING_FILE, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved tracking data to {TRACKING_FILE}")


def get_actual_stats(week: int, season: int = 2025) -> pd.DataFrame:
    """Get actual player stats for a completed week.

    Uses the NFLverse R parquet data (canonical source).

    Args:
        week: NFL week number
        season: NFL season year
    """
    stats_path = PROJECT_ROOT / 'data' / 'nflverse' / 'weekly_stats.parquet'

    if not stats_path.exists():
        raise FileNotFoundError(f"Weekly stats not found: {stats_path}")

    stats = pd.read_parquet(stats_path)
    week_stats = stats[(stats['season'] == season) & (stats['week'] == week)]

    if len(week_stats) == 0:
        raise ValueError(f"No stats found for Week {week}, Season {season}. Games may not have started yet.")

    logger.info(f"Loaded {len(week_stats)} player stats for Week {week}")
    return week_stats


def get_week_recommendations(week: int) -> pd.DataFrame:
    """Load recommendations that were generated for a specific week."""
    import glob

    # Priority 1: live_prop_edges files (most reliable - timestamped)
    live_edge_patterns = [
        PROJECT_ROOT / 'reports' / f'live_prop_edges_week{week}_*.csv',
        PROJECT_ROOT / 'reports' / 'archive' / f'WEEK_{week}_LIVE_EDGES.csv',
    ]

    for pattern in live_edge_patterns:
        matches = sorted(glob.glob(str(pattern)), reverse=True)  # Most recent first
        if matches:
            df = pd.read_csv(matches[0])
            logger.info(f"Using live edges file: {matches[0]}")
            # Normalize column names for compatibility
            if 'direction' in df.columns and 'pick' not in df.columns:
                df['pick'] = df['direction']
            if 'model_prob' in df.columns and 'v5_p_under' not in df.columns:
                # For UNDER bets, model_prob IS the under probability
                # For OVER bets, it's the over probability (1 - under)
                df['v5_p_under'] = df.apply(
                    lambda r: r['model_prob'] if 'UNDER' in str(r.get('direction', '')).upper()
                    else 1 - r['model_prob'], axis=1
                )
            return df

    # Priority 2: Standard recommendation files
    archive_patterns = [
        PROJECT_ROOT / 'reports' / f'WEEK_{week}_RECOMMENDATIONS.csv',
        PROJECT_ROOT / 'reports' / f'WEEK{week}_RECOMMENDATIONS.csv',
        PROJECT_ROOT / 'reports' / 'archive' / f'week{week}_recommendations.csv',
    ]

    for path in archive_patterns:
        if path.exists():
            return pd.read_csv(path)

    # Fall back to current recommendations if it's this week
    current_path = PROJECT_ROOT / 'reports' / 'CURRENT_WEEK_RECOMMENDATIONS.csv'
    if current_path.exists():
        logger.warning(f"Using current recommendations file - may not be for Week {week}")
        return pd.read_csv(current_path)

    raise FileNotFoundError(f"No recommendations found for Week {week}")


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace('.', '').replace("'", "")


def track_week_results(week: int, season: int = 2025) -> dict:
    """
    Track bet results for a completed week (supports partial weeks).

    Returns dict with:
    - bets: list of individual bet results
    - pending: list of bets that couldn't be validated (game not played yet)
    - summary: aggregate metrics
    """
    logger.info(f"Tracking results for Week {week}, Season {season}")

    # Load recommendations
    try:
        recs = get_week_recommendations(week)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    # Load actual stats
    try:
        actuals = get_actual_stats(week, season)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return None

    # Normalize names for matching
    recs['player_norm'] = recs['player'].apply(normalize_name)
    actuals['player_norm'] = actuals['player_display_name'].apply(normalize_name)

    # Map markets to stat columns
    market_to_stat = {
        'player_reception_yds': 'receiving_yards',
        'player_receptions': 'receptions',
        'player_rush_yds': 'rushing_yards',
        'player_pass_yds': 'passing_yards',
    }

    results = []

    # Filter to validated bets
    # If v5_edge_validated column exists, use it; otherwise use all bets in supported markets
    if 'v5_edge_validated' in recs.columns:
        v5_bets = recs[recs['v5_edge_validated'] == True]
        if len(v5_bets) == 0:
            logger.warning("No V5-validated bets found in recommendations")
    else:
        v5_bets = pd.DataFrame()

    if len(v5_bets) == 0:
        # Fall back to all bets in supported markets (live_prop_edges format)
        v5_bets = recs[recs['market'].isin(market_to_stat.keys())]
        logger.info(f"Using all {len(v5_bets)} bets in supported markets")

    logger.info(f"Evaluating {len(v5_bets)} bets")

    pending = []  # Track bets we couldn't validate (game not played yet)

    for _, bet in v5_bets.iterrows():
        player_norm = bet['player_norm']
        market = bet['market']
        line = bet['line']
        pick = bet['pick']

        if market not in market_to_stat:
            continue

        stat_col = market_to_stat[market]

        # Find matching player stats
        player_stats = actuals[actuals['player_norm'] == player_norm]

        if len(player_stats) == 0:
            # Player not found - DNP (injured/bye/benched) = voided bet
            pending.append({
                'player': bet['player'],
                'market': market,
                'pick': pick,
                'line': line,
                'reason': 'DNP (injured/bye/benched) - VOIDED'
            })
            continue

        actual_stat = player_stats.iloc[0].get(stat_col, None)

        if pd.isna(actual_stat):
            pending.append({
                'player': bet['player'],
                'market': market,
                'pick': pick,
                'line': line,
                'reason': 'Stat not available'
            })
            continue

        # Determine if bet won
        is_under = 'under' in pick.lower()
        bet_won = (actual_stat < line) if is_under else (actual_stat > line)
        is_push = (actual_stat == line)

        # Calculate profit (assuming -110 odds)
        if is_push:
            profit = 0
        elif bet_won:
            profit = 0.909  # Win at -110
        else:
            profit = -1.0

        result = {
            'week': week,
            'season': season,
            'player': bet['player'],
            'market': market,
            'pick': pick,
            'line': line,
            'actual': actual_stat,
            'bet_won': bet_won,
            'is_push': is_push,
            'profit': profit,
            'v5_p_under': bet.get('v5_p_under', None),
            'model_projection': bet.get('model_projection', None),
            'tracked_at': datetime.now().isoformat(),
        }

        results.append(result)

    # Calculate summary
    if results:
        results_df = pd.DataFrame(results)
        wins = results_df['bet_won'].sum()
        losses = len(results_df) - wins - results_df['is_push'].sum()
        pushes = results_df['is_push'].sum()
        total_profit = results_df['profit'].sum()
        roi = (total_profit / len(results_df)) * 100

        summary = {
            'week': week,
            'season': season,
            'total_bets': len(results_df),
            'wins': int(wins),
            'losses': int(losses),
            'pushes': int(pushes),
            'win_rate': wins / (wins + losses) if (wins + losses) > 0 else 0,
            'total_profit': round(total_profit, 2),
            'roi': round(roi, 1),
            'by_market': {}
        }

        # By market breakdown
        for market in results_df['market'].unique():
            market_df = results_df[results_df['market'] == market]
            market_wins = market_df['bet_won'].sum()
            market_losses = len(market_df) - market_wins - market_df['is_push'].sum()
            market_profit = market_df['profit'].sum()

            summary['by_market'][market] = {
                'bets': len(market_df),
                'wins': int(market_wins),
                'losses': int(market_losses),
                'win_rate': market_wins / (market_wins + market_losses) if (market_wins + market_losses) > 0 else 0,
                'profit': round(market_profit, 2),
                'roi': round((market_profit / len(market_df)) * 100, 1) if len(market_df) > 0 else 0,
            }

        logger.info(f"Week {week} Results: {wins}W-{losses}L ({summary['win_rate']:.1%}), ROI: {roi:.1f}%")
        if pending:
            logger.info(f"  {len(pending)} bets pending (games not played yet)")

        return {'bets': results, 'pending': pending, 'summary': summary}

    # Even if no results, return pending info
    if pending:
        return {'bets': [], 'pending': pending, 'summary': {'total_bets': 0, 'pending': len(pending)}}

    return None


def show_all_time_summary():
    """Display all-time betting results summary."""
    history = load_tracking_history()

    if not history.get('bets'):
        print("No bet history found. Track some weeks first!")
        return

    all_bets = pd.DataFrame(history['bets'])

    print("=" * 80)
    print("NFL QUANT - ALL-TIME BET RESULTS")
    print("=" * 80)
    print()

    # Overall stats
    wins = all_bets['bet_won'].sum()
    losses = len(all_bets) - wins - all_bets['is_push'].sum()
    total_profit = all_bets['profit'].sum()
    roi = (total_profit / len(all_bets)) * 100

    print(f"Total Bets: {len(all_bets)}")
    print(f"Record: {wins}W - {losses}L")
    print(f"Win Rate: {wins / (wins + losses):.1%}")
    print(f"Total Profit: {total_profit:+.2f} units")
    print(f"ROI: {roi:+.1f}%")
    print()

    # By market
    print("BY MARKET:")
    print("-" * 60)
    for market in all_bets['market'].unique():
        mdf = all_bets[all_bets['market'] == market]
        m_wins = mdf['bet_won'].sum()
        m_losses = len(mdf) - m_wins - mdf['is_push'].sum()
        m_profit = mdf['profit'].sum()
        m_roi = (m_profit / len(mdf)) * 100

        print(f"  {market}: {m_wins}W-{m_losses}L ({m_wins/(m_wins+m_losses):.1%}), ROI: {m_roi:+.1f}%")

    print()

    # By week
    print("BY WEEK:")
    print("-" * 60)
    for week in sorted(all_bets['week'].unique()):
        wdf = all_bets[all_bets['week'] == week]
        w_wins = wdf['bet_won'].sum()
        w_losses = len(wdf) - w_wins - wdf['is_push'].sum()
        w_profit = wdf['profit'].sum()
        w_roi = (w_profit / len(wdf)) * 100

        print(f"  Week {week}: {w_wins}W-{w_losses}L, {w_profit:+.2f} units ({w_roi:+.1f}%)")

    print()

    # V5 P(UNDER) calibration check
    if 'v5_p_under' in all_bets.columns:
        all_bets['p_bucket'] = pd.cut(all_bets['v5_p_under'], bins=[0.5, 0.55, 0.60, 0.65, 0.70, 1.0])
        print("V5 P(UNDER) CALIBRATION:")
        print("-" * 60)
        for bucket in all_bets['p_bucket'].dropna().unique():
            bdf = all_bets[all_bets['p_bucket'] == bucket]
            actual_hit = bdf['bet_won'].mean()
            expected = (bucket.left + bucket.right) / 2
            print(f"  {bucket}: Predicted {expected:.0%}, Actual {actual_hit:.1%} (N={len(bdf)})")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Track NFL QUANT bet results')
    parser.add_argument('--week', type=int, help='Week to track results for')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--summary', action='store_true', help='Show all-time summary')
    args = parser.parse_args()

    if args.summary:
        show_all_time_summary()
        return

    if args.week is None:
        print("Please specify --week or --summary")
        return

    # Track the week
    result = track_week_results(args.week, args.season)

    if result is None:
        print(f"Could not track results for Week {args.week}")
        return

    # Load existing history
    history = load_tracking_history()

    # Remove any existing entries for this week (allow re-tracking)
    history['bets'] = [b for b in history['bets'] if not (b['week'] == args.week and b['season'] == args.season)]

    # Add new results
    history['bets'].extend(result['bets'])

    # Update summary
    if 'weekly_summaries' not in history:
        history['weekly_summaries'] = {}
    history['weekly_summaries'][f"{args.season}_week{args.week}"] = result['summary']

    # Save
    save_tracking_history(history)

    # Display results
    print()
    print("=" * 80)
    print(f"WEEK {args.week} RESULTS")
    print("=" * 80)
    s = result['summary']

    if s['total_bets'] > 0:
        print(f"Validated Bets: {s['total_bets']}")
        print(f"Record: {s['wins']}W - {s['losses']}L")
        print(f"Win Rate: {s['win_rate']:.1%}")
        print(f"Profit: {s['total_profit']:+.2f} units")
        print(f"ROI: {s['roi']:+.1f}%")
        print()
        print("By Market:")
        for market, stats in s.get('by_market', {}).items():
            print(f"  {market}: {stats['wins']}W-{stats['losses']}L, {stats['profit']:+.2f} units ({stats['roi']:+.1f}%)")
    else:
        print("No bets validated yet.")

    # Show voided bets (DNP)
    pending = result.get('pending', [])
    if pending:
        print()
        print(f"VOIDED ({len(pending)} bets - player DNP):")
        print("-" * 60)
        for p in pending[:10]:  # Show first 10
            print(f"  {p['player']}: {p['pick']} {p['line']} - {p['reason']}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")

    print("=" * 80)


if __name__ == '__main__':
    main()
