#!/usr/bin/env python3
"""
Walk-Forward Backtest for Game Line Edge

Tests game line edge predictions using strict walk-forward validation
to ensure no data leakage.

Key anti-leakage measures:
1. EPA calculated using ONLY weeks < test_week
2. Rest days come from schedule (known before game)
3. No future information in any features

Usage:
    python scripts/validation/backtest_game_lines.py
    python scripts/validation/backtest_game_lines.py --start-week 10 --season 2025
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

from nfl_quant.edges.game_line_edge import GameLineEdge
from nfl_quant.config_paths import DATA_DIR


def load_pbp_data() -> pd.DataFrame:
    """Load play-by-play data from NFLverse."""
    pbp_path = DATA_DIR / 'nflverse' / 'pbp.parquet'
    if not pbp_path.exists():
        pbp_path = DATA_DIR / 'nflverse' / 'pbp_2025.parquet'
    if not pbp_path.exists():
        pbp_path = DATA_DIR / 'processed' / 'pbp_2025.parquet'

    if not pbp_path.exists():
        raise FileNotFoundError(f"No PBP data found")

    print(f"Loading PBP from: {pbp_path}")
    return pd.read_parquet(pbp_path)


def load_schedule_data() -> pd.DataFrame:
    """Load schedule data from NFLverse."""
    schedule_path = DATA_DIR / 'nflverse' / 'schedules.parquet'
    if not schedule_path.exists():
        raise FileNotFoundError(f"No schedule data found at {schedule_path}")

    print(f"Loading schedule from: {schedule_path}")
    return pd.read_parquet(schedule_path)


def get_actual_results(schedule_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Get actual game results for a week."""
    games = schedule_df[
        (schedule_df['season'] == season) &
        (schedule_df['week'] == week)
    ].copy()

    if len(games) == 0:
        return pd.DataFrame()

    # Calculate actual results
    # home_score and away_score should be in the schedule after game is played
    if 'home_score' in games.columns and 'away_score' in games.columns:
        games['actual_total'] = games['home_score'] + games['away_score']
        games['actual_home_margin'] = games['home_score'] - games['away_score']
    else:
        # Try to get from result column
        games['actual_total'] = np.nan
        games['actual_home_margin'] = np.nan

    return games


def evaluate_spread_bet(
    rec: dict,
    actual_margin: float,
    market_spread: float,
) -> dict:
    """
    Evaluate if a spread bet would have won.

    Args:
        rec: Recommendation dict with 'direction' (HOME/AWAY)
        actual_margin: Actual home margin (home_score - away_score)
        market_spread: Market spread (negative = home favored)

    Returns:
        Dict with 'won', 'push', 'actual_cover'
    """
    # Actual cover = home margin - spread
    # If spread is -3 (home favored by 3), home covers if they win by more than 3
    actual_cover = actual_margin + market_spread

    if rec['direction'] == 'HOME':
        # Bet on home to cover
        won = actual_cover > 0
        push = actual_cover == 0
    else:
        # Bet on away to cover
        won = actual_cover < 0
        push = actual_cover == 0

    return {
        'won': won,
        'push': push,
        'actual_cover': actual_cover,
        'actual_margin': actual_margin,
    }


def evaluate_total_bet(
    rec: dict,
    actual_total: float,
    market_total: float,
) -> dict:
    """
    Evaluate if a total bet would have won.

    Args:
        rec: Recommendation dict with 'direction' (OVER/UNDER)
        actual_total: Actual game total
        market_total: Market total line

    Returns:
        Dict with 'won', 'push'
    """
    if rec['direction'] == 'OVER':
        won = actual_total > market_total
        push = actual_total == market_total
    else:
        won = actual_total < market_total
        push = actual_total == market_total

    return {
        'won': won,
        'push': push,
        'actual_total': actual_total,
    }


def walk_forward_backtest(
    start_week: int = 10,
    end_week: int = 17,
    season: int = 2025,
) -> pd.DataFrame:
    """
    Run walk-forward backtest for game line edge.

    For each test week:
    1. Calculate team EPA using ONLY prior weeks (no leakage)
    2. Generate predictions
    3. Compare to actual results

    Args:
        start_week: First week to test
        end_week: Last week to test
        season: Season to test

    Returns:
        DataFrame with backtest results
    """
    print(f"\n{'='*60}")
    print("WALK-FORWARD BACKTEST: Game Line Edge")
    print(f"Season {season}, Weeks {start_week}-{end_week}")
    print('='*60)

    # Load data
    pbp_df = load_pbp_data()
    schedule_df = load_schedule_data()

    results = []

    for test_week in range(start_week, end_week + 1):
        print(f"\n--- Testing Week {test_week} ---")

        # Create fresh edge model for each week
        edge = GameLineEdge()

        # Get games for this week
        week_games = schedule_df[
            (schedule_df['season'] == season) &
            (schedule_df['week'] == test_week)
        ]

        if len(week_games) == 0:
            print(f"  No games found for week {test_week}")
            continue

        # Get teams
        teams = set(week_games['home_team'].tolist() + week_games['away_team'].tolist())

        # Calculate team EPA using ONLY prior weeks (critical for no leakage)
        # Filter PBP to weeks < test_week in the same season, plus all prior seasons
        historical_pbp = pbp_df[
            (pbp_df['season'] < season) |
            ((pbp_df['season'] == season) & (pbp_df['week'] < test_week))
        ]

        print(f"  Using {len(historical_pbp)} historical plays for EPA")

        for team in teams:
            edge.team_stats[team] = edge.calculate_team_epa(
                historical_pbp, team, test_week
            )

        # Get actual results
        actual_results = get_actual_results(schedule_df, season, test_week)

        if len(actual_results) == 0 or actual_results['actual_total'].isna().all():
            print(f"  No actual results available for week {test_week}")
            continue

        # Process each game
        for _, game in week_games.iterrows():
            home_team = game['home_team']
            away_team = game['away_team']

            home_epa = edge.team_stats.get(home_team)
            away_epa = edge.team_stats.get(away_team)

            if home_epa is None or away_epa is None:
                continue

            if home_epa['games'] < 3 or away_epa['games'] < 3:
                continue

            # Extract rest days and divisional flag (safe - known before game)
            home_rest = int(game.get('home_rest', 7)) if pd.notna(game.get('home_rest')) else 7
            away_rest = int(game.get('away_rest', 7)) if pd.notna(game.get('away_rest')) else 7
            is_divisional = bool(game.get('div_game', False))

            # Get market lines from schedule (if available)
            market_spread = game.get('spread_line', game.get('point_spread_home'))
            market_total = game.get('total_line', game.get('over_under'))

            if pd.isna(market_spread) and pd.isna(market_total):
                continue

            # Get actual results for this game
            game_actual = actual_results[
                (actual_results['home_team'] == home_team) &
                (actual_results['away_team'] == away_team)
            ]

            if len(game_actual) == 0:
                continue

            actual_margin = game_actual['actual_home_margin'].iloc[0]
            actual_total = game_actual['actual_total'].iloc[0]

            if pd.isna(actual_margin) and pd.isna(actual_total):
                continue

            # --- SPREAD EDGE ---
            if not pd.isna(market_spread) and not pd.isna(actual_margin):
                direction, edge_pts, confidence = edge.calculate_spread_edge(
                    home_epa, away_epa, market_spread,
                    home_rest=home_rest, away_rest=away_rest,
                    is_divisional=is_divisional
                )

                if direction is not None:
                    eval_result = evaluate_spread_bet(
                        {'direction': direction}, actual_margin, market_spread
                    )

                    results.append({
                        'season': season,
                        'week': test_week,
                        'game': f"{away_team} @ {home_team}",
                        'bet_type': 'spread',
                        'direction': direction,
                        'market_line': market_spread,
                        'edge_pts': edge_pts,
                        'confidence': confidence,
                        'won': eval_result['won'],
                        'push': eval_result['push'],
                        'actual_margin': actual_margin,
                        'actual_cover': eval_result['actual_cover'],
                        'home_rest': home_rest,
                        'away_rest': away_rest,
                        'is_divisional': is_divisional,
                    })

            # --- TOTAL EDGE ---
            if not pd.isna(market_total) and not pd.isna(actual_total):
                # V29: calculate_total_edge returns 4 values (direction, edge_pct, confidence, debug_info)
                result = edge.calculate_total_edge(
                    home_epa, away_epa, market_total
                )
                # Handle both old (3 values) and new (4 values) return formats
                if len(result) == 4:
                    direction, edge_pct, confidence, debug_info = result
                else:
                    direction, edge_pct, confidence = result

                if direction is not None:
                    eval_result = evaluate_total_bet(
                        {'direction': direction}, actual_total, market_total
                    )

                    results.append({
                        'season': season,
                        'week': test_week,
                        'game': f"{away_team} @ {home_team}",
                        'bet_type': 'total',
                        'direction': direction,
                        'market_line': market_total,
                        'edge_pts': edge_pct,
                        'confidence': confidence,
                        'won': eval_result['won'],
                        'push': eval_result['push'],
                        'actual_total': actual_total,
                        'home_rest': home_rest,
                        'away_rest': away_rest,
                        'is_divisional': is_divisional,
                    })

    return pd.DataFrame(results)


def analyze_results(results_df: pd.DataFrame):
    """Analyze and print backtest results."""
    if len(results_df) == 0:
        print("\nNo results to analyze")
        return

    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    # Filter out pushes for win rate calculation
    non_push = results_df[~results_df['push']]

    # Overall stats
    total_bets = len(non_push)
    wins = non_push['won'].sum()
    win_rate = wins / total_bets if total_bets > 0 else 0

    print(f"\nOverall: {wins}/{total_bets} ({win_rate:.1%})")
    print(f"Pushes: {results_df['push'].sum()}")

    # By bet type
    print("\nBy Bet Type:")
    for bet_type in results_df['bet_type'].unique():
        bt_df = non_push[non_push['bet_type'] == bet_type]
        bt_wins = bt_df['won'].sum()
        bt_total = len(bt_df)
        bt_rate = bt_wins / bt_total if bt_total > 0 else 0
        print(f"  {bet_type}: {bt_wins}/{bt_total} ({bt_rate:.1%})")

    # By confidence bucket
    print("\nBy Confidence Level:")
    non_push['conf_bucket'] = pd.cut(
        non_push['confidence'],
        bins=[0, 0.52, 0.55, 0.58, 1.0],
        labels=['50-52%', '52-55%', '55-58%', '58%+']
    )
    for bucket in ['50-52%', '52-55%', '55-58%', '58%+']:
        bucket_df = non_push[non_push['conf_bucket'] == bucket]
        if len(bucket_df) > 0:
            bucket_wins = bucket_df['won'].sum()
            bucket_total = len(bucket_df)
            bucket_rate = bucket_wins / bucket_total if bucket_total > 0 else 0
            print(f"  {bucket}: {bucket_wins}/{bucket_total} ({bucket_rate:.1%})")

    # By edge size
    print("\nBy Edge Size:")
    non_push['edge_bucket'] = pd.cut(
        non_push['edge_pts'],
        bins=[0, 2, 3, 4, 100],
        labels=['2-3 pts', '3-4 pts', '4-5 pts', '5+ pts']
    )
    for bucket in ['2-3 pts', '3-4 pts', '4-5 pts', '5+ pts']:
        bucket_df = non_push[non_push['edge_bucket'] == bucket]
        if len(bucket_df) > 0:
            bucket_wins = bucket_df['won'].sum()
            bucket_total = len(bucket_df)
            bucket_rate = bucket_wins / bucket_total if bucket_total > 0 else 0
            print(f"  {bucket}: {bucket_wins}/{bucket_total} ({bucket_rate:.1%})")

    # Divisional vs Non-divisional
    print("\nDivisional Games:")
    for div_val in [True, False]:
        div_df = non_push[non_push['is_divisional'] == div_val]
        if len(div_df) > 0:
            div_wins = div_df['won'].sum()
            div_total = len(div_df)
            div_rate = div_wins / div_total if div_total > 0 else 0
            label = "Divisional" if div_val else "Non-Divisional"
            print(f"  {label}: {div_wins}/{div_total} ({div_rate:.1%})")

    # Check for leakage (suspiciously high win rate)
    print("\n" + "="*60)
    print("LEAKAGE CHECK")
    print("="*60)

    if win_rate > 0.60:
        print(f"\n*** WARNING: Win rate {win_rate:.1%} is suspiciously high! ***")
        print("This may indicate data leakage. Review:")
        print("  1. EPA calculation uses only prior weeks?")
        print("  2. No future information in features?")
        print("  3. Walk-forward split is correct?")
    elif win_rate > 0.55:
        print(f"\nWin rate {win_rate:.1%} is reasonable for game lines.")
        print("Market is efficient, expect 52-55% with edge.")
    else:
        print(f"\nWin rate {win_rate:.1%} suggests model may need tuning.")
        print("Consider adjusting thresholds or features.")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest for Game Lines")
    parser.add_argument('--start-week', type=int, default=10, help='First week to test')
    parser.add_argument('--end-week', type=int, default=17, help='Last week to test')
    parser.add_argument('--season', type=int, default=2025, help='Season to test')
    parser.add_argument('--output', type=str, help='Output CSV path')
    args = parser.parse_args()

    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results_df = walk_forward_backtest(
        start_week=args.start_week,
        end_week=args.end_week,
        season=args.season,
    )

    if len(results_df) > 0:
        analyze_results(results_df)

        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = DATA_DIR / 'backtest' / f'game_line_backtest_{args.season}.csv'

        output_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
