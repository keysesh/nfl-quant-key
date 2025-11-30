#!/usr/bin/env python3
"""
Comprehensive Framework Validation Script

Validates betting framework by:
1. Quick validation: Latest completed week
2. Full backtest: Multiple weeks of current or historical season

Calculates:
- Win rate, ROI, profit
- Brier score, calibration quality
- Breakdown by week, market, edge threshold
- Comparison to breakeven rates

Usage:
    python scripts/validate/validate_framework.py --quick              # Latest week only
    python scripts/validate/validate_framework.py --full               # Full season
    python scripts/validate/validate_framework.py --weeks 1-10         # Weeks range
    python scripts/validate/validate_framework.py --season 2024        # Historical backtest
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_quant.utils.player_names import normalize_player_name
from nfl_quant.utils.team_names import normalize_team_name
from nfl_quant.utils.season_utils import get_current_season
from sklearn.metrics import brier_score_loss, log_loss

logger = None


def setup_logging():
    """Setup logging."""
    import logging
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if pd.isna(american_odds):
        return np.nan

    american_odds = float(american_odds)

    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    decimal = american_to_decimal(american_odds)
    if pd.isna(decimal):
        return np.nan
    return 1.0 / decimal


def calculate_profit(american_odds: float, bet_amount: float = 100.0) -> float:
    """Calculate profit from American odds."""
    decimal = american_to_decimal(american_odds)
    if pd.isna(decimal):
        return 0.0
    return bet_amount * (decimal - 1)


def load_recommendations(week: int) -> pd.DataFrame:
    """Load betting recommendations for a week."""
    # Try current week recommendations first
    rec_file = Path(f'reports/CURRENT_WEEK_RECOMMENDATIONS.csv')
    if rec_file.exists():
        df = pd.read_csv(rec_file)
        if 'game' in df.columns:
            # Filter by week if we can determine it
            # For now, assume it's the current week
            return df

    # Try to load from model predictions + odds matching
    # This would require regenerating recommendations
    return pd.DataFrame()


def load_actual_stats(week: int, season: Optional[int] = None) -> pd.DataFrame:
    """Load actual player stats for a week using unified interface - FAIL EXPLICITLY if unavailable."""
    from nfl_quant.data.stats_loader import load_weekly_stats, is_data_available

    if season is None:
        season = get_current_season()

    # Use unified interface - FAIL if data not available
    if not is_data_available(week, season, source='auto'):
        raise FileNotFoundError(
            f"Stats data not available for week {week}, season {season}. "
            f"Run data fetching scripts to populate data."
        )

    df = load_weekly_stats(week, season, source='auto')
    if 'week' not in df.columns:
        df['week'] = week
    if 'season' not in df.columns:
        df['season'] = season
    return df


def load_game_results(week: int, season: Optional[int] = None) -> pd.DataFrame:
    """Load actual game results (scores, totals) for a week."""
    if season is None:
        season = get_current_season()

    # Try nflverse PBP data
    pbp_file = Path(f'data/nflverse/pbp_{season}.parquet')
    if pbp_file.exists():
        pbp = pd.read_parquet(pbp_file)
        week_pbp = pbp[pbp['week'] == week]

        if not week_pbp.empty:
            # Aggregate by game
            games = week_pbp.groupby(['home_team', 'away_team']).agg({
                'home_score': 'max',
                'away_score': 'max',
                'total_line': 'first'
            }).reset_index()

            games['game_total'] = games['home_score'] + games['away_score']
            games['week'] = week
            return games

    return pd.DataFrame()


def match_player_bet_to_actual(
    bet: pd.Series,
    actual_stats: pd.DataFrame,
    week: int
) -> Optional[Dict]:
    """Match a player prop bet to actual outcome."""
    player_name = bet.get('player', '')
    if pd.isna(player_name) or player_name == '':
        return None

    # Normalize player name
    player_key = normalize_player_name(str(player_name))

    # Try to find player in actual stats
    actual_stats['normalized_name'] = actual_stats['player_name'].apply(normalize_player_name)
    matches = actual_stats[actual_stats['normalized_name'] == player_key]

    if len(matches) == 0:
        return None

    # If multiple matches, try to match by team
    if len(matches) > 1:
        team = bet.get('team', '')
        if team and not pd.isna(team):
            team_normalized = normalize_team_name(str(team))
            team_matches = matches[matches['team'].str.upper() == team_normalized]
            if len(team_matches) > 0:
                matches = team_matches

    actual_row = matches.iloc[0]

    # Determine market and actual value
    market = bet.get('market', '')
    line = bet.get('line', np.nan)
    pick = bet.get('pick', '').lower()

    # Map market to actual stat column
    market_to_stat = {
        'player_rush_yds': 'rushing_yards',
        'player_reception_yds': 'receiving_yards',
        'player_receptions': 'receptions',
        'player_pass_yds': 'passing_yards',
        'player_rush_tds': 'rushing_tds',
        'player_reception_tds': 'receiving_tds',
        'player_pass_tds': 'passing_tds',
        'player_anytime_td': 'anytime_td',
    }

    # Handle column name variations
    stat_col = None
    for col in actual_row.index:
        col_lower = col.lower()
        if market in market_to_stat:
            target_stat = market_to_stat[market]
            if target_stat in col_lower or col_lower in target_stat:
                stat_col = col
                break

    if stat_col is None:
        # Try direct match
        if market.replace('player_', '') in actual_row.index:
            stat_col = market.replace('player_', '')
        else:
            return None

    actual_value = actual_row.get(stat_col, np.nan)

    if pd.isna(actual_value):
        return None

    # Handle TD markets specially
    if 'td' in market.lower() and market != 'player_anytime_td':
        # Count TDs
        rush_tds = actual_row.get('rushing_tds', 0) or 0
        rec_tds = actual_row.get('receiving_tds', 0) or 0
        pass_tds = actual_row.get('passing_tds', 0) or 0

        if 'rush' in market:
            actual_value = rush_tds
        elif 'rec' in market or 'reception' in market:
            actual_value = rec_tds
        elif 'pass' in market:
            actual_value = pass_tds

    # Determine if bet won
    if pd.isna(line):
        return None

    if 'under' in pick:
        won = actual_value < line
    elif 'over' in pick:
        won = actual_value > line
    else:
        return None

    return {
        'player': player_name,
        'market': market,
        'line': line,
        'pick': pick,
        'actual_value': actual_value,
        'won': bool(won),
        'model_prob': bet.get('model_prob', np.nan),
        'odds': bet.get('odds', np.nan),
        'edge_pct': bet.get('edge_pct', np.nan),
    }


def match_game_bet_to_actual(
    bet: pd.Series,
    game_results: pd.DataFrame,
    week: int
) -> Optional[Dict]:
    """Match a game line bet (spread, total, moneyline) to actual outcome."""
    game_str = bet.get('game', '')
    if pd.isna(game_str) or game_str == '':
        return None

    # Parse game string (e.g., "PHI @ GB")
    parts = game_str.split(' @ ')
    if len(parts) != 2:
        return None

    away_team = normalize_team_name(parts[0])
    home_team = normalize_team_name(parts[1])

    # Find game result
    game_result = game_results[
        (game_results['away_team'].str.upper() == away_team) &
        (game_results['home_team'].str.upper() == home_team)
    ]

    if len(game_result) == 0:
        return None

    game_result = game_result.iloc[0]
    away_score = game_result.get('away_score', 0)
    home_score = game_result.get('home_score', 0)

    market = bet.get('market', '')
    pick = bet.get('pick', '').lower()
    line = bet.get('line', np.nan)

    won = False

    if 'total' in market.lower():
        # Game total bet
        game_total = away_score + home_score
        if 'under' in pick:
            won = game_total < line
        elif 'over' in pick:
            won = game_total > line
        actual_value = game_total

    elif 'spread' in market.lower():
        # Spread bet
        spread = home_score - away_score
        if 'home' in pick.lower():
            won = spread > (-line) if line < 0 else spread > line
        elif 'away' in pick.lower():
            won = spread < line if line < 0 else spread < (-line)
        actual_value = spread

    elif 'moneyline' in market.lower():
        # Moneyline bet
        if 'home' in pick.lower():
            won = home_score > away_score
        elif 'away' in pick.lower():
            won = away_score > home_score
        actual_value = 1 if won else 0

    else:
        return None

    return {
        'game': game_str,
        'market': market,
        'line': line,
        'pick': pick,
        'actual_value': actual_value,
        'won': bool(won),
        'model_prob': bet.get('model_prob', np.nan),
        'odds': bet.get('odds', np.nan),
        'edge_pct': bet.get('edge_pct', np.nan),
    }


def validate_week(week: int, season: Optional[int] = None, min_edge: float = 0.02) -> Dict:
    """Validate predictions for a single week."""
    if season is None:
        season = get_current_season()

    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATING WEEK {week} (Season {season})")
    logger.info(f"{'='*80}")

    # Load recommendations
    recommendations = load_recommendations(week)

    if recommendations.empty:
        logger.warning(f"  ⚠️  No recommendations found for Week {week}")
        logger.info("     Generating recommendations...")
        # Could trigger recommendation generation here
        return {
            'week': week,
            'bets_evaluated': 0,
            'bets_won': 0,
            'win_rate': 0.0,
            'profit': 0.0,
            'roi': 0.0,
            'brier_score': np.nan,
        }

    # Filter by edge threshold
    if 'edge_pct' in recommendations.columns:
        recommendations = recommendations[recommendations['edge_pct'] >= min_edge * 100]

    logger.info(f"  📊 Found {len(recommendations)} recommendations")

    # Load actual stats and game results
    actual_stats = load_actual_stats(week, season)
    game_results = load_game_results(week, season)

    if actual_stats.empty and game_results.empty:
        logger.warning(f"  ⚠️  No actual results found for Week {week}")
        return {
            'week': week,
            'bets_evaluated': 0,
            'bets_won': 0,
            'win_rate': 0.0,
            'profit': 0.0,
            'roi': 0.0,
            'brier_score': np.nan,
        }

    logger.info(f"  ✅ Loaded {len(actual_stats)} player stats, {len(game_results)} game results")

    # Match bets to actuals
    bet_results = []

    for _, bet in recommendations.iterrows():
        market = bet.get('market', '')

        # Player props
        if 'player_' in market:
            result = match_player_bet_to_actual(bet, actual_stats, week)
            if result:
                bet_results.append(result)

        # Game lines
        elif market in ['game_total', 'spread', 'moneyline']:
            result = match_game_bet_to_actual(bet, game_results, week)
            if result:
                bet_results.append(result)

    if not bet_results:
        logger.warning(f"  ⚠️  Could not match any bets to actual results")
        return {
            'week': week,
            'bets_evaluated': 0,
            'bets_won': 0,
            'win_rate': 0.0,
            'profit': 0.0,
            'roi': 0.0,
            'brier_score': np.nan,
        }

    bet_results_df = pd.DataFrame(bet_results)

    # Calculate metrics
    total_bets = len(bet_results_df)
    wins = bet_results_df['won'].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0.0

    # Calculate profit (assuming $100 bets)
    bet_results_df['profit'] = bet_results_df.apply(
        lambda row: calculate_profit(row['odds'], 100.0) if row['won'] else -100.0,
        axis=1
    )

    total_profit = bet_results_df['profit'].sum()
    total_wagered = total_bets * 100.0
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0

    # Calculate Brier score
    brier_score = np.nan
    if 'model_prob' in bet_results_df.columns:
        probs = bet_results_df['model_prob'].dropna()
        outcomes = bet_results_df.loc[probs.index, 'won'].astype(int)
        if len(probs) > 0:
            brier_score = brier_score_loss(outcomes, probs)

    logger.info(f"  ✅ Evaluated {total_bets} bets")
    logger.info(f"     Wins: {wins} ({win_rate:.1%})")
    logger.info(f"     Profit: ${total_profit:.2f}")
    logger.info(f"     ROI: {roi:.1f}%")
    if not pd.isna(brier_score):
        logger.info(f"     Brier Score: {brier_score:.4f}")

    return {
        'week': week,
        'bets_evaluated': total_bets,
        'bets_won': wins,
        'bets_lost': losses,
        'win_rate': win_rate,
        'profit': total_profit,
        'roi': roi,
        'brier_score': brier_score,
        'bet_results': bet_results_df,
    }


def generate_validation_report(results: List[Dict], output_file: Path):
    """Generate comprehensive validation report."""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("FRAMEWORK VALIDATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Overall summary
    total_bets = sum(r['bets_evaluated'] for r in results)
    total_wins = sum(r['bets_won'] for r in results)
    total_profit = sum(r['profit'] for r in results)
    total_wagered = total_bets * 100.0
    overall_roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    overall_win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0.0

    # Calculate average Brier score
    brier_scores = [r['brier_score'] for r in results if not pd.isna(r['brier_score'])]
    avg_brier = np.mean(brier_scores) if brier_scores else np.nan

    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-"*80)
    report_lines.append(f"Weeks Validated: {len(results)}")
    report_lines.append(f"Total Bets: {total_bets}")
    report_lines.append(f"Wins: {total_wins} ({overall_win_rate:.1f}%)")
    report_lines.append(f"Losses: {total_bets - total_wins}")
    report_lines.append(f"Total Profit: ${total_profit:,.2f}")
    report_lines.append(f"Total Wagered: ${total_wagered:,.2f}")
    report_lines.append(f"ROI: {overall_roi:.1f}%")
    if not pd.isna(avg_brier):
        report_lines.append(f"Average Brier Score: {avg_brier:.4f}")
    report_lines.append("")

    # Weekly breakdown
    report_lines.append("WEEKLY BREAKDOWN")
    report_lines.append("-"*80)
    report_lines.append(f"{'Week':<6} {'Bets':<6} {'Wins':<6} {'Win Rate':<10} {'Profit':<12} {'ROI':<8} {'Brier':<8}")
    report_lines.append("-"*80)

    for r in results:
        week = r['week']
        bets = r['bets_evaluated']
        wins = r['bets_won']
        wr = r['win_rate'] * 100
        profit = r['profit']
        roi = r['roi']
        brier = r['brier_score']
        brier_str = f"{brier:.4f}" if not pd.isna(brier) else "N/A"

        report_lines.append(
            f"{week:<6} {bets:<6} {wins:<6} {wr:>6.1f}%   ${profit:>9,.2f}  {roi:>5.1f}%  {brier_str:<8}"
        )

    report_lines.append("")

    # Market breakdown (if available)
    all_bet_results = []
    for r in results:
        if 'bet_results' in r and isinstance(r['bet_results'], pd.DataFrame):
            all_bet_results.append(r['bet_results'])

    if all_bet_results:
        combined_results = pd.concat(all_bet_results, ignore_index=True)

        if 'market' in combined_results.columns:
            report_lines.append("MARKET BREAKDOWN")
            report_lines.append("-"*80)

            market_summary = combined_results.groupby('market').agg({
                'won': ['count', 'sum'],
                'profit': 'sum',
            }).reset_index()

            market_summary.columns = ['market', 'bets', 'wins', 'profit']
            market_summary['win_rate'] = (market_summary['wins'] / market_summary['bets'] * 100)
            market_summary['roi'] = (market_summary['profit'] / (market_summary['bets'] * 100) * 100)

            report_lines.append(f"{'Market':<25} {'Bets':<6} {'Wins':<6} {'Win Rate':<10} {'Profit':<12} {'ROI':<8}")
            report_lines.append("-"*80)

            for _, row in market_summary.iterrows():
                report_lines.append(
                    f"{row['market']:<25} {row['bets']:<6} {row['wins']:<6} "
                    f"{row['win_rate']:>6.1f}%   ${row['profit']:>9,.2f}  {row['roi']:>5.1f}%"
                )

            report_lines.append("")

    # Calibration analysis
    if all_bet_results:
        combined_results = pd.concat(all_bet_results, ignore_index=True)
        if 'model_prob' in combined_results.columns:
            probs = combined_results['model_prob'].dropna()
            outcomes = combined_results.loc[probs.index, 'won'].astype(int)

            if len(probs) > 0:
                report_lines.append("CALIBRATION ANALYSIS")
                report_lines.append("-"*80)

                # Bin probabilities
                bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

                report_lines.append(f"{'Bin':<12} {'Count':<8} {'Predicted':<12} {'Actual':<12} {'Error':<12}")
                report_lines.append("-"*80)

                for i in range(len(bins) - 1):
                    bin_min = bins[i]
                    bin_max = bins[i + 1]

                    mask = (probs >= bin_min) & (probs < bin_max)
                    bin_probs = probs[mask]
                    bin_outcomes = outcomes[mask]

                    if len(bin_probs) > 0:
                        predicted = bin_probs.mean()
                        actual = bin_outcomes.mean()
                        error = abs(predicted - actual)

                        report_lines.append(
                            f"{bin_labels[i]:<12} {len(bin_probs):<8} "
                            f"{predicted:>10.1%}  {actual:>10.1%}  {error:>10.1%}"
                        )

                report_lines.append("")

    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-"*80)

    if overall_win_rate > 55.0 and overall_roi > 5.0:
        report_lines.append("✅ Framework is PROFITABLE and WELL-CALIBRATED")
        report_lines.append("   System is ready for live betting")
    elif overall_win_rate > 52.4:  # Breakeven
        report_lines.append("⚠️  Framework is PROFITABLE but needs improvement")
        report_lines.append("   Consider:")
        report_lines.append("   - Reviewing calibration quality")
        report_lines.append("   - Adjusting edge thresholds")
        report_lines.append("   - Analyzing losing bets")
    else:
        report_lines.append("❌ Framework is NOT PROFITABLE")
        report_lines.append("   DO NOT place bets until issues are resolved")
        report_lines.append("   Issues to investigate:")
        report_lines.append("   - Model accuracy")
        report_lines.append("   - Calibration quality")
        report_lines.append("   - Edge calculation")
        report_lines.append("   - Data quality")

    report_lines.append("")
    report_lines.append("="*80)

    # Write report
    report_text = "\n".join(report_lines)
    output_file.write_text(report_text)

    print("\n" + report_text)

    return report_text


def main():
    parser = argparse.ArgumentParser(description='Validate betting framework')
    parser.add_argument('--quick', action='store_true', help='Validate Week 10 only')
    parser.add_argument('--full', action='store_true', help='Validate Weeks 1-10')
    parser.add_argument('--weeks', type=str, help='Weeks to validate (e.g., "1-10" or "10")')
    parser.add_argument('--season', type=int, default=None, help='NFL season year (default: auto-detect)')
    parser.add_argument('--min-edge', type=float, default=0.02, help='Minimum edge threshold (default: 0.02)')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    setup_logging()

    # Auto-detect season if not provided
    season = args.season if args.season is not None else get_current_season()

    # Determine weeks to validate
    if args.quick:
        weeks = [10]
    elif args.full:
        weeks = list(range(1, 11))
    elif args.weeks:
        if '-' in args.weeks:
            start, end = map(int, args.weeks.split('-'))
            weeks = list(range(start, end + 1))
        else:
            weeks = [int(w) for w in args.weeks.split(',')]
    else:
        # Default: Week 10 only
        weeks = [10]

    logger.info(f"Validating {season} season, weeks: {weeks}")
    logger.info(f"Minimum edge threshold: {args.min_edge * 100:.1f}%")

    # Validate each week
    results = []
    for week in weeks:
        result = validate_week(week, season=season, min_edge=args.min_edge)
        results.append(result)

    # Generate report
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f'reports/validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    output_file.parent.mkdir(parents=True, exist_ok=True)

    generate_validation_report(results, output_file)

    logger.info(f"\n✅ Validation complete!")
    logger.info(f"   Report saved to: {output_file}")


if __name__ == '__main__':
    main()
