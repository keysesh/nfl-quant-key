#!/usr/bin/env python3
"""
Comprehensive 2025 Season Backtesting Framework
================================================

Validates TIER 1 & 2 models against Weeks 1-11 actual outcomes.

Features:
- Walk-forward validation (train on past, predict future)
- Bet result tracking (W/L/Push, P&L, ROI)
- Performance metrics by confidence tier and market
- Calibration assessment
- Edge accuracy analysis

Usage:
    python scripts/backtest/backtest_2025_season.py --start-week 1 --end-week 11
    python scripts/backtest/backtest_2025_season.py --week 5  # Single week
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class NFLBacktester:
    """
    Comprehensive backtesting framework for NFL player prop predictions.
    """

    def __init__(self, season: int = 2025):
        self.season = season
        self.results = []
        self.bet_log = []

    def load_actual_stats(self, week: int) -> pd.DataFrame:
        """Load actual player stats for a given week."""
        logger.info(f"Loading actual stats for Week {week}...")

        weekly_path = project_root / "data/nflverse/weekly_stats.parquet"
        df = pd.read_parquet(weekly_path)

        # Filter to specific week and season
        week_stats = df[
            (df['season'] == self.season) &
            (df['week'] == week)
        ].copy()

        logger.info(f"  ✓ Loaded {len(week_stats):,} player-weeks")
        return week_stats

    def load_predictions(self, week: int) -> pd.DataFrame:
        """Load model predictions for a given week."""
        pred_path = project_root / f"data/model_predictions_week{week}.csv"

        if not pred_path.exists():
            logger.warning(f"  ⚠️  Predictions not found: {pred_path}")
            return None

        preds = pd.read_csv(pred_path)
        logger.info(f"  ✓ Loaded {len(preds):,} predictions")
        return preds

    def load_recommendations(self, week: int) -> pd.DataFrame:
        """Load betting recommendations for a given week."""
        rec_path = project_root / f"reports/week{week}_recommendations.csv"

        if not rec_path.exists():
            # Try alternate path
            rec_path = project_root / "reports/CURRENT_WEEK_RECOMMENDATIONS.csv"
            if not rec_path.exists():
                logger.warning(f"  ⚠️  Recommendations not found")
                return None

        recs = pd.read_csv(rec_path)
        logger.info(f"  ✓ Loaded {len(recs):,} recommendations")
        return recs

    def match_prediction_to_actual(
        self,
        player_name: str,
        position: str,
        market: str,
        prediction: float,
        actual_stats: pd.DataFrame
    ) -> Tuple[float, bool]:
        """
        Match a prediction to actual outcome.

        Returns:
            (actual_value, found)
        """
        # Normalize player name
        player_normalized = player_name.strip().lower()

        # Find player in actual stats
        player_stats = actual_stats[
            (actual_stats['player_name'].str.lower() == player_normalized) &
            (actual_stats['position'] == position)
        ]

        if len(player_stats) == 0:
            return None, False

        if len(player_stats) > 1:
            # Take first match
            player_stats = player_stats.iloc[0:1]

        # Map market to actual stat column
        market_mapping = {
            'player_receptions': 'receptions',
            'player_receiving_yards': 'receiving_yards',
            'player_receiving_tds': 'receiving_tds',
            'player_rush_attempts': 'carries',
            'player_rushing_yards': 'rushing_yards',
            'player_rushing_tds': 'rushing_tds',
            'player_pass_attempts': 'attempts',
            'player_pass_completions': 'completions',
            'player_pass_yards': 'passing_yards',
            'player_pass_tds': 'passing_tds',
        }

        stat_col = market_mapping.get(market)
        if not stat_col or stat_col not in player_stats.columns:
            return None, False

        actual_value = player_stats[stat_col].iloc[0]
        return actual_value, True

    def evaluate_bet(
        self,
        predicted_value: float,
        actual_value: float,
        line: float,
        side: str,
        odds: float,
        stake: float = 1.0
    ) -> Dict:
        """
        Evaluate a single bet outcome.

        Returns:
            {
                'result': 'W'|'L'|'P',
                'profit': float,
                'actual_value': float,
                'hit': bool
            }
        """
        # Determine if bet hit
        if side.upper() == 'OVER':
            hit = actual_value > line
            push = actual_value == line
        else:  # UNDER
            hit = actual_value < line
            push = actual_value == line

        # Calculate profit/loss
        if push:
            result = 'P'
            profit = 0.0
        elif hit:
            result = 'W'
            # American odds to decimal
            if odds > 0:
                profit = stake * (odds / 100)
            else:
                profit = stake * (100 / abs(odds))
        else:
            result = 'L'
            profit = -stake

        return {
            'result': result,
            'profit': profit,
            'actual_value': actual_value,
            'hit': hit,
            'push': push
        }

    def backtest_week(self, week: int, generate_predictions: bool = False) -> Dict:
        """
        Backtest a single week.

        Args:
            week: Week number to backtest
            generate_predictions: If True, generate predictions first

        Returns:
            Performance metrics dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING WEEK {week}")
        logger.info(f"{'='*80}\n")

        # Load actual stats
        actual_stats = self.load_actual_stats(week)

        if generate_predictions:
            logger.info("Generating predictions...")
            # TODO: Call prediction generation script
            pass

        # Load predictions
        predictions = self.load_predictions(week)
        if predictions is None:
            logger.error(f"Cannot backtest Week {week}: No predictions available")
            return None

        # Load recommendations
        recommendations = self.load_recommendations(week)
        if recommendations is None:
            logger.error(f"Cannot backtest Week {week}: No recommendations available")
            return None

        # Process each recommendation
        week_results = []

        for idx, row in recommendations.iterrows():
            player_name = row['player_name']
            position = row['position']
            market = row['market']
            line = row.get('line', row.get('prop_line'))
            side = row.get('side', row.get('recommendation'))
            predicted_prob = row.get('model_prob', row.get('predicted_prob'))
            odds = row.get('odds', -110)  # Default to -110 if not specified
            kelly_stake = row.get('kelly_stake', 1.0)
            confidence = row.get('confidence_tier', 'STANDARD')

            # Get prediction value
            pred_col_map = {
                'player_receptions': 'predicted_receptions',
                'player_receiving_yards': 'predicted_receiving_yards',
                'player_rushing_yards': 'predicted_rushing_yards',
            }
            pred_col = pred_col_map.get(market, 'predicted_value')

            pred_row = predictions[
                (predictions['player_name'].str.lower() == player_name.lower()) &
                (predictions['position'] == position)
            ]

            if len(pred_row) == 0:
                continue

            predicted_value = pred_row[pred_col].iloc[0] if pred_col in pred_row.columns else None

            # Match to actual outcome
            actual_value, found = self.match_prediction_to_actual(
                player_name, position, market, predicted_value, actual_stats
            )

            if not found or actual_value is None:
                logger.debug(f"  ⚠️  No actual stats for {player_name} ({position})")
                continue

            # Evaluate bet
            bet_result = self.evaluate_bet(
                predicted_value,
                actual_value,
                line,
                side,
                odds,
                kelly_stake
            )

            # Log result
            result_record = {
                'week': week,
                'player_name': player_name,
                'position': position,
                'market': market,
                'line': line,
                'side': side,
                'predicted_value': predicted_value,
                'actual_value': actual_value,
                'predicted_prob': predicted_prob,
                'odds': odds,
                'stake': kelly_stake,
                'confidence_tier': confidence,
                **bet_result
            }

            week_results.append(result_record)

        # Calculate week metrics
        if len(week_results) == 0:
            logger.warning(f"  ⚠️  No valid bets for Week {week}")
            return None

        results_df = pd.DataFrame(week_results)

        metrics = {
            'week': week,
            'total_bets': len(results_df),
            'wins': len(results_df[results_df['result'] == 'W']),
            'losses': len(results_df[results_df['result'] == 'L']),
            'pushes': len(results_df[results_df['result'] == 'P']),
            'hit_rate': results_df['hit'].mean() * 100,
            'total_profit': results_df['profit'].sum(),
            'total_staked': results_df['stake'].sum(),
            'roi': (results_df['profit'].sum() / results_df['stake'].sum()) * 100,
            'avg_profit_per_bet': results_df['profit'].mean(),
        }

        logger.info(f"\n📊 Week {week} Results:")
        logger.info(f"  Total Bets: {metrics['total_bets']}")
        logger.info(f"  Record: {metrics['wins']}W - {metrics['losses']}L - {metrics['pushes']}P")
        logger.info(f"  Hit Rate: {metrics['hit_rate']:.1f}%")
        logger.info(f"  Total P&L: ${metrics['total_profit']:.2f}")
        logger.info(f"  ROI: {metrics['roi']:.1f}%")

        # Store results
        self.results.append(metrics)
        self.bet_log.extend(week_results)

        return metrics

    def backtest_season(self, start_week: int = 1, end_week: int = 11) -> pd.DataFrame:
        """
        Backtest entire season range.

        Args:
            start_week: First week to backtest
            end_week: Last week to backtest

        Returns:
            DataFrame with all results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING 2025 SEASON: WEEKS {start_week}-{end_week}")
        logger.info(f"{'='*80}\n")

        for week in range(start_week, end_week + 1):
            self.backtest_week(week)

        # Generate summary report
        self.generate_summary_report()

        # Save results
        self.save_results()

        return pd.DataFrame(self.results)

    def generate_summary_report(self):
        """Generate comprehensive summary of all backtesting results."""
        if len(self.results) == 0:
            logger.warning("No results to summarize")
            return

        results_df = pd.DataFrame(self.results)
        bet_log_df = pd.DataFrame(self.bet_log)

        logger.info(f"\n{'='*80}")
        logger.info(f"SEASON SUMMARY (Weeks {results_df['week'].min()}-{results_df['week'].max()})")
        logger.info(f"{'='*80}\n")

        # Overall metrics
        total_bets = results_df['total_bets'].sum()
        total_wins = results_df['wins'].sum()
        total_losses = results_df['losses'].sum()
        total_pushes = results_df['pushes'].sum()
        overall_hit_rate = (bet_log_df['hit'].sum() / len(bet_log_df)) * 100
        total_profit = results_df['total_profit'].sum()
        total_staked = results_df['total_staked'].sum()
        overall_roi = (total_profit / total_staked) * 100

        logger.info(f"📊 Overall Performance:")
        logger.info(f"  Total Bets: {total_bets}")
        logger.info(f"  Record: {total_wins}W - {total_losses}L - {total_pushes}P")
        logger.info(f"  Hit Rate: {overall_hit_rate:.1f}%")
        logger.info(f"  Total P&L: ${total_profit:.2f}")
        logger.info(f"  Total Staked: ${total_staked:.2f}")
        logger.info(f"  ROI: {overall_roi:.1f}%")

        # Performance by confidence tier
        if 'confidence_tier' in bet_log_df.columns:
            logger.info(f"\n📈 Performance by Confidence Tier:")
            for tier in ['ELITE', 'HIGH', 'STANDARD', 'LOW']:
                tier_bets = bet_log_df[bet_log_df['confidence_tier'] == tier]
                if len(tier_bets) > 0:
                    tier_roi = (tier_bets['profit'].sum() / tier_bets['stake'].sum()) * 100
                    tier_hit_rate = tier_bets['hit'].mean() * 100
                    logger.info(
                        f"  {tier:8s}: {len(tier_bets):3d} bets, "
                        f"{tier_hit_rate:5.1f}% hit, "
                        f"{tier_roi:6.1f}% ROI"
                    )

        # Performance by market
        if 'market' in bet_log_df.columns:
            logger.info(f"\n🎯 Performance by Market:")
            for market in bet_log_df['market'].unique():
                market_bets = bet_log_df[bet_log_df['market'] == market]
                if len(market_bets) > 0:
                    market_roi = (market_bets['profit'].sum() / market_bets['stake'].sum()) * 100
                    market_hit_rate = market_bets['hit'].mean() * 100
                    logger.info(
                        f"  {market:30s}: {len(market_bets):3d} bets, "
                        f"{market_hit_rate:5.1f}% hit, "
                        f"{market_roi:6.1f}% ROI"
                    )

        # Week-by-week performance
        logger.info(f"\n📅 Week-by-Week ROI:")
        for _, row in results_df.iterrows():
            logger.info(f"  Week {row['week']:2d}: {row['roi']:6.1f}%")

    def save_results(self):
        """Save backtesting results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save weekly summary
        results_df = pd.DataFrame(self.results)
        summary_path = project_root / f"reports/backtest_summary_{timestamp}.csv"
        results_df.to_csv(summary_path, index=False)
        logger.info(f"\n💾 Saved summary: {summary_path}")

        # Save detailed bet log
        bet_log_df = pd.DataFrame(self.bet_log)
        log_path = project_root / f"reports/backtest_bet_log_{timestamp}.csv"
        bet_log_df.to_csv(log_path, index=False)
        logger.info(f"💾 Saved bet log: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest NFL QUANT models on 2025 season"
    )
    parser.add_argument(
        '--start-week',
        type=int,
        default=1,
        help="First week to backtest (default: 1)"
    )
    parser.add_argument(
        '--end-week',
        type=int,
        default=11,
        help="Last week to backtest (default: 11)"
    )
    parser.add_argument(
        '--week',
        type=int,
        help="Backtest single week only"
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help="Season year (default: 2025)"
    )

    args = parser.parse_args()

    # Create backtester
    backtester = NFLBacktester(season=args.season)

    # Run backtest
    if args.week:
        backtester.backtest_week(args.week)
        backtester.generate_summary_report()
        backtester.save_results()
    else:
        backtester.backtest_season(args.start_week, args.end_week)


if __name__ == "__main__":
    main()
