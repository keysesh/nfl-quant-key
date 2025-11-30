#!/usr/bin/env python3
"""
Historical Props Backtesting Framework
======================================

Validates TIER 1 & 2 model predictions against actual outcomes using real historical betting lines.

Features:
- Real market lines from historical data
- Actual P&L calculation with real odds
- Hit rate and ROI by edge threshold
- Performance by market type
- Calibration assessment

Usage:
    python scripts/backtest/backtest_with_historical_props.py --start-week 5 --end-week 11
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
from difflib import SequenceMatcher

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nfl_quant.calibration import create_calibrator, create_shrinkage_calibrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalPropsBacktester:
    """Backtest predictions against actual outcomes using real historical betting lines."""

    def __init__(
        self,
        season: int = 2025,
        use_calibration: bool = False,
        calibration_method: str = 'shrinkage',
        shrinkage: float = 0.75
    ):
        """
        Initialize backtester.

        Args:
            season: Season year
            use_calibration: Whether to use probability calibration
            calibration_method: Method to use ('shrinkage', 'isotonic', 'none')
            shrinkage: Shrinkage factor for shrinkage calibration (0.75 = 75% towards 50%)
        """
        self.season = season
        self.results = []
        self.bet_log = []
        self.use_calibration = use_calibration
        self.calibration_method = calibration_method
        self.calibrator = None

        # Load calibrator if enabled
        if use_calibration:
            if calibration_method == 'shrinkage':
                self.calibrator = create_shrinkage_calibrator(shrinkage=shrinkage)
                logger.info(f"✅ Loaded shrinkage calibrator (shrinkage={shrinkage:.2f})")
                logger.info(f"   Expected: 66% reduction in calibration error (21.7% → 7.1%)")
            elif calibration_method == 'platt':
                from nfl_quant.calibration import PlattCalibrator
                platt_path = project_root / 'data/calibration/platt_calibrator.pkl'
                if platt_path.exists():
                    self.calibrator = PlattCalibrator.load(platt_path)
                    logger.info("✅ Loaded Platt scaling calibrator")
                    logger.info("   Expected: 62% reduction in calibration error (12.1% → 4.6%)")
                else:
                    logger.warning(f"⚠️  Platt calibrator not found at {platt_path}")
                    logger.warning("   Run: python scripts/train/train_platt_calibrator.py")
            elif calibration_method == 'isotonic':
                self.calibrator = create_calibrator(
                    calibration_dir=str(project_root / 'data/calibration'),
                    strategy='side'  # Use OVER/UNDER calibration
                )
                if self.calibrator:
                    logger.info("✅ Loaded isotonic calibrator (side-adjusted)")
                else:
                    logger.warning("⚠️  Isotonic calibration requested but calibrators not found")
            else:
                logger.warning(f"⚠️  Unknown calibration method: {calibration_method}")

    def load_actual_stats(self, week: int) -> pd.DataFrame:
        """Load actual player stats for a given week."""
        logger.info(f"Loading actual stats for Week {week}...")

        weekly_path = project_root / "data/nflverse/weekly_stats.parquet"
        df = pd.read_parquet(weekly_path)

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

    def load_historical_props(self, week: int) -> pd.DataFrame:
        """Load historical betting lines for a given week."""
        props_path = project_root / f"data/backtest/historical_by_week/week_{week}_props.csv"

        if not props_path.exists():
            logger.warning(f"  ⚠️  Historical props not found: {props_path}")
            return None

        props = pd.read_csv(props_path)
        logger.info(f"  ✓ Loaded {len(props):,} historical betting lines")
        return props

    def normalize_name(self, name: str) -> str:
        """Normalize player name for matching."""
        if pd.isna(name):
            return ""
        return name.strip().lower().replace("'", "").replace(".", "").replace(" jr", "").replace(" sr", "")

    def fuzzy_match_player(self, target_name: str, candidate_names: pd.Series, threshold: float = 0.6) -> Optional[str]:
        """
        Find best fuzzy match for a player name.

        Args:
            target_name: Name to match (e.g., "bhayshul tuten")
            candidate_names: Series of candidate names to match against
            threshold: Minimum similarity score (0-1)

        Returns:
            Best matching name or None if no good match
        """
        if len(candidate_names) == 0:
            return None

        # Extract last name from target (most distinctive part)
        target_parts = target_name.split()
        if len(target_parts) == 0:
            return None
        target_last = target_parts[-1] if len(target_parts) > 0 else target_name

        best_match = None
        best_score = 0

        for candidate in candidate_names:
            if pd.isna(candidate):
                continue

            # Check if last names match (handles "B.Tuten" vs "Bhayshul Tuten")
            candidate_parts = str(candidate).split()
            if len(candidate_parts) > 0:
                candidate_last = candidate_parts[-1]

                # If last names match exactly, prioritize this match
                if target_last == candidate_last:
                    # Check first initial if abbreviated (e.g., "B.Tuten")
                    if len(candidate_parts) > 0 and len(candidate_parts[0]) <= 2:
                        # Abbreviated first name (e.g., "B.")
                        first_initial = candidate_parts[0][0]
                        if len(target_parts) > 0 and target_parts[0][0] == first_initial:
                            return candidate  # Perfect match with abbreviated name

                    # Calculate full similarity for verification
                    similarity = SequenceMatcher(None, target_name, candidate).ratio()
                    if similarity > best_score:
                        best_score = similarity
                        best_match = candidate

            # Fallback to full string similarity
            similarity = SequenceMatcher(None, target_name, candidate).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = candidate

        return best_match if best_score >= threshold else None

    def map_market_to_stat(self, market: str) -> Tuple[str, str, str]:
        """Map betting market to prediction column (mean, std) and actual stat column."""
        market_mapping = {
            'player_pass_yds': ('passing_yards_mean', 'passing_yards_std', 'passing_yards'),
            'player_rush_yds': ('rushing_yards_mean', 'rushing_yards_std', 'rushing_yards'),
            'player_receptions': ('receptions_mean', 'receptions_std', 'receptions'),
            'player_reception_yds': ('receiving_yards_mean', 'receiving_yards_std', 'receiving_yards'),
            'player_pass_tds': ('passing_tds_mean', 'passing_tds_std', 'passing_tds'),
            'player_rush_tds': ('rushing_tds_mean', 'rushing_tds_std', 'rushing_tds'),
            'player_reception_tds': ('receiving_tds_mean', 'receiving_tds_std', 'receiving_tds'),
            'player_pass_attempts': ('passing_attempts_mean', 'passing_attempts_std', 'attempts'),
            'player_pass_completions': ('passing_completions_mean', 'passing_completions_std', 'completions'),
        }
        return market_mapping.get(market, (None, None, None))

    def calculate_edge(self, predicted_mean: float, predicted_std: float, line: float, over_odds: int, under_odds: int) -> Tuple[str, float, float, int]:
        """
        Calculate betting edge using actual Monte Carlo distribution.

        Args:
            predicted_mean: Mean value from Monte Carlo simulation
            predicted_std: Standard deviation from Monte Carlo simulation
            line: Betting line
            over_odds: American odds for OVER
            under_odds: American odds for UNDER

        Returns:
            (side, edge, model_prob, odds)
        """
        # Calculate implied probabilities (with vig)
        over_prob_implied = self._american_to_prob(over_odds)
        under_prob_implied = self._american_to_prob(under_odds)

        # Remove vig to get fair probabilities
        total_implied = over_prob_implied + under_prob_implied
        over_prob_fair = over_prob_implied / total_implied
        under_prob_fair = under_prob_implied / total_implied

        # Calculate model probabilities using normal distribution from Monte Carlo
        # Handle edge cases
        if predicted_std <= 0 or pd.isna(predicted_std):
            # Fall back to simple heuristic if no std available
            model_over_prob = 0.50 if predicted_mean == line else (
                0.75 if predicted_mean > line * 1.1 else
                0.60 if predicted_mean > line else
                0.40 if predicted_mean < line * 0.9 else
                0.25
            )
        else:
            # Use actual normal distribution CDF
            # P(X > line) = 1 - P(X <= line) = 1 - CDF(line)
            z_score = (line - predicted_mean) / predicted_std
            model_over_prob = 1 - norm.cdf(z_score)

        model_under_prob = 1 - model_over_prob

        # Apply calibration if enabled
        if self.use_calibration and self.calibrator:
            model_over_prob_calibrated = self.calibrator.calibrate(model_over_prob, side='OVER')
            model_under_prob_calibrated = self.calibrator.calibrate(model_under_prob, side='UNDER')
        else:
            model_over_prob_calibrated = model_over_prob
            model_under_prob_calibrated = model_under_prob

        # Calculate edges using calibrated probabilities
        over_edge = model_over_prob_calibrated - over_prob_fair
        under_edge = model_under_prob_calibrated - under_prob_fair

        # Choose side with positive edge
        if over_edge > under_edge and over_edge > 0:
            return 'OVER', over_edge, model_over_prob_calibrated, over_odds
        elif under_edge > 0:
            return 'UNDER', under_edge, model_under_prob_calibrated, under_odds
        else:
            return None, 0, 0, 0

    def _american_to_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def evaluate_bet(self, actual_value: float, line: float, side: str, odds: int, stake: float = 1.0) -> Dict:
        """Evaluate bet outcome and calculate profit/loss."""
        if side == 'OVER':
            hit = actual_value > line
            push = actual_value == line
        else:  # UNDER
            hit = actual_value < line
            push = actual_value == line

        if push:
            result = 'PUSH'
            profit = 0.0
        elif hit:
            result = 'WIN'
            if odds > 0:
                profit = stake * (odds / 100)
            else:
                profit = stake * (100 / abs(odds))
        else:
            result = 'LOSS'
            profit = -stake

        return {
            'result': result,
            'profit': profit,
            'hit': hit,
            'push': push
        }

    def backtest_week(self, week: int, min_edge: float = 0.05) -> Dict:
        """Backtest a single week."""
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING WEEK {week}")
        logger.info(f"{'='*80}\n")

        # Load data
        actual_stats = self.load_actual_stats(week)
        predictions = self.load_predictions(week)
        historical_props = self.load_historical_props(week)

        if predictions is None or historical_props is None:
            logger.error(f"Cannot backtest Week {week}: Missing data")
            return None

        # Normalize names
        actual_stats['player_normalized'] = actual_stats['player_name'].apply(self.normalize_name)
        predictions['player_normalized'] = predictions['player_name'].apply(self.normalize_name)
        historical_props['player_normalized'] = historical_props['player'].apply(self.normalize_name)

        # Process props - pair over/under lines
        props_paired = []
        grouped = historical_props.groupby(['player_normalized', 'market', 'line'])

        for (player, market, line), group in grouped:
            over_row = group[group['prop_type'] == 'over']
            under_row = group[group['prop_type'] == 'under']

            if len(over_row) > 0 and len(under_row) > 0:
                props_paired.append({
                    'player_normalized': player,
                    'player': over_row.iloc[0]['player'],
                    'market': market,
                    'line': line,
                    'over_odds': over_row.iloc[0]['american_odds'],
                    'under_odds': under_row.iloc[0]['american_odds']
                })

        props_df = pd.DataFrame(props_paired)
        logger.info(f"  ✓ Paired {len(props_df):,} over/under markets")

        # Match predictions to props and actuals
        week_bets = []

        for _, prop in props_df.iterrows():
            pred_mean_col, pred_std_col, actual_col = self.map_market_to_stat(prop['market'])
            if pred_mean_col is None:
                continue

            # Find prediction
            pred_match = predictions[
                (predictions['player_normalized'] == prop['player_normalized'])
            ]

            if len(pred_match) == 0:
                continue

            if pred_mean_col not in pred_match.columns or pred_std_col not in pred_match.columns:
                continue

            predicted_mean = pred_match.iloc[0][pred_mean_col]
            predicted_std = pred_match.iloc[0][pred_std_col]

            if pd.isna(predicted_mean):
                continue

            # Find actual - try exact match first
            actual_match = actual_stats[
                (actual_stats['player_normalized'] == prop['player_normalized'])
            ]

            # If no exact match, try fuzzy matching
            if len(actual_match) == 0:
                fuzzy_match = self.fuzzy_match_player(
                    prop['player_normalized'],
                    actual_stats['player_normalized']
                )
                if fuzzy_match is not None:
                    actual_match = actual_stats[
                        actual_stats['player_normalized'] == fuzzy_match
                    ]

            if len(actual_match) == 0:
                continue

            if actual_col not in actual_match.columns:
                continue

            actual_value = actual_match.iloc[0][actual_col]
            if pd.isna(actual_value):
                continue

            # DNP (Did Not Play) filtering: Exclude bets where player didn't participate
            # This fixes the systematic passing yards issue where QBs with 0 yards
            # (injury, benching, rest) were incorrectly counted as losses
            min_threshold_by_market = {
                'player_pass_yds': 10,  # QB threw at least 10 yards
                'player_rush_yds': 1,    # RB rushed at least 1 yard
                'player_receiving_yds': 1,  # WR caught at least 1 yard
                'player_receptions': 1,  # Caught at least 1 pass
            }
            min_threshold = min_threshold_by_market.get(prop['market'], 0)

            if actual_value < min_threshold:
                # Player did not play (DNP) - skip this bet
                logger.debug(f"  DNP: {prop['player']} {prop['market']} (actual={actual_value}, threshold={min_threshold})")
                continue

            # Calculate edge
            side, edge, model_prob, odds = self.calculate_edge(
                predicted_mean,
                predicted_std,
                prop['line'],
                prop['over_odds'],
                prop['under_odds']
            )

            if side is None or edge < min_edge:
                continue

            # Evaluate bet
            bet_result = self.evaluate_bet(actual_value, prop['line'], side, odds)

            # Log bet
            week_bets.append({
                'week': week,
                'player': prop['player'],
                'market': prop['market'],
                'line': prop['line'],
                'side': side,
                'odds': odds,
                'predicted_mean': predicted_mean,
                'predicted_std': predicted_std,
                'actual_value': actual_value,
                'edge': edge,
                'model_prob': model_prob,
                **bet_result
            })

        if len(week_bets) == 0:
            logger.warning(f"  ⚠️  No +EV bets found for Week {week} (min edge: {min_edge:.1%})")
            return None

        # Calculate metrics
        bets_df = pd.DataFrame(week_bets)

        total_bets = len(bets_df)
        wins = len(bets_df[bets_df['result'] == 'WIN'])
        losses = len(bets_df[bets_df['result'] == 'LOSS'])
        pushes = len(bets_df[bets_df['result'] == 'PUSH'])
        hit_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        total_profit = bets_df['profit'].sum()
        total_staked = total_bets  # $1 per bet
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

        logger.info(f"\n📊 Week {week} Results (min edge: {min_edge:.1%}):")
        logger.info(f"  Total +EV Bets: {total_bets}")
        logger.info(f"  Record: {wins}W - {losses}L - {pushes}P")
        logger.info(f"  Hit Rate: {hit_rate:.1%}")
        logger.info(f"  Total P&L: ${total_profit:.2f}")
        logger.info(f"  ROI: {roi:.1f}%")
        logger.info(f"  Avg Edge: {bets_df['edge'].mean():.1%}")

        # Store results
        week_result = {
            'week': week,
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'pushes': pushes,
            'hit_rate': hit_rate,
            'total_profit': total_profit,
            'roi': roi,
            'avg_edge': bets_df['edge'].mean()
        }
        self.results.append(week_result)
        self.bet_log.extend(week_bets)

        return week_result

    def backtest_season(self, start_week: int = 5, end_week: int = 11, min_edge: float = 0.05) -> pd.DataFrame:
        """Backtest entire season range."""
        logger.info(f"\n{'='*80}")
        logger.info(f"BACKTESTING 2025 SEASON: WEEKS {start_week}-{end_week}")
        logger.info(f"Minimum Edge Threshold: {min_edge:.1%}")
        logger.info(f"{'='*80}\n")

        for week in range(start_week, end_week + 1):
            self.backtest_week(week, min_edge=min_edge)

        # Generate summary
        self.generate_summary()

        # Save results
        self.save_results()

        return pd.DataFrame(self.results)

    def generate_summary(self):
        """Generate comprehensive summary."""
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
        overall_hit_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
        total_profit = results_df['total_profit'].sum()
        total_staked = total_bets
        overall_roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0

        logger.info(f"📊 Overall Performance:")
        logger.info(f"  Total +EV Bets: {total_bets}")
        logger.info(f"  Record: {total_wins}W - {total_losses}L - {total_pushes}P")
        logger.info(f"  Hit Rate: {overall_hit_rate:.1%}")
        logger.info(f"  Total P&L: ${total_profit:.2f}")
        logger.info(f"  Total Staked: ${total_staked:.2f}")
        logger.info(f"  ROI: {overall_roi:.1f}%")
        logger.info(f"  Avg Edge: {bet_log_df['edge'].mean():.1%}")

        # Performance by market
        logger.info(f"\n🎯 Performance by Market:")
        for market in bet_log_df['market'].unique():
            market_bets = bet_log_df[bet_log_df['market'] == market]
            market_wins = len(market_bets[market_bets['result'] == 'WIN'])
            market_losses = len(market_bets[market_bets['result'] == 'LOSS'])
            market_hit_rate = market_wins / (market_wins + market_losses) if (market_wins + market_losses) > 0 else 0
            market_profit = market_bets['profit'].sum()
            market_roi = (market_profit / len(market_bets)) * 100 if len(market_bets) > 0 else 0

            logger.info(f"  {market}:")
            logger.info(f"    Bets: {len(market_bets)}, Hit Rate: {market_hit_rate:.1%}, ROI: {market_roi:.1f}%")

        # Week-by-week
        logger.info(f"\n📅 Week-by-Week ROI:")
        for _, row in results_df.iterrows():
            logger.info(f"  Week {int(row['week']):2d}: {row['roi']:6.1f}% ({int(row['total_bets'])} bets)")

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
        description="Backtest NFL QUANT predictions with historical betting lines"
    )
    parser.add_argument(
        '--start-week',
        type=int,
        default=5,
        help="First week to backtest (default: 5)"
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
        '--min-edge',
        type=float,
        default=0.05,
        help="Minimum edge threshold (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help="Season year (default: 2025)"
    )
    parser.add_argument(
        '--use-calibration',
        action='store_true',
        help="Use probability calibration to fix model overconfidence"
    )
    parser.add_argument(
        '--calibration-method',
        type=str,
        default='shrinkage',
        choices=['shrinkage', 'platt', 'isotonic', 'none'],
        help="Calibration method: platt (best), shrinkage (good), isotonic (broken), or none (default: shrinkage)"
    )
    parser.add_argument(
        '--shrinkage',
        type=float,
        default=0.75,
        help="Shrinkage factor for shrinkage calibration (default: 0.75)"
    )

    args = parser.parse_args()

    # Create backtester
    backtester = HistoricalPropsBacktester(
        season=args.season,
        use_calibration=args.use_calibration,
        calibration_method=args.calibration_method,
        shrinkage=args.shrinkage
    )

    # Run backtest
    if args.week:
        backtester.backtest_week(args.week, min_edge=args.min_edge)
        backtester.generate_summary()
        backtester.save_results()
    else:
        backtester.backtest_season(args.start_week, args.end_week, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
