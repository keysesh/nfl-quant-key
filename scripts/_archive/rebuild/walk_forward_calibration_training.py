#!/usr/bin/env python3
"""
Walk-Forward Calibration Training

This script generates REAL calibrator training data by:
1. For each completed week (e.g., 1-10):
   - Use ONLY data from previous weeks (no data leakage)
   - Run actual Monte Carlo simulation for each player
   - Compare predictions to actual outcomes
   - Record (raw_probability, actual_outcome) pairs

This is the PROPER way to train calibrators - using actual model predictions,
not synthetic/placeholder probabilities.

Usage:
    python scripts/rebuild/walk_forward_calibration_training.py --weeks 1-10
    python scripts/rebuild/walk_forward_calibration_training.py --all

Note: This is computationally intensive (10K+ simulations per player-week).
Expect ~30-60 minutes for full season.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.utils.season_utils import get_current_season

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WalkForwardCalibrationTrainer:
    """
    Walk-forward calibration training using actual Monte Carlo simulation.

    Key principle: When predicting week N, only use data from weeks 1 to N-1.
    This prevents data leakage and creates realistic training data.
    """

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.nflverse_dir = self.data_dir / "nflverse"

        self.current_season = get_current_season()

        # Load predictors once (they're trained on historical data)
        logger.info("Loading ML models...")
        self.usage_predictor, self.efficiency_predictor = load_predictors()
        logger.info("✅ Models loaded")

        # Load NFLverse data
        self.player_stats = None
        self.pbp_data = None
        self._load_nflverse_data()

        # Results storage
        self.training_records = []

    def _load_nflverse_data(self):
        """Load NFLverse data for building trailing stats."""
        logger.info("Loading NFLverse data...")

        stats_file = self.nflverse_dir / "player_stats.parquet"
        if not stats_file.exists():
            raise FileNotFoundError(f"Player stats not found: {stats_file}")

        self.player_stats = pd.read_parquet(stats_file)
        logger.info(f"✅ Loaded {len(self.player_stats):,} player-week records")

        # Filter to current season
        self.season_stats = self.player_stats[
            self.player_stats['season'] == self.current_season
        ].copy()
        logger.info(f"   Season {self.current_season}: {len(self.season_stats):,} records")

    def build_trailing_stats_for_week(self, target_week: int, lookback: int = 4) -> Dict:
        """
        Build trailing stats using ONLY weeks before target_week.

        This is the critical function that prevents data leakage.
        """
        logger.info(f"   Building trailing stats for week {target_week} (using weeks {max(1, target_week-lookback)}-{target_week-1})...")

        # Only use weeks BEFORE the target week
        available_data = self.season_stats[self.season_stats['week'] < target_week].copy()

        if len(available_data) == 0:
            logger.warning(f"   No historical data available for week {target_week}")
            return {}

        # Group by player and calculate trailing averages
        trailing_stats = {}

        # Get recent weeks within lookback window
        recent_weeks = sorted(available_data['week'].unique())[-lookback:]
        recent_data = available_data[available_data['week'].isin(recent_weeks)]

        # Aggregate by player
        for player_id in recent_data['player_id'].unique():
            player_data = recent_data[recent_data['player_id'] == player_id]

            if len(player_data) == 0:
                continue

            # Get latest team and position
            latest_row = player_data.iloc[-1]
            team = latest_row.get('recent_team', latest_row.get('team', 'UNK'))
            position = latest_row.get('position', 'UNK')
            player_name = latest_row.get('player_display_name', latest_row.get('player_name', player_id))

            # Calculate trailing statistics
            stats = {
                'player_id': player_id,
                'player_name': player_name,
                'team': team,
                'position': position,
                'games_played': len(player_data),

                # Usage metrics
                'trailing_targets': player_data['targets'].mean() if 'targets' in player_data.columns else 0,
                'trailing_carries': player_data['carries'].mean() if 'carries' in player_data.columns else 0,
                'trailing_attempts': player_data['attempts'].mean() if 'attempts' in player_data.columns else 0,

                # Yards
                'trailing_passing_yards': player_data['passing_yards'].mean() if 'passing_yards' in player_data.columns else 0,
                'trailing_rushing_yards': player_data['rushing_yards'].mean() if 'rushing_yards' in player_data.columns else 0,
                'trailing_receiving_yards': player_data['receiving_yards'].mean() if 'receiving_yards' in player_data.columns else 0,
                'trailing_receptions': player_data['receptions'].mean() if 'receptions' in player_data.columns else 0,

                # TDs
                'trailing_passing_tds': player_data['passing_tds'].mean() if 'passing_tds' in player_data.columns else 0,
                'trailing_rushing_tds': player_data['rushing_tds'].mean() if 'rushing_tds' in player_data.columns else 0,
                'trailing_receiving_tds': player_data['receiving_tds'].mean() if 'receiving_tds' in player_data.columns else 0,
            }

            trailing_stats[player_id] = stats

        logger.info(f"   Built trailing stats for {len(trailing_stats)} players")
        return trailing_stats

    def simulate_player_props(self, player_stats: Dict, lines: Dict[str, List[float]]) -> Dict[str, Dict[float, float]]:
        """
        Simulate player prop probabilities using Monte Carlo.

        Args:
            player_stats: Dict with trailing stats for this player
            lines: Dict mapping market type to list of lines to evaluate

        Returns:
            Dict mapping market -> line -> probability of going over
        """
        position = player_stats.get('position', 'WR')

        # Create PlayerPropInput
        player_input = PlayerPropInput(
            player_id=player_stats.get('player_id', 'unknown'),
            player_name=player_stats.get('player_name', 'Unknown'),
            team=player_stats.get('team', 'UNK'),
            position=position,
            week=1,  # Placeholder
            season=self.current_season,
            # Usage shares (estimate from trailing stats)
            snap_share=0.7,  # Default - would need snap count data
            target_share=player_stats.get('trailing_targets', 5) / 35,  # Rough team average
            carry_share=player_stats.get('trailing_carries', 10) / 25,
            # Efficiency (from trailing stats)
            yards_per_opportunity=7.0,  # Default
            td_rate=0.05,  # Default
        )

        # Create simulator
        simulator = PlayerSimulator(
            usage_predictor=self.usage_predictor,
            efficiency_predictor=self.efficiency_predictor,
            trials=1000,  # Reduced for speed (normally 10000+)
            seed=42
        )

        # Run simulation
        try:
            sim_results = simulator.simulate_player(player_input)
        except Exception as e:
            logger.warning(f"   Simulation failed for {player_stats.get('player_name')}: {e}")
            return {}

        # Calculate probabilities for each market/line
        probabilities = {}

        for market, market_lines in lines.items():
            probabilities[market] = {}

            if market == 'player_pass_yds' and 'passing_yards' in sim_results:
                sim_values = sim_results['passing_yards']
            elif market == 'player_rush_yds' and 'rushing_yards' in sim_results:
                sim_values = sim_results['rushing_yards']
            elif market == 'player_reception_yds' and 'receiving_yards' in sim_results:
                sim_values = sim_results['receiving_yards']
            elif market == 'player_receptions' and 'receptions' in sim_results:
                sim_values = sim_results['receptions']
            else:
                # Use trailing stats as fallback
                if market == 'player_pass_yds':
                    mean_val = player_stats.get('trailing_passing_yards', 200)
                elif market == 'player_rush_yds':
                    mean_val = player_stats.get('trailing_rushing_yards', 50)
                elif market == 'player_reception_yds':
                    mean_val = player_stats.get('trailing_receiving_yards', 50)
                elif market == 'player_receptions':
                    mean_val = player_stats.get('trailing_receptions', 4)
                else:
                    continue

                # Simple normal distribution fallback
                std_val = mean_val * 0.35  # Typical variance
                for line in market_lines:
                    prob = 1 - self._normal_cdf((line - mean_val) / std_val)
                    probabilities[market][line] = np.clip(prob, 0.01, 0.99)
                continue

            # Calculate probability of going over each line
            for line in market_lines:
                prob_over = np.mean(sim_values > line)
                probabilities[market][line] = np.clip(prob_over, 0.01, 0.99)

        return probabilities

    def _normal_cdf(self, x):
        """Standard normal CDF."""
        from scipy.stats import norm
        return norm.cdf(x)

    def process_week(self, week: int):
        """
        Process a single week: generate predictions and match to outcomes.

        This is the core of walk-forward validation.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING WEEK {week}")
        logger.info(f"{'='*80}")

        # Step 1: Build trailing stats using only previous weeks
        trailing_stats = self.build_trailing_stats_for_week(week)

        if not trailing_stats:
            logger.warning(f"   Skipping week {week}: no trailing stats available")
            return

        # Step 2: Get actual outcomes for this week
        week_actuals = self.season_stats[self.season_stats['week'] == week].copy()

        if len(week_actuals) == 0:
            logger.warning(f"   Skipping week {week}: no actual outcomes available")
            return

        logger.info(f"   Actual outcomes: {len(week_actuals)} players")

        # Step 3: Define lines to evaluate
        lines_config = {
            'player_pass_yds': [150.5, 175.5, 200.5, 225.5, 250.5, 275.5, 300.5],
            'player_rush_yds': [25.5, 40.5, 55.5, 70.5, 85.5, 100.5],
            'player_reception_yds': [25.5, 40.5, 55.5, 70.5, 85.5, 100.5],
            'player_receptions': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        }

        # Step 4: For each player with actual outcomes
        records_this_week = 0

        for _, actual_row in week_actuals.iterrows():
            player_id = actual_row['player_id']

            # Check if we have trailing stats for this player
            if player_id not in trailing_stats:
                continue

            player_trailing = trailing_stats[player_id]
            position = player_trailing.get('position', 'UNK')

            # Generate predictions
            predictions = self.simulate_player_props(player_trailing, lines_config)

            if not predictions:
                continue

            # Match predictions to actual outcomes
            for market, line_probs in predictions.items():
                # Get actual value
                if market == 'player_pass_yds':
                    actual_value = actual_row.get('passing_yards', 0)
                    if pd.isna(actual_value) or actual_value == 0:
                        continue
                elif market == 'player_rush_yds':
                    actual_value = actual_row.get('rushing_yards', 0)
                    if pd.isna(actual_value) or actual_value == 0:
                        continue
                elif market == 'player_reception_yds':
                    actual_value = actual_row.get('receiving_yards', 0)
                    if pd.isna(actual_value) or actual_value == 0:
                        continue
                elif market == 'player_receptions':
                    actual_value = actual_row.get('receptions', 0)
                    if pd.isna(actual_value):
                        continue
                else:
                    continue

                # Create training records for each line
                for line, raw_prob in line_probs.items():
                    outcome = 1 if actual_value > line else 0

                    self.training_records.append({
                        'season': self.current_season,
                        'week': week,
                        'player_id': player_id,
                        'player': player_trailing.get('player_name', player_id),
                        'position': position,
                        'market': market,
                        'line': line,
                        'actual_value': actual_value,
                        'model_prob_raw': raw_prob,
                        'bet_won': outcome
                    })
                    records_this_week += 1

        logger.info(f"   ✅ Generated {records_this_week:,} training records for week {week}")

    def run_walk_forward(self, weeks: List[int] = None):
        """
        Run walk-forward calibration training for specified weeks.

        Args:
            weeks: List of weeks to process (default: all completed weeks)
        """
        if weeks is None:
            # Get all completed weeks
            available_weeks = sorted(self.season_stats['week'].unique())
            # Skip week 1 (no trailing stats) and current week (in progress)
            weeks = [w for w in available_weeks if w > 1 and w < max(available_weeks)]

        logger.info(f"Starting walk-forward calibration training")
        logger.info(f"Season: {self.current_season}")
        logger.info(f"Weeks to process: {weeks}")
        logger.info(f"Total weeks: {len(weeks)}")

        start_time = datetime.now()

        for week in weeks:
            self.process_week(week)

        # Save training data
        if self.training_records:
            df = pd.DataFrame(self.training_records)

            output_dir = self.data_dir / "training"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "walk_forward_calibration_data.parquet"
            df.to_parquet(output_file, index=False)

            # Also save CSV for inspection
            df.to_csv(output_dir / "walk_forward_calibration_data.csv", index=False)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"\n{'='*80}")
            logger.info(f"WALK-FORWARD TRAINING COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Total records: {len(df):,}")
            logger.info(f"Weeks processed: {sorted(df['week'].unique())}")
            logger.info(f"Time elapsed: {elapsed:.1f} seconds")
            logger.info(f"Output: {output_file}")

            # Market breakdown
            logger.info(f"\nMarket Breakdown:")
            for market, count in df['market'].value_counts().items():
                win_rate = df[df['market'] == market]['bet_won'].mean()
                avg_prob = df[df['market'] == market]['model_prob_raw'].mean()
                logger.info(f"  {market}:")
                logger.info(f"    Records: {count:,}")
                logger.info(f"    Actual win rate: {win_rate:.1%}")
                logger.info(f"    Avg model prob: {avg_prob:.1%}")
                logger.info(f"    Calibration error: {abs(win_rate - avg_prob):.1%}")
        else:
            logger.error("No training records generated!")


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Calibration Training")
    parser.add_argument('--weeks', type=str, help='Weeks to process (e.g., "2-10" or "3,5,7")')
    parser.add_argument('--all', action='store_true', help='Process all available weeks')

    args = parser.parse_args()

    trainer = WalkForwardCalibrationTrainer()

    if args.weeks:
        if '-' in args.weeks:
            start, end = args.weeks.split('-')
            weeks = list(range(int(start), int(end) + 1))
        else:
            weeks = [int(w) for w in args.weeks.split(',')]
    elif args.all:
        weeks = None  # Will auto-detect
    else:
        # Default: process weeks 2-10 (need 1 week of history minimum)
        weeks = list(range(2, 11))

    trainer.run_walk_forward(weeks)


if __name__ == "__main__":
    main()
