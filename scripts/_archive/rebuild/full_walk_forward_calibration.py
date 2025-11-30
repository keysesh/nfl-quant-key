#!/usr/bin/env python3
"""
FULL Walk-Forward Calibration Training with Game Simulation Integration

This script generates REAL calibrator training data by:
1. Loading historical NFLverse data (play-by-play, player stats, schedules)
2. For each completed week (e.g., 2-10):
   - Build trailing stats using ONLY prior weeks (no data leakage)
   - Simulate game context for each matchup (GameScriptEngine)
   - Create proper PlayerPropInput with all required fields
   - Run Monte Carlo simulation with correlation modeling (V3)
   - Compare predictions to actual outcomes
   - Record (raw_probability, actual_outcome) pairs

This is the COMPLETE and PROPER way to train calibrators - using actual
game simulations and Monte Carlo predictions, not shortcuts or synthetic data.

Usage:
    python scripts/rebuild/full_walk_forward_calibration.py --weeks 2-10
    python scripts/rebuild/full_walk_forward_calibration.py --all

Note: This is computationally intensive (1000+ trials per player-week).
Expect ~45-90 minutes for weeks 2-10 with proper simulation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Tuple

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.simulation.game_script_engine import GameScriptEngine, GameScriptConfig
from nfl_quant.schemas import PlayerPropInput
from nfl_quant.utils.season_utils import get_current_season

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class FullWalkForwardCalibrationTrainer:
    """
    Complete walk-forward calibration training using:
    1. Historical trailing stats from NFLverse
    2. Game simulation for team-level context
    3. Full Monte Carlo simulation with correlations (V3)
    4. Proper defensive EPA from play-by-play data

    Key principle: When predicting week N, only use data from weeks 1 to N-1.
    """

    def __init__(self, trials: int = 1000):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data"
        self.nflverse_dir = self.data_dir / "nflverse"
        self.trials = trials

        self.current_season = get_current_season()

        # Load predictors once (they're trained on historical data)
        logger.info("Loading ML models...")
        self.usage_predictor, self.efficiency_predictor = load_predictors()
        logger.info("Models loaded successfully")

        # Initialize game script engine
        self.game_engine = GameScriptEngine()

        # Load NFLverse data
        self.player_stats = None
        self.schedules = None
        self.team_defense_epa = None
        self.team_pace = None
        self._load_nflverse_data()

        # Results storage
        self.training_records = []

    def _load_nflverse_data(self):
        """Load all required NFLverse data."""
        logger.info("Loading NFLverse data...")

        # Player stats
        stats_file = self.nflverse_dir / "player_stats.parquet"
        if not stats_file.exists():
            raise FileNotFoundError(f"Player stats not found: {stats_file}")
        self.player_stats = pd.read_parquet(stats_file)
        logger.info(f"  Player stats: {len(self.player_stats):,} records")

        # Filter to current season
        self.season_stats = self.player_stats[
            self.player_stats['season'] == self.current_season
        ].copy()
        logger.info(f"  {self.current_season} season: {len(self.season_stats):,} records")

        # Schedules
        schedules_file = self.nflverse_dir / "schedules.parquet"
        if schedules_file.exists():
            self.schedules = pd.read_parquet(schedules_file)
            self.schedules = self.schedules[self.schedules['season'] == self.current_season].copy()
            logger.info(f"  Schedules: {len(self.schedules):,} games")
        else:
            logger.warning("  Schedules file not found, will derive from player stats")
            self.schedules = self._derive_schedules_from_stats()

        # Team defensive EPA (extracted from PBP)
        defense_epa_file = self.nflverse_dir / "team_defensive_epa.parquet"
        if defense_epa_file.exists():
            self.team_defense_epa = pd.read_parquet(defense_epa_file)
            logger.info(f"  Team defensive EPA: {len(self.team_defense_epa):,} records")
        else:
            logger.warning("  Team defensive EPA not found, will use defaults")
            self.team_defense_epa = None

        # Team pace
        team_pace_file = self.nflverse_dir / "team_pace.parquet"
        if team_pace_file.exists():
            self.team_pace = pd.read_parquet(team_pace_file)
            logger.info(f"  Team pace: {len(self.team_pace):,} records")
        else:
            logger.warning("  Team pace not found, will use defaults")
            self.team_pace = None

        logger.info("NFLverse data loaded successfully")

    def _derive_schedules_from_stats(self) -> pd.DataFrame:
        """Derive schedule from player stats if schedule file missing."""
        games = []
        for week in self.season_stats['week'].unique():
            week_data = self.season_stats[self.season_stats['week'] == week]
            teams = week_data['recent_team'].unique()

            # Group teams into matchups (approximation)
            for team in teams:
                games.append({
                    'season': self.current_season,
                    'week': week,
                    'home_team': team,
                    'away_team': 'UNK'  # Will need to be inferred
                })

        return pd.DataFrame(games)

    def build_trailing_stats_for_week(self, target_week: int, lookback: int = 4) -> Dict:
        """
        Build trailing stats using ONLY weeks before target_week.
        This prevents data leakage.
        """
        logger.info(f"  Building trailing stats for week {target_week}...")

        # Only use weeks BEFORE the target week
        available_data = self.season_stats[self.season_stats['week'] < target_week].copy()

        if len(available_data) == 0:
            logger.warning(f"  No historical data available for week {target_week}")
            return {}

        # Get recent weeks within lookback window
        recent_weeks = sorted(available_data['week'].unique())[-lookback:]
        recent_data = available_data[available_data['week'].isin(recent_weeks)]

        trailing_stats = {}

        # Aggregate by player
        for player_id in recent_data['player_id'].unique():
            player_data = recent_data[recent_data['player_id'] == player_id]

            if len(player_data) < 2:  # Need at least 2 weeks of data
                continue

            # Get latest team and position
            latest_row = player_data.iloc[-1]
            team = latest_row.get('recent_team', latest_row.get('team', 'UNK'))
            position = latest_row.get('position', 'UNK')

            if position not in ['QB', 'RB', 'WR', 'TE']:
                continue  # Skip non-skill positions

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
                'trailing_completions': player_data['completions'].mean() if 'completions' in player_data.columns else 0,

                # Yards
                'trailing_passing_yards': player_data['passing_yards'].mean() if 'passing_yards' in player_data.columns else 0,
                'trailing_rushing_yards': player_data['rushing_yards'].mean() if 'rushing_yards' in player_data.columns else 0,
                'trailing_receiving_yards': player_data['receiving_yards'].mean() if 'receiving_yards' in player_data.columns else 0,
                'trailing_receptions': player_data['receptions'].mean() if 'receptions' in player_data.columns else 0,

                # TDs
                'trailing_passing_tds': player_data['passing_tds'].mean() if 'passing_tds' in player_data.columns else 0,
                'trailing_rushing_tds': player_data['rushing_tds'].mean() if 'rushing_tds' in player_data.columns else 0,
                'trailing_receiving_tds': player_data['receiving_tds'].mean() if 'receiving_tds' in player_data.columns else 0,

                # Variance (for simulation)
                'std_passing_yards': player_data['passing_yards'].std() if 'passing_yards' in player_data.columns else 50,
                'std_rushing_yards': player_data['rushing_yards'].std() if 'rushing_yards' in player_data.columns else 20,
                'std_receiving_yards': player_data['receiving_yards'].std() if 'receiving_yards' in player_data.columns else 25,
                'std_receptions': player_data['receptions'].std() if 'receptions' in player_data.columns else 2,
            }

            trailing_stats[player_id] = stats

        logger.info(f"    Built trailing stats for {len(trailing_stats)} players")
        return trailing_stats

    def get_team_game_context(self, team: str, opponent: str, week: int) -> Dict:
        """
        Get game context for a team's matchup.
        Uses GameScriptEngine to simulate expected game flow.
        """
        # Get team's offensive strength (points per game proxy)
        team_offense_strength = 2.0  # Default: 2 points per drive

        # Get opponent's defensive strength
        opponent_defense_epa = 0.0  # Default: league average

        if self.team_defense_epa is not None:
            # Look up opponent's defensive EPA (season aggregate)
            opp_defense = self.team_defense_epa[
                self.team_defense_epa['team'] == opponent
            ]
            if len(opp_defense) > 0:
                # Use season EPA (positive = bad defense, negative = good defense)
                opponent_defense_epa = opp_defense.iloc[0].get('def_epa_allowed', 0.0)

        # Adjust team offense based on opponent defense
        # Positive opponent EPA = bad defense = increase team strength
        adjusted_team_strength = team_offense_strength * (1 + opponent_defense_epa * 0.5)

        # Get team pace
        team_pace_val = 30.0  # Default: 30 seconds per play
        if self.team_pace is not None:
            pace_data = self.team_pace[
                self.team_pace['team'] == team
            ]
            if len(pace_data) > 0:
                # Convert plays per game to seconds per play
                plays_per_game = pace_data.iloc[0].get('plays_per_game', 64)
                team_pace_val = 3600 / max(plays_per_game, 50)  # 60 minutes / plays

        # Simulate game flow to get expected pass/rush attempts
        game_flow = self.game_engine.simulate_game_flow(
            home_team_strength=adjusted_team_strength,
            away_team_strength=2.0,  # Assume opponent is average
            n_quarters=4,
            possessions_per_quarter=6,
            seed=42 + week  # Deterministic per week
        )

        # Calculate team's expected usage from simulation
        team_possessions = game_flow[game_flow['offensive_team'] == 'home']
        avg_pass_rate = team_possessions['pass_rate'].mean()

        # Estimate total plays (pace * game time)
        total_plays = int(60 * 60 / team_pace_val)  # 60 minutes
        team_plays = total_plays // 2  # Half the plays

        # Calculate team attempts
        team_pass_attempts = team_plays * avg_pass_rate
        team_rush_attempts = team_plays * (1 - avg_pass_rate)
        team_targets = team_pass_attempts * 0.85  # ~85% of pass attempts are targets

        # Calculate expected team total (points)
        total_points_scored = team_possessions['points_scored'].sum()
        projected_team_total = total_points_scored

        # Game script (expected point differential)
        final_home_score = game_flow.iloc[-1]['home_score']
        final_away_score = game_flow.iloc[-1]['away_score']
        game_script = final_home_score - final_away_score

        return {
            'projected_pace': team_pace_val,
            'projected_team_pass_attempts': team_pass_attempts,
            'projected_team_rush_attempts': team_rush_attempts,
            'projected_team_targets': team_targets,
            'projected_team_total': projected_team_total,
            'projected_opponent_total': final_away_score,
            'projected_game_script': game_script,
            'opponent_defense_epa': opponent_defense_epa
        }

    def get_opponent_for_team_week(self, team: str, week: int) -> str:
        """Get opponent for a team in a given week."""
        if self.schedules is not None and len(self.schedules) > 0:
            week_games = self.schedules[self.schedules['week'] == week]

            # Check home games
            home_game = week_games[week_games['home_team'] == team]
            if len(home_game) > 0:
                return home_game.iloc[0]['away_team']

            # Check away games
            away_game = week_games[week_games['away_team'] == team]
            if len(away_game) > 0:
                return away_game.iloc[0]['home_team']

        # Fallback: try to infer from player stats
        week_data = self.season_stats[
            (self.season_stats['week'] == week) &
            (self.season_stats['recent_team'] == team)
        ]
        if len(week_data) > 0 and 'opponent_team' in week_data.columns:
            return week_data.iloc[0]['opponent_team']

        return 'UNK'

    def create_player_prop_input(
        self,
        player_stats: Dict,
        game_context: Dict,
        opponent: str,
        week: int
    ) -> PlayerPropInput:
        """
        Create a fully populated PlayerPropInput with game context.
        """
        position = player_stats.get('position', 'WR')

        # Calculate usage shares
        team_targets = game_context['projected_team_targets']
        team_rush_attempts = game_context['projected_team_rush_attempts']

        if position in ['WR', 'TE']:
            target_share = player_stats.get('trailing_targets', 5) / max(team_targets, 1)
            carry_share = 0.0
        elif position == 'RB':
            target_share = player_stats.get('trailing_targets', 3) / max(team_targets, 1)
            carry_share = player_stats.get('trailing_carries', 10) / max(team_rush_attempts, 1)
        elif position == 'QB':
            target_share = 0.0
            carry_share = player_stats.get('trailing_carries', 3) / max(team_rush_attempts, 1)
        else:
            target_share = 0.0
            carry_share = 0.0

        # Clamp shares to reasonable ranges
        target_share = np.clip(target_share, 0.0, 0.5)
        carry_share = np.clip(carry_share, 0.0, 0.8)

        # Calculate efficiency metrics
        if position == 'QB':
            if player_stats.get('trailing_attempts', 0) > 0:
                yards_per_opp = player_stats.get('trailing_passing_yards', 0) / player_stats.get('trailing_attempts', 1)
                td_rate = player_stats.get('trailing_passing_tds', 0) / player_stats.get('trailing_attempts', 1)
            else:
                yards_per_opp = 7.0
                td_rate = 0.04
        elif position == 'RB':
            total_opps = player_stats.get('trailing_carries', 0) + player_stats.get('trailing_targets', 0)
            if total_opps > 0:
                total_yards = player_stats.get('trailing_rushing_yards', 0) + player_stats.get('trailing_receiving_yards', 0)
                yards_per_opp = total_yards / total_opps
                total_tds = player_stats.get('trailing_rushing_tds', 0) + player_stats.get('trailing_receiving_tds', 0)
                td_rate = total_tds / total_opps
            else:
                yards_per_opp = 5.0
                td_rate = 0.04
        else:  # WR/TE
            if player_stats.get('trailing_targets', 0) > 0:
                yards_per_opp = player_stats.get('trailing_receiving_yards', 0) / player_stats.get('trailing_targets', 1)
                td_rate = player_stats.get('trailing_receiving_tds', 0) / player_stats.get('trailing_targets', 1)
            else:
                yards_per_opp = 8.0
                td_rate = 0.05

        # Create PlayerPropInput with ALL required fields
        return PlayerPropInput(
            player_id=player_stats.get('player_id', 'unknown'),
            player_name=player_stats.get('player_name', 'Unknown'),
            team=player_stats.get('team', 'UNK'),
            position=position,
            week=week,
            opponent=opponent,
            season=self.current_season,

            # Game context (from simulation)
            projected_team_total=game_context['projected_team_total'],
            projected_opponent_total=game_context['projected_opponent_total'],
            projected_game_script=game_context['projected_game_script'],
            projected_pace=game_context['projected_pace'],

            # Team usage (from simulation)
            projected_team_pass_attempts=game_context['projected_team_pass_attempts'],
            projected_team_rush_attempts=game_context['projected_team_rush_attempts'],
            projected_team_targets=game_context['projected_team_targets'],

            # Historical player stats (trailing)
            trailing_snap_share=0.7,  # Default - would need snap count data
            trailing_target_share=target_share,
            trailing_carry_share=carry_share,
            trailing_yards_per_opportunity=yards_per_opp,
            trailing_td_rate=td_rate,

            # Opponent defense
            opponent_def_epa_vs_position=game_context['opponent_defense_epa'],
        )

    def simulate_player_props(
        self,
        player_input: PlayerPropInput,
        lines: Dict[str, List[float]]
    ) -> Dict[str, Dict[float, float]]:
        """
        Simulate player prop probabilities using Monte Carlo with V3 backend.
        """
        # Create simulator with proper game context
        simulator = PlayerSimulator(
            usage_predictor=self.usage_predictor,
            efficiency_predictor=self.efficiency_predictor,
            trials=self.trials,
            seed=42
        )

        # Provide game context explicitly
        game_context = {
            'pace': player_input.projected_pace,
            'game_script': player_input.projected_game_script,
            'team_pass_attempts': player_input.projected_team_pass_attempts,
            'team_rush_attempts': player_input.projected_team_rush_attempts,
            'team_targets': player_input.projected_team_targets
        }

        # Run simulation
        try:
            # Use V3's simulate_team_players with explicit game context
            results = simulator.simulate_team_players([player_input], game_context=game_context)
            sim_results = results.get(player_input.player_id, {})
        except Exception as e:
            logger.warning(f"  Simulation failed for {player_input.player_name}: {e}")
            # Fallback to simple normal distribution
            return self._fallback_simulation(player_input, lines)

        # Calculate probabilities for each market/line
        probabilities = {}

        for market, market_lines in lines.items():
            probabilities[market] = {}

            # Map market to simulation result key
            if market == 'player_pass_yds':
                sim_key = 'passing_yards'
            elif market == 'player_rush_yds':
                sim_key = 'rushing_yards'
            elif market == 'player_reception_yds':
                sim_key = 'receiving_yards'
            elif market == 'player_receptions':
                sim_key = 'receptions'
            else:
                continue

            if sim_key in sim_results and len(sim_results[sim_key]) > 0:
                sim_values = sim_results[sim_key]

                # Calculate probability of going over each line
                for line in market_lines:
                    prob_over = np.mean(sim_values > line)
                    probabilities[market][line] = np.clip(prob_over, 0.01, 0.99)
            else:
                # Fallback for missing stat
                for line in market_lines:
                    probabilities[market][line] = 0.5  # Unknown

        return probabilities

    def _fallback_simulation(
        self,
        player_input: PlayerPropInput,
        lines: Dict[str, List[float]]
    ) -> Dict[str, Dict[float, float]]:
        """Fallback to simple normal distribution when V3 simulation fails."""
        probabilities = {}

        for market, market_lines in lines.items():
            probabilities[market] = {}

            # Estimate mean from input
            if market == 'player_pass_yds':
                mean_val = player_input.trailing_yards_per_opportunity * 30  # ~30 attempts
                std_val = mean_val * 0.35
            elif market == 'player_rush_yds':
                mean_val = player_input.trailing_yards_per_opportunity * 15  # ~15 carries
                std_val = mean_val * 0.40
            elif market == 'player_reception_yds':
                mean_val = player_input.trailing_yards_per_opportunity * 6  # ~6 targets
                std_val = mean_val * 0.45
            elif market == 'player_receptions':
                mean_val = player_input.trailing_target_share * 35 * 0.65  # catch rate
                std_val = mean_val * 0.35
            else:
                continue

            # Calculate probabilities
            for line in market_lines:
                if std_val > 0:
                    z_score = (line - mean_val) / std_val
                    prob_over = 1 - self._normal_cdf(z_score)
                else:
                    prob_over = 0.5

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
            logger.warning(f"  Skipping week {week}: no trailing stats available")
            return

        # Step 2: Get actual outcomes for this week
        week_actuals = self.season_stats[self.season_stats['week'] == week].copy()

        if len(week_actuals) == 0:
            logger.warning(f"  Skipping week {week}: no actual outcomes available")
            return

        logger.info(f"  Actual outcomes: {len(week_actuals)} players")

        # Step 3: Define lines to evaluate (typical sportsbook lines)
        lines_config = {
            'player_pass_yds': [150.5, 175.5, 200.5, 225.5, 250.5, 275.5, 300.5],
            'player_rush_yds': [25.5, 40.5, 55.5, 70.5, 85.5, 100.5],
            'player_reception_yds': [25.5, 40.5, 55.5, 70.5, 85.5, 100.5],
            'player_receptions': [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
        }

        # Step 4: Cache game contexts by team
        game_contexts = {}

        # Step 5: For each player with actual outcomes
        records_this_week = 0
        players_processed = 0

        for _, actual_row in week_actuals.iterrows():
            player_id = actual_row['player_id']

            # Check if we have trailing stats for this player
            if player_id not in trailing_stats:
                continue

            player_trailing = trailing_stats[player_id]
            position = player_trailing.get('position', 'UNK')
            team = player_trailing.get('team', 'UNK')

            # Get opponent
            opponent = self.get_opponent_for_team_week(team, week)
            if opponent == 'UNK':
                continue

            # Get or calculate game context for this team
            if team not in game_contexts:
                game_contexts[team] = self.get_team_game_context(team, opponent, week)

            game_context = game_contexts[team]

            # Create proper PlayerPropInput
            try:
                player_input = self.create_player_prop_input(
                    player_trailing, game_context, opponent, week
                )
            except Exception as e:
                logger.debug(f"  Failed to create input for {player_trailing.get('player_name')}: {e}")
                continue

            # Generate predictions
            predictions = self.simulate_player_props(player_input, lines_config)

            if not predictions:
                continue

            players_processed += 1

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
                        'team': team,
                        'opponent': opponent,
                        'market': market,
                        'line': line,
                        'actual_value': actual_value,
                        'model_prob_raw': raw_prob,
                        'bet_won': outcome
                    })
                    records_this_week += 1

        logger.info(f"  Players processed: {players_processed}")
        logger.info(f"  Training records generated: {records_this_week:,}")

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

        logger.info(f"Starting FULL walk-forward calibration training")
        logger.info(f"Season: {self.current_season}")
        logger.info(f"Weeks to process: {weeks}")
        logger.info(f"Total weeks: {len(weeks)}")
        logger.info(f"Trials per player: {self.trials}")

        start_time = datetime.now()

        for week in weeks:
            self.process_week(week)

        # Save training data
        if self.training_records:
            df = pd.DataFrame(self.training_records)

            output_dir = self.data_dir / "training"
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "full_walk_forward_calibration_data.parquet"
            df.to_parquet(output_file, index=False)

            # Also save CSV for inspection
            df.to_csv(output_dir / "full_walk_forward_calibration_data.csv", index=False)

            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"\n{'='*80}")
            logger.info(f"WALK-FORWARD TRAINING COMPLETE")
            logger.info(f"{'='*80}")
            logger.info(f"Total records: {len(df):,}")
            logger.info(f"Weeks processed: {sorted(df['week'].unique().tolist())}")
            logger.info(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            logger.info(f"Output: {output_file}")

            # Market breakdown
            logger.info(f"\nMarket Breakdown:")
            for market in df['market'].unique():
                market_data = df[df['market'] == market]
                count = len(market_data)
                win_rate = market_data['bet_won'].mean()
                avg_prob = market_data['model_prob_raw'].mean()
                calibration_error = abs(win_rate - avg_prob)

                logger.info(f"  {market}:")
                logger.info(f"    Records: {count:,}")
                logger.info(f"    Actual win rate: {win_rate:.1%}")
                logger.info(f"    Avg model prob: {avg_prob:.1%}")
                logger.info(f"    Calibration error: {calibration_error:.1%}")

            # Position breakdown
            logger.info(f"\nPosition Breakdown:")
            for pos in df['position'].unique():
                pos_data = df[df['position'] == pos]
                count = len(pos_data)
                win_rate = pos_data['bet_won'].mean()
                avg_prob = pos_data['model_prob_raw'].mean()
                logger.info(f"  {pos}: {count:,} records, win rate {win_rate:.1%}, avg prob {avg_prob:.1%}")

        else:
            logger.error("No training records generated!")


def main():
    parser = argparse.ArgumentParser(description="Full Walk-Forward Calibration Training with Game Simulation")
    parser.add_argument('--weeks', type=str, help='Weeks to process (e.g., "2-10" or "3,5,7")')
    parser.add_argument('--all', action='store_true', help='Process all available weeks')
    parser.add_argument('--trials', type=int, default=1000, help='Number of Monte Carlo trials (default 1000)')

    args = parser.parse_args()

    trainer = FullWalkForwardCalibrationTrainer(trials=args.trials)

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
