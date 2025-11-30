"""
Unified Orchestrator for Game and Player Simulations

Coordinates the integration between:
- Game-level simulator (MonteCarloSimulator)
- Player-level simulator (PlayerSimulator)

Ensures game context flows correctly from game predictions to player predictions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pandas as pd

from nfl_quant.simulation.simulator import MonteCarloSimulator
from nfl_quant.simulation.player_simulator import PlayerSimulator, load_predictors
from nfl_quant.schemas import SimulationInput, SimulationOutput, PlayerPropInput
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator

logger = logging.getLogger(__name__)


class UnifiedOrchestrator:
    """
    Coordinates game-level and player-level simulators.

    Responsibilities:
    1. Run game simulations to get team totals and context
    2. Extract game context from simulation output
    3. Feed game context to player simulator
    4. Ensure both simulators use same data sources and calibrators
    """

    def __init__(
        self,
        trials: int = 50000,
        seed: int = 42,
        player_calibrator: Optional[NFLProbabilityCalibrator] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            trials: Number of Monte Carlo trials for simulations
            seed: Random seed for reproducibility
            player_calibrator: Calibrator for player prop probabilities
        """
        self.trials = trials
        self.seed = seed

        # Initialize game simulator
        self.game_simulator = MonteCarloSimulator(seed=seed)

        # Initialize player simulator with predictors
        logger.info("Loading ML predictors for player simulator...")
        usage_predictor, efficiency_predictor = load_predictors()

        self.player_simulator = PlayerSimulator(
            usage_predictor=usage_predictor,
            efficiency_predictor=efficiency_predictor,
            trials=trials,
            seed=seed,
            calibrator=player_calibrator,
        )

        logger.info("Orchestrator initialized successfully")

    def extract_game_context(
        self,
        game_output: SimulationOutput,
        sim_input: SimulationInput
    ) -> Dict[str, float]:
        """
        Extract game context from simulation output for player simulator.

        Args:
            game_output: Output from game simulation allocation
            sim_input: Original game simulation input

        Returns:
            Dictionary with game context variables
        """
        # Calculate game script (score differential)
        game_script = game_output.home_score_median - game_output.away_score_median

        # Get pace (average of both teams)
        avg_pace = (sim_input.home_pace + sim_input.away_pace) / 2

        return {
            'home_total': game_output.home_score_median,
            'away_total': game_output.away_score_median,
            'game_script_home': game_script,
            'game_script_away': -game_script,  # Inverted for away team
            'total_median': game_output.total_median,
            'margin_median': game_script,
            'pace': avg_pace,
            'total_std': game_output.total_std,
        }

    def build_player_input_from_context(
        self,
        player_data: Dict,
        game_context: Dict,
        is_home: bool,
        week: int
    ) -> Optional[PlayerPropInput]:
        """
        Build PlayerPropInput using game context.

        Args:
            player_data: Dictionary with player information
                Required keys: player_id, player_name, position, trailing_snap_share
                Optional keys: trailing_target_share, trailing_carry_share,
                             trailing_yards_per_opportunity, trailing_td_rate, opponent
            game_context: Game context from extract_game_context()
            is_home: Whether player's team is home team
            week: Week number

        Returns:
            PlayerPropInput object or None if data insufficient
        """
        # Determine which team total to use
        team_total = game_context['home_total'] if is_home else game_context['away_total']
        opponent_total = game_context['away_total'] if is_home else game_context['home_total']
        game_script = game_context['game_script_home'] if is_home else game_context['game_script_away']

        # Build player input
        try:
            player_input = PlayerPropInput(
                player_id=player_data.get('player_id', player_data['player_name']),
                player_name=player_data['player_name'],
                team=player_data.get('team', ''),
                position=player_data['position'],
                week=week,
                opponent=player_data.get('opponent', ''),
                projected_team_total=team_total,
                projected_opponent_total=opponent_total,
                projected_game_script=game_script,
                projected_pace=game_context['pace'],
                trailing_snap_share=player_data.get('trailing_snap_share'),  # No default - must have data
                trailing_target_share=player_data.get('trailing_target_share'),
                trailing_carry_share=player_data.get('trailing_carry_share'),
                trailing_yards_per_opportunity=player_data.get('trailing_yards_per_opportunity'),  # No default
                trailing_td_rate=player_data.get('trailing_td_rate'),  # No default
                opponent_def_epa_vs_position=player_data.get('opponent_def_epa_vs_position'),  # No default
            )
            return player_input
        except KeyError as e:
            logger.warning(f"Missing required field for {player_data.get('player_name', 'unknown')}: {e}")
            return None

    def simulate_game_and_players(
        self,
        game_input: SimulationInput,
        players: List[Dict],
        home_team: str,
        away_team: str,
        week: int
    ) -> Tuple[Dict, List]:
        """
        Run complete simulation: game + players.

        Args:
            game_input: Game simulation input
            players: List of player dictionaries
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            week: Week number

        Returns:
            Tuple of (game_context, player_outputs)
        """
        # Step 1: Simulate game
        logger.info(f"Simulating game: {away_team} @ {home_team}")
        game_output = self.game_simulator.simulate_game(game_input, trials=self.trials)

        # Step 2: Extract game context
        game_context = self.extract_game_context(game_output, game_input)
        logger.debug(f"Game context: home_total={game_context['home_total']:.1f}, "
                    f"away_total={game_context['away_total']:.1f}")

        # Step 3: Simulate each player with game context
        player_outputs = []
        for player_data in players:
            is_home = player_data.get('team') == home_team

            player_input = self.build_player_input_from_context(
                player_data, game_context, is_home, week
            )

            if player_input is None:
                continue

            # Run player simulation
            try:
                outputs = self.player_simulator.simulate_all_props(
                    player_input,
                    prop_lines=None  # Simulate all prop types
                )
                player_outputs.append({
                    'player': player_data['player_name'],
                    'position': player_data['position'],
                    'team': player_data.get('team', ''),
                    'outputs': outputs,
                })
            except Exception as e:
                logger.warning(f"Error simulating {player_data['player_name']}: {e}")
                continue

        logger.info(f"Successfully simulated {len(player_outputs)} players")

        return game_context, player_outputs

    def get_game_context_for_team(
        self,
        game_context: Dict,
        team: str,
        is_home: bool
    ) -> Dict[str, float]:
        """
        Extract team-specific context from game context.

        Args:
            game_context: Full game context from extract_game_context()
            team: Team abbreviation
            is_home: Whether team is home

        Returns:
            Team-specific context dictionary
        """
        return {
            'team_total': game_context['home_total'] if is_home else game_context['away_total'],
            'opponent_total': game_context['away_total'] if is_home else game_context['home_total'],
            'game_script': game_context['game_script_home'] if is_home else game_context['game_script_away'],
            'pace': game_context['pace'],
            'total': game_context['total_median'],
        }


def create_orchestrator(calibrator_path: Optional[str] = None) -> UnifiedOrchestrator:
    """
    Factory function to create orchestrator with optional calibrator.

    Args:
        calibrator_path: Path to calibrator JSON file

    Returns:
        UnifiedOrchestrator instance
    """
    player_calibrator = None
    if calibrator_path and Path(calibrator_path).exists():
        try:
            player_calibrator = NFLProbabilityCalibrator()
            player_calibrator.load(calibrator_path)
            logger.info(f"Loaded calibrator from {calibrator_path}")
        except Exception as e:
            logger.warning(f"Could not load calibrator: {e}")
    else:
        logger.info("Using calibrator from default path")
        calibrator_path = Path('configs/calibrator.json')
        if calibrator_path.exists():
            try:
                player_calibrator = NFLProbabilityCalibrator()
                player_calibrator.load(str(calibrator_path))
                logger.info("Loaded default calibrator")
            except Exception as e:
                logger.warning(f"Could not load default calibrator: {e}")

    return UnifiedOrchestrator(player_calibrator=player_calibrator)
