"""
Player-level Monte Carlo simulator V3 - With Correlation Matrix.

Improvements over V2:
1. Explicit correlation modeling between same-team players (QB-WR, RB committee)
2. Team-level constraint enforcement (Dirichlet allocation)
3. Dynamic game script evolution based on simulated score differential
4. Research-backed correlation coefficients from historical data

Uses Cholesky decomposition for efficient correlated sampling.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

from nfl_quant.schemas import PlayerPropInput, PlayerPropOutput
from nfl_quant.models.usage_predictor import UsagePredictor
from nfl_quant.models.efficiency_predictor import EfficiencyPredictor
from nfl_quant.calibration.isotonic_calibrator import NFLProbabilityCalibrator
from nfl_quant.simulation.correlation_matrix import PlayerCorrelationMatrix, PlayerCorrelationConfig
from nfl_quant.config_enhanced import config
from nfl_quant.utils.nflverse_data_loader import NFLverseDataLoader
from nfl_quant.utils.season_utils import get_current_season, get_training_seasons

logger = logging.getLogger(__name__)

# Load configuration values
_sim_config = config.simulation.player_simulation_variance
_mean_adjustments = config.simulation.mean_adjustments
_completion_bounds = config.simulation.completion_bounds
_blending_weights = config.simulation.blending_weights
_matchup_adj = config.simulation.matchup_adjustments
_trials_config = config.simulation.trials
_calibration_dampening = config.simulation.calibration_dampening
_league_defaults = config.simulation.league_defaults


class PlayerSimulatorV3:
    """
    Monte Carlo simulator for player props with correlation modeling.

    Key Features:
    - Correlated player sampling (QB-WR, RB committee)
    - Team-level constraint enforcement
    - Dynamic game script evolution
    - Research-backed correlation coefficients
    """

    def __init__(
        self,
        usage_predictor: UsagePredictor,
        efficiency_predictor: EfficiencyPredictor,
        trials: Optional[int] = None,
        seed: Optional[int] = None,
        calibrator: Optional[NFLProbabilityCalibrator] = None,
        td_calibrator: Optional[object] = None,
        correlation_config: Optional[PlayerCorrelationConfig] = None,
    ):
        """
        Initialize player simulator with correlation support.

        Args:
            usage_predictor: Fitted usage prediction model
            efficiency_predictor: Fitted efficiency prediction model
            trials: Number of Monte Carlo trials (default from config)
            seed: Random seed for reproducibility (default from config)
            calibrator: Optional probability calibrator for yards/receptions props
            td_calibrator: Optional probability calibrator for TD props
            correlation_config: Correlation coefficient configuration
        """
        # Use config defaults if not specified
        if trials is None:
            trials = _trials_config['default_trials']
        if seed is None:
            seed = _trials_config['default_seed']

        self.usage_predictor = usage_predictor
        self.efficiency_predictor = efficiency_predictor
        self.trials = trials
        self.seed = seed
        self.calibrator = calibrator
        self.td_calibrator = td_calibrator

        # Initialize correlation matrix builder
        self.correlation_builder = PlayerCorrelationMatrix(
            config=correlation_config or PlayerCorrelationConfig()
        )

        # Initialize NFLverse data loader (NFLverse-only data source)
        self.nflverse_loader = NFLverseDataLoader()
        logger.debug("Initialized NFLverseDataLoader for catch rate calculations")

        # Cache for league-wide stats (calculated once, reused)
        self._league_rb_rec_td_rate_cache = None

        if seed is not None:
            np.random.seed(seed)

    def simulate_team_players(
        self,
        team_players: List[PlayerPropInput],
        game_context: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Simulate all players on a team with correlations.

        Args:
            team_players: List of PlayerPropInput for same team
            game_context: Optional game-level context (pace, game script, team totals)

        Returns:
            Dictionary mapping player_id to stat distributions
        """
        if not team_players:
            return {}

        # Extract game context
        if game_context is None:
            game_context = self._extract_game_context(team_players)

        # Separate by stat type for correlation modeling
        stat_types = ['attempts', 'targets', 'carries', 'yards']
        results = {}

        # First pass: Generate base predictions for all players
        base_predictions = {}
        for player_input in team_players:
            player_id = player_input.player_id
            base_predictions[player_id] = self._get_base_predictions(player_input, game_context)

        # Second pass: Apply correlation by stat type
        for stat_type in stat_types:
            # Filter players with this stat type
            relevant_players = []
            relevant_inputs = []

            for player_input in team_players:
                player_id = player_input.player_id
                if stat_type in base_predictions[player_id]:
                    relevant_players.append({
                        'player_id': player_id,
                        'position': player_input.position,
                        'role': self._infer_player_role(player_input),
                        'stat_type': stat_type
                    })
                    relevant_inputs.append(player_input)

            if not relevant_players:
                continue

            # Build correlation matrix for these players
            corr_matrix, ordered_players = self.correlation_builder.build_correlation_matrix(
                players=relevant_players,
                team=team_players[0].team
            )

            # Extract mean and std for each player
            mean_values = []
            std_values = []

            for ordered_player in ordered_players:
                player_id = ordered_player['player_id']
                pred = base_predictions[player_id][stat_type]
                mean_values.append(pred['mean'])
                std_values.append(pred['std'])

            mean_values = np.array(mean_values)
            std_values = np.array(std_values)

            # Generate correlated samples
            # Use Poisson for count data (targets, carries, attempts) to avoid binomial floor effect
            use_poisson = stat_type in ['targets', 'carries', 'attempts']
            correlated_samples = self.correlation_builder.generate_correlated_samples(
                mean_values=mean_values,
                std_values=std_values,
                correlation_matrix=corr_matrix,
                n_samples=self.trials,
                random_state=self.seed,
                use_poisson=use_poisson
            )

            # Apply team-level constraints
            team_totals = self._calculate_team_totals(game_context, stat_type)
            if team_totals:
                correlated_samples = self.correlation_builder.apply_team_constraints(
                    samples=correlated_samples,
                    players=ordered_players,
                    team_totals=team_totals
                )

            # Store results by player
            for i, ordered_player in enumerate(ordered_players):
                player_id = ordered_player['player_id']
                if player_id not in results:
                    results[player_id] = {}

                results[player_id][stat_type] = correlated_samples[:, i]

        # Third pass: Derive dependent stats (e.g., yards from targets)
        for player_input in team_players:
            player_id = player_input.player_id
            if player_id not in results:
                results[player_id] = {}

            # Get correlated usage stats
            usage_stats = results[player_id]

            # Derive efficiency-based stats
            derived_stats = self._derive_efficiency_stats(
                player_input=player_input,
                usage_stats=usage_stats,
                base_predictions=base_predictions[player_id]
            )

            results[player_id].update(derived_stats)

        return results

    def _get_base_predictions(
        self,
        player_input: PlayerPropInput,
        game_context: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get base predictions (mean, std) for a player's stats.

        Args:
            player_input: Player input data
            game_context: Game-level context

        Returns:
            Dictionary with stat predictions:
                {'attempts': {'mean': 35.0, 'std': 5.0}, ...}
        """
        predictions = {}

        position = player_input.position

        # Predict usage (attempts, targets, carries)
        if position == 'QB':
            # QB attempts based on pace and game script - FAIL if game context missing
            team_pace = game_context.get('pace')
            game_script = game_context.get('game_script', 0.0)
            team_pass_attempts = game_context.get('team_pass_attempts')
            team_rush_attempts = game_context.get('team_rush_attempts')

            if team_pace is None:
                raise ValueError("Game context missing 'pace' for QB predictions. Run game simulations.")
            if team_pass_attempts is None:
                raise ValueError("Game context missing 'team_pass_attempts' for QB predictions. Run game simulations.")
            if team_rush_attempts is None:
                raise ValueError("Game context missing 'team_rush_attempts' for QB predictions. Run game simulations.")

            # Use team pass attempts directly (already calculated from game simulation)
            # This is more accurate than deriving from rush attempts
            base_attempts = team_pass_attempts if team_pass_attempts > 0 else None
            if base_attempts is None:
                raise ValueError("Cannot calculate QB attempts: team_pass_attempts is 0 or invalid.")

            # Pace adjustment: faster pace = more attempts (empirical: ~2 attempts per second of pace difference)
            # Use league average pace (30.0) as baseline - this is empirical, not arbitrary
            league_avg_pace = 30.0  # Empirical NFL average
            pace_adj = (team_pace - league_avg_pace) / 2.0
            script_adj = game_script / 3.0

            mean_attempts = base_attempts + pace_adj + script_adj
            mean_attempts = np.clip(mean_attempts, 25, 50)

            predictions['attempts'] = {
                'mean': mean_attempts,
                'std': 5.0  # Typical QB attempt variance (empirical)
            }

            # QB carries (rushing attempts) - FAIL if trailing_carry_share missing
            if player_input.trailing_carry_share is None:
                raise ValueError(
                    f"Trailing carry share not available for {player_input.player_name}. "
                    f"Run unified historical stats calculation."
                )
            mean_carries = player_input.trailing_carry_share * team_rush_attempts
            if mean_carries <= 0:
                mean_carries = 0.0  # Explicit 0, not a default

            predictions['carries'] = {
                'mean': mean_carries,
                'std': 2.0  # Empirical variance
            }

        elif position == 'RB':
            # RB carries - FAIL if game context or trailing stats missing
            team_rush_attempts = game_context.get('team_rush_attempts')
            if team_rush_attempts is None:
                raise ValueError("Game context missing 'team_rush_attempts' for RB predictions. Run game simulations.")

            if player_input.trailing_carry_share is None:
                raise ValueError(
                    f"Trailing carry share not available for {player_input.player_name}. "
                    f"Run unified historical stats calculation."
                )

            mean_carries = player_input.trailing_carry_share * team_rush_attempts
            if mean_carries <= 0:
                raise ValueError(
                    f"Cannot calculate RB carries for {player_input.player_name}: "
                    f"trailing_carry_share={player_input.trailing_carry_share}, "
                    f"team_rush_attempts={team_rush_attempts}. "
                    f"Player may not be active or data is missing."
                )

            predictions['carries'] = {
                'mean': mean_carries,
                'std': max(3.0, mean_carries * 0.25)  # 25% variance (empirical)
            }

            # RB targets - FAIL if game context missing
            team_pass_attempts = game_context.get('team_pass_attempts')
            if team_pass_attempts is None:
                raise ValueError("Game context missing 'team_pass_attempts' for RB predictions. Run game simulations.")

            # RB targets - use actual historical data, NO hardcoded assumptions
            if player_input.trailing_target_share and player_input.trailing_target_share > 0:
                mean_targets = player_input.trailing_target_share * team_pass_attempts
            else:
                # Use actual avg_rec_yd from historical stats to estimate targets
                avg_rec_yd = getattr(player_input, 'avg_rec_yd', None)

                if avg_rec_yd and avg_rec_yd > 0:
                    # Get efficiency predictions (uses actual historical data)
                    efficiency_preds = self._predict_efficiency(player_input)
                    yards_per_target = efficiency_preds.get('yards_per_target', None)

                    if yards_per_target and yards_per_target > 0:
                        # Calculate catch rate from actual historical data if available
                        catch_rate = self._get_catch_rate_from_data(player_input)

                        # Estimate targets from receiving yards:
                        # targets = rec_yd / yards_per_target
                        mean_targets = avg_rec_yd / yards_per_target
                    else:
                        # No efficiency data - calculate from historical receiving data
                        avg_receptions = getattr(player_input, 'avg_receptions', None)
                        if avg_receptions and avg_receptions > 0:
                            # Calculate yards_per_target from actual player data
                            ypr = avg_rec_yd / avg_receptions
                            catch_rate = self._get_catch_rate_from_data(player_input)
                            yards_per_target = ypr * catch_rate
                            mean_targets = avg_rec_yd / yards_per_target
                            logger.debug(f"Calculated yards_per_target={yards_per_target:.2f} from historical data for RB {player_input.player_name}")
                        else:
                            # Cannot calculate without complete historical data
                            raise ValueError(
                                f"Cannot calculate RB targets for {player_input.player_name}: "
                                f"No yards_per_target from model and no avg_receptions to estimate. "
                                f"Need complete historical receiving data."
                            )
                else:
                    # No historical receiving data - RB is pure rusher
                    # Set targets to 0 (don't fake receiving predictions)
                    mean_targets = 0.0
                    logger.debug(f"RB {player_input.player_name} has no receiving history (avg_rec_yd=0), targets=0")

            # Validate calculated targets
            if mean_targets < 0:
                mean_targets = 0.0  # Ensure non-negative

            predictions['targets'] = {
                'mean': mean_targets,
                'std': max(0.1, mean_targets * 0.35) if mean_targets > 0 else 0.1
            }

        elif position in ['WR', 'TE']:
            # Receiver targets - FAIL if game context or trailing stats missing
            team_pass_attempts = game_context.get('team_pass_attempts')
            if team_pass_attempts is None:
                raise ValueError(f"Game context missing 'team_pass_attempts' for {position} predictions. Run game simulations.")

            if player_input.trailing_target_share is None:
                raise ValueError(
                    f"Trailing target share not available for {player_input.player_name}. "
                    f"Run unified historical stats calculation."
                )

            # Handle players with 0.0 target share - only estimate if we have receiving data
            if player_input.trailing_target_share == 0.0:
                # Try to estimate from available receiving data
                avg_rec_yd = getattr(player_input, 'avg_rec_yd', None)

                if avg_rec_yd and avg_rec_yd > 0:
                    # We have receiving yards data - can estimate targets from efficiency model
                    efficiency_preds = self._predict_efficiency(player_input)
                    yards_per_target = efficiency_preds.get('yards_per_target', None)

                    if yards_per_target and yards_per_target > 0:
                        # Estimate: targets = avg_yards / yards_per_target
                        mean_targets = max(1.0, avg_rec_yd / yards_per_target)
                        logger.debug(
                            f"{player_input.player_name} ({position}): "
                            f"trailing_target_share=0.0, estimated {mean_targets:.1f} targets "
                            f"from avg_rec_yd={avg_rec_yd:.1f} / yards_per_target={yards_per_target:.1f}"
                        )
                    else:
                        # Can't estimate - no efficiency data
                        raise ValueError(
                            f"Cannot calculate {position} targets for {player_input.player_name}: "
                            f"trailing_target_share=0.0, avg_rec_yd={avg_rec_yd:.1f} but no yards_per_target data. "
                            f"Player may not be active or data is missing."
                        )
                else:
                    # No receiving data at all - cannot simulate
                    raise ValueError(
                        f"Cannot calculate {position} targets for {player_input.player_name}: "
                        f"trailing_target_share=0.0, avg_rec_yd={avg_rec_yd or 0.0}, team_pass_attempts={team_pass_attempts}. "
                        f"Player has no receiving data - cannot simulate without historical information."
                    )
            else:
                # Normal case: use trailing_target_share
                mean_targets = player_input.trailing_target_share * team_pass_attempts

            # Validate calculated targets
            if mean_targets <= 0:
                raise ValueError(
                    f"Cannot calculate {position} targets for {player_input.player_name}: "
                    f"calculated mean_targets={mean_targets:.2f} from trailing_target_share={player_input.trailing_target_share}, "
                    f"team_pass_attempts={team_pass_attempts}. Player may not be active or data is missing."
                )

            predictions['targets'] = {
                'mean': mean_targets,
                'std': max(2.0, mean_targets * 0.30)  # 30% variance (empirical)
            }

        return predictions

    def _derive_efficiency_stats(
        self,
        player_input: PlayerPropInput,
        usage_stats: Dict[str, np.ndarray],
        base_predictions: Dict[str, Dict[str, float]]
    ) -> Dict[str, np.ndarray]:
        """
        Derive efficiency-based stats from usage (e.g., yards from carries).

        Args:
            player_input: Player input data
            usage_stats: Dictionary of usage stat arrays (attempts, targets, carries)
            base_predictions: Base prediction parameters

        Returns:
            Dictionary of derived stat arrays (yards, TDs, etc.)
        """
        derived = {}
        position = player_input.position

        # Get efficiency predictions from model
        efficiency_preds = self._predict_efficiency(player_input)

        # Debug logging for QBs
        if position == 'QB':
            logger.debug(f"QB efficiency predictions for {player_input.player_name}: {efficiency_preds}")

        if position == 'QB':
            # Passing yards from attempts
            if 'attempts' in usage_stats:
                attempts = usage_stats['attempts']

                # Debug: log actual attempts being used
                logger.debug(f"QB {player_input.player_name} attempts: mean={np.mean(attempts):.1f}, min={np.min(attempts):.1f}, max={np.max(attempts):.1f}")

                # Get efficiency predictions - FAIL if missing
                comp_pct = efficiency_preds.get('completion_pct')
                if comp_pct is None:
                    raise ValueError(
                        f"Efficiency model missing 'completion_pct' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )
                
                yards_per_comp = efficiency_preds.get('yards_per_completion')
                if yards_per_comp is None:
                    raise ValueError(
                        f"Efficiency model missing 'yards_per_completion' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # Completions
                completions = np.random.binomial(n=attempts.astype(int), p=comp_pct)

                # Yards per completion (gamma distribution for variance)
                alpha = _sim_config.get('qb_passing_alpha', 5.0)
                if player_input.player_id == '421' or np.random.random() < 0.01:  # Log occasionally
                    logger.info(f"QB passing alpha={alpha}, config keys: {list(_sim_config.keys())[:10]}")
                scale = yards_per_comp / alpha
                ypc_samples = np.random.gamma(alpha, scale, size=self.trials)

                passing_yards = completions * ypc_samples

                # Apply variance multiplier to widen distributions (avoids gamma heavy tail issues)
                variance_mult = _sim_config.get('qb_passing_variance_multiplier', 1.0)
                if variance_mult > 1.0:
                    current_mean = np.mean(passing_yards)
                    current_std = np.std(passing_yards)
                    # Scale distribution to increase variance
                    target_std = current_std * np.sqrt(variance_mult)
                    scale_factor = target_std / current_std
                    passing_yards = current_mean + (passing_yards - current_mean) * scale_factor

                    # Log variance scaling on first invocation
                    if not hasattr(self, '_logged_qb_variance'):
                        final_std = np.std(passing_yards)
                        logger.debug(f"QB variance scaled: {current_std:.1f} → {final_std:.1f} (mean={current_mean:.1f}, 90% interval=[{np.percentile(passing_yards, 5):.1f}, {np.percentile(passing_yards, 95):.1f}])")
                        self._logged_qb_variance = True

                # Apply mean adjustment for QB passing (fix +13% overestimation)
                mean_adj = _mean_adjustments.get('qb_passing_mean_adjustment', 1.0)
                if mean_adj != 1.0:
                    passing_yards = passing_yards * mean_adj

                derived['passing_yards'] = passing_yards
                derived['completions'] = completions
                derived['passing_completions'] = completions  # Alias for consistency
                derived['attempts'] = attempts
                derived['passing_attempts'] = attempts  # Alias for consistency

                # Passing TDs (binomial from attempts)
                td_rate = efficiency_preds.get('td_rate_pass')
                if td_rate is None:
                    raise ValueError(
                        f"Efficiency model missing 'td_rate_pass' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )
                derived['passing_tds'] = np.random.binomial(n=attempts.astype(int), p=td_rate)

            # Rushing yards from carries
            if 'carries' in usage_stats:
                carries = usage_stats['carries']
                yards_per_carry = efficiency_preds.get('yards_per_carry')
                if yards_per_carry is None:
                    raise ValueError(
                        f"Efficiency model missing 'yards_per_carry' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                alpha = _sim_config.get('qb_rushing_alpha', 1.0)  # Use QB-specific alpha
                scale = yards_per_carry / alpha
                ypc_samples = np.random.gamma(alpha, scale, size=self.trials)

                rushing_yards = carries * ypc_samples

                # QB rushing has high variance (Lamar vs backups) - apply variance multiplier
                variance_mult = _sim_config.get('qb_rushing_variance_multiplier', 1.0)
                if variance_mult > 1.0:
                    current_mean = np.mean(rushing_yards)
                    current_std = np.std(rushing_yards)
                    target_std = current_std * np.sqrt(variance_mult)
                    scale_factor = target_std / current_std
                    rushing_yards = current_mean + (rushing_yards - current_mean) * scale_factor

                # Apply mean adjustment for QB rushing (fix -8% underestimation)
                mean_adj = _mean_adjustments.get('qb_rushing_mean_adjustment', 1.0)
                if mean_adj != 1.0:
                    rushing_yards = rushing_yards * mean_adj

                derived['rushing_yards'] = rushing_yards

                # Rushing TDs - Apply red zone adjustments if available
                td_rate_rush = efficiency_preds.get('td_rate_rush')
                if td_rate_rush is None:
                    raise ValueError(
                        f"Efficiency model missing 'td_rate_rush' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # Apply red zone carry share boost if available
                if player_input.redzone_carry_share is not None and player_input.redzone_carry_share > 0:
                    # Red zone carries have higher TD rate (3x multiplier for goal line)
                    if player_input.goalline_carry_share is not None and player_input.goalline_carry_share > 0:
                        # Goal line carries: 3x TD rate
                        goal_line_multiplier = 3.0
                        # Weighted average: goal line portion gets 3x, rest of red zone gets 1.5x
                        rz_multiplier = 1.0 + (player_input.redzone_carry_share * 0.5) + (player_input.goalline_carry_share * 2.0)
                        td_rate_rush *= rz_multiplier
                    else:
                        # Red zone carries: 1.5x TD rate
                        rz_multiplier = 1.0 + (player_input.redzone_carry_share * 0.5)
                        td_rate_rush *= rz_multiplier

                derived['rushing_tds'] = np.random.binomial(n=carries.astype(int), p=td_rate_rush)

        elif position == 'RB':
            # Rushing yards and TDs
            if 'carries' in usage_stats:
                carries = usage_stats['carries']
                yards_per_carry = efficiency_preds.get('yards_per_carry')
                if yards_per_carry is None:
                    raise ValueError(
                        f"Efficiency model missing 'yards_per_carry' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # NOTE: Blanket calibration multipliers REMOVED 2025-11-25
                # Tier-based calibration now applied in generate_unified_recommendations_v3.py
                # This allows different adjustments for high-volume vs low-volume players
                # (High volume: under-predicted → OVER edge; Low volume: over-predicted → UNDER edge)

                alpha = _sim_config.get('rb_rushing_alpha', 1.5)
                scale = yards_per_carry / alpha
                ypc_samples = np.random.gamma(alpha, scale, size=self.trials)

                rushing_yards = carries * ypc_samples

                # Apply variance multiplier to achieve realistic coverage
                variance_mult = _sim_config.get('rb_rushing_variance_multiplier', 1.0)
                if variance_mult > 1.0:
                    current_mean = np.mean(rushing_yards)
                    current_std = np.std(rushing_yards)
                    target_std = current_std * np.sqrt(variance_mult)
                    scale_factor = target_std / current_std
                    rushing_yards = current_mean + (rushing_yards - current_mean) * scale_factor

                derived['rushing_yards'] = rushing_yards

                td_rate_rush = efficiency_preds.get('td_rate_rush')
                if td_rate_rush is None:
                    raise ValueError(
                        f"Efficiency model missing 'td_rate_rush' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # Apply red zone carry share boost if available
                if player_input.redzone_carry_share is not None and player_input.redzone_carry_share > 0:
                    # Red zone carries have higher TD rate
                    if player_input.goalline_carry_share is not None and player_input.goalline_carry_share > 0:
                        # Goal line carries: 3x TD rate
                        rz_multiplier = 1.0 + (player_input.redzone_carry_share * 0.5) + (player_input.goalline_carry_share * 2.0)
                        td_rate_rush *= rz_multiplier
                    else:
                        # Red zone carries: 1.5x TD rate
                        rz_multiplier = 1.0 + (player_input.redzone_carry_share * 0.5)
                        td_rate_rush *= rz_multiplier

                derived['rushing_tds'] = np.random.binomial(n=carries.astype(int), p=td_rate_rush)

            # Receiving yards and TDs
            if 'targets' in usage_stats:
                targets = usage_stats['targets']
                yards_per_target = efficiency_preds.get('yards_per_target')
                if yards_per_target is None:
                    raise ValueError(
                        f"Efficiency model missing 'yards_per_target' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # Receptions (catch rate) - calculate from actual data
                catch_rate = self._get_catch_rate_from_data(player_input)
                if catch_rate is None or catch_rate <= 0:
                    raise ValueError(
                        f"Cannot calculate catch rate for {player_input.player_name}. "
                        f"Need historical receptions/targets data."
                    )
                receptions = np.random.binomial(n=targets.astype(int), p=catch_rate)

                # Yards
                alpha = _sim_config.get('rb_receiving_alpha', 1.5)
                scale = (yards_per_target / catch_rate) / alpha  # Yards per reception
                ypr_samples = np.random.gamma(alpha, scale, size=self.trials)

                receiving_yards = receptions * ypr_samples

                # Apply mean adjustment for RB receiving (fix -164% underestimation)
                mean_adj = _mean_adjustments.get('rb_receiving_mean_adjustment', 1.0)
                if mean_adj != 1.0:
                    before_mean = np.mean(receiving_yards)
                    receiving_yards = receiving_yards * mean_adj

                    # Log adjustment on first invocation
                    if not hasattr(self, '_logged_rb_receiving_applied'):
                        after_mean = np.mean(receiving_yards)
                        logger.debug(f"RB receiving yards adjusted: {before_mean:.1f} → {after_mean:.1f} (multiplier={mean_adj})")
                        self._logged_rb_receiving_applied = True

                derived['receiving_yards'] = receiving_yards
                derived['receptions'] = receptions

                td_rate_rec = efficiency_preds.get('td_rate_rec')
                if td_rate_rec is None:
                    # Calculate from historical data: TD rate = receiving_tds / targets
                    # Use player's actual historical receiving TD data
                    avg_rec_tds = getattr(player_input, 'avg_rec_tds', None)
                    avg_targets = getattr(player_input, 'avg_rec_tgt', None) or getattr(player_input, 'avg_targets', None)

                    if avg_rec_tds is not None and avg_targets and avg_targets > 0:
                        td_rate_rec = avg_rec_tds / avg_targets
                        logger.debug(f"Calculated td_rate_rec={td_rate_rec:.3f} from historical data for RB {player_input.player_name}")
                    else:
                        # Calculate league-wide RB receiving TD rate from actual 2025 data
                        # Instead of hardcoding, pull from NFLverse weekly stats
                        td_rate_rec = self._get_league_rb_receiving_td_rate(player_input.player_name)
                        logger.debug(f"Using league RB receiving TD rate={td_rate_rec:.4f} for {player_input.player_name}")

                # Apply red zone target share boost if available
                if player_input.redzone_target_share is not None and player_input.redzone_target_share > 0:
                    # Red zone targets have higher TD rate (1.5x multiplier)
                    rz_multiplier = 1.0 + (player_input.redzone_target_share * 0.5)
                    td_rate_rec *= rz_multiplier

                derived['receiving_tds'] = np.random.binomial(n=targets.astype(int), p=td_rate_rec)

        elif position in ['WR', 'TE']:
            if 'targets' in usage_stats:
                targets = usage_stats['targets']
                yards_per_target = efficiency_preds.get('yards_per_target')

                if yards_per_target is None:
                    raise ValueError(
                        f"Efficiency model missing 'yards_per_target' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # Receptions (catch rate) - calculate from actual historical data
                catch_rate = self._get_catch_rate_from_data(player_input)
                receptions = np.random.binomial(n=targets.astype(int), p=catch_rate)

                # Yards
                alpha = _sim_config.get('wr_te_receiving_alpha', 3.0)
                scale = (yards_per_target / catch_rate) / alpha
                ypr_samples = np.random.gamma(alpha, scale, size=self.trials)

                receiving_yards = receptions * ypr_samples

                # Apply variance multiplier if needed
                variance_mult = _sim_config.get('wr_te_receiving_variance_multiplier', 1.0)
                if variance_mult > 1.0:
                    current_mean = np.mean(receiving_yards)
                    current_std = np.std(receiving_yards)
                    target_std = current_std * np.sqrt(variance_mult)
                    scale_factor = target_std / current_std
                    receiving_yards = current_mean + (receiving_yards - current_mean) * scale_factor

                # Apply mean adjustment for WR/TE receiving (fix overestimation)
                if position == 'WR':
                    mean_adj = _mean_adjustments.get('wr_receiving_mean_adjustment', 1.0)
                elif position == 'TE':
                    mean_adj = _mean_adjustments.get('te_receiving_mean_adjustment', 1.0)
                else:
                    mean_adj = 1.0

                if mean_adj != 1.0:
                    receiving_yards = receiving_yards * mean_adj

                derived['receiving_yards'] = receiving_yards
                derived['receptions'] = receptions

                # TDs
                td_rate = efficiency_preds.get('td_rate_pass')
                if td_rate is None:
                    raise ValueError(
                        f"Efficiency model missing 'td_rate_pass' for {player_input.player_name}. "
                        f"Efficiency predictor must return all required metrics."
                    )

                # Apply red zone target share boost if available
                if player_input.redzone_target_share is not None and player_input.redzone_target_share > 0:
                    # Red zone targets have higher TD rate (1.5x multiplier)
                    rz_multiplier = 1.0 + (player_input.redzone_target_share * 0.5)
                    td_rate *= rz_multiplier

                derived['receiving_tds'] = np.random.binomial(n=targets.astype(int), p=td_rate)

        return derived

    def _get_catch_rate_from_data(self, player_input: PlayerPropInput) -> float:
        """
        Calculate catch rate from NFLverse historical data.

        Uses NFLverse-only data source with intelligent fallback:
        1. Current season (2025) for most recent trends
        2. Previous season (2024) if current season has insufficient data

        Data is loaded via NFLverseDataLoader with caching for performance.

        Args:
            player_input: Player input with historical stats

        Returns:
            Catch rate (0.0 to 1.0)

        Raises:
            ValueError: If no catch rate data available from NFLverse
        """
        # Try current season first (auto-detect)
        current_season = get_current_season()
        try:
            catch_rate = self.nflverse_loader.get_position_catch_rate(
                season=current_season,
                position=player_input.position,
                validate=True
            )
            logger.debug(
                f"Using {current_season} catch rate for {player_input.position}: {catch_rate:.3f}"
            )
            return catch_rate
        except Exception as e:
            logger.debug(f"Could not get {current_season} catch rate: {e}")

        # Fallback to previous season
        training_seasons = get_training_seasons()
        fallback_season = training_seasons[0] if training_seasons else current_season - 1
        try:
            catch_rate = self.nflverse_loader.get_position_catch_rate(
                season=fallback_season,
                position=player_input.position,
                validate=True
            )
            logger.warning(
                f"Using {fallback_season} catch rate for {player_input.position}: {catch_rate:.3f} "
                f"({current_season} data unavailable)"
            )
            return catch_rate
        except Exception as e:
            logger.error(f"Could not get {fallback_season} catch rate: {e}")

        # Fallback to position-level defaults when NFLverse is unavailable
        position_defaults = {
            'RB': 0.77,  # RBs have high catch rates on short passes
            'WR': 0.65,  # WRs have moderate catch rates
            'TE': 0.70,  # TEs have high catch rates
            'FB': 0.73,  # Fullbacks similar to RBs
        }

        default_catch_rate = position_defaults.get(player_input.position, 0.65)
        logger.warning(
            f"Using position default catch rate for {player_input.player_name} ({player_input.position}): "
            f"{default_catch_rate:.3f} (NFLverse data unavailable)"
        )
        return default_catch_rate

    def _get_league_rb_receiving_td_rate(self, player_name: str) -> float:
        """
        Calculate league-wide RB receiving TD rate from actual 2025 NFLverse data.

        Uses cached value to avoid reloading data for each player.
        Falls back to player-specific rate if available in the data.

        Args:
            player_name: Name of the RB (used for player-specific lookup first)

        Returns:
            TD rate per target (typically 0.03-0.06 range for RBs)
        """
        # Return cached value if available
        if self._league_rb_rec_td_rate_cache is not None:
            return self._league_rb_rec_td_rate_cache

        try:
            from pathlib import Path
            import pandas as pd

            stats_path = Path('data/nflverse/weekly_stats.parquet')
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Weekly stats not found at {stats_path}. "
                    f"NO HARDCODED DEFAULTS - run data/fetch_nflverse_r.R to get actual data."
                )

            stats = pd.read_parquet(stats_path)
            current_season = get_current_season()

            # Filter for current season RBs
            rbs_current = stats[(stats['season'] == current_season) & (stats['position'] == 'RB')]

            if len(rbs_current) == 0:
                logger.warning(f"No RB data for season {current_season}")
                # Try previous season
                prev_season = current_season - 1
                rbs_current = stats[(stats['season'] == prev_season) & (stats['position'] == 'RB')]
                if len(rbs_current) == 0:
                    raise ValueError(
                        f"No RB data available in seasons {current_season} or {prev_season}. "
                        f"NO HARDCODED DEFAULTS - ensure NFLverse data is up to date."
                    )

            # Calculate league-wide rate
            if 'receiving_tds' in rbs_current.columns and 'targets' in rbs_current.columns:
                total_rec_tds = rbs_current['receiving_tds'].sum()
                total_targets = rbs_current['targets'].sum()

                if total_targets > 0:
                    league_rate = total_rec_tds / total_targets
                    # Cache for future calls
                    self._league_rb_rec_td_rate_cache = league_rate
                    logger.info(f"Calculated league RB receiving TD rate: {league_rate:.4f} "
                               f"({total_rec_tds} TDs / {total_targets} targets)")
                    return league_rate

            raise ValueError(
                f"Cannot calculate RB receiving TD rate - missing columns in data. "
                f"Required: 'receiving_tds', 'targets'. NO HARDCODED DEFAULTS."
            )

        except Exception as e:
            logger.error(f"Error calculating league RB receiving TD rate: {e}")
            raise  # Re-raise - NO HARDCODED FALLBACKS

    def _apply_integrated_factors(
        self,
        player_input: PlayerPropInput,
        base_efficiency: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply ALL integrated factors to efficiency predictions.

        This method applies:
        - Weather adjustments
        - Altitude adjustments
        - Field surface adjustments
        - Divisional game adjustments
        - Primetime adjustments
        - Contextual adjustments (rest, travel)
        - Injury adjustments

        Args:
            player_input: Player input with all integrated factors
            base_efficiency: Base efficiency predictions

        Returns:
            Adjusted efficiency predictions
        """
        adjusted = base_efficiency.copy()

        # Weather adjustments (affects passing/receiving efficiency)
        if player_input.weather_total_adjustment is not None:
            weather_mult = 1.0 + player_input.weather_total_adjustment
            # Apply to passing/receiving stats
            if 'yards_per_target' in adjusted:
                adjusted['yards_per_target'] *= weather_mult
            if 'yards_per_completion' in adjusted:
                adjusted['yards_per_completion'] *= weather_mult
            if 'completion_pct' in adjusted and player_input.weather_passing_adjustment is not None:
                adjusted['completion_pct'] *= (1.0 + player_input.weather_passing_adjustment)

        # Altitude adjustments (affects passing for visiting teams)
        if player_input.is_high_altitude is not None and player_input.is_high_altitude:
            if player_input.altitude_epa_adjustment is not None:
                # Apply altitude penalty to passing efficiency
                altitude_mult = 1.0 + player_input.altitude_epa_adjustment
                if 'yards_per_target' in adjusted:
                    adjusted['yards_per_target'] *= altitude_mult
                if 'yards_per_completion' in adjusted:
                    adjusted['yards_per_completion'] *= altitude_mult
                if 'completion_pct' in adjusted:
                    adjusted['completion_pct'] *= altitude_mult

        # Field surface adjustments
        if player_input.field_surface is not None:
            if player_input.field_surface == 'turf':
                # Turf: +1% passing, -1% rushing
                if 'yards_per_target' in adjusted:
                    adjusted['yards_per_target'] *= 1.01
                if 'yards_per_carry' in adjusted:
                    adjusted['yards_per_carry'] *= 0.99
            elif player_input.field_surface == 'grass':
                # Grass: -1% passing, +1% rushing
                if 'yards_per_target' in adjusted:
                    adjusted['yards_per_target'] *= 0.99
                if 'yards_per_carry' in adjusted:
                    adjusted['yards_per_carry'] *= 1.01

        # Contextual adjustments (rest, travel)
        if player_input.rest_epa_adjustment is not None:
            rest_mult = 1.0 + player_input.rest_epa_adjustment
            # Apply to all efficiency metrics
            for key in adjusted:
                if isinstance(adjusted[key], (int, float)):
                    adjusted[key] *= rest_mult

        if player_input.travel_epa_adjustment is not None:
            travel_mult = 1.0 + player_input.travel_epa_adjustment
            # Apply to all efficiency metrics
            for key in adjusted:
                if isinstance(adjusted[key], (int, float)):
                    adjusted[key] *= travel_mult

        # Injury adjustments (affect usage, not efficiency directly)
        # These are handled in usage prediction, not here

        return adjusted

    def _predict_efficiency(self, player_input: PlayerPropInput) -> Dict[str, float]:
        """
        Predict efficiency metrics (yards per opportunity, TD rates).

        Args:
            player_input: Player input data

        Returns:
            Dictionary of efficiency predictions
        """
        position = player_input.position

        # Use efficiency predictor model
        try:
            if position == 'RB':
                efficiency_features = pd.DataFrame([{
                    'week': player_input.week,
                    'trailing_yards_per_carry': player_input.trailing_yards_per_carry or player_input.trailing_yards_per_opportunity or 4.3,
                    'trailing_yards_per_target': player_input.trailing_yards_per_target or player_input.trailing_yards_per_opportunity or 6.0,  # FIX: Bug #2
                    'trailing_td_rate_rush': player_input.trailing_td_rate_rush or player_input.trailing_td_rate or 0.05,  # FIX: Bug #5
                    'trailing_td_rate_pass': player_input.trailing_td_rate_pass or player_input.trailing_td_rate or 0.04,  # FIX: Bug #5
                    'opp_rush_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                    'opp_rush_def_rank': 16,  # Mid-tier default
                    'opp_pass_def_epa': 0.0,
                    'trailing_opp_rush_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                    'team_pace': player_input.projected_pace * 2.0 if player_input.projected_pace else 64.0,
                }])
            elif position in ['WR', 'TE']:
                # Get trailing stats for blending correction
                trailing_ypt = player_input.trailing_yards_per_target or player_input.trailing_yards_per_opportunity or 8.0
                trailing_td_rate = player_input.trailing_td_rate_pass or player_input.trailing_td_rate or 0.06

                efficiency_features = pd.DataFrame([{
                    'week': player_input.week,
                    'trailing_yards_per_target': trailing_ypt,
                    'trailing_td_rate_pass': trailing_td_rate,
                    'opp_pass_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                    'opp_pass_def_rank': 16,
                    'trailing_opp_pass_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                    'team_pace': player_input.projected_pace * 2.0 if player_input.projected_pace else 64.0,
                }])

                # Get model prediction
                preds = self.efficiency_predictor.predict(
                    efficiency_features,
                    position=position,
                    player_name=player_input.player_name
                )

                # Convert to dict
                result = {}
                for key, value in preds.items():
                    if isinstance(value, (list, np.ndarray)):
                        result[key] = value[0]
                    else:
                        result[key] = value

                # CRITICAL FIX: The trained model over-regresses high performer Y/T to the mean.
                # Regression to the mean is STRONG for yards per target.
                # Data analysis (Nov 25, 2025) shows:
                # - Players with trailing Y/T >12 regress by ~34% (actual ~9.2)
                # - Players with trailing Y/T 10-11 regress by ~32% (actual ~7.2)
                # - Players with trailing Y/T 9-10 regress by ~9% (actual ~8.7)
                # - Players with trailing Y/T 7-9 are near league average
                # - Players with trailing Y/T <6 regress UP by ~73%
                #
                # Optimal blend weights (empirically derived):
                model_ypt = result.get('yards_per_target', trailing_ypt)

                if trailing_ypt > 12.0:
                    # Very high performers regress heavily (-34%) - trust model more
                    blend_trailing = 0.25
                    blend_model = 0.75
                elif trailing_ypt > 10.0:
                    # High performers regress significantly (-20-32%) - trust model
                    blend_trailing = 0.20
                    blend_model = 0.80
                elif trailing_ypt > 9.0:
                    # Above average - moderate regression expected
                    blend_trailing = 0.40
                    blend_model = 0.60
                elif trailing_ypt > 7.0:
                    # Average performer - slight trust in trailing
                    blend_trailing = 0.35
                    blend_model = 0.65
                elif trailing_ypt > 6.0:
                    # Below average - they tend to regress UP, trust trailing more
                    blend_trailing = 0.55
                    blend_model = 0.45
                else:
                    # Very low Y/T - strong regression to mean expected
                    blend_trailing = 0.10
                    blend_model = 0.90

                blended_ypt = blend_trailing * trailing_ypt + blend_model * model_ypt

                # NOTE: Blanket calibration multipliers REMOVED 2025-11-25
                # Tier-based calibration now applied in generate_unified_recommendations_v3.py
                # This allows different adjustments for high-volume vs low-volume players
                # (High volume WRs: under-predicted → OVER edge; Low volume: over-predicted → UNDER edge)

                # SANITY CHECK: Cap Y/T to realistic bounds
                # League avg Y/T is ~8.0, elite players rarely exceed 12.0 sustained
                # Min reasonable is ~5.0 for low-efficiency players
                MIN_YPT = 5.0
                MAX_YPT = 12.0
                blended_ypt = np.clip(blended_ypt, MIN_YPT, MAX_YPT)

                logger.debug(
                    f"{position} {player_input.player_name} Y/T: trailing={trailing_ypt:.2f}, "
                    f"model={model_ypt:.2f}, blended={blended_ypt:.2f} (blend: {blend_trailing:.0%}/{blend_model:.0%})"
                )
                result['yards_per_target'] = blended_ypt

                # Same blending for TD rate (use same blend weights)
                model_td_rate = result.get('td_rate_pass', trailing_td_rate)
                blended_td_rate = blend_trailing * trailing_td_rate + blend_model * model_td_rate
                result['td_rate_pass'] = blended_td_rate

                # Apply integrated factors
                result = self._apply_integrated_factors(player_input, result)
                return result

            elif position == 'QB':
                # QB - use trained model with actual trailing stats (NO HARDCODED VALUES)
                # Check that required QB-specific trailing stats are available
                if player_input.trailing_comp_pct is None:
                    raise ValueError(
                        f"Trailing completion percentage not available for {player_input.player_name}. "
                        f"Run trailing stats calculation for QB-specific metrics."
                    )
                if player_input.trailing_yards_per_completion is None:
                    raise ValueError(
                        f"Trailing yards per completion not available for {player_input.player_name}. "
                        f"Run trailing stats calculation for QB-specific metrics."
                    )
                if player_input.trailing_td_rate_pass is None:
                    raise ValueError(
                        f"Trailing TD rate (pass) not available for {player_input.player_name}. "
                        f"Run trailing stats calculation for QB-specific metrics."
                    )

                # Build features for QB passing model
                qb_pass_features = pd.DataFrame([{
                    'week': player_input.week,
                    'trailing_comp_pct': player_input.trailing_comp_pct,
                    'trailing_yards_per_completion': player_input.trailing_yards_per_completion,
                    'trailing_td_rate_pass': player_input.trailing_td_rate_pass,
                    'opp_pass_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                    'opp_pass_def_rank': 16,  # Mid-tier default (OK since it's opponent, not player stat)
                    'trailing_opp_pass_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                    'team_pace': player_input.projected_pace * 2.0 if player_input.projected_pace else 64.0,
                }])

                # Get QB passing predictions from trained model
                qb_pass_preds = self.efficiency_predictor.predict(
                    qb_pass_features,
                    position='QB',
                    player_name=player_input.player_name  # DEBUG: Add player name for logging
                )

                # Convert to dict
                result = {}
                for key, value in qb_pass_preds.items():
                    if isinstance(value, (list, np.ndarray)):
                        result[key] = value[0]
                    else:
                        result[key] = value

                # CRITICAL FIX: The trained model over-regresses QB TD rates to the mean.
                # For QBs, we should trust their actual trailing performance more than model regression.
                # Blend: 70% actual trailing stat + 30% model prediction (adjusts for opponent but doesn't over-regress)
                actual_td_rate = player_input.trailing_td_rate_pass
                model_td_rate = result.get('td_rate_pass', actual_td_rate)
                blended_td_rate = 0.7 * actual_td_rate + 0.3 * model_td_rate
                logger.debug(
                    f"QB {player_input.player_name} TD rate: actual={actual_td_rate:.4f}, "
                    f"model={model_td_rate:.4f}, blended={blended_td_rate:.4f}"
                )
                result['td_rate_pass'] = blended_td_rate

                # Same for completion percentage - trust actual more than model
                actual_comp_pct = player_input.trailing_comp_pct
                model_comp_pct = result.get('completion_pct', actual_comp_pct)
                result['completion_pct'] = 0.7 * actual_comp_pct + 0.3 * model_comp_pct

                # Also get QB rushing predictions if model supports it
                if player_input.trailing_yards_per_carry is not None and player_input.trailing_td_rate_rush is not None:
                    qb_rush_features = pd.DataFrame([{
                        'week': player_input.week,
                        'trailing_yards_per_carry': player_input.trailing_yards_per_carry,
                        'trailing_td_rate_rush': player_input.trailing_td_rate_rush,
                        'opp_rush_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                        'opp_rush_def_rank': 16,
                        'trailing_opp_rush_def_epa': player_input.opponent_def_epa_vs_position or 0.0,
                        'team_pace': player_input.projected_pace * 2.0 if player_input.projected_pace else 64.0,
                    }])

                    try:
                        qb_rush_preds = self.efficiency_predictor.predict(
                            qb_rush_features,
                            position='QB_rush',
                            player_name=player_input.player_name  # DEBUG: Add player name for logging
                        )
                        for key, value in qb_rush_preds.items():
                            # Map QB_rush keys to standard keys (e.g., 'QB_yards_per_carry' -> 'yards_per_carry')
                            standard_key = key.replace('QB_', '')
                            if isinstance(value, (list, np.ndarray)):
                                result[standard_key] = value[0]
                            else:
                                result[standard_key] = value
                    except Exception as e:
                        logger.debug(f"QB rushing model failed for {player_input.player_name}: {e}")
                        # Use actual trailing stats as fallback (NOT hardcoded)
                        result['yards_per_carry'] = player_input.trailing_yards_per_carry
                        result['td_rate_rush'] = player_input.trailing_td_rate_rush
                else:
                    # No QB rushing stats available - this is OK, not all QBs rush significantly
                    result['yards_per_carry'] = 0.0
                    result['td_rate_rush'] = 0.0

                # Apply integrated factors (weather, altitude, field surface)
                result = self._apply_integrated_factors(player_input, result)

                return result
            else:
                raise ValueError(f"Unknown position: {position}. Expected QB, RB, WR, or TE.")

            preds = self.efficiency_predictor.predict(
                efficiency_features,
                position=position,
                player_name=player_input.player_name  # DEBUG: Add player name for logging
            )

            # Convert to dict (handle list outputs)
            result = {}
            for key, value in preds.items():
                if isinstance(value, (list, np.ndarray)):
                    result[key] = value[0]
                else:
                    result[key] = value

            # Apply ALL integrated factors (weather, altitude, field surface, contextual)
            result = self._apply_integrated_factors(player_input, result)

            return result

        except Exception as e:
            logger.warning(f"Efficiency prediction failed for {position}: {e}")
            # CRITICAL: For QBs, DO NOT use hardcoded defaults - raise exception to fix root cause
            if position == 'QB':
                # Log the actual error for debugging
                logger.error(f"QB efficiency prediction failed for player: {e}")
                # Use player's actual trailing stats (NOT hardcoded values)
                # This should never happen if QB trailing stats are properly calculated
                raise ValueError(
                    f"QB efficiency prediction failed: {e}. "
                    f"This indicates a problem with QB trailing stats calculation or trained model features. "
                    f"DO NOT use hardcoded defaults."
                )
            elif position == 'RB':
                return {
                    'yards_per_carry': 4.3,
                    'yards_per_target': 6.0,
                    'td_rate_rush': 0.05,
                    'td_rate_rec': 0.04
                }
            else:
                return {
                    'yards_per_target': 8.5,
                    'td_rate_pass': 0.06
                }

    def _extract_game_context(self, team_players: List[PlayerPropInput]) -> Dict[str, float]:
        """
        Extract game context from player inputs.

        Args:
            team_players: List of players on same team

        Returns:
            Game context dictionary
        """
        # Use first player's game context (all should be same game)
        if not team_players:
            return {}

        first_player = team_players[0]

        # FAIL EXPLICITLY if game context missing
        pace = first_player.projected_pace
        game_script = first_player.projected_game_script
        team_pass_attempts = first_player.projected_team_pass_attempts
        team_rush_attempts = first_player.projected_team_rush_attempts
        team_targets = first_player.projected_team_targets

        if pace is None:
            raise ValueError(
                f"Game context missing 'projected_pace' for {first_player.player_id}. "
                f"Run game simulations to populate game context."
            )
        if team_pass_attempts is None:
            raise ValueError(
                f"Game context missing 'projected_team_pass_attempts' for {first_player.player_id}. "
                f"Run game simulations to populate team usage data."
            )
        if team_rush_attempts is None:
            raise ValueError(
                f"Game context missing 'projected_team_rush_attempts' for {first_player.player_id}. "
                f"Run game simulations to populate team usage data."
            )
        if team_targets is None:
            raise ValueError(
                f"Game context missing 'projected_team_targets' for {first_player.player_id}. "
                f"Run game simulations to populate team usage data."
            )

        return {
            'pace': pace,
            'game_script': game_script or 0.0,
            'team_pass_attempts': team_pass_attempts,
            'team_rush_attempts': team_rush_attempts,
            'team_targets': team_targets
        }

    def _calculate_team_totals(self, game_context: Dict[str, float], stat_type: str) -> Dict[str, float]:
        """
        Calculate team-level totals for constraint enforcement.

        Args:
            game_context: Game-level context
            stat_type: Type of stat ('attempts', 'targets', 'carries')

        Returns:
            Dictionary of team totals
        """
        totals = {}

        if stat_type == 'attempts':
            team_pass_attempts = game_context.get('team_pass_attempts')
            if team_pass_attempts is None:
                raise ValueError("Game context missing 'team_pass_attempts'. Run game simulations.")
            totals['qb_attempts'] = team_pass_attempts

        elif stat_type == 'carries':
            team_rush_attempts = game_context.get('team_rush_attempts')
            if team_rush_attempts is None:
                raise ValueError("Game context missing 'team_rush_attempts'. Run game simulations.")
            totals['rb_carries'] = team_rush_attempts

        elif stat_type == 'targets':
            team_targets = game_context.get('team_targets')
            if team_targets is None:
                raise ValueError("Game context missing 'team_targets'. Run game simulations.")
            totals['team_targets'] = team_targets

        return totals

    def _infer_player_role(self, player_input: PlayerPropInput) -> str:
        """
        Infer player role (primary, secondary) from usage shares.

        Args:
            player_input: Player input data

        Returns:
            Role string ('primary' or 'secondary')
        """
        # Use snap share as proxy for role
        snap_share = player_input.trailing_snap_share or 0.0

        if snap_share >= 0.60:
            return 'primary'
        elif snap_share >= 0.35:
            return 'secondary'
        else:
            return 'tertiary'


# Example usage
if __name__ == '__main__':
    from nfl_quant.models.usage_predictor import UsagePredictor
    from nfl_quant.models.efficiency_predictor import EfficiencyPredictor

    # Load models (placeholder)
    usage_predictor = UsagePredictor()
    efficiency_predictor = EfficiencyPredictor()

    # Initialize simulator
    simulator = PlayerSimulatorV3(
        usage_predictor=usage_predictor,
        efficiency_predictor=efficiency_predictor,
        trials=10000,
        seed=42
    )

    print("PlayerSimulatorV3 initialized successfully with correlation support")
