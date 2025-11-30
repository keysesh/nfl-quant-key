"""Monte Carlo simulation engine for NFL games - FIXED VERSION with centralized config."""

import json
import logging
from typing import Optional

import numpy as np

from nfl_quant.schemas import SimulationInput, SimulationOutput
from nfl_quant.config_enhanced import config

logger = logging.getLogger(__name__)

# Load configuration values from centralized config
_sim_config = config.simulation.game_simulation
BASE_POSSESSIONS_PER_GAME = _sim_config['base_possessions_per_game']
BASE_TD_RATE = _sim_config['base_td_rate']
BASE_FG_RATE = _sim_config['base_fg_rate']
BASE_SAFETY_RATE = _sim_config['base_safety_rate']

# Load adjustment multipliers
_game_adj = config.simulation.game_adjustments
_weather_adj = config.simulation.weather_adjustments

logger.info(f"✅ Loaded simulation config: BASE_TD_RATE={BASE_TD_RATE}, trials={config.simulation.trials['default_trials']}")


class MonteCarloSimulator:
    """Monte Carlo simulator for NFL game outcomes."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize simulator.

        Args:
            seed: Random seed for reproducibility (default from config)
        """
        self.seed = seed if seed is not None else config.simulation.trials['default_seed']
        self._game_counter = 0

    def _apply_adjustments(
        self,
        base_epa: float,
        injury_adj: float,
        sim_input: SimulationInput
    ) -> float:
        """
        Apply all contextual adjustments to EPA

        Args:
            base_epa: Base EPA value
            injury_adj: Injury adjustment
            sim_input: Full simulation input with context

        Returns:
            Adjusted EPA
        """
        adjusted_epa = base_epa + injury_adj

        # Divisional game adjustment (from config)
        if sim_input.is_divisional:
            adjusted_epa *= (1.0 + _game_adj['divisional_game'])

        # Primetime game adjustment (varies by type, from config)
        if sim_input.game_type in ['SNF', 'MNF']:
            adjusted_epa *= (1.0 + _game_adj['snf_mnf_primetime'])
        elif sim_input.game_type == 'TNF':
            adjusted_epa *= (1.0 + _game_adj['thursday_night'])

        # Weather adjustments (only if not dome, from config)
        if not sim_input.is_dome and sim_input.temperature is not None:
            temp_config = _weather_adj['temperature']
            # Cold weather
            if sim_input.temperature < temp_config['extreme_cold_threshold_f']:
                adjusted_epa *= temp_config['extreme_cold_multiplier']
            elif sim_input.temperature < temp_config['freezing_threshold_f']:
                adjusted_epa *= temp_config['freezing_multiplier']

            # Wind impact
            if sim_input.wind_speed is not None:
                wind_config = _weather_adj['wind']
                if sim_input.wind_speed > wind_config['high_wind_threshold_mph']:
                    adjusted_epa *= wind_config['high_wind_multiplier']
                elif sim_input.wind_speed > wind_config['moderate_wind_threshold_mph']:
                    adjusted_epa *= wind_config['moderate_wind_multiplier']

            # Precipitation
            precip_config = _weather_adj['precipitation']
            if sim_input.precipitation is not None and sim_input.precipitation > precip_config['threshold_inches']:
                adjusted_epa *= precip_config['rain_snow_multiplier']

        return adjusted_epa

    def simulate_game(
        self, sim_input: SimulationInput, trials: Optional[int] = None
    ) -> SimulationOutput:
        """Run Monte Carlo simulation for a single game.

        Args:
            sim_input: SimulationInput with EPAs and context
            trials: Number of Monte Carlo trials (default from config)

        Returns:
            SimulationOutput with distributions
        """
        # Use config default if trials not specified
        if trials is None:
            trials = config.simulation.trials['default_trials']

        # Create game-specific RNG using seed + game counter
        # This ensures each game gets different random numbers while being reproducible
        game_seed = self.seed + self._game_counter
        self._game_counter += 1
        rng = np.random.default_rng(game_seed)

        home_scores = []
        away_scores = []

        # Apply context and injury adjustments to EPA
        home_off_epa_adj = self._apply_adjustments(
            sim_input.home_offensive_epa,
            sim_input.home_injury_offensive_adjustment,
            sim_input
        )
        away_off_epa_adj = self._apply_adjustments(
            sim_input.away_offensive_epa,
            sim_input.away_injury_offensive_adjustment,
            sim_input
        )
        home_def_epa_adj = sim_input.home_defensive_epa + sim_input.home_injury_defensive_adjustment
        away_def_epa_adj = sim_input.away_defensive_epa + sim_input.away_injury_defensive_adjustment

        for _ in range(trials):
            home_score = self._simulate_team_score(
                rng,
                home_off_epa_adj,
                away_def_epa_adj,
                sim_input.home_pace,
                sim_input,
                is_home=True
            )
            away_score = self._simulate_team_score(
                rng,
                away_off_epa_adj,
                home_def_epa_adj,
                sim_input.away_pace,
                sim_input,
                is_home=False
            )
            home_scores.append(home_score)
            away_scores.append(away_score)

        home_scores = np.array(home_scores)
        away_scores = np.array(away_scores)
        totals = home_scores + away_scores
        margins = home_scores - away_scores

        # Calculate probabilities
        home_win_prob = float((home_scores > away_scores).sum() / trials)
        away_win_prob = float((away_scores > home_scores).sum() / trials)
        tie_prob = float((home_scores == away_scores).sum() / trials)

        # Fair prices (implied odds)
        fair_spread = float(np.median(margins))
        fair_total = float(np.median(totals))

        # Distributions
        home_score_median = float(np.median(home_scores))
        away_score_median = float(np.median(away_scores))
        home_score_std = float(np.std(home_scores))
        away_score_std = float(np.std(away_scores))
        total_median = float(np.median(totals))
        total_std = float(np.std(totals))

        # Calculate percentiles for total distribution
        total_p5 = float(np.percentile(totals, 5))
        total_p25 = float(np.percentile(totals, 25))
        total_p50 = float(np.percentile(totals, 50))  # Same as median
        total_p75 = float(np.percentile(totals, 75))
        total_p95 = float(np.percentile(totals, 95))

        return SimulationOutput(
            game_id=sim_input.game_id,
            trial_count=trials,
            seed=game_seed,  # Store the actual seed used for this game
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            tie_prob=tie_prob,
            fair_spread=fair_spread,
            fair_total=fair_total,
            home_score_median=home_score_median,
            away_score_median=away_score_median,
            home_score_std=home_score_std,
            away_score_std=away_score_std,
            total_median=total_median,
            total_std=total_std,
            total_p5=total_p5,
            total_p25=total_p25,
            total_p50=total_p50,
            total_p75=total_p75,
            total_p95=total_p95,
            pace=(sim_input.home_pace + sim_input.away_pace) / 2.0 if sim_input.home_pace and sim_input.away_pace else None,
            home_pace=sim_input.home_pace,
            away_pace=sim_input.away_pace,
        )

    def _simulate_team_score(
        self,
        rng: np.random.Generator,
        offensive_epa: float,
        defensive_epa: float,
        pace: float,
        sim_input: SimulationInput = None,
        is_home: bool = False
    ) -> int:
        """Simulate final score for one team in one trial.

        Args:
            rng: NumPy random generator for this specific game
            offensive_epa: Team's offensive EPA per play (already adjusted)
            defensive_epa: Opponent's defensive EPA per play (already adjusted)
            pace: Adjusted pace (seconds per play)
            sim_input: Full simulation input (for additional context if needed)
            is_home: Whether this is the home team

        Returns:
            Simulated score
        """
        # Possessions based on ACTUAL NFL data from config (derived from 2025 NFLverse data)
        # Config value is calculated from actual game data, not hardcoded
        BASE_POSSESSIONS = BASE_POSSESSIONS_PER_GAME  # From centralized config

        # Adjust possessions based on pace
        # Pace can be in two formats:
        # 1. Seconds per play (typically 25-35, like 64.6)
        # 2. Plays per game (typically 60-70, like 65.0)
        # We need to detect which format and normalize

        LEAGUE_AVG_PACE_PLAYS = _sim_config.get('league_avg_pace_plays', 65.0)  # From config
        LEAGUE_AVG_SECONDS_PER_PLAY = _sim_config.get('league_avg_pace_seconds', 30.0)  # From config

        # Determine if pace is seconds/play or plays/game
        if pace > 50:  # This is seconds per play (e.g., 64.6)
            # Convert to plays per game: 3600 seconds / seconds_per_play = plays_per_game
            # But this doesn't quite work... 3600/64.6 = 55.7 plays per team
            # Actually, pace=64.6 means each team averages 64.6 plays per game
            # If pace > 50, it's likely plays per team per game
            plays_per_game = pace
            pace_factor = plays_per_game / LEAGUE_AVG_PACE_PLAYS
        elif pace > 10:  # This is plays per game (e.g., 65.0)
            plays_per_game = pace
            pace_factor = plays_per_game / LEAGUE_AVG_PACE_PLAYS
        else:  # This is seconds per play (e.g., 30.0)
            plays_per_game = 3600 / pace  # Total plays per game
            plays_per_team = plays_per_game / 2.0  # Each team gets half
            pace_factor = plays_per_team / LEAGUE_AVG_PACE_PLAYS

        avg_possessions = BASE_POSSESSIONS * pace_factor

        # Add stochasticity to possessions (from config)
        possessions = rng.poisson(avg_possessions)
        possessions = max(_sim_config['possessions_min'], min(possessions, _sim_config['possessions_max']))

        logger.debug(f"Pace={pace:.1f} → pace_factor={pace_factor:.2f} → {avg_possessions:.1f} avg possessions → {possessions} actual")

        # Net EPA per play (offense vs defense)
        # CRITICAL: defensive_epa represents what OPPONENTS score
        # - Positive defensive_epa = bad defense (opponents score well)
        # - Negative defensive_epa = good defense (opponents score poorly)
        # Therefore: net_epa = offensive_epa - defensive_epa
        # Example: offense +0.10, opponent defense +0.05 (bad) → net = +0.10 - 0.05 = +0.05
        net_epa = offensive_epa - defensive_epa

        # Apply HOME FIELD ADVANTAGE
        # Research: NFL home teams win ~57% and score ~1.5-2.5 more points on average
        # Since 2020: ~54-57% home win rate with ~1.5 points average advantage
        # 1.5 points / ~12 possessions = ~0.125 points/possession
        # In EPA terms: ~0.015-0.020 EPA/play boost (empirically validated)
        if is_home:
            home_epa_boost = 0.018  # Stronger HFA to match actual 54% home win rate
            net_epa += home_epa_boost
            logger.debug(f"  Home field advantage applied: +{home_epa_boost:.3f} EPA")

        # Points per possession with EPA adjustment
        points = 0
        for _ in range(int(possessions)):
            # EPA directly converts to expected points per play
            # Key insight: EPA IS the expected points added per play
            # So if net_epa = 0.10, team adds 0.10 expected points per PLAY
            # Over ~6 plays per possession = 0.60 extra expected points per possession
            #
            # Calculate team-specific scoring rate from their EPA
            # EPA of 0.0 = league average (23.3 pts / 10.7 poss = 2.18 pts/poss)
            # EPA of +0.10 per play = +0.6 pts/poss = 2.78 pts/poss
            # EPA of -0.10 per play = -0.6 pts/poss = 1.58 pts/poss

            plays_per_possession = _sim_config.get('plays_per_possession', 6.5)
            expected_pts_per_poss = _sim_config['base_possessions_per_game'] * (BASE_TD_RATE * 7 + BASE_FG_RATE * 3) / _sim_config['base_possessions_per_game']

            # Adjust expected points based on EPA
            epa_pts_adjustment = net_epa * plays_per_possession
            adjusted_pts_per_poss = expected_pts_per_poss + epa_pts_adjustment

            # Convert back to TD and FG rates
            # Assume FG rate stays proportional to TD rate
            base_pts_from_tds = BASE_TD_RATE * 7
            base_pts_from_fgs = BASE_FG_RATE * 3
            total_base_pts = base_pts_from_tds + base_pts_from_fgs

            td_share_of_pts = base_pts_from_tds / total_base_pts
            fg_share_of_pts = base_pts_from_fgs / total_base_pts

            # Scale rates to match adjusted points
            scale_factor = adjusted_pts_per_poss / total_base_pts
            td_prob = BASE_TD_RATE * scale_factor
            fg_prob = BASE_FG_RATE * scale_factor

            # Clamp probabilities to reasonable ranges
            td_prob = max(_sim_config['td_prob_min'], min(_sim_config['td_prob_max'], td_prob))
            fg_prob = max(0.05, min(0.25, fg_prob))  # FG rate reasonable bounds

            # Stochastic scoring outcomes
            rand = rng.random()

            if rand < td_prob:
                points += 7

            # Field Goal (3 pts) - scaled with TD rate
            elif rand < td_prob + fg_prob:
                points += 3

            # Safety (2 pts) - rare
            elif rand < td_prob + fg_prob + BASE_SAFETY_RATE:
                points += 2

            # Otherwise: punt, turnover (0 pts)

        return int(points)

    def export_results(
        self, output: SimulationOutput, file_path: str
    ) -> None:
        """Export simulation results to JSON.

        Args:
            output: SimulationOutput
            file_path: Path to save JSON
        """
        data = output.model_dump()
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved simulation results to {file_path}")
