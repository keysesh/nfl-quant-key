"""
Touchdown Prediction Model
==========================

Statistically robust TD probability prediction using:
- Historical TD rates by position
- Usage patterns (carries, targets, snap share)
- Red zone opportunity metrics
- Team scoring environment
- Poisson regression for TD counts

Based on 2024 NFL research:
- 85.7% of rushing TDs from red zone
- 69.9% of receiving TDs from red zone
- Base receiving TD rate: 0.347 per game (20+ routes)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from nfl_quant.features.defensive_metrics import DefensiveMetricsExtractor
from nfl_quant.config_enhanced import config

logger = logging.getLogger(__name__)

# Load TD prediction configuration
_td_config = config.td_prediction


class TouchdownPredictor:
    """
    Predicts touchdown probabilities using position-specific models
    based on historical rates, usage, and game context.
    """

    # Load baseline TD rates from config
    # Calibrated based on backtest showing 8-10% underestimation
    # Applied 1.15x multiplier to improve calibration
    BASELINE_TD_RATES = _td_config.baseline_td_rates

    # Red zone multipliers from config (based on 2024 research)
    RED_ZONE_MULTIPLIER_RUSH = _td_config.red_zone_multipliers['rushing']  # 85.7% of rush TDs from RZ
    RED_ZONE_MULTIPLIER_PASS = _td_config.red_zone_multipliers['passing']  # 69.9% of pass TDs from RZ

    def __init__(
        self,
        historical_stats_path: Optional[Path] = None,
        pbp_path: Optional[Path] = None,
        season: Optional[int] = None
    ):
        """
        Initialize TD predictor.

        Args:
            historical_stats_path: Path to nflverse stats file
            pbp_path: Path to play-by-play data for defensive metrics
            season: Season to use (defaults to current season)
        """
        self.historical_stats = None
        self.defensive_metrics = None

        if historical_stats_path and historical_stats_path.exists():
            try:
                self.historical_stats = pd.read_csv(historical_stats_path)
                logger.info(
                    f"✅ Loaded historical stats: "
                    f"{len(self.historical_stats)} rows"
                )
            except Exception as e:
                logger.warning(f"⚠️  Could not load historical stats: {e}")

        # Initialize defensive metrics extractor
        # Use nflverse data as single source of truth (includes most recent week)
        if pbp_path is None:
            if season is None:
                from nfl_quant.utils.season_utils import get_current_season
                season = get_current_season()
            pbp_path = Path(f'data/nflverse/pbp_{season}.parquet')

        if pbp_path.exists():
            try:
                self.defensive_metrics = DefensiveMetricsExtractor(pbp_path, season=season)
                logger.info("✅ Initialized defensive metrics extractor")
            except Exception as e:
                logger.warning(
                    f"⚠️  Could not initialize defensive metrics: {e}"
                )

    def calculate_player_td_rates(
        self,
        player_name: str,
        position: str,
        weeks_back: int = 4
    ) -> Dict[str, float]:
        """
        Calculate player's recent TD rates from historical data.

        Args:
            player_name: Player name
            position: Position (QB, RB, WR, TE)
            weeks_back: Number of recent weeks to average

        Returns:
            Dict with TD rates per touch type
        """
        rates = {}

        if self.historical_stats is None:
            # No historical data - use position baselines
            return self._get_baseline_rates(position)

        # Get player's recent games
        player_stats = self.historical_stats[
            self.historical_stats['player_name'].str.contains(player_name, case=False, na=False)
        ].copy()

        if player_stats.empty:
            return self._get_baseline_rates(position)

        # Sort by week and get recent games
        player_stats = player_stats.sort_values('week', ascending=False).head(weeks_back)

        # Calculate rates based on position
        if position == 'QB':
            attempts = player_stats['attempts'].sum()
            carries = player_stats['carries'].sum()
            pass_tds = player_stats['passing_tds'].sum()
            rush_tds = player_stats['rushing_tds'].sum()

            rates['passing_td_per_attempt'] = pass_tds / attempts if attempts > 0 else self.BASELINE_TD_RATES['QB']['passing_td_per_attempt']
            rates['rushing_td_per_carry'] = rush_tds / carries if carries > 0 else self.BASELINE_TD_RATES['QB']['rushing_td_per_carry']

        elif position == 'RB':
            carries = player_stats['carries'].sum()
            targets = player_stats['targets'].sum()
            rush_tds = player_stats['rushing_tds'].sum()
            rec_tds = player_stats['receiving_tds'].sum()

            rates['rushing_td_per_carry'] = rush_tds / carries if carries > 0 else self.BASELINE_TD_RATES['RB']['rushing_td_per_carry']
            rates['receiving_td_per_target'] = rec_tds / targets if targets > 0 else self.BASELINE_TD_RATES['RB']['receiving_td_per_target']

        elif position in ['WR', 'TE']:
            targets = player_stats['targets'].sum()
            rec_tds = player_stats['receiving_tds'].sum()

            baseline = self.BASELINE_TD_RATES.get(position, self.BASELINE_TD_RATES['WR'])
            rates['receiving_td_per_target'] = rec_tds / targets if targets > 0 else baseline['receiving_td_per_target']

        # Apply Bayesian shrinkage toward baseline (handle small samples)
        rates = self._apply_shrinkage(rates, position, player_stats)

        return rates

    def _get_baseline_rates(self, position: str) -> Dict[str, float]:
        """Get baseline TD rates for position."""
        return self.BASELINE_TD_RATES.get(position, self.BASELINE_TD_RATES['RB']).copy()

    def _apply_shrinkage(
        self,
        rates: Dict[str, float],
        position: str,
        player_stats: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Apply Bayesian shrinkage to handle small sample sizes.

        More games = more weight to observed rate
        Fewer games = more weight to baseline
        """
        n_games = len(player_stats)
        baseline = self._get_baseline_rates(position)

        # Shrinkage factor: more games = less shrinkage
        # Get shrinkage parameters from config
        min_weight = _td_config.shrinkage['min_weight']
        per_game_decay = _td_config.shrinkage['per_game_decay']
        shrinkage_weight = max(min_weight, 1.0 - (n_games * per_game_decay))

        shrunken_rates = {}
        for key, observed_rate in rates.items():
            baseline_rate = baseline.get(key, observed_rate)
            shrunken_rates[key] = (
                shrinkage_weight * baseline_rate +
                (1 - shrinkage_weight) * observed_rate
            )

        return shrunken_rates

    def calculate_game_script_factor(
        self,
        position: str,
        projected_point_differential: float
    ) -> Dict[str, float]:
        """
        Calculate game script adjustment factors for TD probabilities.

        Teams that are winning typically run more (boost RB rush TDs)
        Teams that are losing typically pass more (boost pass TDs)

        Args:
            position: Player position
            projected_point_differential: Team points - Opponent points
                                        (positive = winning, negative = losing)

        Returns:
            Dict with multipliers for different TD types
        """
        # Neutral game script = [-3, +3] points (no adjustment)
        # Strong negative script = -7+ points (trailing badly)
        # Strong positive script = +7+ points (winning big)

        factors = {
            'rushing_multiplier': 1.0,
            'receiving_multiplier': 1.0,
            'passing_multiplier': 1.0,
        }

        # Get game script sensitivity and bounds from config
        sensitivities = _td_config.game_script_sensitivity
        bounds = _td_config.game_script_bounds

        if position == 'QB':
            # QB passing increases when trailing
            # Formula from config: 1.0 at neutral, sensitivity controls adjustment rate
            qb_pass_sens = sensitivities['qb_passing']
            factors['passing_multiplier'] = 1.0 - (projected_point_differential * qb_pass_sens)
            # Cap at bounds from config
            factors['passing_multiplier'] = max(bounds['qb_passing'][0], min(bounds['qb_passing'][1], factors['passing_multiplier']))

            # QB rushing slightly affected (designed runs in RZ)
            qb_rush_sens = sensitivities['qb_rushing']
            factors['rushing_multiplier'] = 1.0 + (projected_point_differential * qb_rush_sens)
            factors['rushing_multiplier'] = max(bounds['qb_rushing'][0], min(bounds['qb_rushing'][1], factors['rushing_multiplier']))

        elif position == 'RB':
            # RB rushing increases when leading (clock management)
            rb_rush_sens = sensitivities['rb_rushing']
            factors['rushing_multiplier'] = 1.0 + (projected_point_differential * rb_rush_sens)
            factors['rushing_multiplier'] = max(bounds['rb_rushing'][0], min(bounds['rb_rushing'][1], factors['rushing_multiplier']))

            # RB receiving increases when trailing (checkdowns)
            rb_rec_sens = sensitivities['rb_receiving']
            factors['receiving_multiplier'] = 1.0 - (projected_point_differential * rb_rec_sens)
            factors['receiving_multiplier'] = max(bounds['rb_receiving'][0], min(bounds['rb_receiving'][1], factors['receiving_multiplier']))

        elif position in ['WR', 'TE']:
            # WR/TE receiving increases when trailing
            wr_te_sens = sensitivities['wr_te_receiving']
            factors['receiving_multiplier'] = 1.0 - (projected_point_differential * wr_te_sens)
            factors['receiving_multiplier'] = max(bounds['wr_te_receiving'][0], min(bounds['wr_te_receiving'][1], factors['receiving_multiplier']))

        return factors

    def get_opponent_td_multiplier(
        self,
        opponent_team: str,
        position: str,
        current_week: int
    ) -> float:
        """
        Calculate opponent TD rate multiplier based on defensive strength.

        Args:
            opponent_team: Opponent team abbreviation (e.g., 'KC')
            position: Player position ('QB', 'RB', 'WR', 'TE')
            current_week: Current week number

        Returns:
            Multiplier for TD rate (1.0 = average, >1.0 = easier, <1.0 = harder)
        """
        if self.defensive_metrics is None:
            logger.debug("No defensive metrics available, using neutral")
            return 1.0

        try:
            # Get defensive stats vs position
            def_stats = self.defensive_metrics.get_defense_vs_position(
                defense_team=opponent_team,
                position=position,
                current_week=current_week,
                trailing_weeks=4
            )

            # Extract TD rate allowed (league average from config)
            league_avg = _td_config.league_averages['td_rate_for_defensive_rating']
            td_rate_allowed = def_stats.get('td_rate_allowed', league_avg)

            # Convert to multiplier (league average = 1.0)
            multiplier = td_rate_allowed / league_avg

            # Cap multiplier at bounds from config
            min_mult = _td_config.league_averages['defensive_multiplier_min']
            max_mult = _td_config.league_averages['defensive_multiplier_max']
            multiplier = max(min_mult, min(max_mult, multiplier))

            logger.debug(
                f"{opponent_team} vs {position}: "
                f"TD rate {td_rate_allowed:.3f} -> {multiplier:.2f}x"
            )

            return multiplier

        except Exception as e:
            logger.warning(
                f"Error calculating opponent TD multiplier for "
                f"{opponent_team} vs {position}: {e}"
            )
            return 1.0

    def predict_touchdown_probability(
        self,
        player_name: str,
        position: str,
        projected_carries: float = 0.0,
        projected_targets: float = 0.0,
        projected_pass_attempts: float = 0.0,
        red_zone_share: float = 0.5,
        goal_line_role: float = 0.5,
        team_projected_total: float = 24.0,
        opponent_td_rate_allowed: float = 1.0,
        opponent_team: Optional[str] = None,
        current_week: Optional[int] = None,
        projected_point_differential: Optional[float] = None,
        projected_snap_share: float = 1.0,
    ) -> Dict[str, float]:
        """
        Predict TD probabilities for a player using Poisson model.

        Args:
            player_name: Player name
            position: Position (QB, RB, WR, TE)
            projected_carries: Expected rushing attempts
            projected_targets: Expected receiving targets
            projected_pass_attempts: Expected pass attempts (QB only)
            red_zone_share: % of team's RZ opportunities (0.0-1.0)
            goal_line_role: Goal line usage (0.0=none, 1.0=primary)
            team_projected_total: Team's projected point total
            opponent_td_rate_allowed: Multiplier for opponent (deprecated)
            opponent_team: Opponent team abbreviation (e.g., 'KC')
            current_week: Current week number
            projected_point_differential: Team - Opponent points (for game script)
            projected_snap_share: Expected snap share (0.0-1.0)

        Returns:
            Dict with TD probabilities and expected TD counts
        """
        # Get player's historical TD rates
        td_rates = self.calculate_player_td_rates(player_name, position)

        # Calculate opponent TD multiplier from defensive metrics
        if opponent_team and current_week:
            opponent_td_rate_allowed = self.get_opponent_td_multiplier(
                opponent_team=opponent_team,
                position=position,
                current_week=current_week
            )
        # If opponent_td_rate_allowed still at default, it will be used

        # Calculate game script factors
        game_script_factors = {'rushing_multiplier': 1.0, 'receiving_multiplier': 1.0, 'passing_multiplier': 1.0}
        if projected_point_differential is not None:
            game_script_factors = self.calculate_game_script_factor(
                position=position,
                projected_point_differential=projected_point_differential
            )

        # Calculate expected TDs (λ) using Poisson model
        lambda_values = {}

        # Team scoring environment factor (higher scoring = more TDs)
        # Get scoring environment params from config
        base_total = _td_config.scoring_environment['base_team_total']
        exponent = _td_config.scoring_environment['scoring_factor_exponent']
        scoring_factor = (team_projected_total / base_total) ** exponent

        if position == 'QB':
            # Passing TDs - REQUIRE actual TD rate, NO hardcoded default
            if 'passing_td_per_attempt' not in td_rates:
                raise ValueError(
                    f"QB {player_name} missing 'passing_td_per_attempt' in td_rates. "
                    f"NO HARDCODED DEFAULTS - must calculate from actual player data."
                )
            pass_td_rate = td_rates['passing_td_per_attempt']
            lambda_pass = (
                projected_pass_attempts *
                pass_td_rate *
                scoring_factor *
                opponent_td_rate_allowed *
                game_script_factors['passing_multiplier']
            )
            lambda_values['passing_tds'] = lambda_pass

            # Rushing TDs (designed plays, red zone) - REQUIRE actual TD rate
            if 'rushing_td_per_carry' not in td_rates:
                raise ValueError(
                    f"QB {player_name} missing 'rushing_td_per_carry' in td_rates. "
                    f"NO HARDCODED DEFAULTS - must calculate from actual player data."
                )
            rush_td_rate = td_rates['rushing_td_per_carry']
            lambda_rush = (
                projected_carries *
                rush_td_rate *
                red_zone_share *
                scoring_factor *
                opponent_td_rate_allowed *
                game_script_factors['rushing_multiplier']
            )
            lambda_values['rushing_tds'] = lambda_rush

        elif position == 'RB':
            # Rushing TDs - REQUIRE actual TD rate, NO hardcoded default
            if 'rushing_td_per_carry' not in td_rates:
                raise ValueError(
                    f"RB {player_name} missing 'rushing_td_per_carry' in td_rates. "
                    f"NO HARDCODED DEFAULTS - must calculate from actual player data."
                )
            rush_td_rate = td_rates['rushing_td_per_carry']

            # Apply red zone and goal line boosts
            # Goal line backs get 2x multiplier, committee backs get less
            goal_line_boost = 1.0 + (goal_line_role * 1.0)  # Up to 2x for primary goal line back
            red_zone_boost = 1.0 + (red_zone_share * 0.8)    # Up to 1.8x for high RZ usage

            lambda_rush = (
                projected_carries *
                rush_td_rate *
                red_zone_boost *
                goal_line_boost *
                scoring_factor *
                opponent_td_rate_allowed *
                game_script_factors['rushing_multiplier']
            )
            lambda_values['rushing_tds'] = lambda_rush

            # Receiving TDs - REQUIRE actual TD rate
            if 'receiving_td_per_target' not in td_rates:
                raise ValueError(
                    f"RB {player_name} missing 'receiving_td_per_target' in td_rates. "
                    f"NO HARDCODED DEFAULTS - must calculate from actual player data."
                )
            rec_td_rate = td_rates['receiving_td_per_target']
            lambda_rec = (
                projected_targets *
                rec_td_rate *
                (1.0 + red_zone_share * 0.5) *  # RBs get some RZ targets
                scoring_factor *
                opponent_td_rate_allowed *
                game_script_factors['receiving_multiplier']
            )
            lambda_values['receiving_tds'] = lambda_rec

        elif position in ['WR', 'TE']:
            # Receiving TDs - REQUIRE actual TD rate
            if 'receiving_td_per_target' not in td_rates:
                raise ValueError(
                    f"{position} {player_name} missing 'receiving_td_per_target' in td_rates. "
                    f"NO HARDCODED DEFAULTS - must calculate from actual player data."
                )
            rec_td_rate = td_rates['receiving_td_per_target']

            # Red zone targets are crucial for WRs/TEs
            red_zone_boost = 1.0 + (red_zone_share * 1.2)  # Up to 2.2x for heavy RZ usage

            lambda_rec = (
                projected_targets *
                rec_td_rate *
                red_zone_boost *
                scoring_factor *
                opponent_td_rate_allowed *
                game_script_factors['receiving_multiplier']
            )
            lambda_values['receiving_tds'] = lambda_rec

        # Apply snap share weighting to lambda values
        # Snap share directly affects opportunity: 50% snaps ≈ 50% TDs
        # This discriminates starters (100%) from backups (30-50%)
        for td_type in lambda_values:
            lambda_values[td_type] *= projected_snap_share

        # Calculate P(any TD) using Poisson distribution
        # P(X >= 1) = 1 - P(X = 0) = 1 - exp(-λ)
        prob_no_td = 1.0

        for td_type, lambda_val in lambda_values.items():
            prob_no_td *= np.exp(-lambda_val)

        prob_any_td = 1.0 - prob_no_td

        # Cap probabilities at bounds from config
        min_prob = _td_config.probability_bounds['min_any_td_prob']
        max_prob = _td_config.probability_bounds['max_any_td_prob']
        prob_any_td = max(min_prob, min(max_prob, prob_any_td))

        return {
            'prob_any_td': prob_any_td,
            'lambda_total': sum(lambda_values.values()),
            'snap_share_used': projected_snap_share,
            **{f'{k}_mean': v for k, v in lambda_values.items()}
        }

    def predict_first_touchdown_probability(
        self,
        player_name: str,
        position: str,
        prob_any_td: float,
        team_first_score_prob: float = 0.5,
        team_td_share: float = 0.15,
    ) -> float:
        """
        Estimate probability of scoring FIRST TD.

        P(First TD) ≈ P(Team scores first) × P(Player scores it | team scores)

        Args:
            player_name: Player name
            position: Position
            prob_any_td: Probability of scoring any TD
            team_first_score_prob: P(team scores first) - typically 0.5
            team_td_share: Player's share of team's TDs (0.15 = 15%)

        Returns:
            Probability of first TD
        """
        # Simple approximation
        # If player has 60% chance of any TD and 15% share of team TDs
        # P(First) ≈ 0.5 * 0.15 * (1 + boost for high TD prob)

        # Boost factor: players more likely to score any TD are more likely to score first
        td_boost = 1.0 + (prob_any_td - 0.3) * 0.5

        prob_first_td = (
            team_first_score_prob *
            team_td_share *
            td_boost
        )

        # Cap at bounds from config
        min_prob = _td_config.probability_bounds['min_first_td_prob']
        max_prob = _td_config.probability_bounds['max_first_td_prob']
        prob_first_td = max(min_prob, min(max_prob, prob_first_td))

        return prob_first_td


def estimate_usage_factors(
    player_data: pd.Series,
    position: str
) -> Dict[str, float]:
    """
    Estimate red zone share, goal line role, and team TD share from available data.

    Args:
        player_data: Row from predictions dataframe
        position: Player position

    Returns:
        Dict with estimated usage factors
    """
    # NOTE: Red zone share and goal line role estimated from overall usage patterns
    # This provides reasonable proxies until actual RZ-specific data is available

    if position == 'QB':
        # QBs typically have high red zone participation - get from config
        qb_usage = _td_config.usage_scenarios_by_position['QB']['default']
        return {
            'red_zone_share': qb_usage['red_zone_share'],
            'goal_line_role': qb_usage['goal_line_role'],
            'team_td_share': qb_usage['team_td_share'],
        }

    elif position == 'RB':
        # Estimate based on usage - get thresholds from config
        rush_attempts = player_data.get('rushing_attempts_mean', 10.0)
        rb_scenarios = _td_config.usage_scenarios_by_position['RB']

        if rush_attempts > rb_scenarios['workhorse']['rush_attempts_threshold']:
            # Workhorse back
            scenario = rb_scenarios['workhorse']
        elif rush_attempts > rb_scenarios['primary_committee']['rush_attempts_threshold']:
            # Primary back in committee
            scenario = rb_scenarios['primary_committee']
        elif rush_attempts > rb_scenarios['secondary']['rush_attempts_threshold']:
            # Secondary back
            scenario = rb_scenarios['secondary']
        else:
            # Change of pace / backup
            scenario = rb_scenarios['backup']

        return {
            'red_zone_share': scenario['red_zone_share'],
            'goal_line_role': scenario['goal_line_role'],
            'team_td_share': scenario['team_td_share'],
        }

    elif position in ['WR', 'TE']:
        # Estimate based on target share - get thresholds from config
        targets = player_data.get('targets_mean', 0.0) if position == 'TE' else player_data.get('receptions_mean', 0.0) * 1.5
        wr_te_scenarios = _td_config.usage_scenarios_by_position['WR_TE']

        if targets > wr_te_scenarios['wr1_te1']['targets_threshold']:
            # WR1 / primary TE
            scenario = wr_te_scenarios['wr1_te1']
        elif targets > wr_te_scenarios['wr2_te2']['targets_threshold']:
            # WR2 / secondary TE
            scenario = wr_te_scenarios['wr2_te2']
        elif targets > wr_te_scenarios['wr3']['targets_threshold']:
            # WR3
            scenario = wr_te_scenarios['wr3']
        else:
            # Depth player
            scenario = wr_te_scenarios['depth']

        return {
            'red_zone_share': scenario['red_zone_share'],
            'goal_line_role': 0.0,  # WR/TE rarely get goal line carries
            'team_td_share': scenario['team_td_share'],
        }

    else:
        # Default
        return {
            'red_zone_share': 0.3,
            'goal_line_role': 0.3,
            'team_td_share': 0.1,
        }
