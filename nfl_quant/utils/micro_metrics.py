"""
Micro Metrics Module - Advanced Feature Engineering for NFL QUANT

This module extracts granular matchup metrics beyond macro EPA:
- Trench matchups (pressure rates, sack conversion)
- Explosive play profiles
- Turnover luck regression
- Game-state play-calling tendencies
- 4th down aggressiveness
- NGS advanced metrics integration
- Playoff leverage tags

Author: NFL QUANT System
Created: 2025-11-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Use centralized path configuration
from nfl_quant.config_paths import NFLVERSE_DIR

logger = logging.getLogger(__name__)

# Data paths (from centralized config)
DATA_DIR = NFLVERSE_DIR


class MicroMetricsCalculator:
    """Calculate advanced micro-level metrics from play-by-play data."""

    def __init__(self, season: int = 2025):
        self.season = season
        self._pbp = None
        self._ngs_passing = None
        self._ngs_receiving = None
        self._ngs_rushing = None
        self._target_shares = None
        self._carry_shares = None
        self._schedules = None

    def _load_pbp(self) -> pd.DataFrame:
        """Load play-by-play data."""
        if self._pbp is None:
            pbp_path = DATA_DIR / f'pbp_{self.season}.parquet'
            if not pbp_path.exists():
                pbp_path = DATA_DIR / 'pbp.parquet'
            self._pbp = pd.read_parquet(pbp_path)
            if 'season' in self._pbp.columns:
                self._pbp = self._pbp[self._pbp['season'] == self.season]
            logger.info(f"Loaded {len(self._pbp)} plays for {self.season}")
        return self._pbp

    def _load_ngs_passing(self) -> pd.DataFrame:
        """Load NGS passing data."""
        if self._ngs_passing is None:
            path = DATA_DIR / 'ngs_passing.parquet'
            if path.exists():
                self._ngs_passing = pd.read_parquet(path)
                self._ngs_passing = self._ngs_passing[self._ngs_passing['season'] == self.season]
            else:
                self._ngs_passing = pd.DataFrame()
        return self._ngs_passing

    def _load_ngs_receiving(self) -> pd.DataFrame:
        """Load NGS receiving data."""
        if self._ngs_receiving is None:
            path = DATA_DIR / 'ngs_receiving.parquet'
            if path.exists():
                self._ngs_receiving = pd.read_parquet(path)
                self._ngs_receiving = self._ngs_receiving[self._ngs_receiving['season'] == self.season]
            else:
                self._ngs_receiving = pd.DataFrame()
        return self._ngs_receiving

    def _load_ngs_rushing(self) -> pd.DataFrame:
        """Load NGS rushing data."""
        if self._ngs_rushing is None:
            path = DATA_DIR / 'ngs_rushing.parquet'
            if path.exists():
                self._ngs_rushing = pd.read_parquet(path)
                self._ngs_rushing = self._ngs_rushing[self._ngs_rushing['season'] == self.season]
            else:
                self._ngs_rushing = pd.DataFrame()
        return self._ngs_rushing

    def _load_target_shares(self) -> pd.DataFrame:
        """Load target share data."""
        if self._target_shares is None:
            path = DATA_DIR / 'player_target_shares.parquet'
            if path.exists():
                self._target_shares = pd.read_parquet(path)
            else:
                self._target_shares = pd.DataFrame()
        return self._target_shares

    def _load_carry_shares(self) -> pd.DataFrame:
        """Load carry share data."""
        if self._carry_shares is None:
            path = DATA_DIR / 'player_carry_shares.parquet'
            if path.exists():
                self._carry_shares = pd.read_parquet(path)
            else:
                self._carry_shares = pd.DataFrame()
        return self._carry_shares

    def _load_schedules(self) -> pd.DataFrame:
        """Load schedule data for playoff leverage."""
        if self._schedules is None:
            path = DATA_DIR / 'schedules.parquet'
            if path.exists():
                self._schedules = pd.read_parquet(path)
                self._schedules = self._schedules[self._schedules['season'] == self.season]
            else:
                self._schedules = pd.DataFrame()
        return self._schedules

    # =========================================================================
    # TRENCH METRICS (Pressure, Sacks, QB Hits)
    # =========================================================================

    def calculate_team_pressure_metrics(self, through_week: int) -> pd.DataFrame:
        """
        Calculate team-level pressure metrics through a given week.

        Returns:
            DataFrame with columns:
            - team
            - pressure_rate_allowed (OL metric)
            - pressure_rate_generated (DL metric)
            - sack_rate_allowed
            - sack_rate_generated
            - qb_hit_rate_allowed
            - qb_hit_rate_generated
            - pressure_to_sack_conversion (how often pressure becomes sack)
        """
        pbp = self._load_pbp()

        # Filter to pass plays through the week
        pass_plays = pbp[
            (pbp['week'] <= through_week) &
            (pbp['play_type'] == 'pass') &
            (pbp['qb_dropback'] == 1)
        ].copy()

        if pass_plays.empty:
            logger.warning(f"No pass plays found through week {through_week}")
            return pd.DataFrame()

        # Offensive metrics (pressure/sacks ALLOWED)
        off_metrics = pass_plays.groupby('posteam').agg({
            'qb_hit': 'sum',
            'sack': 'sum',
            'play_id': 'count'
        }).rename(columns={'play_id': 'dropbacks'})

        off_metrics['pressure_rate_allowed'] = (off_metrics['qb_hit'] + off_metrics['sack']) / off_metrics['dropbacks']
        off_metrics['sack_rate_allowed'] = off_metrics['sack'] / off_metrics['dropbacks']
        off_metrics['qb_hit_rate_allowed'] = off_metrics['qb_hit'] / off_metrics['dropbacks']

        # Defensive metrics (pressure/sacks GENERATED)
        def_metrics = pass_plays.groupby('defteam').agg({
            'qb_hit': 'sum',
            'sack': 'sum',
            'play_id': 'count'
        }).rename(columns={'play_id': 'dropbacks_faced'})

        def_metrics['pressure_rate_generated'] = (def_metrics['qb_hit'] + def_metrics['sack']) / def_metrics['dropbacks_faced']
        def_metrics['sack_rate_generated'] = def_metrics['sack'] / def_metrics['dropbacks_faced']
        def_metrics['qb_hit_rate_generated'] = def_metrics['qb_hit'] / def_metrics['dropbacks_faced']

        # Calculate pressure-to-sack conversion
        pressures_generated = def_metrics['qb_hit'] + def_metrics['sack']
        def_metrics['pressure_to_sack_conversion'] = np.where(
            pressures_generated > 0,
            def_metrics['sack'] / pressures_generated,
            0
        )

        # Merge offensive and defensive
        result = off_metrics[['pressure_rate_allowed', 'sack_rate_allowed', 'qb_hit_rate_allowed']].copy()
        result = result.join(
            def_metrics[['pressure_rate_generated', 'sack_rate_generated',
                        'qb_hit_rate_generated', 'pressure_to_sack_conversion']]
        )
        result.index.name = 'team'
        result = result.reset_index()

        return result

    # =========================================================================
    # EXPLOSIVE PLAY METRICS
    # =========================================================================

    def calculate_team_explosive_metrics(self, through_week: int) -> pd.DataFrame:
        """
        Calculate explosive play rates (20+ yards) for teams.

        Returns:
            DataFrame with:
            - explosive_pass_rate (20+ yard passes / pass attempts)
            - explosive_rush_rate (20+ yard rushes / rush attempts)
            - explosive_play_rate (combined)
            - explosive_allowed_rate (defense)
            - big_play_variance (std dev of yards gained - boom/bust indicator)
        """
        pbp = self._load_pbp()

        plays = pbp[
            (pbp['week'] <= through_week) &
            (pbp['play_type'].isin(['pass', 'run']))
        ].copy()

        if plays.empty:
            return pd.DataFrame()

        # Define explosive plays
        plays['is_explosive'] = plays['yards_gained'] >= 20
        plays['is_explosive_pass'] = (plays['play_type'] == 'pass') & plays['is_explosive']
        plays['is_explosive_rush'] = (plays['play_type'] == 'run') & plays['is_explosive']

        # Offensive explosive rates
        off_metrics = plays.groupby('posteam').agg({
            'is_explosive': 'sum',
            'is_explosive_pass': 'sum',
            'is_explosive_rush': 'sum',
            'play_id': 'count',
            'yards_gained': ['std', 'mean']
        })
        off_metrics.columns = ['explosive_plays', 'explosive_passes', 'explosive_rushes',
                              'total_plays', 'yards_std', 'yards_mean']

        # Pass and rush attempt counts
        pass_attempts = plays[plays['play_type'] == 'pass'].groupby('posteam').size()
        rush_attempts = plays[plays['play_type'] == 'run'].groupby('posteam').size()

        off_metrics['pass_attempts'] = pass_attempts
        off_metrics['rush_attempts'] = rush_attempts
        off_metrics = off_metrics.fillna(0)

        off_metrics['explosive_play_rate'] = off_metrics['explosive_plays'] / off_metrics['total_plays']
        off_metrics['explosive_pass_rate'] = np.where(
            off_metrics['pass_attempts'] > 0,
            off_metrics['explosive_passes'] / off_metrics['pass_attempts'],
            0
        )
        off_metrics['explosive_rush_rate'] = np.where(
            off_metrics['rush_attempts'] > 0,
            off_metrics['explosive_rushes'] / off_metrics['rush_attempts'],
            0
        )
        off_metrics['big_play_variance'] = off_metrics['yards_std']

        # Defensive explosive allowed
        def_metrics = plays.groupby('defteam').agg({
            'is_explosive': 'sum',
            'play_id': 'count'
        })
        def_metrics.columns = ['explosive_allowed', 'plays_faced']
        def_metrics['explosive_allowed_rate'] = def_metrics['explosive_allowed'] / def_metrics['plays_faced']

        # Combine
        result = off_metrics[['explosive_play_rate', 'explosive_pass_rate',
                             'explosive_rush_rate', 'big_play_variance']].copy()
        result = result.join(def_metrics[['explosive_allowed_rate']])
        result.index.name = 'team'
        result = result.reset_index()

        return result

    # =========================================================================
    # TURNOVER LUCK / REGRESSION
    # =========================================================================

    def calculate_turnover_luck_metrics(self, through_week: int) -> pd.DataFrame:
        """
        Calculate turnover metrics with luck/regression indicators.

        Returns:
            - fumbles_forced
            - fumbles_lost
            - fumble_recovery_rate (luck indicator - expected ~50%)
            - fumble_luck_factor (deviation from expected)
            - interceptions
            - turnover_margin
        """
        pbp = self._load_pbp()

        plays = pbp[pbp['week'] <= through_week].copy()

        if plays.empty:
            return pd.DataFrame()

        # Offensive turnovers (bad)
        off_to = plays.groupby('posteam').agg({
            'fumble': 'sum',
            'fumble_lost': 'sum',
            'interception': 'sum'
        }).rename(columns={
            'fumble': 'fumbles',
            'fumble_lost': 'fumbles_lost',
            'interception': 'ints_thrown'
        })

        # Defensive turnovers created (good)
        def_to = plays.groupby('defteam').agg({
            'fumble_forced': 'sum',
            'fumble_lost': 'sum',  # This is fumbles recovered by defense
            'interception': 'sum'
        }).rename(columns={
            'fumble_forced': 'fumbles_forced_def',
            'fumble_lost': 'fumbles_recovered',
            'interception': 'ints_created'
        })

        # Calculate luck factors
        # Fumble recovery rate should regress to ~50%
        off_to['fumble_recovery_rate'] = np.where(
            off_to['fumbles'] > 0,
            off_to['fumbles_lost'] / off_to['fumbles'],
            0.5
        )
        # Luck factor: >0.5 means unlucky (lost more than expected)
        off_to['fumble_luck_factor'] = off_to['fumble_recovery_rate'] - 0.5

        # Combine and calculate turnover margin
        result = off_to.join(def_to, how='outer').fillna(0)
        result['turnovers_committed'] = result['fumbles_lost'] + result['ints_thrown']
        result['turnovers_forced'] = result['fumbles_recovered'] + result['ints_created']
        result['turnover_margin'] = result['turnovers_forced'] - result['turnovers_committed']

        # Expected turnover margin based on luck regression
        # If team has been lucky with fumble recoveries, expect regression
        result['expected_turnover_regression'] = -result['fumble_luck_factor'] * result['fumbles']

        result.index.name = 'team'
        result = result.reset_index()

        return result[['team', 'turnovers_committed', 'turnovers_forced', 'turnover_margin',
                       'fumble_recovery_rate', 'fumble_luck_factor', 'expected_turnover_regression']]

    # =========================================================================
    # 4TH DOWN AGGRESSIVENESS
    # =========================================================================

    def calculate_fourth_down_metrics(self, through_week: int) -> pd.DataFrame:
        """
        Calculate 4th down aggressiveness and success rates.

        Returns:
            - fourth_down_go_rate (how often they go for it)
            - fourth_down_success_rate
            - fourth_down_attempts
            - aggressiveness_index (compared to league average)
        """
        pbp = self._load_pbp()

        fourth_downs = pbp[
            (pbp['week'] <= through_week) &
            (pbp['down'] == 4)
        ].copy()

        if fourth_downs.empty:
            return pd.DataFrame()

        # Categorize 4th down decisions
        fourth_downs['went_for_it'] = fourth_downs['play_type'].isin(['pass', 'run'])
        fourth_downs['converted'] = fourth_downs['fourth_down_converted'] == 1

        # Team-level aggregation
        metrics = fourth_downs.groupby('posteam').agg({
            'went_for_it': ['sum', 'count'],
            'converted': 'sum',
            'fourth_down_converted': 'sum',
            'fourth_down_failed': 'sum'
        })
        metrics.columns = ['go_for_it_count', 'fourth_down_situations',
                          'conversions', 'converted_total', 'failed_total']

        metrics['fourth_down_go_rate'] = metrics['go_for_it_count'] / metrics['fourth_down_situations']
        metrics['fourth_down_success_rate'] = np.where(
            metrics['go_for_it_count'] > 0,
            metrics['conversions'] / metrics['go_for_it_count'],
            0
        )

        # League average for aggressiveness index
        league_avg_go_rate = metrics['fourth_down_go_rate'].mean()
        metrics['aggressiveness_index'] = metrics['fourth_down_go_rate'] / league_avg_go_rate

        metrics.index.name = 'team'
        metrics = metrics.reset_index()

        return metrics[['team', 'fourth_down_go_rate', 'fourth_down_success_rate',
                       'go_for_it_count', 'aggressiveness_index']]

    # =========================================================================
    # GAME STATE PLAY-CALLING
    # =========================================================================

    def calculate_game_state_tendencies(self, through_week: int) -> pd.DataFrame:
        """
        Calculate play-calling tendencies by game state.

        Returns:
            - neutral_pass_rate (1st half, score within 7)
            - leading_pass_rate (when up by 7+)
            - trailing_pass_rate (when down by 7+)
            - pass_rate_delta (how much they change when leading vs trailing)
            - early_down_pass_rate (1st and 2nd down)
        """
        pbp = self._load_pbp()

        plays = pbp[
            (pbp['week'] <= through_week) &
            (pbp['play_type'].isin(['pass', 'run']))
        ].copy()

        if plays.empty:
            return pd.DataFrame()

        plays['is_pass'] = plays['play_type'] == 'pass'
        plays['score_diff'] = plays['posteam_score'] - plays['defteam_score']

        # Define game states
        plays['neutral_script'] = (
            (plays['qtr'] <= 2) &  # First half
            (abs(plays['score_diff']) <= 7)  # Close game
        )
        plays['leading_big'] = plays['score_diff'] >= 7
        plays['trailing_big'] = plays['score_diff'] <= -7
        plays['early_down'] = plays['down'].isin([1, 2])

        def calc_pass_rate(df, mask_col):
            subset = df[df[mask_col]]
            return subset.groupby('posteam')['is_pass'].mean()

        # Calculate various pass rates
        neutral_rate = calc_pass_rate(plays, 'neutral_script')
        leading_rate = calc_pass_rate(plays, 'leading_big')
        trailing_rate = calc_pass_rate(plays, 'trailing_big')
        early_down_rate = calc_pass_rate(plays, 'early_down')
        overall_rate = plays.groupby('posteam')['is_pass'].mean()

        result = pd.DataFrame({
            'neutral_pass_rate': neutral_rate,
            'leading_pass_rate': leading_rate,
            'trailing_pass_rate': trailing_rate,
            'early_down_pass_rate': early_down_rate,
            'overall_pass_rate': overall_rate
        }).fillna(0.5)  # Default to 50% if no data

        # Calculate delta (how much behavior changes)
        result['pass_rate_delta_leading'] = result['leading_pass_rate'] - result['neutral_pass_rate']
        result['pass_rate_delta_trailing'] = result['trailing_pass_rate'] - result['neutral_pass_rate']

        result.index.name = 'team'
        result = result.reset_index()

        return result

    # =========================================================================
    # NGS ADVANCED METRICS
    # =========================================================================

    def get_player_ngs_metrics(self, player_id: str, week: int) -> Dict:
        """
        Get NGS metrics for a specific player through a given week.

        Returns dict with relevant NGS stats based on position.
        """
        result = {}

        # Try passing metrics
        ngs_pass = self._load_ngs_passing()
        if not ngs_pass.empty:
            player_pass = ngs_pass[
                (ngs_pass['player_gsis_id'] == player_id) &
                (ngs_pass['week'] <= week)
            ]
            if not player_pass.empty:
                latest = player_pass.iloc[-1]
                result.update({
                    'ngs_avg_time_to_throw': latest.get('avg_time_to_throw'),
                    'ngs_aggressiveness': latest.get('aggressiveness'),
                    'ngs_avg_air_yards': latest.get('avg_intended_air_yards'),
                    'ngs_completion_pct_above_expected': latest.get('completion_percentage_above_expectation'),
                })

        # Try receiving metrics
        ngs_rec = self._load_ngs_receiving()
        if not ngs_rec.empty:
            player_rec = ngs_rec[
                (ngs_rec['player_gsis_id'] == player_id) &
                (ngs_rec['week'] <= week)
            ]
            if not player_rec.empty:
                latest = player_rec.iloc[-1]
                result.update({
                    'ngs_avg_cushion': latest.get('avg_cushion'),
                    'ngs_avg_separation': latest.get('avg_separation'),
                    'ngs_avg_yac': latest.get('avg_yac'),
                    'ngs_yac_above_expected': latest.get('avg_yac_above_expectation'),
                    'ngs_catch_pct': latest.get('catch_percentage'),
                })

        # Try rushing metrics
        ngs_rush = self._load_ngs_rushing()
        if not ngs_rush.empty:
            player_rush = ngs_rush[
                (ngs_rush['player_gsis_id'] == player_id) &
                (ngs_rush['week'] <= week)
            ]
            if not player_rush.empty:
                latest = player_rush.iloc[-1]
                result.update({
                    'ngs_efficiency': latest.get('efficiency'),
                    'ngs_rush_yards_over_expected': latest.get('rush_yards_over_expected_per_att'),
                    'ngs_pct_8plus_defenders': latest.get('percent_attempts_gte_eight_defenders'),
                })

        return result

    def get_team_ngs_aggregates(self, through_week: int) -> pd.DataFrame:
        """
        Aggregate NGS metrics at team level.
        """
        ngs_pass = self._load_ngs_passing()
        ngs_rec = self._load_ngs_receiving()
        ngs_rush = self._load_ngs_rushing()

        results = []

        # Passing aggregates by team
        if not ngs_pass.empty:
            pass_agg = ngs_pass[ngs_pass['week'] <= through_week].groupby('team_abbr').agg({
                'avg_time_to_throw': 'mean',
                'aggressiveness': 'mean',
                'avg_intended_air_yards': 'mean',
                'completion_percentage_above_expectation': 'mean'
            }).rename(columns={
                'avg_time_to_throw': 'team_avg_time_to_throw',
                'aggressiveness': 'team_qb_aggressiveness',
                'avg_intended_air_yards': 'team_avg_intended_air_yards',
                'completion_percentage_above_expectation': 'team_cpoe'
            })
            results.append(pass_agg)

        # Receiving aggregates
        if not ngs_rec.empty:
            rec_agg = ngs_rec[ngs_rec['week'] <= through_week].groupby('team_abbr').agg({
                'avg_separation': 'mean',
                'avg_yac_above_expectation': 'mean'
            }).rename(columns={
                'avg_separation': 'team_avg_separation',
                'avg_yac_above_expectation': 'team_yac_over_expected'
            })
            results.append(rec_agg)

        # Rushing aggregates
        if not ngs_rush.empty:
            rush_agg = ngs_rush[ngs_rush['week'] <= through_week].groupby('team_abbr').agg({
                'rush_yards_over_expected_per_att': 'mean',
                'efficiency': 'mean'
            }).rename(columns={
                'rush_yards_over_expected_per_att': 'team_ryoe_per_att',
                'efficiency': 'team_rush_efficiency'
            })
            results.append(rush_agg)

        if results:
            combined = pd.concat(results, axis=1)
            combined.index.name = 'team'
            return combined.reset_index()

        return pd.DataFrame()

    # =========================================================================
    # TARGET / CARRY SHARES (Already Downloaded)
    # =========================================================================

    def get_player_share_metrics(self, player_id: str, team: str, week: int) -> Dict:
        """
        Get target and carry share for a player.
        """
        result = {}

        # Target share
        ts = self._load_target_shares()
        if not ts.empty:
            player_ts = ts[
                (ts['receiver_player_id'] == player_id) &
                (ts['posteam'] == team) &
                (ts['week'] == week - 1)  # Use previous week's data
            ]
            if not player_ts.empty:
                result['target_share'] = player_ts.iloc[0]['target_share']
                result['targets_last_week'] = player_ts.iloc[0]['targets']

            # Also get trailing average
            player_ts_all = ts[
                (ts['receiver_player_id'] == player_id) &
                (ts['posteam'] == team) &
                (ts['week'] < week)
            ]
            if len(player_ts_all) > 0:
                result['target_share_trailing'] = player_ts_all['target_share'].mean()

        # Carry share
        cs = self._load_carry_shares()
        if not cs.empty:
            player_cs = cs[
                (cs['rusher_player_id'] == player_id) &
                (cs['posteam'] == team) &
                (cs['week'] == week - 1)
            ]
            if not player_cs.empty:
                result['carry_share'] = player_cs.iloc[0]['carry_share']
                result['carries_last_week'] = player_cs.iloc[0]['carries']

            player_cs_all = cs[
                (cs['rusher_player_id'] == player_id) &
                (cs['posteam'] == team) &
                (cs['week'] < week)
            ]
            if len(player_cs_all) > 0:
                result['carry_share_trailing'] = player_cs_all['carry_share'].mean()

        return result

    # =========================================================================
    # PLAYOFF LEVERAGE / MOTIVATION
    # =========================================================================

    def calculate_playoff_leverage(self, week: int) -> pd.DataFrame:
        """
        Calculate playoff leverage/motivation indicators.

        Note: This is a simplified version. Full implementation would
        require playoff probability models.

        Returns:
            - games_remaining
            - is_late_season (week >= 14)
            - division_games_remaining
            - motivation_score (heuristic)
        """
        schedules = self._load_schedules()

        if schedules.empty:
            return pd.DataFrame()

        # Get remaining games for each team
        remaining = schedules[schedules['week'] > week].copy()

        team_metrics = []
        for team in schedules['home_team'].unique():
            home_games = remaining[remaining['home_team'] == team]
            away_games = remaining[remaining['away_team'] == team]

            games_remaining = len(home_games) + len(away_games)

            # Division games (simplified - same first 2 letters often = division)
            # In reality would need division mapping
            div_home = len(home_games[home_games['div_game'] == 1]) if 'div_game' in home_games.columns else 0
            div_away = len(away_games[away_games['div_game'] == 1]) if 'div_game' in away_games.columns else 0
            div_games_remaining = div_home + div_away

            team_metrics.append({
                'team': team,
                'games_remaining': games_remaining,
                'is_late_season': week >= 14,
                'division_games_remaining': div_games_remaining,
                # Simple motivation heuristic: late season + games remaining
                'motivation_score': 1.0 + (0.1 * (17 - week)) if week >= 10 else 1.0
            })

        return pd.DataFrame(team_metrics)

    # =========================================================================
    # MAIN AGGREGATION METHOD
    # =========================================================================

    def get_all_team_micro_metrics(self, through_week: int) -> pd.DataFrame:
        """
        Get all team-level micro metrics in one DataFrame.

        This is the main method to call for feature engineering.
        """
        logger.info(f"Calculating all micro metrics through week {through_week}")

        # Calculate each metric type
        pressure = self.calculate_team_pressure_metrics(through_week)
        explosive = self.calculate_team_explosive_metrics(through_week)
        turnover = self.calculate_turnover_luck_metrics(through_week)
        fourth_down = self.calculate_fourth_down_metrics(through_week)
        game_state = self.calculate_game_state_tendencies(through_week)
        ngs = self.get_team_ngs_aggregates(through_week)

        # Merge all together
        result = None
        for df in [pressure, explosive, turnover, fourth_down, game_state, ngs]:
            if df is not None and not df.empty:
                if result is None:
                    result = df
                else:
                    result = result.merge(df, on='team', how='outer')

        if result is not None:
            logger.info(f"Generated {len(result.columns)} micro metric features for {len(result)} teams")

        return result if result is not None else pd.DataFrame()

    def get_matchup_micro_metrics(self, team: str, opponent: str, week: int) -> Dict:
        """
        Get matchup-specific micro metrics for a game.

        This combines offensive team metrics vs defensive opponent metrics.
        """
        all_metrics = self.get_all_team_micro_metrics(week - 1)

        if all_metrics.empty:
            return {}

        team_row = all_metrics[all_metrics['team'] == team]
        opp_row = all_metrics[all_metrics['team'] == opponent]

        if team_row.empty or opp_row.empty:
            return {}

        team_data = team_row.iloc[0].to_dict()
        opp_data = opp_row.iloc[0].to_dict()

        # Create matchup differentials
        matchup = {}

        # Pressure matchup: team's pressure allowed vs opp pressure generated
        if 'pressure_rate_allowed' in team_data and 'pressure_rate_generated' in opp_data:
            matchup['pressure_matchup'] = team_data['pressure_rate_allowed'] - opp_data['pressure_rate_generated']

        # Explosive matchup: team's explosive rate vs opp allowed
        if 'explosive_play_rate' in team_data and 'explosive_allowed_rate' in opp_data:
            matchup['explosive_matchup'] = team_data['explosive_play_rate'] - opp_data['explosive_allowed_rate']

        # Include raw values for model
        for key, val in team_data.items():
            if key != 'team':
                matchup[f'team_{key}'] = val

        for key, val in opp_data.items():
            if key != 'team':
                matchup[f'opp_{key}'] = val

        return matchup


# Convenience function for quick access
def get_micro_metrics(season: int = 2025, through_week: int = None) -> pd.DataFrame:
    """
    Quick function to get all team micro metrics.

    Usage:
        from nfl_quant.utils.micro_metrics import get_micro_metrics
        metrics = get_micro_metrics(2025, through_week=12)
    """
    calc = MicroMetricsCalculator(season)
    if through_week is None:
        # Auto-detect current week
        pbp = calc._load_pbp()
        through_week = pbp['week'].max() if not pbp.empty else 1

    return calc.get_all_team_micro_metrics(through_week)


if __name__ == '__main__':
    # Test the module
    logging.basicConfig(level=logging.INFO)

    calc = MicroMetricsCalculator(2025)

    print("\n=== Testing Pressure Metrics ===")
    pressure = calc.calculate_team_pressure_metrics(12)
    print(pressure.head())

    print("\n=== Testing Explosive Play Metrics ===")
    explosive = calc.calculate_team_explosive_metrics(12)
    print(explosive.head())

    print("\n=== Testing Turnover Luck Metrics ===")
    turnover = calc.calculate_turnover_luck_metrics(12)
    print(turnover.head())

    print("\n=== Testing 4th Down Metrics ===")
    fourth = calc.calculate_fourth_down_metrics(12)
    print(fourth.head())

    print("\n=== Testing Game State Tendencies ===")
    game_state = calc.calculate_game_state_tendencies(12)
    print(game_state.head())

    print("\n=== All Combined Metrics ===")
    all_metrics = calc.get_all_team_micro_metrics(12)
    print(f"Total columns: {len(all_metrics.columns)}")
    print(all_metrics.columns.tolist())
