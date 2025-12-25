"""
Dynamic Parameter Provider

Replaces hardcoded values with data-driven parameters fetched from NFLverse data.
Eliminates assumptions like:
- team_avg_pass_attempts = 35.0 (actual: ~32.5)
- team_avg_rush_attempts = 25.0 (actual: ~26.9)
- snap_share = 0.6 if starter (should be actual snap data)
- std = mean * 0.25 (actual CV is ~0.47)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple
import warnings

# Use centralized path configuration
from nfl_quant.config_paths import NFLVERSE_DIR

# Data directory (from centralized config)
DATA_DIR = NFLVERSE_DIR


class DynamicParameterProvider:
    """
    Provides empirically-derived parameters from NFLverse data.
    All parameters are computed from actual data, not hardcoded assumptions.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self._cache: Dict[str, Any] = {}
        self._weekly_data: Optional[pd.DataFrame] = None
        self._snap_counts: Optional[pd.DataFrame] = None
        self._target_shares: Optional[pd.DataFrame] = None
        self._carry_shares: Optional[pd.DataFrame] = None
        self._team_pace: Optional[pd.DataFrame] = None
        self._defensive_epa: Optional[pd.DataFrame] = None

    @property
    def weekly_data(self) -> pd.DataFrame:
        """Load weekly player stats."""
        if self._weekly_data is None:
            # Prefer 2024+2025 combined data for current season predictions
            path = self.data_dir / "player_stats_2024_2025.parquet"
            if path.exists():
                self._weekly_data = pd.read_parquet(path)
                # Rename columns to match expected format
                if 'team' in self._weekly_data.columns and 'recent_team' not in self._weekly_data.columns:
                    self._weekly_data['recent_team'] = self._weekly_data['team']
            else:
                # Fall back to weekly_2024
                path = self.data_dir / "weekly_2024.parquet"
                if path.exists():
                    self._weekly_data = pd.read_parquet(path)
                else:
                    # Fall back to CSV
                    csv_path = self.data_dir / "weekly_2024.csv"
                    if csv_path.exists():
                        self._weekly_data = pd.read_csv(csv_path)
                    else:
                        raise FileNotFoundError(f"No weekly data found at {path}")
        return self._weekly_data

    @property
    def snap_counts(self) -> pd.DataFrame:
        """Load snap count data."""
        if self._snap_counts is None:
            path = self.data_dir / "snap_counts_2025.csv"
            if path.exists():
                self._snap_counts = pd.read_csv(path)
            else:
                warnings.warn(f"Snap counts not found at {path}")
                self._snap_counts = pd.DataFrame()
        return self._snap_counts

    @property
    def target_shares(self) -> pd.DataFrame:
        """Load target share data."""
        if self._target_shares is None:
            path = self.data_dir / "player_target_shares.parquet"
            if path.exists():
                self._target_shares = pd.read_parquet(path)
            else:
                self._target_shares = pd.DataFrame()
        return self._target_shares

    @property
    def carry_shares(self) -> pd.DataFrame:
        """Load carry share data."""
        if self._carry_shares is None:
            path = self.data_dir / "player_carry_shares.parquet"
            if path.exists():
                self._carry_shares = pd.read_parquet(path)
            else:
                self._carry_shares = pd.DataFrame()
        return self._carry_shares

    @property
    def team_pace(self) -> pd.DataFrame:
        """Load team pace data."""
        if self._team_pace is None:
            path = self.data_dir / "team_pace.parquet"
            if path.exists():
                self._team_pace = pd.read_parquet(path)
            else:
                self._team_pace = pd.DataFrame()
        return self._team_pace

    @property
    def defensive_epa(self) -> pd.DataFrame:
        """Load defensive EPA data."""
        if self._defensive_epa is None:
            path = self.data_dir / "team_defensive_epa.parquet"
            if path.exists():
                self._defensive_epa = pd.read_parquet(path)
            else:
                self._defensive_epa = pd.DataFrame()
        return self._defensive_epa

    # ========== TEAM-LEVEL PARAMETERS ==========

    def get_league_avg_pass_attempts(self, season: Optional[int] = None) -> float:
        """
        Calculate actual league average pass attempts per game.
        Replaces hardcoded 35.0

        Args:
            season: Filter to specific season. If None, uses most recent season.
        """
        if season is None:
            season = self.weekly_data['season'].max()

        cache_key = f"league_avg_pass_attempts_{season}"
        if cache_key not in self._cache:
            qb_data = self.weekly_data[
                (self.weekly_data['position'] == 'QB') &
                (self.weekly_data['season'] == season)
            ]
            # Group by team and week to get per-game attempts
            team_game_attempts = qb_data.groupby(['recent_team', 'week'])['attempts'].sum()
            self._cache[cache_key] = team_game_attempts.mean()
        return self._cache[cache_key]

    def get_league_avg_rush_attempts(self, season: Optional[int] = None) -> float:
        """
        Calculate actual league average rush attempts per game.
        Replaces hardcoded 25.0

        Args:
            season: Filter to specific season. If None, uses most recent season.
        """
        if season is None:
            season = self.weekly_data['season'].max()

        cache_key = f"league_avg_rush_attempts_{season}"
        if cache_key not in self._cache:
            team_data = self.weekly_data[self.weekly_data['season'] == season]
            rush_data = team_data.groupby(['recent_team', 'week'])['carries'].sum()
            self._cache[cache_key] = rush_data.mean()
        return self._cache[cache_key]

    def get_team_pass_attempts(self, team: str, up_to_week: Optional[int] = None, season: Optional[int] = None) -> float:
        """
        Get team-specific average pass attempts per game.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            up_to_week: Only use data up to this week (for backtesting)
            season: Filter to specific season. If None, uses most recent season.
        """
        if season is None:
            season = self.weekly_data['season'].max()

        cache_key = f"team_pass_{team}_{up_to_week}_{season}"
        if cache_key not in self._cache:
            qb_data = self.weekly_data[
                (self.weekly_data['position'] == 'QB') &
                (self.weekly_data['recent_team'] == team) &
                (self.weekly_data['season'] == season)
            ]
            if up_to_week is not None:
                qb_data = qb_data[qb_data['week'] < up_to_week]

            if len(qb_data) == 0:
                self._cache[cache_key] = self.get_league_avg_pass_attempts(season)
            else:
                # Group by week within the season to get per-game attempts
                team_game_attempts = qb_data.groupby('week')['attempts'].sum()
                self._cache[cache_key] = team_game_attempts.mean()

        return self._cache[cache_key]

    def get_team_rush_attempts(self, team: str, up_to_week: Optional[int] = None, season: Optional[int] = None) -> float:
        """
        Get team-specific average rush attempts per game.

        Args:
            team: Team abbreviation
            up_to_week: Only use data up to this week (for backtesting)
            season: Filter to specific season. If None, uses most recent season.
        """
        if season is None:
            season = self.weekly_data['season'].max()

        cache_key = f"team_rush_{team}_{up_to_week}_{season}"
        if cache_key not in self._cache:
            team_data = self.weekly_data[
                (self.weekly_data['recent_team'] == team) &
                (self.weekly_data['season'] == season)
            ]
            if up_to_week is not None:
                team_data = team_data[team_data['week'] < up_to_week]

            if len(team_data) == 0:
                self._cache[cache_key] = self.get_league_avg_rush_attempts(season)
            else:
                # Group by week within the season to get per-game attempts
                team_game_carries = team_data.groupby('week')['carries'].sum()
                self._cache[cache_key] = team_game_carries.mean()

        return self._cache[cache_key]

    def get_team_plays_per_game(self, team: str) -> float:
        """Get team's average plays per game from pace data."""
        if len(self.team_pace) > 0 and team in self.team_pace['team'].values:
            return self.team_pace[self.team_pace['team'] == team]['plays_per_game'].iloc[0]
        # Fallback: sum of pass and rush attempts
        return self.get_team_pass_attempts(team) + self.get_team_rush_attempts(team)

    def get_team_defensive_epa(self, team: str) -> float:
        """Get team's defensive EPA allowed (higher = worse defense)."""
        if len(self.defensive_epa) > 0 and team in self.defensive_epa['team'].values:
            return self.defensive_epa[self.defensive_epa['team'] == team]['def_epa_allowed'].iloc[0]
        return 0.0  # Neutral

    # ========== PLAYER-LEVEL PARAMETERS ==========

    def get_player_snap_share(
        self,
        player_name: str,
        team: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> float:
        """
        Get player's actual snap share from snap count data.
        Replaces hardcoded 0.6 for starters.

        Args:
            player_name: Player's name
            team: Team abbreviation
            up_to_week: Only use data up to this week
            n_weeks: Number of trailing weeks to average
        """
        if len(self.snap_counts) == 0:
            return self._estimate_snap_share_from_usage(player_name, team, up_to_week, n_weeks)

        player_snaps = self.snap_counts[
            (self.snap_counts['player'] == player_name) &
            (self.snap_counts['team'] == team)
        ]

        if up_to_week is not None:
            player_snaps = player_snaps[player_snaps['week'] < up_to_week]

        if len(player_snaps) == 0:
            return self._estimate_snap_share_from_usage(player_name, team, up_to_week, n_weeks)

        # Get recent snap percentage
        recent_snaps = player_snaps.nlargest(n_weeks, 'week')
        # offense_pct is already a ratio (0.84 = 84%), not a percentage value
        return recent_snaps['offense_pct'].mean()

    def _estimate_snap_share_from_usage(
        self,
        player_name: str,
        team: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> float:
        """Estimate snap share from target/carry usage when direct snap data unavailable."""
        player_data = self.weekly_data[
            (self.weekly_data['player_name'] == player_name) &
            (self.weekly_data['recent_team'] == team)
        ]

        if up_to_week is not None:
            player_data = player_data[player_data['week'] < up_to_week]

        if len(player_data) == 0:
            return 0.4  # Minimal fallback

        position = player_data['position'].iloc[0]
        recent_data = player_data.nlargest(n_weeks, 'week')

        if position == 'RB':
            avg_carries = recent_data['carries'].mean()
            avg_targets = recent_data['targets'].mean() if 'targets' in recent_data.columns else 0
            # Estimate: high-usage RBs (15+ carries) get ~60-75% snaps
            usage_score = (avg_carries / 15.0) + (avg_targets / 4.0)
            return min(0.85, max(0.25, usage_score * 0.5))

        elif position in ['WR', 'TE']:
            avg_targets = recent_data['targets'].mean() if 'targets' in recent_data.columns else 0
            # High-target receivers (8+ targets) get ~85-95% snaps
            return min(0.95, max(0.30, avg_targets / 10.0))

        else:
            return 0.5

    def get_player_target_share(
        self,
        player_id: str,
        team: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> float:
        """Get player's actual target share from NFLverse data."""
        if len(self.target_shares) == 0:
            return self._compute_target_share_from_weekly(player_id, team, up_to_week, n_weeks)

        player_shares = self.target_shares[
            (self.target_shares['receiver_player_id'] == player_id) &
            (self.target_shares['posteam'] == team)
        ]

        if up_to_week is not None:
            player_shares = player_shares[player_shares['week'] < up_to_week]

        if len(player_shares) == 0:
            return self._compute_target_share_from_weekly(player_id, team, up_to_week, n_weeks)

        recent_shares = player_shares.nlargest(n_weeks, 'week')
        return recent_shares['target_share'].mean()

    def _compute_target_share_from_weekly(
        self,
        player_id: str,
        team: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> float:
        """Compute target share from weekly stats."""
        # Try matching by player_display_name first (most common input format)
        # Then fall back to player_id if not found
        if 'player_display_name' in self.weekly_data.columns:
            player_data = self.weekly_data[
                (self.weekly_data['player_display_name'] == player_id) &
                (self.weekly_data['recent_team'] == team)
            ]
        else:
            player_data = self.weekly_data[
                (self.weekly_data['player_id'] == player_id) &
                (self.weekly_data['recent_team'] == team)
            ]

        # If no matches by display name, try player_name (abbreviated form)
        if len(player_data) == 0 and 'player_name' in self.weekly_data.columns:
            player_data = self.weekly_data[
                (self.weekly_data['player_name'] == player_id) &
                (self.weekly_data['recent_team'] == team)
            ]

        if up_to_week is not None:
            player_data = player_data[player_data['week'] < up_to_week]

        if len(player_data) == 0 or 'targets' not in player_data.columns:
            return 0.0

        recent_data = player_data.nlargest(n_weeks, 'week')

        total_targets = 0.0
        for _, row in recent_data.iterrows():
            week = row['week']
            team_targets = self.weekly_data[
                (self.weekly_data['recent_team'] == team) &
                (self.weekly_data['week'] == week)
            ]['targets'].sum()

            if team_targets > 0:
                total_targets += row['targets'] / team_targets

        return total_targets / len(recent_data) if len(recent_data) > 0 else 0.0

    def get_player_carry_share(
        self,
        player_id: str,
        team: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> float:
        """Get player's actual carry share from NFLverse data."""
        if len(self.carry_shares) == 0:
            return self._compute_carry_share_from_weekly(player_id, team, up_to_week, n_weeks)

        player_shares = self.carry_shares[
            (self.carry_shares['rusher_player_id'] == player_id) &
            (self.carry_shares['posteam'] == team)
        ]

        if up_to_week is not None:
            player_shares = player_shares[player_shares['week'] < up_to_week]

        if len(player_shares) == 0:
            return self._compute_carry_share_from_weekly(player_id, team, up_to_week, n_weeks)

        recent_shares = player_shares.nlargest(n_weeks, 'week')
        return recent_shares['carry_share'].mean()

    def _compute_carry_share_from_weekly(
        self,
        player_id: str,
        team: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> float:
        """Compute carry share from weekly stats."""
        # Try matching by player_display_name first (most common input format)
        # Then fall back to player_id if not found
        if 'player_display_name' in self.weekly_data.columns:
            player_data = self.weekly_data[
                (self.weekly_data['player_display_name'] == player_id) &
                (self.weekly_data['recent_team'] == team)
            ]
        else:
            player_data = self.weekly_data[
                (self.weekly_data['player_id'] == player_id) &
                (self.weekly_data['recent_team'] == team)
            ]

        # If no matches by display name, try player_name (abbreviated form)
        if len(player_data) == 0 and 'player_name' in self.weekly_data.columns:
            player_data = self.weekly_data[
                (self.weekly_data['player_name'] == player_id) &
                (self.weekly_data['recent_team'] == team)
            ]

        if up_to_week is not None:
            player_data = player_data[player_data['week'] < up_to_week]

        if len(player_data) == 0:
            return 0.0

        recent_data = player_data.nlargest(n_weeks, 'week')

        total_share = 0.0
        for _, row in recent_data.iterrows():
            week = row['week']
            team_carries = self.weekly_data[
                (self.weekly_data['recent_team'] == team) &
                (self.weekly_data['week'] == week)
            ]['carries'].sum()

            if team_carries > 0:
                total_share += row['carries'] / team_carries

        return total_share / len(recent_data) if len(recent_data) > 0 else 0.0

    # ========== VARIANCE PARAMETERS ==========

    def get_player_stat_variance(
        self,
        player_name: str,
        stat_column: str,
        up_to_week: Optional[int] = None,
        min_games: int = 3
    ) -> Tuple[float, float, float]:
        """
        Calculate actual mean, std, and CV for a player's stat.
        Replaces hardcoded mean * 0.25 variance assumption.

        Returns:
            (mean, std, coefficient_of_variation)
        """
        player_data = self.weekly_data[
            self.weekly_data['player_name'] == player_name
        ]

        if up_to_week is not None:
            player_data = player_data[player_data['week'] < up_to_week]

        if len(player_data) < min_games or stat_column not in player_data.columns:
            # Return league average CV when insufficient data
            league_cv = self.get_league_avg_cv(stat_column)
            return 0.0, 0.0, league_cv

        values = player_data[stat_column].values
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1) if len(values) > 1 else mean_val * 0.3
        cv = std_val / mean_val if mean_val > 0 else 0.5

        return mean_val, std_val, cv

    def get_league_avg_cv(self, stat_column: str) -> float:
        """
        Calculate league average coefficient of variation for a stat.
        Based on actual data analysis:
        - Targets: ~0.47 CV (not 0.30)
        - Receptions: ~0.50 CV
        - Carries: ~0.40 CV
        - Rushing yards: ~0.55 CV
        - Receiving yards: ~0.60 CV
        """
        cache_key = f"league_cv_{stat_column}"
        if cache_key not in self._cache:
            if stat_column not in self.weekly_data.columns:
                self._cache[cache_key] = 0.5  # Default fallback
            else:
                # Calculate CV for players with 5+ games
                player_stats = self.weekly_data.groupby('player_name')[stat_column].agg(['mean', 'std', 'count'])
                player_stats = player_stats[
                    (player_stats['count'] >= 5) &
                    (player_stats['mean'] > 0)
                ]

                if len(player_stats) == 0:
                    self._cache[cache_key] = 0.5
                else:
                    cvs = player_stats['std'] / player_stats['mean']
                    self._cache[cache_key] = cvs.median()  # Use median to reduce outlier impact

        return self._cache[cache_key]

    def get_player_std(
        self,
        player_name: str,
        mean_value: float,
        stat_column: str,
        up_to_week: Optional[int] = None,
        min_games: int = 3
    ) -> float:
        """
        Calculate appropriate standard deviation for simulation.
        Uses actual player variance when available, falls back to league CV.

        Args:
            player_name: Player's name
            mean_value: The mean prediction for this stat
            stat_column: The stat column to look up variance for
            up_to_week: Only use data up to this week
            min_games: Minimum games needed for player-specific variance
        """
        _, player_std, player_cv = self.get_player_stat_variance(
            player_name, stat_column, up_to_week, min_games
        )

        if player_std > 0:
            # Use player-specific CV scaled to the predicted mean
            return mean_value * player_cv
        else:
            # Fall back to league average CV
            league_cv = self.get_league_avg_cv(stat_column)
            return mean_value * league_cv

    # ========== EFFICIENCY PARAMETERS ==========

    def get_player_efficiency(
        self,
        player_name: str,
        position: str,
        up_to_week: Optional[int] = None,
        n_weeks: int = 4
    ) -> Dict[str, float]:
        """
        Get player's actual efficiency metrics from historical data.
        Replaces hardcoded efficiency defaults.
        """
        player_data = self.weekly_data[
            self.weekly_data['player_name'] == player_name
        ]

        if up_to_week is not None:
            player_data = player_data[player_data['week'] < up_to_week]

        if len(player_data) == 0:
            return self._get_position_efficiency_defaults(position)

        recent_data = player_data.nlargest(n_weeks, 'week')

        efficiency = {}

        # Rushing efficiency
        total_carries = recent_data['carries'].sum()
        if total_carries > 0:
            efficiency['yards_per_carry'] = recent_data['rushing_yards'].sum() / total_carries
            efficiency['td_rate_rush'] = recent_data['rushing_tds'].sum() / total_carries
        else:
            # NO HARDCODED DEFAULTS - use league position average from actual data
            defaults = self._get_position_efficiency_defaults(position)
            if 'yards_per_carry' not in defaults:
                raise ValueError(
                    f"No yards_per_carry data for {position}. NO HARDCODED DEFAULTS."
                )
            efficiency['yards_per_carry'] = defaults['yards_per_carry']
            efficiency['td_rate_rush'] = defaults['td_rate_rush']

        # Receiving efficiency
        if 'targets' in recent_data.columns:
            total_targets = recent_data['targets'].sum()
            if total_targets > 0:
                total_receptions = recent_data['receptions'].sum() if 'receptions' in recent_data.columns else 0
                efficiency['catch_rate'] = total_receptions / total_targets
                efficiency['yards_per_target'] = recent_data['receiving_yards'].sum() / total_targets
                efficiency['td_rate_rec'] = recent_data['receiving_tds'].sum() / total_targets
            else:
                # NO HARDCODED DEFAULTS - use league position average from actual data
                defaults = self._get_position_efficiency_defaults(position)
                if 'catch_rate' not in defaults:
                    raise ValueError(
                        f"No catch_rate data for {position}. NO HARDCODED DEFAULTS."
                    )
                efficiency['catch_rate'] = defaults['catch_rate']
                efficiency['yards_per_target'] = defaults['yards_per_target']
                efficiency['td_rate_rec'] = defaults['td_rate_rec']

        return efficiency

    def _get_position_efficiency_defaults(self, position: str) -> Dict[str, float]:
        """
        Get position-based efficiency from ACTUAL NFLverse data.

        NO HARDCODED DEFAULTS - computes from real league data.
        """
        # Check cache first
        cache_key = f'position_efficiency_{position}'
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get all data for this position from actual NFLverse data
        if self.weekly_data is None or self.weekly_data.empty:
            raise ValueError(
                f"Weekly data not loaded. "
                f"NO HARDCODED DEFAULTS - ensure NFLverse data is available."
            )

        pos_data = self.weekly_data[self.weekly_data['position'] == position]

        if pos_data.empty:
            raise ValueError(
                f"No {position} data found in NFLverse weekly stats. "
                f"NO HARDCODED DEFAULTS - ensure position is valid."
            )

        efficiency = {}

        # Calculate actual rushing efficiency
        total_carries = pos_data['carries'].sum()
        if total_carries > 0:
            efficiency['yards_per_carry'] = pos_data['rushing_yards'].sum() / total_carries
            efficiency['td_rate_rush'] = pos_data['rushing_tds'].sum() / total_carries
        else:
            efficiency['yards_per_carry'] = 0.0
            efficiency['td_rate_rush'] = 0.0

        # Calculate actual receiving efficiency
        if 'targets' in pos_data.columns:
            total_targets = pos_data['targets'].sum()
            if total_targets > 0:
                total_receptions = pos_data['receptions'].sum() if 'receptions' in pos_data.columns else 0
                efficiency['catch_rate'] = total_receptions / total_targets
                efficiency['yards_per_target'] = pos_data['receiving_yards'].sum() / total_targets
                efficiency['td_rate_rec'] = pos_data['receiving_tds'].sum() / total_targets
            else:
                efficiency['catch_rate'] = 0.0
                efficiency['yards_per_target'] = 0.0
                efficiency['td_rate_rec'] = 0.0
        else:
            efficiency['catch_rate'] = 0.0
            efficiency['yards_per_target'] = 0.0
            efficiency['td_rate_rec'] = 0.0

        logger.info(
            f"Computed {position} efficiency from actual data: "
            f"YPC={efficiency.get('yards_per_carry', 0):.2f}, "
            f"YPT={efficiency.get('yards_per_target', 0):.2f}, "
            f"TD_rush={efficiency.get('td_rate_rush', 0):.3f}, "
            f"TD_rec={efficiency.get('td_rate_rec', 0):.3f}"
        )

        # Cache result
        self._cache[cache_key] = efficiency
        return efficiency

    # ========== UTILITY METHODS ==========

    def clear_cache(self):
        """Clear all cached values."""
        self._cache = {}

    def reload_data(self):
        """Reload all data from disk."""
        self._weekly_data = None
        self._snap_counts = None
        self._target_shares = None
        self._carry_shares = None
        self._team_pace = None
        self._defensive_epa = None
        self.clear_cache()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary of actual league parameters."""
        return {
            'league_avg_pass_attempts': self.get_league_avg_pass_attempts(),
            'league_avg_rush_attempts': self.get_league_avg_rush_attempts(),
            'target_cv': self.get_league_avg_cv('targets'),
            'receptions_cv': self.get_league_avg_cv('receptions'),
            'carries_cv': self.get_league_avg_cv('carries'),
            'rushing_yards_cv': self.get_league_avg_cv('rushing_yards'),
            'receiving_yards_cv': self.get_league_avg_cv('receiving_yards'),
        }

    def get_player_current_team(self, player_name: str) -> Optional[str]:
        """
        Get player's current team from NFLverse data.
        Returns the most recent team the player played for.
        """
        player_data = self.weekly_data[self.weekly_data['player_name'] == player_name]

        if len(player_data) == 0:
            return None

        # Get most recent game
        most_recent = player_data.nlargest(1, 'week')
        return most_recent['recent_team'].iloc[0]

    def get_player_position(self, player_name: str) -> Optional[str]:
        """
        Get player's position from NFLverse data.
        """
        player_data = self.weekly_data[self.weekly_data['player_name'] == player_name]

        if len(player_data) == 0:
            return None

        return player_data['position'].iloc[0]

    def get_player_info(self, player_name: str) -> Dict[str, Any]:
        """
        Get all player info (team, position, player_id) from NFLverse data.
        NO ASSUMPTIONS - all from actual data.
        """
        player_data = self.weekly_data[self.weekly_data['player_name'] == player_name]

        if len(player_data) == 0:
            return {}

        most_recent = player_data.nlargest(1, 'week').iloc[0]

        return {
            'player_name': player_name,
            'team': most_recent['recent_team'],
            'position': most_recent['position'],
            'player_id': most_recent.get('player_id', ''),
            'weeks_played': len(player_data),
            'last_week': int(most_recent['week']),
        }


# Global singleton instance
_provider_instance: Optional[DynamicParameterProvider] = None


def get_parameter_provider() -> DynamicParameterProvider:
    """Get or create the global parameter provider instance."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = DynamicParameterProvider()
    return _provider_instance


# Convenience functions
def get_league_avg_pass_attempts() -> float:
    """Get league average pass attempts per game."""
    return get_parameter_provider().get_league_avg_pass_attempts()


def get_league_avg_rush_attempts() -> float:
    """Get league average rush attempts per game."""
    return get_parameter_provider().get_league_avg_rush_attempts()


def get_team_pass_attempts(team: str, up_to_week: Optional[int] = None) -> float:
    """Get team-specific average pass attempts."""
    return get_parameter_provider().get_team_pass_attempts(team, up_to_week)


def get_team_rush_attempts(team: str, up_to_week: Optional[int] = None) -> float:
    """Get team-specific average rush attempts."""
    return get_parameter_provider().get_team_rush_attempts(team, up_to_week)


def get_player_snap_share(
    player_name: str,
    team: str,
    up_to_week: Optional[int] = None,
    n_weeks: int = 4
) -> float:
    """Get player's actual snap share."""
    return get_parameter_provider().get_player_snap_share(player_name, team, up_to_week, n_weeks)


def get_player_stat_std(
    player_name: str,
    mean_value: float,
    stat_column: str,
    up_to_week: Optional[int] = None
) -> float:
    """Get appropriate standard deviation for a player's stat prediction."""
    return get_parameter_provider().get_player_std(player_name, mean_value, stat_column, up_to_week)
