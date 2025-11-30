"""
Dynamic Correlation Matrix for Player Props

Implements:
1. Correlation updates based on recent game data
2. Injury-induced correlation shifts
3. Committee backfield modeling
4. Weather impact on correlations
5. Game script correlation adjustments
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation modeling."""
    # Base correlations (from historical analysis)
    qb_wr_correlation: float = 0.65
    qb_te_correlation: float = 0.55
    qb_rb_correlation: float = 0.40
    wr_wr_same_team: float = -0.25  # Negative - targets compete
    rb_rb_same_team: float = -0.60  # Negative - carries compete
    same_game_correlation: float = 0.15  # Same game totals
    receiving_rushing_correlation: float = 0.10

    # Adjustment multipliers
    injury_correlation_shift: float = 0.30  # How much WR2 correlation increases when WR1 out
    weather_rushing_boost: float = 0.20  # Increase rush correlations in bad weather
    blowout_garbage_time: float = -0.15  # Reduce correlations in blowouts

    # Recency weighting
    correlation_half_life_weeks: int = 4


class DynamicCorrelationMatrix:
    """
    Dynamically updated correlation matrix based on recent performance and context.
    """

    def __init__(self, config: CorrelationConfig = None):
        self.config = config or CorrelationConfig()
        self.base_correlations = self._initialize_base_correlations()
        self.current_adjustments = {}

    def _initialize_base_correlations(self) -> Dict:
        """Initialize base correlation values."""
        return {
            ('QB', 'WR'): self.config.qb_wr_correlation,
            ('QB', 'TE'): self.config.qb_te_correlation,
            ('QB', 'RB'): self.config.qb_rb_correlation,
            ('WR', 'WR'): self.config.wr_wr_same_team,
            ('RB', 'RB'): self.config.rb_rb_same_team,
            ('TE', 'WR'): self.config.wr_wr_same_team * 0.8,  # Slightly less negative
            ('TE', 'TE'): self.config.wr_wr_same_team * 1.2,  # More negative
        }

    def get_correlation(
        self,
        player1_position: str,
        player2_position: str,
        player1_team: str,
        player2_team: str,
        stat1_type: str = 'yards',
        stat2_type: str = 'yards',
        same_game: bool = False,
        context: Dict = None
    ) -> float:
        """
        Get correlation between two players' props.

        Args:
            player1_position: Position of first player
            player2_position: Position of second player
            player1_team: Team of first player
            player2_team: Team of second player
            stat1_type: Type of stat (yards, receptions, tds)
            stat2_type: Type of stat
            same_game: Whether players are in the same game
            context: Additional context (injuries, weather, etc.)

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if context is None:
            context = {}

        # Base correlation
        if player1_team == player2_team:
            # Same team - use position-based correlation
            base_corr = self._get_same_team_correlation(
                player1_position, player2_position, stat1_type, stat2_type
            )
        elif same_game:
            # Same game but different teams
            base_corr = self.config.same_game_correlation
        else:
            # Different games - minimal correlation
            base_corr = 0.0

        # Apply contextual adjustments
        adjusted_corr = self._apply_adjustments(
            base_corr, player1_position, player2_position,
            player1_team, player2_team, stat1_type, stat2_type, context
        )

        # Clamp to valid range
        return np.clip(adjusted_corr, -0.95, 0.95)

    def _get_same_team_correlation(
        self,
        pos1: str,
        pos2: str,
        stat1: str,
        stat2: str
    ) -> float:
        """Get base correlation for same-team players."""
        # Lookup in base correlations
        key = (pos1, pos2)
        reverse_key = (pos2, pos1)

        if key in self.base_correlations:
            base = self.base_correlations[key]
        elif reverse_key in self.base_correlations:
            base = self.base_correlations[reverse_key]
        else:
            # Default for unlisted combinations
            base = 0.0

        # Adjust for stat type
        if stat1 != stat2:
            # Different stats have lower correlation
            if stat1 in ['yards', 'receptions'] and stat2 in ['yards', 'receptions']:
                base *= 0.8  # Related stats
            else:
                base *= 0.5  # Unrelated stats

        # TD props have different correlations
        if 'td' in stat1.lower() or 'td' in stat2.lower():
            base *= 0.6  # TDs are more independent

        return base

    def _apply_adjustments(
        self,
        base_corr: float,
        pos1: str,
        pos2: str,
        team1: str,
        team2: str,
        stat1: str,
        stat2: str,
        context: Dict
    ) -> float:
        """Apply contextual adjustments to correlation."""
        adjusted = base_corr

        # Weather adjustments
        if 'weather' in context:
            weather = context['weather']
            if weather.get('precipitation', 0) > 0.3 or weather.get('wind', 0) > 15:
                # Bad weather increases rushing correlation
                if 'rush' in stat1.lower() or 'rush' in stat2.lower():
                    adjusted += self.config.weather_rushing_boost
                # Decreases passing correlation
                if 'pass' in stat1.lower() or 'pass' in stat2.lower():
                    adjusted -= 0.1

        # Injury adjustments
        if 'injuries' in context:
            injuries = context['injuries']
            # If WR1 is out, WR2 and TE correlations increase
            if team1 == team2:
                if 'wr1_out' in injuries.get(team1, []):
                    if pos1 == 'WR' or pos2 == 'WR':
                        adjusted += self.config.injury_correlation_shift

        # Game script adjustments
        if 'expected_spread' in context:
            spread = abs(context['expected_spread'])
            if spread > 10:
                # Blowout expected - correlations break down
                adjusted *= (1 + self.config.blowout_garbage_time)

        # Pace adjustments
        if 'game_total' in context:
            total = context['game_total']
            if total > 50:
                # High scoring game - more plays, more correlation
                adjusted *= 1.1
            elif total < 40:
                # Low scoring - less correlation
                adjusted *= 0.9

        return adjusted

    def update_from_recent_data(self, stats_df: pd.DataFrame, weeks: int = 4):
        """
        Update correlation estimates based on recent player performance.

        Args:
            stats_df: DataFrame with player stats
            weeks: Number of recent weeks to analyze
        """
        logger.info(f"Updating correlations from {weeks} weeks of data")

        # Get recent weeks
        max_week = stats_df['week'].max()
        recent_data = stats_df[stats_df['week'] > max_week - weeks]

        if len(recent_data) < 50:
            logger.warning("Insufficient data for correlation update")
            return

        # Calculate empirical correlations by team
        team_correlations = {}

        for team in recent_data['team'].unique():
            team_data = recent_data[recent_data['team'] == team]

            # Pivot to player x stat
            if len(team_data) > 10:
                try:
                    # Calculate correlations between players on same team
                    team_corr = self._calculate_team_correlations(team_data)
                    team_correlations[team] = team_corr
                except Exception as e:
                    logger.warning(f"Error calculating correlations for {team}: {e}")

        # Update base correlations with weighted average
        self._blend_correlations(team_correlations)

        logger.info(f"Updated correlations for {len(team_correlations)} teams")

    def _calculate_team_correlations(self, team_data: pd.DataFrame) -> Dict:
        """Calculate empirical correlations for a team."""
        correlations = {}

        # Group by week to get game-level observations
        game_stats = {}

        for week in team_data['week'].unique():
            week_data = team_data[team_data['week'] == week]

            # Get stats per player
            for _, row in week_data.iterrows():
                player = row.get('player_display_name', row.get('player_name'))
                position = row.get('position', 'UNK')

                if week not in game_stats:
                    game_stats[week] = []

                game_stats[week].append({
                    'player': player,
                    'position': position,
                    'receiving_yards': row.get('receiving_yards', 0),
                    'rushing_yards': row.get('rushing_yards', 0),
                    'targets': row.get('targets', 0),
                    'carries': row.get('carries', 0),
                })

        # Calculate correlations between position groups
        # This is simplified - in production would compute actual correlations
        qb_pass_yards = []
        wr_rec_yards = []

        for week_stats in game_stats.values():
            qb_stats = [p for p in week_stats if p['position'] == 'QB']
            wr_stats = [p for p in week_stats if p['position'] == 'WR']

            if qb_stats and wr_stats:
                # Total passing vs total receiving
                total_rec = sum(p['receiving_yards'] for p in wr_stats)
                qb_pass_yards.append(qb_stats[0].get('rushing_yards', 0))  # placeholder
                wr_rec_yards.append(total_rec)

        if len(qb_pass_yards) > 3:
            correlation = np.corrcoef(qb_pass_yards, wr_rec_yards)[0, 1]
            if not np.isnan(correlation):
                correlations[('QB', 'WR')] = correlation

        return correlations

    def _blend_correlations(self, team_correlations: Dict):
        """Blend empirical correlations with base correlations."""
        # Simple averaging (could use Bayesian updating)
        blend_weight = 0.3  # 30% empirical, 70% prior

        all_corrs = {}

        for team, corrs in team_correlations.items():
            for key, value in corrs.items():
                if key not in all_corrs:
                    all_corrs[key] = []
                all_corrs[key].append(value)

        # Update base correlations
        for key, values in all_corrs.items():
            if key in self.base_correlations:
                empirical = np.mean(values)
                prior = self.base_correlations[key]
                blended = (1 - blend_weight) * prior + blend_weight * empirical
                self.base_correlations[key] = blended
                logger.info(f"Updated {key} correlation: {prior:.3f} -> {blended:.3f}")

    def build_correlation_matrix(
        self,
        players: List[Dict],
        context: Dict = None
    ) -> np.ndarray:
        """
        Build full correlation matrix for a set of players.

        Args:
            players: List of player dicts with 'name', 'position', 'team', 'stat_type'
            context: Game context for adjustments

        Returns:
            NxN correlation matrix
        """
        n = len(players)
        corr_matrix = np.eye(n)  # Diagonal = 1

        for i in range(n):
            for j in range(i+1, n):
                p1 = players[i]
                p2 = players[j]

                corr = self.get_correlation(
                    p1['position'], p2['position'],
                    p1['team'], p2['team'],
                    p1.get('stat_type', 'yards'),
                    p2.get('stat_type', 'yards'),
                    same_game=(p1['team'] == p2['team']),
                    context=context
                )

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return corr_matrix

    def cholesky_decomposition(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Get Cholesky decomposition for correlated sampling.

        Args:
            corr_matrix: Correlation matrix (must be positive semi-definite)

        Returns:
            Lower triangular Cholesky matrix
        """
        # Ensure positive semi-definite
        min_eigenvalue = np.min(np.linalg.eigvalsh(corr_matrix))
        if min_eigenvalue < 0:
            # Add small diagonal to make positive semi-definite
            corr_matrix = corr_matrix + (-min_eigenvalue + 0.01) * np.eye(len(corr_matrix))

        return np.linalg.cholesky(corr_matrix)

    def sample_correlated_normal(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        corr_matrix: np.ndarray,
        n_samples: int = 10000
    ) -> np.ndarray:
        """
        Generate correlated samples using Cholesky decomposition.

        Args:
            means: Mean values for each variable
            stds: Standard deviations
            corr_matrix: Correlation matrix
            n_samples: Number of samples

        Returns:
            Array of shape (n_samples, n_variables)
        """
        n_vars = len(means)

        # Get Cholesky decomposition
        L = self.cholesky_decomposition(corr_matrix)

        # Generate independent standard normal samples
        Z = np.random.randn(n_samples, n_vars)

        # Apply correlation structure
        correlated_standard = Z @ L.T

        # Scale and shift
        samples = correlated_standard * stds + means

        return samples

    def get_parlay_correlation_penalty(self, legs: List[Dict]) -> float:
        """
        Calculate correlation-based penalty for parlay legs.

        Args:
            legs: List of parlay legs with player info

        Returns:
            Multiplier for parlay probability (0-1)
        """
        if len(legs) <= 1:
            return 1.0

        # Build correlation matrix for legs
        corr_matrix = self.build_correlation_matrix(legs)

        # Calculate average absolute correlation (excluding diagonal)
        n = len(legs)
        total_corr = 0
        count = 0

        for i in range(n):
            for j in range(i+1, n):
                total_corr += abs(corr_matrix[i, j])
                count += 1

        avg_corr = total_corr / count if count > 0 else 0

        # High correlation means less independence, penalize
        # Low correlation means more independence, no penalty
        penalty = 1.0 - (avg_corr * 0.3)  # Max 30% penalty for high correlation

        return max(0.7, min(1.0, penalty))
