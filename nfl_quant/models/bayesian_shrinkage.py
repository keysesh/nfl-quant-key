"""
Bayesian Shrinkage / James-Stein Estimator
===========================================

Handles small-sample adjustments by shrinking individual player estimates
toward league/position priors when sample size is limited.

Theory:
-------
With limited data, player-specific estimates have high variance. Bayesian
shrinkage improves estimates by blending:
- Individual player sample mean (high variance, unbiased)
- League/position prior (low variance, stable)

Weight allocation depends on sample size:
- Small samples (n < 30): Shrink heavily toward prior (70-90%)
- Medium samples (30 <= n < 100): Balanced shrinkage (30-70%)
- Large samples (n >= 100): Trust player-specific estimate (10-30%)

Applications:
-------------
1. Early season (2025 weeks 1-4): Players have 1-4 games
2. Backup QBs/RBs: Limited snaps even late in season
3. Injured players returning: Old data is stale
4. Rookies: No NFL history

Mathematical Formula:
---------------------
shrunk_estimate = w * player_mean + (1 - w) * prior_mean

where:
    w = n / (n + k)
    n = sample size
    k = shrinkage parameter (controls aggressiveness)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BayesianShrinker:
    """
    Implements Bayesian shrinkage for player statistics.
    """

    # Shrinkage parameters by statistic type
    # Higher k = more aggressive shrinkage (trust prior more)
    SHRINKAGE_PARAMS = {
        'yards_per_carry': 50,      # RB rushing efficiency
        'yards_per_target': 40,     # WR/RB receiving efficiency
        'yards_per_completion': 40, # QB passing efficiency
        'comp_pct': 60,             # QB completion%
        'td_rate': 80,              # TD rates (high variance)
        'usage': 30,                # Snap share, target share (more stable)
    }

    def __init__(self, position: str = None):
        """
        Initialize shrinkage estimator.

        Args:
            position: Player position (QB, RB, WR) for position-specific priors
        """
        self.position = position
        self.priors = {}  # Will be populated from historical data

    def set_prior(self, stat_name: str, mean: float, std: float):
        """
        Set league/position prior for a statistic.

        Args:
            stat_name: Statistic name (e.g., 'yards_per_carry')
            mean: Prior mean (e.g., league average)
            std: Prior standard deviation (measure of typical variance)
        """
        self.priors[stat_name] = {
            'mean': mean,
            'std': std,
        }
        logger.debug(f"Set prior for {stat_name}: μ={mean:.3f}, σ={std:.3f}")

    def calculate_shrinkage_weight(
        self,
        sample_size: int,
        stat_name: str
    ) -> float:
        """
        Calculate shrinkage weight based on sample size.

        Args:
            sample_size: Number of observations for player
            stat_name: Statistic name (determines k parameter)

        Returns:
            Weight for player-specific estimate (0 = all prior, 1 = all player)
        """
        k = self.SHRINKAGE_PARAMS.get(stat_name, 50)  # Default k=50
        weight = sample_size / (sample_size + k)
        return weight

    def shrink_estimate(
        self,
        player_mean: float,
        player_sample_size: int,
        stat_name: str,
        prior_mean: float = None
    ) -> Tuple[float, float, Dict]:
        """
        Apply Bayesian shrinkage to player estimate.

        Args:
            player_mean: Player's sample mean
            player_sample_size: Number of observations
            stat_name: Statistic name
            prior_mean: Override prior (uses self.priors if None)

        Returns:
            (shrunk_estimate, weight_used, metadata)
        """
        # Get prior
        if prior_mean is None:
            if stat_name not in self.priors:
                logger.warning(f"No prior set for {stat_name}, returning player mean")
                return player_mean, 1.0, {'shrinkage_applied': False}
            prior_mean = self.priors[stat_name]['mean']

        # Calculate shrinkage weight
        weight = self.calculate_shrinkage_weight(player_sample_size, stat_name)

        # Apply shrinkage
        shrunk = weight * player_mean + (1 - weight) * prior_mean

        # Metadata
        metadata = {
            'shrinkage_applied': True,
            'player_mean': player_mean,
            'prior_mean': prior_mean,
            'shrunk_mean': shrunk,
            'weight': weight,
            'sample_size': player_sample_size,
            'shrinkage_amount': abs(shrunk - player_mean),
        }

        return shrunk, weight, metadata

    def shrink_dataframe(
        self,
        df: pd.DataFrame,
        stat_col: str,
        sample_size_col: str,
        stat_name: str,
        output_col: str = None
    ) -> pd.DataFrame:
        """
        Apply Bayesian shrinkage to entire DataFrame.

        Args:
            df: DataFrame with player statistics
            stat_col: Column containing player means
            sample_size_col: Column containing sample sizes (e.g., 'games_played')
            stat_name: Statistic type (for shrinkage parameter)
            output_col: Output column name (defaults to stat_col + '_shrunk')

        Returns:
            DataFrame with shrunk estimates added
        """
        if output_col is None:
            output_col = f"{stat_col}_shrunk"

        # Get prior
        if stat_name not in self.priors:
            logger.warning(f"No prior set for {stat_name}, skipping shrinkage")
            df[output_col] = df[stat_col]
            return df

        prior_mean = self.priors[stat_name]['mean']

        # Apply shrinkage row by row
        results = []
        for _, row in df.iterrows():
            player_mean = row[stat_col]
            sample_size = row[sample_size_col]

            shrunk, weight, metadata = self.shrink_estimate(
                player_mean=player_mean,
                player_sample_size=sample_size,
                stat_name=stat_name,
                prior_mean=prior_mean
            )

            results.append({
                output_col: shrunk,
                f"{output_col}_weight": weight,
            })

        # Add shrunk estimates to DataFrame
        results_df = pd.DataFrame(results)
        df = pd.concat([df.reset_index(drop=True), results_df], axis=1)

        return df


def load_position_priors(position: str, season: int = 2025) -> Dict[str, Dict]:
    """
    Load league/position priors from ACTUAL historical NFLverse data.

    NO HARDCODED DEFAULTS - computes priors from real data.

    Args:
        position: Player position (QB, RB, WR, TE)
        season: Historical season to compute priors from (uses season-1)

    Returns:
        Dictionary of priors: {stat_name: {'mean': float, 'std': float}}
    """
    from pathlib import Path

    logger.info(f"Loading {position} priors from {season-1} season actual data...")

    project_root = Path(__file__).parent.parent.parent
    stats_path = project_root / 'data' / 'nflverse' / 'weekly_stats.parquet'

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Weekly stats not found at {stats_path}. "
            f"NO HARDCODED DEFAULTS - run data/fetch_nflverse_r.R to get actual data."
        )

    weekly = pd.read_parquet(stats_path)

    # Use previous season for priors (more complete data)
    prior_season = season - 1
    season_data = weekly[weekly['season'] == prior_season]

    if season_data.empty:
        raise ValueError(
            f"No data found for season {prior_season}. "
            f"NO HARDCODED DEFAULTS - ensure NFLverse data includes {prior_season} season."
        )

    priors = {}

    if position == 'QB':
        qb_data = season_data[season_data['position'] == 'QB']

        if qb_data.empty:
            raise ValueError(f"No QB data found for season {prior_season}.")

        # Completion percentage: completions / attempts
        qb_with_attempts = qb_data[qb_data['attempts'] > 0]
        if len(qb_with_attempts) == 0:
            raise ValueError(f"No QB passing attempts found for season {prior_season}.")

        comp_pct_vals = qb_with_attempts['completions'] / qb_with_attempts['attempts']
        priors['comp_pct'] = {
            'mean': float(comp_pct_vals.mean()),
            'std': float(comp_pct_vals.std())
        }

        # Yards per completion
        qb_with_completions = qb_data[qb_data['completions'] > 0]
        if len(qb_with_completions) > 0:
            ypc_vals = qb_with_completions['passing_yards'] / qb_with_completions['completions']
            priors['yards_per_completion'] = {
                'mean': float(ypc_vals.mean()),
                'std': float(ypc_vals.std())
            }

        # Yards per carry (QB rushing)
        qb_with_carries = qb_data[qb_data['carries'] > 0]
        if len(qb_with_carries) > 0:
            ypc_rush_vals = qb_with_carries['rushing_yards'] / qb_with_carries['carries']
            priors['yards_per_carry'] = {
                'mean': float(ypc_rush_vals.mean()),
                'std': float(ypc_rush_vals.std())
            }

        # TD rates
        if len(qb_with_attempts) > 0:
            td_rate_pass_vals = qb_with_attempts['passing_tds'] / qb_with_attempts['attempts']
            priors['td_rate_pass'] = {
                'mean': float(td_rate_pass_vals.mean()),
                'std': float(td_rate_pass_vals.std())
            }

        if len(qb_with_carries) > 0:
            td_rate_rush_vals = qb_with_carries['rushing_tds'] / qb_with_carries['carries']
            priors['td_rate_rush'] = {
                'mean': float(td_rate_rush_vals.mean()),
                'std': float(td_rate_rush_vals.std())
            }

    elif position == 'RB':
        rb_data = season_data[season_data['position'] == 'RB']

        if rb_data.empty:
            raise ValueError(f"No RB data found for season {prior_season}.")

        # Yards per carry
        rb_with_carries = rb_data[rb_data['carries'] > 0]
        if len(rb_with_carries) == 0:
            raise ValueError(f"No RB carries found for season {prior_season}.")

        ypc_vals = rb_with_carries['rushing_yards'] / rb_with_carries['carries']
        priors['yards_per_carry'] = {
            'mean': float(ypc_vals.mean()),
            'std': float(ypc_vals.std())
        }

        # Yards per target (receiving)
        rb_with_targets = rb_data[rb_data['targets'] > 0]
        if len(rb_with_targets) > 0:
            ypt_vals = rb_with_targets['receiving_yards'] / rb_with_targets['targets']
            priors['yards_per_target'] = {
                'mean': float(ypt_vals.mean()),
                'std': float(ypt_vals.std())
            }

        # TD rates
        td_rate_rush_vals = rb_with_carries['rushing_tds'] / rb_with_carries['carries']
        priors['td_rate_rush'] = {
            'mean': float(td_rate_rush_vals.mean()),
            'std': float(td_rate_rush_vals.std())
        }

        if len(rb_with_targets) > 0:
            td_rate_pass_vals = rb_with_targets['receiving_tds'] / rb_with_targets['targets']
            priors['td_rate_pass'] = {
                'mean': float(td_rate_pass_vals.mean()),
                'std': float(td_rate_pass_vals.std())
            }

    elif position in ['WR', 'TE']:
        wr_te_data = season_data[season_data['position'] == position]

        if wr_te_data.empty:
            raise ValueError(f"No {position} data found for season {prior_season}.")

        # Yards per target
        with_targets = wr_te_data[wr_te_data['targets'] > 0]
        if len(with_targets) == 0:
            raise ValueError(f"No {position} targets found for season {prior_season}.")

        ypt_vals = with_targets['receiving_yards'] / with_targets['targets']
        priors['yards_per_target'] = {
            'mean': float(ypt_vals.mean()),
            'std': float(ypt_vals.std())
        }

        # TD rate
        td_rate_vals = with_targets['receiving_tds'] / with_targets['targets']
        priors['td_rate_pass'] = {
            'mean': float(td_rate_vals.mean()),
            'std': float(td_rate_vals.std())
        }

    else:
        raise ValueError(
            f"Unknown position: {position}. "
            f"NO HARDCODED DEFAULTS - position must be QB, RB, WR, or TE."
        )

    logger.info(f"Computed {position} priors from {prior_season} actual data: {list(priors.keys())}")
    return priors


# Example usage
if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("="*80)
    print("BAYESIAN SHRINKAGE DEMONSTRATION")
    print("="*80)

    # Create shrinkage estimator for RBs
    shrinker = BayesianShrinker(position='RB')

    # Load priors
    priors = load_position_priors('RB', season=2024)
    for stat_name, prior in priors.items():
        shrinker.set_prior(stat_name, prior['mean'], prior['std'])

    print("\nRB Position Priors:")
    for stat_name, prior in priors.items():
        print(f"  {stat_name}: μ={prior['mean']:.3f}, σ={prior['std']:.3f}")

    # Example: Backup RB with limited sample
    print("\n" + "="*80)
    print("EXAMPLE 1: Backup RB (2 games, 8 carries, 6.5 YPC)")
    print("="*80)

    player_ypc = 6.5  # Very high YPC (small sample luck)
    sample_size = 8   # Only 8 carries

    shrunk, weight, metadata = shrinker.shrink_estimate(
        player_mean=player_ypc,
        player_sample_size=sample_size,
        stat_name='yards_per_carry'
    )

    print(f"\nPlayer Sample: {player_ypc:.2f} YPC (n={sample_size})")
    print(f"League Prior: {priors['yards_per_carry']['mean']:.2f} YPC")
    print(f"Shrinkage Weight: {weight:.2%} (player) / {1-weight:.2%} (prior)")
    print(f"Shrunk Estimate: {shrunk:.2f} YPC")
    print(f"Adjustment: {metadata['shrinkage_amount']:.2f} YPC pulled toward prior")

    # Example: Established starter
    print("\n" + "="*80)
    print("EXAMPLE 2: Established Starter (8 games, 120 carries, 4.8 YPC)")
    print("="*80)

    player_ypc = 4.8
    sample_size = 120

    shrunk, weight, metadata = shrinker.shrink_estimate(
        player_mean=player_ypc,
        player_sample_size=sample_size,
        stat_name='yards_per_carry'
    )

    print(f"\nPlayer Sample: {player_ypc:.2f} YPC (n={sample_size})")
    print(f"League Prior: {priors['yards_per_carry']['mean']:.2f} YPC")
    print(f"Shrinkage Weight: {weight:.2%} (player) / {1-weight:.2%} (prior)")
    print(f"Shrunk Estimate: {shrunk:.2f} YPC")
    print(f"Adjustment: {metadata['shrinkage_amount']:.2f} YPC (minimal shrinkage)")

    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("\nBayesian shrinkage automatically adjusts for sample size:")
    print("  • Small samples → Shrink heavily toward prior (avoid overfitting)")
    print("  • Large samples → Trust player-specific estimate (capture true skill)")
    print("\nThis is critical for early-season predictions and backup players!")
