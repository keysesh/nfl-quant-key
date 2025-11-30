"""
V4 Output Schemas with Full Distributional Information.

V4 enhancements over V3:
- Percentiles: 5th, 25th, 50th (median), 75th, 95th
- Multiple stats per player (not just single prop_type)
- Distribution properties: CV, IQR
- Backward compatible with V3 schemas
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class V4StatDistribution(BaseModel):
    """
    V4 statistical distribution for a single stat (e.g., receiving_yards).

    Replaces V3's single prop_type with detailed distributional info.
    """
    # Point estimates
    mean: float = Field(..., description="Mean of distribution")
    median: float = Field(..., description="Median (50th percentile)")
    std: float = Field(..., description="Standard deviation")

    # Percentiles for upside/downside analysis
    p5: float = Field(..., description="5th percentile (downside risk)")
    p25: float = Field(..., description="25th percentile (lower bound)")
    p75: float = Field(..., description="75th percentile (upper bound)")
    p95: float = Field(..., description="95th percentile (upside potential)")

    # Distribution properties
    cv: float = Field(..., description="Coefficient of variation (std/mean)")
    iqr: float = Field(..., description="Interquartile range (p75 - p25)")

    class Config:
        frozen = True


class PlayerPropOutputV4(BaseModel):
    """
    V4 Player Prop Output with full distributional information.

    Supports multiple stats per player (QB: passing_yards, passing_tds, etc.)
    """
    # Player identification
    player_id: str
    player_name: str
    position: str
    team: str
    week: int

    # Simulation metadata
    trial_count: int = Field(default=10000, description="Number of Monte Carlo trials")
    seed: Optional[int] = Field(default=None, description="Random seed used")

    # Stat distributions (position-specific)
    # QB
    attempts: Optional[V4StatDistribution] = None
    completions: Optional[V4StatDistribution] = None
    passing_yards: Optional[V4StatDistribution] = None
    passing_tds: Optional[V4StatDistribution] = None
    interceptions: Optional[V4StatDistribution] = None

    # RB
    carries: Optional[V4StatDistribution] = None
    rushing_yards: Optional[V4StatDistribution] = None
    rushing_tds: Optional[V4StatDistribution] = None

    # WR/TE/RB receiving
    targets: Optional[V4StatDistribution] = None
    receptions: Optional[V4StatDistribution] = None
    receiving_yards: Optional[V4StatDistribution] = None
    receiving_tds: Optional[V4StatDistribution] = None

    # Calibrated probabilities (if prop lines provided)
    prop_lines: Optional[Dict[str, float]] = Field(
        default=None,
        description="Sportsbook prop lines (e.g., {'receiving_yards': 45.5})"
    )
    over_probs_raw: Optional[Dict[str, float]] = Field(
        default=None,
        description="Raw P(over) before calibration"
    )
    over_probs_calibrated: Optional[Dict[str, float]] = Field(
        default=None,
        description="Calibrated P(over) after isotonic calibration"
    )

    class Config:
        frozen = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        result = {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'position': self.position,
            'team': self.team,
            'week': self.week,
            'trial_count': self.trial_count,
            'seed': self.seed,
        }

        # Add all stat distributions
        for stat_name in [
            'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions',
            'carries', 'rushing_yards', 'rushing_tds',
            'targets', 'receptions', 'receiving_yards', 'receiving_tds'
        ]:
            stat_dist = getattr(self, stat_name, None)
            if stat_dist is not None:
                # Flatten distribution into columns
                result[f'{stat_name}_mean'] = stat_dist.mean
                result[f'{stat_name}_median'] = stat_dist.median
                result[f'{stat_name}_std'] = stat_dist.std
                result[f'{stat_name}_p5'] = stat_dist.p5
                result[f'{stat_name}_p25'] = stat_dist.p25
                result[f'{stat_name}_p75'] = stat_dist.p75
                result[f'{stat_name}_p95'] = stat_dist.p95
                result[f'{stat_name}_cv'] = stat_dist.cv
                result[f'{stat_name}_iqr'] = stat_dist.iqr

        # Add calibrated probabilities
        if self.prop_lines:
            for stat_name, line in self.prop_lines.items():
                result[f'{stat_name}_line'] = line
                if self.over_probs_raw:
                    result[f'{stat_name}_over_prob_raw'] = self.over_probs_raw.get(stat_name)
                if self.over_probs_calibrated:
                    result[f'{stat_name}_over_prob_cal'] = self.over_probs_calibrated.get(stat_name)

        return result


class PlayerComparisonV4(BaseModel):
    """
    V4 Player Comparison (V4 vs V3 validation).

    Used for validating V4 improvements over V3.
    """
    player_id: str
    player_name: str
    position: str
    week: int

    # V3 predictions
    v3_mean: float
    v3_std: float

    # V4 predictions
    v4_mean: float
    v4_median: float
    v4_std: float
    v4_p5: float
    v4_p25: float
    v4_p75: float
    v4_p95: float

    # Actual outcome (for validation)
    actual: Optional[float] = None

    # Error metrics
    v3_error: Optional[float] = None  # |v3_mean - actual|
    v4_error: Optional[float] = None  # |v4_median - actual|
    improvement_pct: Optional[float] = None  # (v3_error - v4_error) / v3_error

    class Config:
        frozen = True


# Conversion utilities for backward compatibility

def v4_to_v3_output(v4_output: PlayerPropOutputV4, stat_name: str) -> Dict:
    """
    Convert V4 output to V3 format for a specific stat.

    Args:
        v4_output: V4 player prop output
        stat_name: Stat to extract (e.g., 'receiving_yards')

    Returns:
        Dictionary with V3-compatible fields

    Example:
        >>> v4_out = PlayerPropOutputV4(...)
        >>> v3_dict = v4_to_v3_output(v4_out, 'receiving_yards')
        >>> v3_dict['mean_stat']
        61.2
    """
    stat_dist = getattr(v4_output, stat_name, None)

    if stat_dist is None:
        raise ValueError(f"Stat '{stat_name}' not found in V4 output")

    return {
        'player_id': v4_output.player_id,
        'player_name': v4_output.player_name,
        'position': v4_output.position,
        'prop_type': stat_name,
        'trial_count': v4_output.trial_count,
        'seed': v4_output.seed or 42,
        'median_stat': stat_dist.median,
        'mean_stat': stat_dist.mean,
        'std_stat': stat_dist.std,
        'p10_stat': stat_dist.p5,  # V3 used p10, V4 uses p5 (more conservative)
        'p90_stat': stat_dist.p95,  # V3 used p90, V4 uses p95 (wider range)
        # Prop line probabilities (if available)
        'prop_line': v4_output.prop_lines.get(stat_name) if v4_output.prop_lines else None,
        'over_prob_raw': v4_output.over_probs_raw.get(stat_name) if v4_output.over_probs_raw else None,
        'over_prob_calibrated': v4_output.over_probs_calibrated.get(stat_name) if v4_output.over_probs_calibrated else None,
    }


def v3_to_v4_stat_dist(v3_output: Dict) -> V4StatDistribution:
    """
    Convert V3 output to V4 StatDistribution.

    Args:
        v3_output: V3 output dictionary

    Returns:
        V4StatDistribution

    Example:
        >>> v3_dict = {'mean_stat': 61.2, 'std_stat': 18.5, ...}
        >>> v4_dist = v3_to_v4_stat_dist(v3_dict)
        >>> v4_dist.p25
        48.7
    """
    mean = v3_output['mean_stat']
    std = v3_output['std_stat']
    median = v3_output['median_stat']

    # Estimate percentiles from Normal distribution (V3 assumption)
    from scipy import stats as scipy_stats
    norm_dist = scipy_stats.norm(loc=mean, scale=std)

    return V4StatDistribution(
        mean=mean,
        median=median,
        std=std,
        p5=float(norm_dist.ppf(0.05)),
        p25=float(norm_dist.ppf(0.25)),
        p75=float(norm_dist.ppf(0.75)),
        p95=float(norm_dist.ppf(0.95)),
        cv=std / mean if mean > 0 else 0.0,
        iqr=float(norm_dist.ppf(0.75) - norm_dist.ppf(0.25))
    )
