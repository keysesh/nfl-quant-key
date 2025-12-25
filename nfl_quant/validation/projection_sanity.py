"""
Projection Sanity Checks

Flags suspicious projections that are likely due to recency bias,
EWMA over-weighting outliers, or other model artifacts.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ProjectionFlags:
    """Results of projection sanity check."""
    projection: float
    trailing: float
    season_avg: float
    line: float
    flags: List[str]
    is_suspicious: bool
    recommendation: str


def validate_projection(
    projected: float,
    trailing: float,
    season_avg: float,
    line: float,
    market: str = None,
    using_line_as_baseline: bool = False,
) -> ProjectionFlags:
    """
    Flag suspicious projections.

    Args:
        projected: Model's projection
        trailing: 4-week trailing average
        season_avg: Season average
        line: Vegas line
        market: Market type (optional, for context)

    Returns:
        ProjectionFlags with any warnings
    """
    flags = []

    # Guard against division by zero
    if season_avg <= 0:
        season_avg = max(trailing, 1)
    if trailing <= 0:
        trailing = max(season_avg, 1)

    # Check 1: Projection vs season average
    pct_above_season = (projected - season_avg) / season_avg
    if pct_above_season > 0.30:  # 30% above season avg
        flags.append(f"Projection {pct_above_season:.0%} above season avg ({season_avg:.1f})")
    elif pct_above_season < -0.30:  # 30% below season avg
        flags.append(f"Projection {pct_above_season:.0%} below season avg ({season_avg:.1f})")

    # Check 2: Projection vs trailing average
    pct_above_trailing = (projected - trailing) / trailing
    if pct_above_trailing > 0.25:  # 25% above trailing
        flags.append(f"Projection {pct_above_trailing:.0%} above trailing ({trailing:.1f})")
    elif pct_above_trailing < -0.25:  # 25% below trailing
        flags.append(f"Projection {pct_above_trailing:.0%} below trailing ({trailing:.1f})")

    # Check 3: Projection vs line (extreme divergence)
    pct_vs_line = (projected - line) / line
    if pct_vs_line > 0.20:  # 20% above line
        flags.append(f"Projection {projected:.1f} >> line {line} ({pct_vs_line:.0%})")
    elif pct_vs_line < -0.20:  # 20% below line
        flags.append(f"Projection {projected:.1f} << line {line} ({pct_vs_line:.0%})")

    # Check 4: Trailing vs season divergence (indicates regime change or noise)
    trailing_vs_season = (trailing - season_avg) / season_avg
    if abs(trailing_vs_season) > 0.20:
        flags.append(f"Trailing ({trailing:.1f}) diverges from season ({season_avg:.1f}) by {trailing_vs_season:.0%}")

    # Determine if suspicious
    is_suspicious = len(flags) >= 2  # 2+ flags = suspicious

    # Generate recommendation
    if len(flags) >= 3:
        recommendation = "REJECT - Multiple sanity flags triggered"
    elif len(flags) == 2:
        recommendation = "CAUTION - Projection may be unreliable"
    elif len(flags) == 1:
        recommendation = "REVIEW - Minor flag, verify manually"
    else:
        recommendation = "OK - Projection within normal bounds"

    return ProjectionFlags(
        projection=projected,
        trailing=trailing,
        season_avg=season_avg,
        line=line,
        flags=flags,
        is_suspicious=is_suspicious,
        recommendation=recommendation,
    )


def check_recency_bias(
    recent_values: List[float],
    projection: float,
    weights: List[float] = None,
) -> Tuple[bool, str]:
    """
    Check if projection is overly influenced by a single recent outlier.

    Args:
        recent_values: Last N game values (most recent first)
        projection: Model's projection
        weights: EWMA weights (optional)

    Returns:
        (has_bias, reason)
    """
    if len(recent_values) < 3:
        return False, "Insufficient data"

    # Default EWMA-like weights if not provided
    if weights is None:
        weights = [0.4, 0.27, 0.18, 0.12][:len(recent_values)]

    # Calculate simple mean (unweighted)
    simple_mean = sum(recent_values) / len(recent_values)

    # Check if most recent value is an outlier
    most_recent = recent_values[0]
    other_mean = sum(recent_values[1:]) / len(recent_values[1:])

    # If most recent is >50% above/below the rest, and projection tracks it
    if other_mean > 0:
        recent_deviation = (most_recent - other_mean) / other_mean
        projection_vs_other = (projection - other_mean) / other_mean

        # Check if projection is tracking the outlier
        if abs(recent_deviation) > 0.40 and abs(projection_vs_other) > 0.25:
            if (recent_deviation > 0 and projection_vs_other > 0) or \
               (recent_deviation < 0 and projection_vs_other < 0):
                return True, f"Most recent ({most_recent:.1f}) is outlier vs others ({other_mean:.1f}), projection tracking it"

    return False, "No recency bias detected"


def format_sanity_report(flags: ProjectionFlags, player: str = None) -> str:
    """Format sanity check results for display."""
    lines = []

    header = f"PROJECTION SANITY CHECK"
    if player:
        header += f" - {player}"
    lines.append("=" * 60)
    lines.append(header)
    lines.append("=" * 60)

    lines.append(f"Projection: {flags.projection:.1f}")
    lines.append(f"Trailing:   {flags.trailing:.1f}")
    lines.append(f"Season Avg: {flags.season_avg:.1f}")
    lines.append(f"Line:       {flags.line:.1f}")
    lines.append("")

    if flags.flags:
        lines.append("⚠️  FLAGS:")
        for flag in flags.flags:
            lines.append(f"  • {flag}")
        lines.append("")

    status = "🚨 SUSPICIOUS" if flags.is_suspicious else "✅ OK"
    lines.append(f"Status: {status}")
    lines.append(f"Recommendation: {flags.recommendation}")
    lines.append("=" * 60)

    return "\n".join(lines)
