"""
V4 Narrative Generator - Upside/Downside Pathway Analysis.

Uses V4 percentile outputs to generate human-readable betting narratives.

Key Features:
- Upside pathways: What needs to happen for 75th/95th percentile outcomes
- Downside risks: Scenarios leading to 5th/25th percentile outcomes
- Volatility analysis: How wide is the range of outcomes?
- Conditional probabilities: "If X happens, then Y% chance of Z"
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from nfl_quant.v4_output import V4StatDistribution, PlayerPropOutputV4


@dataclass
class UpsidePathway:
    """Description of an upside scenario."""
    percentile: int  # 75 or 95
    value: float  # Yards/receptions at this percentile
    probability: float  # Probability of reaching this
    conditions: List[str]  # What needs to happen
    narrative: str  # Human-readable explanation


@dataclass
class DownsideRisk:
    """Description of a downside scenario."""
    percentile: int  # 5 or 25
    value: float  # Yards/receptions at this percentile
    probability: float  # Probability of this or worse
    causes: List[str]  # What could cause this
    narrative: str  # Human-readable explanation


@dataclass
class VolatilityAnalysis:
    """Analysis of outcome range and volatility."""
    cv: float  # Coefficient of variation
    iqr: float  # Interquartile range
    range_p5_p95: float  # 90% outcome range
    volatility_tier: str  # LOW, MEDIUM, HIGH
    narrative: str  # Human-readable explanation


def generate_upside_pathways(
    stat_dist: V4StatDistribution,
    player_name: str,
    stat_name: str,
    position: str
) -> List[UpsidePathway]:
    """
    Generate upside pathway narratives from V4 distribution.

    Args:
        stat_dist: V4 statistical distribution
        player_name: Player name
        stat_name: Stat name (e.g., 'receiving_yards')
        position: Player position

    Returns:
        List of upside pathways (75th and 95th percentiles)

    Example:
        >>> pathways = generate_upside_pathways(stat_dist, 'George Kittle', 'receiving_yards', 'TE')
        >>> print(pathways[0].narrative)
        "75th percentile (74.5 yards): Kittle gets 8+ targets in competitive game..."
    """
    pathways = []

    # 75th percentile pathway
    p75_conditions = _get_p75_conditions(stat_name, stat_dist, position)
    p75_narrative = _create_p75_narrative(
        player_name, stat_name, stat_dist.p75, p75_conditions
    )

    pathways.append(UpsidePathway(
        percentile=75,
        value=stat_dist.p75,
        probability=0.25,  # 25% chance of exceeding p75
        conditions=p75_conditions,
        narrative=p75_narrative
    ))

    # 95th percentile pathway (boom scenario)
    p95_conditions = _get_p95_conditions(stat_name, stat_dist, position)
    p95_narrative = _create_p95_narrative(
        player_name, stat_name, stat_dist.p95, p95_conditions
    )

    pathways.append(UpsidePathway(
        percentile=95,
        value=stat_dist.p95,
        probability=0.05,  # 5% chance of exceeding p95
        conditions=p95_conditions,
        narrative=p95_narrative
    ))

    return pathways


def generate_downside_risks(
    stat_dist: V4StatDistribution,
    player_name: str,
    stat_name: str,
    position: str
) -> List[DownsideRisk]:
    """
    Generate downside risk narratives from V4 distribution.

    Args:
        stat_dist: V4 statistical distribution
        player_name: Player name
        stat_name: Stat name
        position: Player position

    Returns:
        List of downside risks (5th and 25th percentiles)
    """
    risks = []

    # 25th percentile risk
    p25_causes = _get_p25_causes(stat_name, stat_dist, position)
    p25_narrative = _create_p25_narrative(
        player_name, stat_name, stat_dist.p25, p25_causes
    )

    risks.append(DownsideRisk(
        percentile=25,
        value=stat_dist.p25,
        probability=0.25,  # 25% chance of p25 or worse
        causes=p25_causes,
        narrative=p25_narrative
    ))

    # 5th percentile risk (bust scenario)
    p5_causes = _get_p5_causes(stat_name, stat_dist, position)
    p5_narrative = _create_p5_narrative(
        player_name, stat_name, stat_dist.p5, p5_causes
    )

    risks.append(DownsideRisk(
        percentile=5,
        value=stat_dist.p5,
        probability=0.05,  # 5% chance of p5 or worse
        causes=p5_causes,
        narrative=p5_narrative
    ))

    return risks


def generate_volatility_analysis(
    stat_dist: V4StatDistribution,
    stat_name: str
) -> VolatilityAnalysis:
    """
    Analyze volatility and outcome range.

    Args:
        stat_dist: V4 statistical distribution
        stat_name: Stat name

    Returns:
        VolatilityAnalysis with tier and narrative
    """
    # Determine volatility tier
    if stat_dist.cv < 0.35:
        tier = "LOW"
        tier_desc = "narrow range of outcomes, predictable"
    elif stat_dist.cv < 0.55:
        tier = "MEDIUM"
        tier_desc = "moderate variance, typical NFL player"
    else:
        tier = "HIGH"
        tier_desc = "wide range of outcomes, boom/bust potential"

    # Calculate range
    range_90 = stat_dist.p95 - stat_dist.p5

    # Create narrative
    narrative = (
        f"{tier} VOLATILITY (CV={stat_dist.cv:.2f}): {tier_desc}. "
        f"90% of outcomes fall between {stat_dist.p5:.1f} and {stat_dist.p95:.1f} "
        f"({stat_name.replace('_', ' ')}), a range of {range_90:.1f}. "
        f"Interquartile range (50% of outcomes): {stat_dist.iqr:.1f}."
    )

    return VolatilityAnalysis(
        cv=stat_dist.cv,
        iqr=stat_dist.iqr,
        range_p5_p95=range_90,
        volatility_tier=tier,
        narrative=narrative
    )


def generate_full_narrative(
    output: PlayerPropOutputV4,
    stat_name: str,
    prop_line: Optional[float] = None
) -> str:
    """
    Generate complete betting narrative for a player stat.

    Args:
        output: V4 player output
        stat_name: Stat to analyze (e.g., 'receiving_yards')
        prop_line: Optional sportsbook line

    Returns:
        Full narrative string

    Example:
        >>> narrative = generate_full_narrative(output, 'receiving_yards', 45.5)
        >>> print(narrative)
        '''
        George Kittle (TE) - Receiving Yards
        Line: 45.5 yards

        PROJECTION:
        Mean: 58.0 yards
        Median: 51.6 yards
        Range: 15.3 - 123.0 yards (5th-95th percentile)

        VOLATILITY: HIGH (CV=0.40)
        Wide range of outcomes, boom/bust potential...

        UPSIDE PATHWAYS:
        75th percentile (74.5 yards): Kittle gets 8+ targets...
        95th percentile (123.0 yards): Big play game with 60+ yard TD...

        DOWNSIDE RISKS:
        25th percentile (34.1 yards): Game script turns negative...
        5th percentile (15.3 yards): Blowout or injury concerns...

        RECOMMENDATION: OVER 45.5 ✅ (78% probability)
        '''
    """
    stat_dist = getattr(output, stat_name, None)
    if stat_dist is None:
        return f"No {stat_name} data available for {output.player_name}"

    # Header
    lines = [
        f"{output.player_name} ({output.position}) - {stat_name.replace('_', ' ').title()}",
        "=" * 60
    ]

    if prop_line is not None:
        lines.append(f"Line: {prop_line:.1f} {stat_name.replace('_', ' ')}")
        lines.append("")

    # Projection
    lines.append("PROJECTION:")
    lines.append(f"  Mean: {stat_dist.mean:.1f}")
    lines.append(f"  Median: {stat_dist.median:.1f}")
    lines.append(f"  25th-75th percentile: {stat_dist.p25:.1f} - {stat_dist.p75:.1f}")
    lines.append(f"  5th-95th percentile: {stat_dist.p5:.1f} - {stat_dist.p95:.1f}")
    lines.append("")

    # Volatility
    vol_analysis = generate_volatility_analysis(stat_dist, stat_name)
    lines.append(vol_analysis.narrative)
    lines.append("")

    # Upside pathways
    upside = generate_upside_pathways(
        stat_dist, output.player_name, stat_name, output.position
    )
    lines.append("UPSIDE PATHWAYS:")
    for pathway in upside:
        lines.append(f"  {pathway.percentile}th percentile ({pathway.value:.1f}): {pathway.narrative}")
    lines.append("")

    # Downside risks
    downside = generate_downside_risks(
        stat_dist, output.player_name, stat_name, output.position
    )
    lines.append("DOWNSIDE RISKS:")
    for risk in downside:
        lines.append(f"  {risk.percentile}th percentile ({risk.value:.1f}): {risk.narrative}")
    lines.append("")

    # Recommendation (if line provided)
    if prop_line is not None:
        prob_over = _calculate_prob_over(stat_dist, prop_line)
        recommendation = "OVER" if prob_over > 0.55 else "UNDER"
        symbol = "✅" if prob_over > 0.55 else "❌"

        lines.append(f"RECOMMENDATION: {recommendation} {prop_line:.1f} {symbol}")
        lines.append(f"  Probability: {prob_over:.1%}")
        lines.append(f"  Edge: {abs(prob_over - 0.5):.1%}")

    return "\n".join(lines)


# Helper functions

def _get_p75_conditions(stat_name: str, stat_dist: V4StatDistribution, position: str) -> List[str]:
    """Get conditions needed for 75th percentile outcome."""
    if 'receiving' in stat_name or 'targets' in stat_name:
        return [
            "Game stays competitive (within 1 score)",
            f"Player gets {_estimate_targets_for_p75(stat_dist, position):.0f}+ targets",
            "No early injury or benching",
            "Offensive game script favorable"
        ]
    elif 'rushing' in stat_name:
        return [
            "Team leads or game stays close",
            f"Player gets {_estimate_carries_for_p75(stat_dist):.0f}+ carries",
            "Good run blocking performance",
            "Goal line opportunities"
        ]
    else:
        return ["Above-average performance", "Favorable game conditions"]


def _get_p95_conditions(stat_name: str, stat_dist: V4StatDistribution, position: str) -> List[str]:
    """Get conditions needed for 95th percentile outcome (boom)."""
    if 'receiving' in stat_name:
        return [
            "Big play opportunity (40+ yard gain)",
            f"High target volume ({_estimate_targets_for_p95(stat_dist, position):.0f}+ targets)",
            "Touchdown catch",
            "Defensive breakdown or busted coverage"
        ]
    elif 'rushing' in stat_name:
        return [
            "Breakaway run (50+ yards)",
            "Multiple red zone carries",
            "Touchdown run",
            "Dominant game script (large lead)"
        ]
    else:
        return ["Elite performance", "Multiple scoring opportunities"]


def _get_p25_causes(stat_name: str, stat_dist: V4StatDistribution, position: str) -> List[str]:
    """Get causes of 25th percentile outcome."""
    if 'receiving' in stat_name:
        return [
            "Game script turns negative (blowout)",
            f"Low target volume (<{_estimate_targets_for_p25(stat_dist, position):.0f} targets)",
            "Poor QB performance or weather",
            "Defensive scheme limits opportunities"
        ]
    elif 'rushing' in stat_name:
        return [
            "Team trails early (pass-heavy script)",
            "Poor offensive line performance",
            "Goal line opportunities go to another player",
            "Limited carries due to game flow"
        ]
    else:
        return ["Below-average performance", "Unfavorable conditions"]


def _get_p5_causes(stat_name: str, stat_dist: V4StatDistribution, position: str) -> List[str]:
    """Get causes of 5th percentile outcome (bust)."""
    return [
        "Early injury or illness",
        "Extreme blowout (benched in 2nd half)",
        "Worst-case game script",
        "Complete defensive shutdown",
        "Minimal playing time"
    ]


def _create_p75_narrative(player_name: str, stat_name: str, value: float, conditions: List[str]) -> str:
    """Create narrative for 75th percentile."""
    return f"{player_name} reaches {value:.1f} if: {conditions[0].lower()}, {conditions[1].lower()}"


def _create_p95_narrative(player_name: str, stat_name: str, value: float, conditions: List[str]) -> str:
    """Create narrative for 95th percentile."""
    return f"Boom scenario: {conditions[0]}, {conditions[1].lower()}"


def _create_p25_narrative(player_name: str, stat_name: str, value: float, causes: List[str]) -> str:
    """Create narrative for 25th percentile."""
    return f"Risk: {causes[0].lower()} or {causes[1].lower()}"


def _create_p5_narrative(player_name: str, stat_name: str, value: float, causes: List[str]) -> str:
    """Create narrative for 5th percentile."""
    return f"Bust scenario: {causes[0].lower()} or {causes[3].lower()}"


def _estimate_targets_for_p75(stat_dist: V4StatDistribution, position: str) -> float:
    """Estimate targets needed for 75th percentile yards."""
    # Rough heuristic: p75 yards / typical Y/T
    typical_ypt = {'WR': 9.0, 'TE': 8.5, 'RB': 7.0}.get(position, 8.5)
    return stat_dist.p75 / typical_ypt


def _estimate_targets_for_p95(stat_dist: V4StatDistribution, position: str) -> float:
    """Estimate targets needed for 95th percentile yards."""
    typical_ypt = {'WR': 11.0, 'TE': 10.0, 'RB': 8.5}.get(position, 10.0)  # Higher Y/T for boom
    return stat_dist.p95 / typical_ypt


def _estimate_targets_for_p25(stat_dist: V4StatDistribution, position: str) -> float:
    """Estimate targets for 25th percentile yards."""
    typical_ypt = {'WR': 7.5, 'TE': 7.0, 'RB': 6.0}.get(position, 7.0)  # Lower Y/T for bust
    return stat_dist.p25 / typical_ypt


def _estimate_carries_for_p75(stat_dist: V4StatDistribution) -> float:
    """Estimate carries needed for 75th percentile yards."""
    typical_ypc = 4.5
    return stat_dist.p75 / typical_ypc


def _calculate_prob_over(stat_dist: V4StatDistribution, line: float) -> float:
    """
    Calculate probability of exceeding line from V4 distribution.

    Uses percentiles to interpolate probability.
    """
    # Find where line falls in distribution
    if line <= stat_dist.p5:
        return 0.95
    elif line <= stat_dist.p25:
        # Interpolate between p5 and p25
        frac = (line - stat_dist.p5) / (stat_dist.p25 - stat_dist.p5)
        return 0.95 - frac * 0.20  # 95% to 75%
    elif line <= stat_dist.median:
        # Interpolate between p25 and median
        frac = (line - stat_dist.p25) / (stat_dist.median - stat_dist.p25)
        return 0.75 - frac * 0.25  # 75% to 50%
    elif line <= stat_dist.p75:
        # Interpolate between median and p75
        frac = (line - stat_dist.median) / (stat_dist.p75 - stat_dist.median)
        return 0.50 - frac * 0.25  # 50% to 25%
    elif line <= stat_dist.p95:
        # Interpolate between p75 and p95
        frac = (line - stat_dist.p75) / (stat_dist.p95 - stat_dist.p75)
        return 0.25 - frac * 0.20  # 25% to 5%
    else:
        return 0.05
