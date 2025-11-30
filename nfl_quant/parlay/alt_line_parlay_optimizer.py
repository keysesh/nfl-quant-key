"""
Alt-Line Parlay Optimizer

Integrates AltLineStrategy with parlay generation to create optimized parlays
that use safer alt-lines to improve hit rates while preserving acceptable EV.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from nfl_quant.analysis.alt_line_strategy import AltLineStrategy, LineRecommendation, BetContext
    from nfl_quant.analysis.alt_lines import AltLine
    from .recommendation import SingleBet, ParlayRecommendation, ParlayRecommender
except ImportError:
    # Fallback for direct execution
    from nfl_quant.analysis.alt_line_strategy import AltLineStrategy, LineRecommendation, BetContext
    from nfl_quant.analysis.alt_lines import AltLine
    from nfl_quant.parlay.recommendation import SingleBet, ParlayRecommendation, ParlayRecommender


@dataclass
class AltLineBet:
    """Bet with alt-line recommendations."""
    original_bet: SingleBet
    main_line: float
    alt_lines: List[AltLine]
    parlay_recommendation: Optional[LineRecommendation] = None
    single_recommendation: Optional[LineRecommendation] = None
    pick_direction: str = 'over'  # 'over' or 'under'


@dataclass
class OptimizedParlayRecommendation(ParlayRecommendation):
    """Parlay recommendation with alt-line optimization details."""
    leg_line_recommendations: List[LineRecommendation] = field(default_factory=list)
    total_cushion: float = 0.0
    ev_impact: Optional[float] = None
    hit_rate_boost: Optional[float] = None
    optimization_summary: str = ""


class AltLineParlayOptimizer:
    """
    Optimizes parlay combinations using intelligent alt-line selection.

    Key features:
    1. Applies safer lines to parlay legs (more cushion)
    2. Recommends aggressive lines for high-confidence singles
    3. Respects risk mode settings
    4. Tracks EV impact vs hit rate improvement
    """

    def __init__(
        self,
        risk_mode: str = 'balanced',
        correlation_threshold: float = 0.70,
        max_legs: int = 4
    ):
        """
        Initialize optimizer.

        Args:
            risk_mode: 'conservative', 'balanced', or 'aggressive'
            correlation_threshold: Maximum allowed correlation
            max_legs: Maximum legs per parlay
        """
        self.risk_mode = risk_mode
        self.strategy = AltLineStrategy()
        self.parlay_recommender = ParlayRecommender(
            correlation_threshold=correlation_threshold,
            max_legs=max_legs
        )

    def optimize_bet_for_context(
        self,
        bet: AltLineBet,
        context: str  # 'single' or 'parlay'
    ) -> LineRecommendation:
        """
        Get optimal line recommendation for bet in given context.

        Args:
            bet: Bet with alt-line data
            context: 'single' or 'parlay'

        Returns:
            LineRecommendation with optimal line for context
        """

        # Get confidence and edge from original bet
        confidence = bet.original_bet.our_prob or 0.5
        edge = bet.original_bet.edge or 0.0

        # Get main line EV (if available)
        main_line_ev = None
        if bet.alt_lines:
            for alt in bet.alt_lines:
                if abs(alt.line - bet.main_line) < 0.01:  # Found main line
                    main_line_ev = alt.ev_over if bet.pick_direction == 'over' else alt.ev_under
                    break

        # Get recommendation from strategy
        recommendation = self.strategy.recommend_line_for_context(
            main_line=bet.main_line,
            alt_lines=bet.alt_lines,
            confidence=confidence,
            edge=edge,
            bet_context=context,
            risk_mode=self.risk_mode,
            pick_direction=bet.pick_direction,
            main_line_ev=main_line_ev
        )

        return recommendation

    def generate_optimized_parlays(
        self,
        bets_with_alt_lines: List[AltLineBet],
        num_parlays: int = 10
    ) -> List[OptimizedParlayRecommendation]:
        """
        Generate parlays using optimized alt-lines for each leg.

        Args:
            bets_with_alt_lines: List of bets with their alt-line data
            num_parlays: Number of parlays to generate

        Returns:
            List of optimized parlay recommendations
        """

        # Step 1: Get line recommendations for each bet in parlay context
        for bet in bets_with_alt_lines:
            bet.parlay_recommendation = self.optimize_bet_for_context(bet, 'parlay')
            bet.single_recommendation = self.optimize_bet_for_context(bet, 'single')

        # Step 2: Filter to bets suitable for parlays
        # (exclude very low confidence or those that don't improve with safer lines)
        parlay_suitable = [
            bet for bet in bets_with_alt_lines
            if bet.parlay_recommendation is not None
            and bet.original_bet.our_prob and bet.original_bet.our_prob >= 0.55
        ]

        if len(parlay_suitable) < 2:
            return []

        # Step 3: Generate standard parlays using original bets
        single_bets = [bet.original_bet for bet in parlay_suitable]
        base_parlays = self.parlay_recommender.generate_parlays(
            single_bets=single_bets,
            num_parlays=num_parlays * 2  # Generate more, then optimize
        )

        # Step 4: Enhance parlays with alt-line optimization
        optimized_parlays = []
        for parlay in base_parlays:
            # Find the alt-line data for each leg
            leg_recommendations = []
            total_cushion = 0.0
            total_ev_delta = 0.0

            for leg in parlay.legs:
                # Find matching bet in our alt-line data
                matching_bet = None
                for bet in parlay_suitable:
                    if bet.original_bet.name == leg.name:
                        matching_bet = bet
                        break

                if matching_bet and matching_bet.parlay_recommendation:
                    leg_recommendations.append(matching_bet.parlay_recommendation)
                    total_cushion += matching_bet.parlay_recommendation.cushion_points

                    if matching_bet.parlay_recommendation.ev_delta is not None:
                        total_ev_delta += matching_bet.parlay_recommendation.ev_delta

            # Calculate metrics
            hit_rate_boost = total_cushion * 0.025  # Rough estimate: 2.5% per point

            # Create optimization summary
            optimization_summary = self._create_optimization_summary(
                leg_recommendations=leg_recommendations,
                total_cushion=total_cushion,
                hit_rate_boost=hit_rate_boost
            )

            # Create optimized recommendation
            optimized = OptimizedParlayRecommendation(
                legs=parlay.legs,
                true_odds=parlay.true_odds,
                model_odds=parlay.model_odds,
                true_prob=parlay.true_prob,
                model_prob=parlay.model_prob,
                edge=parlay.edge,
                correlation_valid=parlay.correlation_valid,
                correlation_issues=parlay.correlation_issues,
                recommended_stake=parlay.recommended_stake,
                potential_win=parlay.potential_win,
                expected_value=parlay.expected_value,
                leg_line_recommendations=leg_recommendations,
                total_cushion=total_cushion,
                ev_impact=total_ev_delta,
                hit_rate_boost=hit_rate_boost,
                optimization_summary=optimization_summary
            )

            optimized_parlays.append(optimized)

        # Step 5: Sort by adjusted EV (considering cushion benefit)
        # Parlays with better cushion should rank higher if EV is close
        optimized_parlays.sort(
            key=lambda p: p.edge + (p.hit_rate_boost or 0) * 0.5,
            reverse=True
        )

        return optimized_parlays[:num_parlays]

    def _create_optimization_summary(
        self,
        leg_recommendations: List[LineRecommendation],
        total_cushion: float,
        hit_rate_boost: Optional[float]
    ) -> str:
        """Create human-readable summary of parlay optimization."""

        num_legs = len(leg_recommendations)

        if not leg_recommendations:
            return "No optimization data available"

        # Count line type distribution
        line_types = [rec.line_type.value for rec in leg_recommendations]
        aggressive = line_types.count('aggressive') + line_types.count('very_aggressive')
        main = line_types.count('main')
        safe = line_types.count('safe') + line_types.count('very_safe')

        summary_parts = []

        summary_parts.append(f"{num_legs}-leg parlay")

        if safe > 0:
            summary_parts.append(f"{safe} safer lines")
        if main > 0:
            summary_parts.append(f"{main} main lines")
        if aggressive > 0:
            summary_parts.append(f"{aggressive} aggressive lines")

        summary_parts.append(f"{total_cushion:+.1f}pt total cushion")

        if hit_rate_boost:
            summary_parts.append(f"~{hit_rate_boost:.1%} hit rate boost")

        return " | ".join(summary_parts)

    def get_single_bet_recommendations(
        self,
        bets_with_alt_lines: List[AltLineBet]
    ) -> List[Dict[str, Any]]:
        """
        Get optimized single bet recommendations.

        Args:
            bets_with_alt_lines: List of bets with alt-line data

        Returns:
            List of dictionaries with single bet recommendations
        """

        recommendations = []

        for bet in bets_with_alt_lines:
            single_rec = self.optimize_bet_for_context(bet, 'single')
            parlay_rec = self.optimize_bet_for_context(bet, 'parlay')

            recommendations.append({
                'bet': bet.original_bet,
                'main_line': bet.main_line,
                'single_recommended_line': single_rec.recommended_line,
                'single_line_type': single_rec.line_type.value,
                'single_reasoning': single_rec.reasoning,
                'single_cushion': single_rec.cushion_points,
                'parlay_recommended_line': parlay_rec.recommended_line,
                'parlay_line_type': parlay_rec.line_type.value,
                'parlay_reasoning': parlay_rec.reasoning,
                'parlay_cushion': parlay_rec.cushion_points,
                'pick_direction': bet.pick_direction
            })

        return recommendations


def create_alt_line_bet_from_recommendation(
    recommendation: Dict[str, Any],
    alt_lines: List[AltLine]
) -> AltLineBet:
    """
    Helper function to create AltLineBet from recommendation dictionary.

    Args:
        recommendation: Dictionary with bet data
        alt_lines: List of alternative lines

    Returns:
        AltLineBet object
    """

    # Extract pick direction from pick string
    pick_str = str(recommendation.get('pick', ''))
    pick_direction = 'over' if 'Over' in pick_str or '+' in pick_str else 'under'

    # Extract main line from pick
    import re
    line_match = re.search(r'[\+\-]?(\d+\.?\d*)', pick_str)
    main_line = float(line_match.group(1)) if line_match else 50.0

    # Create SingleBet
    single_bet = SingleBet(
        name=recommendation.get('pick', ''),
        bet_type=recommendation.get('bet_type', ''),
        game=recommendation.get('game', ''),
        team=recommendation.get('team'),
        player=recommendation.get('player'),
        market=recommendation.get('market'),
        odds=recommendation.get('market_odds'),
        our_prob=recommendation.get('our_prob'),
        market_prob=recommendation.get('market_prob'),
        edge=recommendation.get('edge'),
        bet_size=recommendation.get('bet_size'),
        potential_profit=recommendation.get('potential_profit')
    )

    return AltLineBet(
        original_bet=single_bet,
        main_line=main_line,
        alt_lines=alt_lines,
        pick_direction=pick_direction
    )
