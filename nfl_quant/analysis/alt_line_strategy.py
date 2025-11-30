"""
Alt-Line Strategy Engine

Provides intelligent alt-line recommendations based on:
- Bet confidence level
- Bet context (single vs parlay)
- Risk mode (conservative/balanced/aggressive)

Key insight: Alt-lines have context-dependent value
- Singles: High confidence → Aggressive lines (better odds, tighter spread)
- Parlays: Need cushion → Safer lines (more room for error)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class LineType(Enum):
    """Classification of alt-line aggressiveness."""
    VERY_AGGRESSIVE = "very_aggressive"  # Tightest line, best odds, highest risk
    AGGRESSIVE = "aggressive"             # Tighter than main
    MAIN = "main"                         # Market main line
    SAFE = "safe"                         # Wider than main, more cushion
    VERY_SAFE = "very_safe"              # Maximum cushion


class BetContext(Enum):
    """Context in which bet will be placed."""
    SINGLE = "single"
    PARLAY = "parlay"


@dataclass
class AltLine:
    """Represents an alternative line with its properties."""
    line: float
    prob_over: float
    prob_under: float
    odds_over: Optional[int] = None
    odds_under: Optional[int] = None
    ev_over: Optional[float] = None
    ev_under: Optional[float] = None
    delta_from_main: float = 0.0


@dataclass
class LineRecommendation:
    """Recommendation for which line to use in a given context."""
    recommended_line: float
    line_type: LineType
    original_main_line: float
    reasoning: str
    cushion_points: float  # Positive = safer, Negative = more aggressive
    ev_main: Optional[float]
    ev_recommended: Optional[float]
    ev_delta: Optional[float]  # Change in EV from using alt-line
    confidence: float
    edge: float
    bet_context: BetContext
    risk_mode: str


class AltLineStrategy:
    """
    Intelligent alt-line selection based on confidence, context, and risk tolerance.

    Strategy Matrix:

    SINGLES:
    - High Confidence (>85%) + High Edge (>10%) → AGGRESSIVE line (better odds)
    - Medium Confidence (70-85%) → MAIN line
    - Lower Confidence (<70%) → SAFE line (more cushion)

    PARLAYS:
    - High Confidence (>85%) + High Edge (>10%) → SAFE line (some cushion)
    - Medium Confidence (70-85%) → SAFE line (significant cushion)
    - Lower Confidence (<70%) → VERY_SAFE line or exclude

    Risk Mode Adjustments:
    - Conservative: Shift 1 level safer
    - Balanced: Use base strategy
    - Aggressive: Allow 1 level more aggressive (but parlays still safer than singles)
    """

    def __init__(self):
        """Initialize strategy engine."""
        # Confidence thresholds
        self.HIGH_CONFIDENCE = 0.85
        self.MEDIUM_CONFIDENCE = 0.70
        self.LOW_CONFIDENCE = 0.55

        # Edge thresholds
        self.HIGH_EDGE = 0.10
        self.MEDIUM_EDGE = 0.05
        self.LOW_EDGE = 0.02

        # Risk mode shift values (how many levels to shift)
        self.RISK_SHIFTS = {
            'conservative': +1,   # Shift safer
            'balanced': 0,        # No shift
            'aggressive': -1      # Shift more aggressive
        }

    def recommend_line_for_context(
        self,
        main_line: float,
        alt_lines: List[AltLine],
        confidence: float,
        edge: float,
        bet_context: str,  # 'single' or 'parlay'
        risk_mode: str = 'balanced',
        pick_direction: str = 'over',  # 'over' or 'under'
        main_line_ev: Optional[float] = None
    ) -> LineRecommendation:
        """
        Recommend the optimal line for given context.

        Args:
            main_line: Market main line value
            alt_lines: List of alternative lines available
            confidence: Model confidence (our_prob)
            edge: Model edge over market
            bet_context: 'single' or 'parlay'
            risk_mode: 'conservative', 'balanced', or 'aggressive'
            pick_direction: 'over' or 'under'
            main_line_ev: EV of main line

        Returns:
            LineRecommendation with optimal line and reasoning
        """
        context_enum = BetContext.SINGLE if bet_context == 'single' else BetContext.PARLAY

        # Step 1: Determine base line type from confidence/edge
        base_line_type = self._get_base_line_type(
            confidence=confidence,
            edge=edge,
            context=context_enum
        )

        # Step 2: Apply risk mode adjustment
        adjusted_line_type = self._apply_risk_mode_shift(
            base_line_type=base_line_type,
            risk_mode=risk_mode
        )

        # Step 3: Find the best matching alt-line
        selected_line, ev_recommended = self._select_alt_line(
            main_line=main_line,
            alt_lines=alt_lines,
            target_line_type=adjusted_line_type,
            pick_direction=pick_direction
        )

        # Step 4: Calculate metrics
        cushion_points = selected_line - main_line
        ev_delta = None
        if main_line_ev is not None and ev_recommended is not None:
            ev_delta = ev_recommended - main_line_ev

        # Step 5: Generate reasoning
        reasoning = self._generate_reasoning(
            confidence=confidence,
            edge=edge,
            context=context_enum,
            line_type=adjusted_line_type,
            cushion_points=cushion_points,
            risk_mode=risk_mode
        )

        return LineRecommendation(
            recommended_line=selected_line,
            line_type=adjusted_line_type,
            original_main_line=main_line,
            reasoning=reasoning,
            cushion_points=cushion_points,
            ev_main=main_line_ev,
            ev_recommended=ev_recommended,
            ev_delta=ev_delta,
            confidence=confidence,
            edge=edge,
            bet_context=context_enum,
            risk_mode=risk_mode
        )

    def _get_base_line_type(
        self,
        confidence: float,
        edge: float,
        context: BetContext
    ) -> LineType:
        """Determine base line type before risk mode adjustment."""

        if context == BetContext.SINGLE:
            # Singles: Optimize for edge when confident
            if confidence >= self.HIGH_CONFIDENCE and edge >= self.HIGH_EDGE:
                return LineType.AGGRESSIVE
            elif confidence >= self.MEDIUM_CONFIDENCE and edge >= self.MEDIUM_EDGE:
                return LineType.MAIN
            elif confidence >= self.LOW_CONFIDENCE:
                return LineType.SAFE
            else:
                return LineType.VERY_SAFE

        else:  # PARLAY
            # Parlays: Prioritize certainty (all legs must hit)
            if confidence >= self.HIGH_CONFIDENCE and edge >= self.HIGH_EDGE:
                return LineType.SAFE  # Even high confidence needs cushion
            elif confidence >= self.MEDIUM_CONFIDENCE:
                return LineType.SAFE
            elif confidence >= self.LOW_CONFIDENCE and edge >= self.MEDIUM_EDGE:
                return LineType.VERY_SAFE
            else:
                # Too risky for parlay - recommend exclusion
                return LineType.VERY_SAFE

    def _apply_risk_mode_shift(
        self,
        base_line_type: LineType,
        risk_mode: str
    ) -> LineType:
        """Apply risk mode adjustment to base line type."""

        # Define line type ordering (from aggressive to safe)
        line_order = [
            LineType.VERY_AGGRESSIVE,
            LineType.AGGRESSIVE,
            LineType.MAIN,
            LineType.SAFE,
            LineType.VERY_SAFE
        ]

        current_index = line_order.index(base_line_type)
        shift = self.RISK_SHIFTS.get(risk_mode, 0)

        # Apply shift (clamped to valid range)
        new_index = max(0, min(len(line_order) - 1, current_index + shift))

        return line_order[new_index]

    def _select_alt_line(
        self,
        main_line: float,
        alt_lines: List[AltLine],
        target_line_type: LineType,
        pick_direction: str
    ) -> tuple[float, Optional[float]]:
        """
        Select the alt-line that best matches target line type.

        Returns:
            (selected_line_value, ev_of_selected_line)
        """

        # If no alt-lines, return main line
        if not alt_lines:
            return main_line, None

        # Define target delta based on line type
        # For OVER bets: Lower line = aggressive, Higher line = safe
        # For UNDER bets: Lower line = aggressive, Higher line = safe
        # Example: Under 51.5
        #   - Aggressive: Under 45.5 (delta -6, harder to hit, better odds)
        #   - Safe: Under 57.5 (delta +6, easier to hit, worse odds)
        target_delta_map = {
            LineType.VERY_AGGRESSIVE: -5.0,  # Always move DOWN (harder)
            LineType.AGGRESSIVE: -2.5,        # Move DOWN (harder)
            LineType.MAIN: 0.0,
            LineType.SAFE: +2.5,              # Move UP (easier)
            LineType.VERY_SAFE: +5.0          # Always move UP (easier)
        }

        target_delta = target_delta_map[target_line_type]

        # Find closest alt-line to target delta
        best_line = None
        best_distance = float('inf')
        best_ev = None

        for alt_line in alt_lines:
            delta_distance = abs(alt_line.delta_from_main - target_delta)

            if delta_distance < best_distance:
                best_distance = delta_distance
                best_line = alt_line.line

                # Get EV for the direction we're betting
                if pick_direction == 'over':
                    best_ev = alt_line.ev_over
                else:
                    best_ev = alt_line.ev_under

        # Fallback to main line if no good match
        if best_line is None:
            return main_line, None

        return best_line, best_ev

    def _generate_reasoning(
        self,
        confidence: float,
        edge: float,
        context: BetContext,
        line_type: LineType,
        cushion_points: float,
        risk_mode: str
    ) -> str:
        """Generate human-readable reasoning for line recommendation."""

        # Confidence description
        if confidence >= self.HIGH_CONFIDENCE:
            conf_desc = "high confidence"
        elif confidence >= self.MEDIUM_CONFIDENCE:
            conf_desc = "medium confidence"
        else:
            conf_desc = "lower confidence"

        # Edge description
        if edge >= self.HIGH_EDGE:
            edge_desc = "strong edge"
        elif edge >= self.MEDIUM_EDGE:
            edge_desc = "moderate edge"
        else:
            edge_desc = "thin edge"

        # Context reasoning
        if context == BetContext.SINGLE:
            context_reason = "single bet can take more risk"
        else:
            context_reason = "parlay needs all legs to hit - prioritize certainty"

        # Line type reasoning
        if line_type == LineType.AGGRESSIVE:
            line_reason = "Using aggressive line for better odds"
        elif line_type == LineType.VERY_AGGRESSIVE:
            line_reason = "Using very aggressive line to maximize edge"
        elif line_type == LineType.SAFE:
            line_reason = "Using safer line for cushion"
        elif line_type == LineType.VERY_SAFE:
            line_reason = "Using maximum cushion for protection"
        else:
            line_reason = "Using main line as optimal balance"

        # Cushion description
        cushion_desc = f"{abs(cushion_points):.1f}pt cushion" if cushion_points != 0 else "no adjustment"

        reasoning = (
            f"{conf_desc.capitalize()} ({confidence:.1%}), {edge_desc} ({edge:.1%}). "
            f"{context_reason}. {line_reason} ({cushion_desc})"
        )

        return reasoning

    def evaluate_parlay_legs(
        self,
        legs: List[Dict[str, Any]],
        risk_mode: str = 'balanced'
    ) -> Dict[str, Any]:
        """
        Evaluate all legs in a parlay and recommend optimal alt-lines.

        Args:
            legs: List of parlay legs with confidence, edge, main_line, alt_lines
            risk_mode: Risk mode setting

        Returns:
            Dictionary with:
            - recommendations: List of line recommendations per leg
            - total_cushion: Sum of all cushion points
            - total_ev_delta: Change in EV from using alt-lines
            - hit_rate_boost: Estimated hit rate improvement (simplified)
        """

        recommendations = []
        total_cushion = 0.0
        total_ev_delta = 0.0

        for leg in legs:
            rec = self.recommend_line_for_context(
                main_line=leg['main_line'],
                alt_lines=leg.get('alt_lines', []),
                confidence=leg['confidence'],
                edge=leg['edge'],
                bet_context='parlay',
                risk_mode=risk_mode,
                pick_direction=leg.get('pick_direction', 'over'),
                main_line_ev=leg.get('main_line_ev')
            )

            recommendations.append(rec)
            total_cushion += rec.cushion_points

            if rec.ev_delta is not None:
                total_ev_delta += rec.ev_delta

        # Estimate hit rate boost (simplified model)
        # Each point of cushion ≈ 2-3% hit rate improvement
        estimated_hit_rate_boost = total_cushion * 0.025

        return {
            'recommendations': recommendations,
            'total_cushion': total_cushion,
            'total_ev_delta': total_ev_delta,
            'hit_rate_boost': estimated_hit_rate_boost,
            'num_legs': len(legs)
        }
